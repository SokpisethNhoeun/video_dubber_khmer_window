from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import uuid
from pathlib import Path
from threading import Event
from typing import Callable

from PyQt6.QtCore import QLocale, QSize, QThread, QTimer, QUrl, pyqtSlot, Qt
from PyQt6.QtGui import QDesktopServices

try:
    from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
    MULTIMEDIA_AVAILABLE = True
except ImportError:
    MULTIMEDIA_AVAILABLE = False

from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QInputDialog,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from config.env import env_path, load_project_env
from config.models import LANGUAGES, STAGES
from core.context import PipelineSettings
from core.draft_queue import DraftQueue, RERUNNABLE_STATUSES, STATUS_COMPLETED, STATUS_QUEUED
from gui.dialogs import ModelDownloadsDialog, SpeakerMappingDialog, TranscriptReviewDialog
from gui.icons import icon as themed_icon
from gui.pages import (
    AudioPage,
    ClonePage,
    ExportPage,
    ImportPage,
    LogsPage,
    SettingsPage,
    SpeakersPage,
    SponsorPage,
    TranslatePage,
    VoicePage,
)
from gui.pages.sessions_page import SessionsPage
from gui.pages.editor_page import EditorPage
from gui.pages.voice_page import FEMALE_EDGE_VOICES, MALE_EDGE_VOICES
from gui.sidebar import Sidebar
from gui.status_bar import StatusBar
from gui.theme import (
    THEME_DARK,
    THEME_LIGHT,
    build_stylesheet,
    get_saved_theme,
    recording_style,
    save_theme,
)
from gui.workers import (
    PipelineWorker,
    PreviewSegmentWorker,
    RedubWorker,
    SetupCheckWorker,
    SpeakerDetectionWorker,
    VideoImportWorker,
)
from modules.audio_quality import prepare_reference_audio, validate_reference_audio
from modules.audio_utils import extract_audio, has_audio_stream, remove_tree
from modules.diarizer import detect_speakers, speaker_ids_from_turns
from modules.transcript_review import parse_srt_text
from modules.video_import import VideoImportService
from modules.voice_profiles import create_voice_profile, delete_voice_profile, list_voice_profiles

FILE_DIALOG_OPTIONS = QFileDialog.Option(0)
DEFAULT_CLONE_COMMAND_SETUP_MESSAGE = (
    "The old clone command uses infer.py, but this project does not include infer.py. "
    "Use the built-in command or replace the Command field with a real installed neural voice conversion tool."
)
ADVANCED_RVC_COMMAND = (
    'python infer.py --input "{input}" --output "{output}" --model "{model}" --index "{index}"'
)
INTERNAL_PER_PERSON_CLONE_BACKENDS = {"xtts", "cosyvoice", "qwen3"}


def _uses_generated_profile_clone(settings) -> bool:
    return (
        getattr(settings, "tts_provider", "edge") != "gemini"
        and (
            settings.voice_female_reference_path is not None
            or settings.voice_male_reference_path is not None
        )
    )


def clone_setup_status(settings) -> str:
    workflow = getattr(settings, "clone_workflow", "auto_per_person")
    per_person = settings.voice_gender in {"per_person", "per_person_auto"}
    generated_profile_clone = _uses_generated_profile_clone(settings)
    emotion_capable = settings.clone_backend in {"xtts", "cosyvoice", "qwen3"}
    emotion_enabled = settings.emotion_aware_clone and emotion_capable
    emotion_note = ""
    if workflow == "auto_per_person" or per_person:
        if emotion_enabled:
            emotion_note = " Emotion matching is ON and will use source clips for segment prosody."
        elif settings.emotion_aware_clone:
            emotion_note = " Emotion matching needs Qwen3-TTS, CosyVoice 2, or XTTS-v2; OpenVoice cannot carry emotion clips."
        else:
            emotion_note = " Emotion matching is OFF."

    if workflow == "gender_profiles" or generated_profile_clone:
        missing = []
        if settings.voice_female_reference_path is None:
            missing.append("female")
        if settings.voice_male_reference_path is None:
            missing.append("male")
        if missing:
            return (
                "Warning: select generated female and male profiles in Voice & TTS Settings "
                f"to avoid same-voice output. Missing: {', '.join(missing)}."
            )
        return (
            "Male/Female clone profiles: the app will first synthesize Khmer TTS, then replace the "
            "voice timbre using separate female and male clone references."
        )

    if not settings.rvc_enabled:
        return (
            "Clone stage is OFF. The app will use the TTS voices from Voice & TTS Settings only; "
            "no post-TTS voice conversion will run."
        )

    if workflow == "auto_per_person" or per_person:
        return (
            "Auto per-person clone: the app will detect speakers and build a separate "
            f"reference from the source video for each person.{emotion_note}"
        )

    if workflow == "single_reference":
        if settings.rvc_reference_audio_path is None:
            return "Single reference clone: select one reference audio file before running."
        return (
            "Warning: single reference clone runs after TTS and uses one voice reference for all speakers, "
            "so male and female voices may sound similar."
        )

    if settings.rvc_enabled and settings.rvc_reference_audio_path is not None:
        return (
            "Warning: one clone reference is selected. Use Auto per-person or Male/Female "
            "profiles when speakers should sound different."
        )
    return ""


def internal_per_person_clone_backend_error(settings) -> str | None:
    if settings.clone_backend == "xtts":
        python_path = os.getenv("OPENVOICE_PYTHON", "").strip()
        if not python_path:
            return "XTTS-v2 requires OPENVOICE_PYTHON to be set."
        if not Path(python_path).expanduser().exists():
            return f"OPENVOICE_PYTHON path does not exist: {python_path}"
    elif settings.clone_backend == "cosyvoice":
        python_path = os.getenv("COSYVOICE_PYTHON", "").strip()
        if not python_path:
            return "CosyVoice 2 requires COSYVOICE_PYTHON to be set."
        if not Path(python_path).expanduser().exists():
            return f"COSYVOICE_PYTHON path does not exist: {python_path}"
    elif settings.clone_backend == "qwen3":
        python_path = os.getenv("QWEN3_TTS_PYTHON", "").strip()
        if not python_path:
            return "Qwen3-TTS 1.7B requires QWEN3_TTS_PYTHON to be set."
        if not Path(python_path).expanduser().exists():
            return f"QWEN3_TTS_PYTHON path does not exist: {python_path}"
        try:
            completed = subprocess.run(
                [str(Path(python_path).expanduser()), "-c", "import shutil; print(shutil.which('sox') or '')"],
                capture_output=True,
                text=True,
                timeout=20,
            )
        except Exception as exc:
            return (
                "Failed to validate QWEN3_TTS_PYTHON environment. Make sure it is a working Python "
                f"and can find SoX on PATH. {exc}"
            )
        if completed.returncode != 0:
            detail = completed.stderr.strip()
            return (
                "Failed to validate QWEN3_TTS_PYTHON environment. Make sure it is a working Python "
                "and can find SoX on PATH."
                + (f" {detail}" if detail else "")
            )
        if not completed.stdout.strip():
            return (
                "Qwen3-TTS 1.7B requires SoX to be installed and available in the "
                "QWEN3_TTS_PYTHON environment."
            )
    return None


class AppWindow(QMainWindow):
    def __init__(self, project_root: Path, keep_temp: bool = False) -> None:
        super().__init__()
        self.project_root = project_root
        self.keep_temp = keep_temp
        self.default_reference_command = self._default_reference_command()
        self.draft_queue = DraftQueue.for_project(project_root)
        self.draft_queue = DraftQueue.load(self.draft_queue.path)
        self.draft_queue.reset_running_to_paused()
        self.thread: QThread | None = None
        self.setup_check_thread: QThread | None = None
        self.setup_check_worker: SetupCheckWorker | None = None
        self.last_output_video: Path | None = None
        self.selected_input_videos: list[Path] = []
        self.recording_process = None
        self.worker: PipelineWorker | None = None
        self.preview_thread: QThread | None = None
        self.preview_worker: PreviewSegmentWorker | None = None
        self.speaker_detection_thread: QThread | None = None
        self.speaker_detection_worker: SpeakerDetectionWorker | None = None
        self.url_import_thread: QThread | None = None
        self.url_import_worker: VideoImportWorker | None = None
        self._url_import_auto_start = False
        self._speaker_detection_videos: list[Path] = []
        self._speaker_detection_index = 0
        self._speaker_detection_settings: PipelineSettings | None = None
        self._speaker_detection_after_complete: Callable[[bool], None] | None = None
        self._speaker_detection_continue_after_thread = False
        self._speaker_detection_final_success: bool | None = None
        self.speaker_voice_mappings: dict[str, dict[str, dict[str, str]]] = {}
        self.diarization_turns: dict[str, list[dict[str, float | str]]] = {}
        self.voice_profiles: list = []
        self._is_script_only = False
        self._syncing_clone_workflow = False

        if MULTIMEDIA_AVAILABLE:
            self.player = QMediaPlayer()
            self.audio_output = QAudioOutput()
            self.player.setAudioOutput(self.audio_output)
        else:
            self.player = None
            self.audio_output = None

        self._theme = get_saved_theme()
        self.setWindowTitle("Khmer Video Dubber")
        self.resize(1280, 820)
        self._build_ui()
        self._apply_styles()
        self._refresh_voice_profiles()
        self._load_settings()
        self.settings_page.gemini_api_key.setText(os.getenv("GEMINI_API_KEY", ""))
        from config.user_secrets import load_user_secrets
        self.settings_page.update_license_display()
        if not os.getenv("GEMINI_API_KEY", "").strip():
            QTimer.singleShot(250, self._show_gemini_setup_notice)
        self._refresh_voice_profiles()
        self._connect_signals()
        self._refresh_draft_queue()
        self._sync_clone_workflow_from_voice_mode()
        self._update_voice_mode_controls()

    # ── Reference command builders ──

    def _default_reference_command(self) -> str:
        if os.getenv("VOICE_CLONE_BACKEND", "").strip().lower() == "local_openvoice":
            return self._openvoice_reference_command()
        return self._built_in_reference_command()

    def _built_in_reference_command(self) -> str:
        python_bin = self.project_root / ".venv" / "bin" / "python"
        executable = python_bin if python_bin.exists() else Path("python")
        return (
            f'{shlex.quote(str(executable))} -m modules.reference_voice_clone '
            '--input "{input}" --output "{output}" --reference "{reference}"'
        )

    def _openvoice_reference_command(self) -> str:
        python_bin = self.project_root / ".venv" / "bin" / "python"
        executable = python_bin if python_bin.exists() else Path("python")
        return (
            f'{shlex.quote(str(executable))} -m modules.openvoice_voice_clone '
            '--input "{input}" --output "{output}" --reference "{reference}"'
        )

    def _production_reference_command(self) -> str:
        python_bin = self.project_root / ".venv" / "bin" / "python"
        executable = python_bin if python_bin.exists() else Path("python")
        return (
            f'{shlex.quote(str(executable))} -m modules.elevenlabs_voice_clone '
            '--input "{input}" --output "{output}" --reference "{reference}"'
        )

    # ── UI Building ──

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("Root")
        main_layout = QVBoxLayout(root)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Header
        header = QWidget()
        header.setObjectName("HeaderBar")
        header.setFixedHeight(52)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(16, 0, 16, 0)

        self.logo_icon = QLabel()
        self.logo_icon.setPixmap(themed_icon("mdi.movie-open-outline").pixmap(QSize(22, 22)))
        header_layout.addWidget(self.logo_icon)

        logo = QLabel("Khmer Video Dubber")
        logo.setObjectName("LogoLabel")
        header_layout.addWidget(logo)
        header_layout.addStretch(1)

        self.downloads_button = QPushButton("Downloads")
        self.downloads_button.setIcon(themed_icon("mdi.download-circle-outline"))
        self.downloads_button.clicked.connect(self._open_model_downloads)
        header_layout.addWidget(self.downloads_button)

        self.theme_toggle_btn = QPushButton()
        self.theme_toggle_btn.setIcon(
            themed_icon("mdi.weather-night" if self._theme == THEME_DARK else "mdi.white-balance-sunny")
        )
        self.theme_toggle_btn.setIconSize(QSize(18, 18))
        self.theme_toggle_btn.setObjectName("ThemeToggle")
        self.theme_toggle_btn.setFixedSize(36, 36)
        self.theme_toggle_btn.clicked.connect(self._toggle_theme)
        header_layout.addWidget(self.theme_toggle_btn)

        main_layout.addWidget(header)

        # Body: Sidebar + Content
        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)

        self.sidebar = Sidebar()
        self.sidebar.page_selected.connect(self._navigate_to)
        body_layout.addWidget(self.sidebar)

        # Stacked pages
        self.page_stack = QStackedWidget()
        self._pages: dict[str, QWidget] = {}

        self.import_page = ImportPage()
        self.voice_page = VoicePage()
        self.translate_page = TranslatePage()
        self.speakers_page = SpeakersPage()
        self.clone_page = ClonePage()
        self.audio_page = AudioPage()
        self.sponsor_page = SponsorPage()
        self.export_page = ExportPage()
        self.sessions_page = SessionsPage(self.project_root)
        self.editor_page = EditorPage(self.project_root)
        self.settings_page = SettingsPage()
        self.logs_page = LogsPage()

        for key, page in [
            ("import", self.import_page),
            ("speakers", self.speakers_page),
            ("voice", self.voice_page),
            ("translate", self.translate_page),
            ("clone", self.clone_page),
            ("audio", self.audio_page),
            ("sponsor", self.sponsor_page),
            ("export", self.export_page),
            ("sessions", self.sessions_page),
            ("editor", self.editor_page),
            ("logs", self.logs_page),
            ("settings", self.settings_page),
        ]:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(page)
            scroll.setObjectName("PageScroll")
            self.page_stack.addWidget(scroll)
            self._pages[key] = scroll

        body_layout.addWidget(self.page_stack, 1)
        main_layout.addWidget(body, 1)

        # Status bar
        self.status_bar = StatusBar()
        main_layout.addWidget(self.status_bar)

        self.setCentralWidget(root)

    def _connect_signals(self) -> None:
        # Import page
        self.import_page.start_requested.connect(lambda: self._start(False))
        self.import_page.generate_script_button.clicked.connect(lambda: self._start(True))
        self.import_page.cancel_button.clicked.connect(self._cancel)
        self.import_page.open_button.clicked.connect(self._open_finished_video)
        self.import_page.files_changed.connect(self._on_files_dropped)
        self.import_page.urls_import_requested.connect(self._start_url_import)
        self.import_page.file_drop.urls_dropped.connect(self._handle_dropped_urls)
        self.import_page.add_to_queue_requested.connect(self._add_selected_to_draft_queue)
        self.import_page.start_queue_requested.connect(self._start_draft_queue)
        self.import_page.pause_after_current_requested.connect(self._pause_queue_after_current)
        self.import_page.remove_draft_requested.connect(self._remove_draft)
        self.import_page.move_draft_requested.connect(self._move_draft)
        self.import_page.open_draft_output_requested.connect(self._open_draft_output)

        # Voice page
        self.voice_page.detect_speakers_button.clicked.connect(self._detect_and_map_speakers)
        self.voice_page.test_female_button.clicked.connect(self._test_female_voice)
        self.voice_page.test_male_button.clicked.connect(self._test_male_voice)
        self.voice_page.voice_gender.currentIndexChanged.connect(lambda: self._update_voice_mode_controls())
        self.voice_page.tts_provider.currentIndexChanged.connect(lambda: self._update_voice_mode_controls())
        self.voice_page.voice_female.currentIndexChanged.connect(lambda: self._on_voice_profile_selection_changed())
        self.voice_page.voice_male.currentIndexChanged.connect(lambda: self._on_voice_profile_selection_changed())
        self.voice_page.voice_female.editTextChanged.connect(lambda: self._on_voice_profile_selection_changed())
        self.voice_page.voice_male.editTextChanged.connect(lambda: self._on_voice_profile_selection_changed())
        self.voice_page.simple_tts_flow.toggled.connect(self._on_simple_tts_flow_toggled)

        # Translate page
        self.translate_page.use_json_button.clicked.connect(self._use_review_json_for_video)
        self.translate_page.use_srt_button.clicked.connect(self._use_review_srt_for_video)
        self.translate_page.glossary_button.clicked.connect(self._manage_glossary)

        # Speakers page
        self.speakers_page.detect_button.clicked.connect(self._detect_and_map_speakers)

        # Clone page
        self.clone_page.record_button.clicked.connect(self._toggle_recording)
        self.clone_page.generate_voice_button.clicked.connect(self._generate_voice_profile)
        self.clone_page.import_voices_button.clicked.connect(self._import_voice_profiles)
        self.clone_page.test_saved_button.clicked.connect(self._test_saved_voice_profile)
        self.clone_page.delete_saved_button.clicked.connect(self._delete_saved_voice_profile)
        self.clone_page.saved_voice_profiles.currentIndexChanged.connect(self._select_saved_voice_profile)
        self.clone_page.workflow_changed.connect(self._on_clone_workflow_changed)
        self.clone_page.rvc_enabled.toggled.connect(self._on_rvc_enabled_toggled)
        self.clone_page.clone_backend.currentIndexChanged.connect(lambda: self._on_clone_backend_changed())
        self.clone_page.rvc_reference_audio.textChanged.connect(lambda: self._refresh_clone_guidance())

        # Export page
        self.export_page.edit_review_button.clicked.connect(self._edit_review_json)
        self.export_page.save_defaults_requested.connect(self._save_settings)

        # Settings page
        self.settings_page.setup_check_button.clicked.connect(self._check_setup)
        self.settings_page.test_gemini_button.clicked.connect(self._save_and_test_gemini_key)
        self.settings_page.activate_license_button.clicked.connect(self._activate_license)
        self.settings_page.open_purchase_wizard_button.clicked.connect(self._open_purchase_wizard)
        self.settings_page.theme_button.clicked.connect(self._toggle_theme)

        # Sessions page
        self.sessions_page.resume_requested.connect(self._resume_session)
        self.sessions_page.edit_requested.connect(self._open_editor_for_session)

        # Editor page
        self.editor_page.redub_requested.connect(self._start_redub)
        self.editor_page.preview_requested.connect(self._preview_editor_segment)

    def _apply_styles(self) -> None:
        self.setStyleSheet(build_stylesheet(self._theme))

    def _navigate_to(self, key: str) -> None:
        if key in self._pages:
            self.page_stack.setCurrentWidget(self._pages[key])
        if key == "sessions":
            self.sessions_page.refresh()
        elif key == "editor":
            self.editor_page.refresh_sessions()

    def _open_model_downloads(self) -> None:
        dialog = getattr(self, "_model_downloads_dialog", None)
        if dialog is None:
            dialog = ModelDownloadsDialog(self)
            self._model_downloads_dialog = dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    # ── Convenience accessors for page widgets used across methods ──

    @property
    def voice_gender(self):
        return self.voice_page.voice_gender

    @property
    def voice_female(self):
        return self.voice_page.voice_female

    @property
    def voice_male(self):
        return self.voice_page.voice_male

    @property
    def speech_rate(self):
        return self.voice_page.speech_rate

    @property
    def pitch_hz(self):
        return self.voice_page.pitch_hz

    @property
    def detect_speakers_button(self):
        return self.voice_page.detect_speakers_button

    @property
    def review_json_path(self):
        return self.translate_page.review_json_path

    @property
    def rvc_enabled(self):
        return self.clone_page.rvc_enabled

    @property
    def rvc_reference_audio(self):
        return self.clone_page.rvc_reference_audio

    @property
    def voice_profile_name(self):
        return self.clone_page.voice_profile_name

    @property
    def voice_profile_gender(self):
        return self.clone_page.voice_profile_gender

    @property
    def saved_voice_profiles(self):
        return self.clone_page.saved_voice_profiles

    @property
    def rvc_command(self):
        return self.clone_page.rvc_command

    @property
    def record_button(self):
        return self.clone_page.record_button

    @property
    def generate_voice_button(self):
        return self.clone_page.generate_voice_button

    @property
    def import_voices_button(self):
        return self.clone_page.import_voices_button

    @property
    def test_saved_voice_button(self):
        return self.clone_page.test_saved_button

    @property
    def setup_check_button(self):
        return self.settings_page.setup_check_button

    @property
    def start_button(self):
        return self.import_page.start_button

    @property
    def cancel_button(self):
        return self.import_page.cancel_button

    @property
    def open_video_button(self):
        return self.import_page.open_button

    @property
    def generate_script_button(self):
        return self.import_page.generate_script_button

    @property
    def output_folder(self):
        return self.import_page.output_folder

    @property
    def log_console(self):
        return self.logs_page.log_console

    # ── Theme ──

    def _toggle_theme(self) -> None:
        self._theme = THEME_LIGHT if self._theme == THEME_DARK else THEME_DARK
        save_theme(self._theme)
        self._apply_styles()
        self.theme_toggle_btn.setIcon(
            themed_icon("mdi.weather-night" if self._theme == THEME_DARK else "mdi.white-balance-sunny")
        )
        self.sidebar.refresh_icons()
        self.logo_icon.setPixmap(themed_icon("mdi.movie-open-outline").pixmap(QSize(22, 22)))

    # ── File handling ──

    def _handle_dropped_urls(self, urls: list[str]) -> None:
        self.import_page.append_urls(urls)

    def _on_files_dropped(self, paths: list) -> None:
        self.selected_input_videos = [Path(p) for p in paths]
        if paths:
            self.import_page.video_preview.set_video(Path(paths[0]))
        else:
            self.import_page.video_preview.clear()

    # ── Draft queue ──

    def _reload_draft_queue(self) -> None:
        self.draft_queue = DraftQueue.load(self.draft_queue.path)

    def _refresh_draft_queue(self) -> None:
        self._reload_draft_queue()
        self.import_page.set_queue_jobs(self.draft_queue.jobs)
        running = self.worker is not None and self.worker.draft_queue_path is not None
        self.import_page.set_queue_running(running)

    def _source_urls_for_videos(self, videos: list[Path]) -> dict[Path, str]:
        return {
            Path(video): VideoImportService.source_url_for_video(Path(video))
            for video in videos
        }

    def _add_selected_to_draft_queue(self) -> bool:
        settings = self._settings_from_ui()
        error = self._validate_settings(settings)
        if error:
            QMessageBox.warning(self, "Invalid settings", error)
            return False
        videos = settings.input_videos or [settings.input_video]
        self._reload_draft_queue()
        jobs = self.draft_queue.add_jobs(
            settings,
            videos,
            self._source_urls_for_videos(videos),
        )
        self._append_log(f"Added {len(jobs)} video(s) to Draft Queue.")
        self._refresh_draft_queue()
        return bool(jobs)

    def _start_draft_queue(self) -> None:
        if self.thread is not None:
            QMessageBox.information(self, "Draft Queue", "A dubbing job is already running.")
            return

        self._reload_draft_queue()
        if not self.draft_queue.has_runnable():
            if not self._add_selected_to_draft_queue():
                return
            self._reload_draft_queue()
        if not self.draft_queue.has_runnable():
            QMessageBox.information(self, "Draft Queue", "Add at least one queued draft first.")
            return

        self._apply_current_voice_settings_to_runnable_drafts()
        self._reload_draft_queue()
        settings = self.draft_queue.next_runnable().settings
        error = self._validate_settings(settings)
        if error:
            QMessageBox.warning(self, "Invalid settings", error)
            return

        self.logs_page.reset_progress()
        self.log_console.clear()
        self._navigate_to("logs")
        self.sidebar.select("logs")
        self._start_draft_queue_pipeline(settings)

    def _apply_current_voice_settings_to_runnable_drafts(self) -> None:
        current = self._settings_from_ui()
        voice_setting_fields = (
            "tts_provider",
            "voice_gender",
            "voice_female",
            "voice_male",
            "voice_female_reference_path",
            "voice_male_reference_path",
            "speech_rate",
            "pitch_hz",
            "rvc_enabled",
            "rvc_reference_audio_path",
            "rvc_clone_gender",
            "rvc_command_template",
            "clone_workflow",
            "clone_backend",
            "emotion_aware_clone",
            "emotion_clone_mode",
            "enable_clone_verification",
        )
        changed = 0
        for job in self.draft_queue.jobs:
            if job.status not in RERUNNABLE_STATUSES:
                continue
            job_changed = False
            for field_name in voice_setting_fields:
                new_value = getattr(current, field_name)
                if getattr(job.settings, field_name) == new_value:
                    continue
                setattr(job.settings, field_name, new_value)
                job_changed = True
            if not job_changed:
                continue
            changed += 1
        if changed:
            self.draft_queue.save()
            self._append_log("Updated queued draft(s) to use the current Voice & TTS settings.")

    def _start_draft_queue_pipeline(self, settings: PipelineSettings) -> None:
        self._append_log("Starting Draft Queue")
        thread = QThread()
        worker = PipelineWorker(settings, self.project_root, draft_queue_path=self.draft_queue.path)
        self._attach_pipeline_worker(thread, worker)
        self.start_button.setEnabled(False)
        self.generate_script_button.setEnabled(False)
        self.import_page.start_queue_button.setEnabled(False)
        self.import_page.pause_after_current_button.setEnabled(True)
        self.cancel_button.setEnabled(True)
        self.cancel_button.show()
        self.open_video_button.setEnabled(False)
        self.open_video_button.hide()
        self.last_output_video = None
        self.thread.start()

    def _pause_queue_after_current(self) -> None:
        if self.worker is not None and self.worker.draft_queue_path is not None:
            self.worker.pause_after_current()
            self.import_page.pause_after_current_button.setEnabled(False)
            self._append_log("Draft Queue will pause after the current video finishes.")

    def _remove_draft(self, draft_id: str) -> None:
        self._reload_draft_queue()
        job = self.draft_queue.get(draft_id)
        if job is not None and job.status == "running":
            QMessageBox.warning(self, "Draft Queue", "Cannot remove the draft that is currently running.")
            return
        self.draft_queue.remove(draft_id)
        self._refresh_draft_queue()

    def _move_draft(self, draft_id: str, offset: int) -> None:
        self._reload_draft_queue()
        job = self.draft_queue.get(draft_id)
        if job is not None and job.status != STATUS_QUEUED:
            return
        self.draft_queue.move(draft_id, offset)
        self._refresh_draft_queue()

    def _open_draft_output(self, draft_id: str) -> None:
        self._reload_draft_queue()
        job = self.draft_queue.get(draft_id)
        if job is None or job.status != STATUS_COMPLETED or not job.output_path:
            QMessageBox.information(self, "Draft Queue", "This draft has no completed output yet.")
            return
        self.last_output_video = job.output_path
        self._open_finished_video()

    # ── Voice mode controls ──

    def _update_voice_mode_controls(self) -> None:
        voice_mode = str(self.voice_gender.currentData() or "")
        self.voice_page.set_voice_mode(voice_mode)
        is_per_speaker = voice_mode == "per_speaker_auto"
        self.clone_page.set_tts_only_preferred(voice_mode in {"female", "male", "auto", "per_speaker_auto"})
        is_manual = voice_mode == "per_person"
        is_auto = voice_mode == "per_person_auto"
        self.detect_speakers_button.setEnabled(is_manual and self.speaker_detection_thread is None)
        if is_per_speaker:
            self.rvc_enabled.setChecked(False)
        if (is_manual or is_auto) and not self.rvc_command.text().strip():
            self.rvc_command.setText(self.default_reference_command)
        if not self._syncing_clone_workflow:
            self._sync_clone_workflow_from_voice_mode()
        self._refresh_clone_guidance()

    def _on_simple_tts_flow_toggled(self, enabled: bool) -> None:
        if not enabled:
            return
        auto_index = self.voice_gender.findData("auto")
        if auto_index >= 0 and self.voice_gender.currentIndex() != auto_index:
            self.voice_gender.setCurrentIndex(auto_index)
        self.rvc_enabled.setChecked(False)
        self.clone_page.set_tts_only_preferred(True)
        self._refresh_clone_guidance()

    def _on_rvc_enabled_toggled(self, enabled: bool) -> None:
        if enabled and self.voice_page.simple_tts_flow.isChecked():
            self.voice_page.simple_tts_flow.setChecked(False)
        self._refresh_clone_guidance()

    def _sync_clone_workflow_from_voice_mode(self) -> None:
        if not hasattr(self, "clone_page"):
            return
        voice_mode = self.voice_gender.currentData()
        if voice_mode in {"per_person", "per_person_auto"}:
            workflow = "auto_per_person"
            self._prefer_emotion_clone_backend()
            self.clone_page.emotion_aware_check.setChecked(True)
        elif self._has_generated_profile_selection(self.voice_female) or self._has_generated_profile_selection(self.voice_male):
            workflow = "gender_profiles"
        else:
            workflow = "single_reference"
        index = self.clone_page.clone_workflow.findData(workflow)
        if index >= 0 and self.clone_page.clone_workflow.currentIndex() != index:
            self._syncing_clone_workflow = True
            try:
                self.clone_page.clone_workflow.blockSignals(True)
                self.clone_page.clone_workflow.setCurrentIndex(index)
                self.clone_page.clone_workflow.blockSignals(False)
                self.clone_page._on_workflow_changed()
            finally:
                self._syncing_clone_workflow = False

    def _on_clone_workflow_changed(self, workflow: str) -> None:
        if self._syncing_clone_workflow:
            return
        if self.voice_page.simple_tts_flow.isChecked():
            self.voice_page.simple_tts_flow.setChecked(False)
        if workflow == "auto_per_person":
            self.rvc_enabled.setChecked(True)
            index = self.voice_gender.findData("per_person_auto")
            self._prefer_emotion_clone_backend()
            self.clone_page.emotion_aware_check.setChecked(True)
        elif workflow == "gender_profiles":
            self.rvc_enabled.setChecked(True)
            index = self.voice_gender.findData("auto")
        else:
            index = self.voice_gender.findData("auto") if self.voice_gender.currentData() in {"per_person", "per_person_auto"} else -1
        if index >= 0 and self.voice_gender.currentIndex() != index:
            self._syncing_clone_workflow = True
            try:
                self.voice_gender.setCurrentIndex(index)
            finally:
                self._syncing_clone_workflow = False
        if workflow in {"auto_per_person", "gender_profiles"} and not self.rvc_command.text().strip():
            self.rvc_command.setText(self.default_reference_command)
        self._refresh_clone_guidance()

    def _has_generated_profile_selection(self, combo) -> bool:
        data = combo.currentData()
        if isinstance(data, dict) and data.get("kind") == "profile" and bool(data.get("reference")):
            return True
        return self._profile_from_combo_text(combo.currentText()) is not None

    def _profile_from_combo_text(self, text: str):
        def normalize(value: str) -> str:
            value = re.sub(r"\s*\(generated\)\s*$", "", value.strip(), flags=re.IGNORECASE)
            return re.sub(r"[^a-z0-9]+", "", value.lower())

        label = text.strip()
        if not label:
            return None
        label_key = normalize(label)
        if not label_key:
            return None
        profiles = list(self.voice_profiles)
        for refresh in (False, True):
            for profile in profiles:
                profile_keys = {
                    normalize(profile.name),
                    normalize(profile.slug),
                    normalize(str(profile.reference_audio_path)),
                    normalize(f"{profile.name} (generated)"),
                }
                if label_key in profile_keys:
                    return profile
            if refresh:
                break
            profiles = list_voice_profiles(self.project_root)
            self.voice_profiles = profiles
        return None

    def _profile_reference_from_text(self, text: str, gender: str) -> Path | None:
        profile = self._profile_from_combo_text(text)
        if profile is not None and str(profile.gender).lower() == gender:
            return profile.reference_audio_path
        return None

    def _generated_profile_reference_from_combo(self, combo, gender: str) -> Path | None:
        data = combo.currentData()
        if isinstance(data, dict):
            reference = str(data.get("reference", "")).strip()
            data_gender = str(data.get("gender", "")).lower()
            if reference and data_gender == gender:
                return Path(reference).expanduser()
        return self._profile_reference_from_text(combo.currentText(), gender)

    def _prefer_emotion_clone_backend(self) -> None:
        current_backend = self.clone_page.clone_backend.currentData()
        if current_backend in {"xtts", "cosyvoice", "qwen3"}:
            return
        preferred = None
        qwen3_python = os.getenv("QWEN3_TTS_PYTHON", "").strip()
        if qwen3_python and Path(qwen3_python).expanduser().exists():
            preferred = "qwen3"
        else:
            cosyvoice_python = os.getenv("COSYVOICE_PYTHON", "").strip()
            if cosyvoice_python and Path(cosyvoice_python).expanduser().exists():
                preferred = "cosyvoice"
            else:
                xtts_python = os.getenv("OPENVOICE_PYTHON", "").strip()
                if xtts_python and Path(xtts_python).expanduser().exists():
                    preferred = "xtts"
        if preferred:
            index = self.clone_page.clone_backend.findData(preferred)
            if index >= 0:
                self.clone_page.clone_backend.setCurrentIndex(index)

    def _on_clone_backend_changed(self) -> None:
        if self.clone_page.clone_workflow.currentData() == "auto_per_person":
            backend = self.clone_page.clone_backend.currentData()
            if backend in {"xtts", "cosyvoice", "qwen3"}:
                self.clone_page.emotion_aware_check.setChecked(True)
        self._refresh_clone_guidance()

    def _on_voice_profile_selection_changed(self) -> None:
        if not self._syncing_clone_workflow:
            self._sync_clone_workflow_from_voice_mode()
        self._refresh_clone_guidance()

    def _refresh_clone_guidance(self) -> None:
        if not hasattr(self, "clone_page"):
            return
        try:
            status = clone_setup_status(self._settings_from_ui())
        except Exception:
            status = ""
        self.clone_page.clone_quality_status.setText(status)

    # ── Voice profiles ──

    def _refresh_voice_profiles(self, select_path: Path | None = None) -> None:
        self.voice_profiles = list_voice_profiles(self.project_root)
        selected_text = str(select_path) if select_path else ""
        selected_profile_gender = ""
        for profile in self.voice_profiles:
            if selected_text and str(profile.reference_audio_path) == selected_text:
                selected_profile_gender = profile.gender
                break

        self.saved_voice_profiles.blockSignals(True)
        self.saved_voice_profiles.clear()
        self.saved_voice_profiles.addItem("Custom reference audio", "")
        selected_index = 0
        for profile in self.voice_profiles:
            label = f"{profile.name} ({profile.duration:.0f}s)"
            path_text = str(profile.reference_audio_path)
            self.saved_voice_profiles.addItem(label, path_text)
            if selected_text and path_text == selected_text:
                selected_index = self.saved_voice_profiles.count() - 1
        self.saved_voice_profiles.setCurrentIndex(selected_index)
        self.saved_voice_profiles.blockSignals(False)

        self._refresh_gender_voice_combo(
            self.voice_female, "female", FEMALE_EDGE_VOICES,
            select_path if selected_profile_gender == "female" else None,
        )
        self._refresh_gender_voice_combo(
            self.voice_male, "male", MALE_EDGE_VOICES,
            select_path if selected_profile_gender == "male" else None,
        )

    def _refresh_gender_voice_combo(self, combo, gender, edge_voices, select_path=None):
        current_data = combo.currentData()
        current_reference = ""
        if isinstance(current_data, dict) and current_data.get("kind") == "profile":
            current_reference = str(current_data.get("reference", ""))
        current_text = combo.currentText()
        target_reference = str(select_path) if select_path else current_reference

        combo.blockSignals(True)
        combo.clear()
        for voice in edge_voices:
            combo.addItem(voice, {"kind": "edge", "voice": voice})
        for profile in self.voice_profiles:
            if profile.gender == gender:
                combo.addItem(
                    f"{profile.name} (generated)",
                    {"kind": "profile", "voice": edge_voices[0],
                     "reference": str(profile.reference_audio_path),
                     "name": profile.name, "gender": gender},
                )
        selected_index = 0
        for index in range(combo.count()):
            data = combo.itemData(index)
            if isinstance(data, dict) and target_reference and data.get("reference") == target_reference:
                selected_index = index
                break
            if not target_reference and combo.itemText(index) == current_text:
                selected_index = index
        combo.setCurrentIndex(selected_index)
        combo.blockSignals(False)

    def _voice_selection(self, combo, fallback_voice):
        profile = self._profile_from_combo_text(combo.currentText())
        if profile is not None:
            return fallback_voice, profile.reference_audio_path
        data = combo.currentData()
        if isinstance(data, dict):
            if data.get("kind") == "profile":
                reference = str(data.get("reference", "")).strip()
                return str(data.get("voice") or fallback_voice), Path(reference).expanduser() if reference else None
            if data.get("kind") == "edge":
                return str(data.get("voice") or combo.currentText()), None
        return combo.currentText().strip() or fallback_voice, None

    def _select_saved_voice_profile(self, _index: int = 0) -> None:
        reference_path = self.saved_voice_profiles.currentData()
        if not reference_path:
            return
        if hasattr(self.clone_page, "profile_reference_audio"):
            self.clone_page.profile_reference_audio.setText(str(reference_path))
        for profile in self.voice_profiles:
            if str(profile.reference_audio_path) == str(reference_path):
                gender_index = self.voice_profile_gender.findData(profile.gender)
                if gender_index >= 0:
                    self.voice_profile_gender.setCurrentIndex(gender_index)
                combo = self.voice_female if profile.gender == "female" else self.voice_male
                self._refresh_gender_voice_combo(
                    combo, profile.gender,
                    FEMALE_EDGE_VOICES if profile.gender == "female" else MALE_EDGE_VOICES,
                    profile.reference_audio_path,
                )
                break

    def _profile_by_reference(self, reference_path):
        reference = Path(reference_path).expanduser()
        for profile in self.voice_profiles:
            if profile.reference_audio_path == reference or profile.reference_audio_path.resolve() == reference.resolve():
                return profile
        return None

    def _test_saved_voice_profile(self) -> None:
        reference_path = self.saved_voice_profiles.currentData()
        if not reference_path:
            QMessageBox.warning(self, "Test Saved Voice", "Select a saved voice first.")
            return
        profile = self._profile_by_reference(reference_path)
        fallback_voice = FEMALE_EDGE_VOICES[0]
        if profile is not None and getattr(profile, "gender", "") == "male":
            fallback_voice = MALE_EDGE_VOICES[0]
        self._test_voice_by_name(fallback_voice, self.test_saved_voice_button, Path(reference_path).expanduser())

    def _delete_saved_voice_profile(self) -> None:
        reference_path = self.saved_voice_profiles.currentData()
        if not reference_path:
            QMessageBox.warning(self, "Delete Voice", "Select a saved voice first.")
            return
        profile = self._profile_by_reference(reference_path)
        profile_name = getattr(profile, "name", "this voice")
        answer = QMessageBox.question(
            self, "Delete Voice",
            f"Delete saved voice '{profile_name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            deleted = delete_voice_profile(self.project_root, Path(reference_path).expanduser())
        except Exception as exc:
            QMessageBox.warning(self, "Delete Voice", str(exc))
            return
        self._append_log(f"Deleted voice profile '{deleted.name}'")
        if hasattr(self.clone_page, "profile_reference_audio"):
            self.clone_page.profile_reference_audio.clear()
        self._refresh_voice_profiles()

    def _generate_voice_profile(self) -> None:
        name = self.voice_profile_name.text().strip()
        reference_text = self.clone_page.profile_reference_audio.text().strip()
        if not name:
            QMessageBox.warning(self, "Generate Voice", "Enter a name for this voice.")
            return
        if not reference_text:
            QMessageBox.warning(self, "Generate Voice", "Select an MP3 or WAV reference audio file first.")
            return
        self.generate_voice_button.setEnabled(False)
        self.generate_voice_button.setText("Generating...")
        QApplication.processEvents()
        try:
            profile = create_voice_profile(
                self.project_root, name, Path(reference_text).expanduser(),
                self.voice_profile_gender.currentData(),
                self._settings_from_ui().min_reference_seconds,
            )
        except Exception as exc:
            QMessageBox.warning(self, "Generate Voice", str(exc))
            return
        finally:
            self.generate_voice_button.setEnabled(True)
            self.generate_voice_button.setText("Generate Voice")
        self._refresh_voice_profiles(profile.reference_audio_path)
        self.clone_page.profile_reference_audio.setText(str(profile.reference_audio_path))
        self._append_log(f"Generated voice profile '{profile.name}': {profile.reference_audio_path}")
        QMessageBox.information(self, "Generate Voice", f"Voice '{profile.name}' is ready.")

    def _unique_voice_profile_name(self, base_name: str) -> str:
        existing = {profile.name.lower() for profile in list_voice_profiles(self.project_root)}
        name = base_name.strip() or "Voice"
        if name.lower() not in existing:
            return name
        for number in range(2, 1000):
            candidate = f"{name} {number}"
            if candidate.lower() not in existing:
                return candidate
        return f"{name}_{uuid.uuid4().hex[:6]}"

    def _import_voice_profiles(self) -> None:
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Import voice reference audio files", str(Path.home()),
            "Audio files (*.mp3 *.wav);;All files (*)", options=FILE_DIALOG_OPTIONS,
        )
        if not file_paths:
            return
        gender = self.voice_profile_gender.currentData()
        self.import_voices_button.setEnabled(False)
        self.import_voices_button.setText("Importing...")
        QApplication.processEvents()
        imported, failed = [], []
        last_profile = None
        for file_path in file_paths:
            source_path = Path(file_path).expanduser()
            base_name = source_path.stem.replace("_", " ").replace("-", " ").title()
            name = self._unique_voice_profile_name(base_name)
            try:
                profile = create_voice_profile(
                    self.project_root, name, source_path, gender,
                    self._settings_from_ui().min_reference_seconds,
                )
                imported.append(profile.name)
                last_profile = profile
                self._append_log(f"Imported voice profile '{profile.name}'")
            except Exception as exc:
                failed.append(f"{source_path.name}: {exc}")
        self.import_voices_button.setEnabled(True)
        self.import_voices_button.setText("Import Voices")
        self._refresh_voice_profiles(last_profile.reference_audio_path if last_profile else None)
        if last_profile:
            self.clone_page.profile_reference_audio.setText(str(last_profile.reference_audio_path))
        message = f"Imported {len(imported)} voice profile(s)."
        if failed:
            message += "\n\nFailed:\n" + "\n".join(failed[:8])
        QMessageBox.information(self, "Import Voices", message)

    # ── Clone command helpers ──

    def _clone_command_setup_error(self, command_template: str) -> str | None:
        try:
            parts = shlex.split(command_template)
        except ValueError as exc:
            return f"Invalid voice clone command: {exc}"
        if "modules.elevenlabs_voice_clone" in parts and not os.getenv("ELEVENLABS_API_KEY", "").strip():
            return "Production voice clone requires ELEVENLABS_API_KEY."
        if "modules.openvoice_voice_clone" in parts:
            openvoice_python = os.getenv("OPENVOICE_PYTHON", "").strip()
            if not openvoice_python:
                return "OpenVoice clone requires OPENVOICE_PYTHON to be set."
            if not Path(openvoice_python).expanduser().exists():
                return f"OPENVOICE_PYTHON does not exist: {openvoice_python}"
            checkpoint_dir = os.getenv("OPENVOICE_CHECKPOINT_DIR", "").strip()
            if not checkpoint_dir:
                return "OpenVoice clone requires OPENVOICE_CHECKPOINT_DIR."
            if not Path(checkpoint_dir).expanduser().exists():
                return f"OPENVOICE_CHECKPOINT_DIR does not exist: {checkpoint_dir}"
        if len(parts) >= 2 and Path(parts[1]).name == "infer.py":
            infer_path = Path(parts[1]).expanduser()
            if not infer_path.is_absolute():
                infer_path = self.project_root / infer_path
            if not infer_path.exists():
                return DEFAULT_CLONE_COMMAND_SETUP_MESSAGE
        return None

    def _clone_command_input_error(self, command_template, reference_path=None, require_reference=False):
        setup_error = self._clone_command_setup_error(command_template)
        if setup_error:
            return setup_error
        if "{reference}" in command_template and require_reference:
            selected = reference_path
            if selected is None and self.rvc_reference_audio.text().strip():
                selected = Path(self.rvc_reference_audio.text()).expanduser()
            if selected is None or not selected.exists():
                return "This clone command uses {reference}; select an existing MP3/WAV reference."
        if reference_path is not None:
            if not reference_path.exists():
                return f"Reference voice audio is missing: {reference_path.name}"
            if reference_path.suffix.lower() not in {".mp3", ".wav"}:
                return "Reference voice audio must be an MP3 or WAV file."
        return None

    def _run_clone_command(self, rendered_command: str) -> None:
        command = shlex.split(rendered_command)
        result = subprocess.run(
            command, cwd=str(self.project_root), capture_output=True, text=True, check=False,
        )
        if result.returncode != 0:
            output = (result.stderr or result.stdout or "").strip()
            if len(output) > 1200:
                output = output[-1200:]
            raise RuntimeError(f"Voice clone command exited with status {result.returncode}.\n{output}")

    # ── Voice testing ──

    def _test_female_voice(self) -> None:
        voice, ref = self._voice_selection(self.voice_female, FEMALE_EDGE_VOICES[0])
        self._test_voice_by_name(voice, self.voice_page.test_female_button, ref)

    def _test_male_voice(self) -> None:
        voice, ref = self._voice_selection(self.voice_male, MALE_EDGE_VOICES[0])
        self._test_voice_by_name(voice, self.voice_page.test_male_button, ref)

    def _test_voice_by_name(self, voice, button_widget, reference_path=None):
        if not MULTIMEDIA_AVAILABLE:
            QMessageBox.warning(self, "Feature Unavailable", "Test Voice requires PyQt6.QtMultimedia.")
            return
        if reference_path is not None:
            if not reference_path.exists():
                QMessageBox.warning(self, "Test Voice", "Generated voice reference file is missing.")
                return
            if not self.rvc_command.text().strip():
                self.rvc_command.setText(self.default_reference_command)
            if "{reference}" not in self.rvc_command.text():
                QMessageBox.warning(self, "Test Voice", "Clone command must contain {reference}.")
                return

        rate_val = self.speech_rate.value()
        pitch_val = self.pitch_hz.value()
        rate_str = f"{'+' if rate_val >= 0 else ''}{rate_val}%"
        pitch_str = f"{'+' if pitch_val >= 0 else ''}{pitch_val}Hz"
        test_text = "សួស្តី នេះគឺជាសំឡេងសាកល្បងរបស់កម្មវិធីសម្រួលសំឡេង។"
        temp_dir = self.project_root / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        test_mp3 = temp_dir / "test_voice.mp3"
        cloned_wav = temp_dir / "test_voice_cloned.wav"

        self.current_test_button = button_widget
        self.current_test_button_label = button_widget.text()
        self.current_test_audio_path = cloned_wav if reference_path is not None else test_mp3
        button_widget.setEnabled(False)
        button_widget.setText("Synthesizing...")

        import threading

        def run_synthesis():
            cmd = [
                str(self.project_root / ".venv" / "bin" / "edge-tts"),
                "--voice", voice, "--text", test_text,
                "--rate", rate_str, "--pitch", pitch_str, "--write-media", str(test_mp3),
            ]
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if reference_path is not None:
                    rendered = self.rvc_command.text().strip().format(
                        input=str(test_mp3), output=str(cloned_wav),
                        model="", index="", reference=str(reference_path),
                    )
                    self._run_clone_command(rendered)
                from PyQt6.QtCore import QMetaObject
                QMetaObject.invokeMethod(self, "_play_test_audio", Qt.ConnectionType.QueuedConnection)
            except Exception as exc:
                self.current_test_failure_message = f"Failed: {exc}"
                from PyQt6.QtCore import QMetaObject
                QMetaObject.invokeMethod(self, "_reset_test_button", Qt.ConnectionType.QueuedConnection)

        threading.Thread(target=run_synthesis, daemon=True).start()

    @pyqtSlot()
    def _play_test_audio(self) -> None:
        if hasattr(self, "current_test_button") and self.current_test_button:
            self.current_test_button.setEnabled(True)
            self.current_test_button.setText(getattr(self, "current_test_button_label", "Test"))
        test_audio = getattr(self, "current_test_audio_path", self.project_root / "temp" / "test_voice.mp3")
        if test_audio.exists():
            self.player.setSource(QUrl.fromLocalFile(str(test_audio)))
            self.audio_output.setVolume(1.0)
            self.player.play()

    @pyqtSlot()
    def _reset_test_button(self) -> None:
        if hasattr(self, "current_test_button") and self.current_test_button:
            self.current_test_button.setEnabled(True)
            self.current_test_button.setText(getattr(self, "current_test_button_label", "Test"))
        QMessageBox.warning(self, "Error", getattr(self, "current_test_failure_message", "Failed."))

    # ── Review JSON/SRT ──

    def _use_review_json_for_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select review JSON", "", "JSON files (*.json);;All files (*)",
            options=FILE_DIALOG_OPTIONS,
        )
        if path:
            self.review_json_path.setText(path)

    def _use_review_srt_for_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select SRT file", "", "SRT files (*.srt);;All files (*)",
            options=FILE_DIALOG_OPTIONS,
        )
        if path:
            self.review_json_path.setText(path)

    def _manage_glossary(self) -> None:
        from gui.dialogs.glossary_manager import GlossaryManagerDialog

        path_text = self.translate_page.glossary_path.text().strip()
        if path_text:
            glossary_path = Path(path_text).expanduser()
        else:
            glossary_path = Path("glossary.txt")

        dialog = GlossaryManagerDialog(glossary_path, self)
        if dialog.exec():
            self.translate_page.glossary_path.setText(str(glossary_path))

    def _edit_review_json(self) -> None:
        if not self.selected_input_videos:
            QMessageBox.warning(self, "Edit Review", "Select a video first.")
            return
        video = self.selected_input_videos[0]
        output_dir = Path(self.output_folder.text()).expanduser()
        review_path = output_dir / f"{video.stem}_transcript_review.json"
        if not review_path.exists():
            QMessageBox.warning(self, "Edit Review", f"Review JSON not found:\n{review_path}")
            return
        dialog = TranscriptReviewDialog(review_path, self._preview_review_text, self)
        dialog.exec()

    def _preview_review_text(self, text: str) -> None:
        if not MULTIMEDIA_AVAILABLE or not text.strip():
            return
        temp_dir = self.project_root / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        preview_mp3 = temp_dir / "review_preview.mp3"
        voice, _ = self._voice_selection(self.voice_female, FEMALE_EDGE_VOICES[0])
        cmd = [
            str(self.project_root / ".venv" / "bin" / "edge-tts"),
            "--voice", voice, "--text", text, "--write-media", str(preview_mp3),
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if preview_mp3.exists():
                self.player.setSource(QUrl.fromLocalFile(str(preview_mp3)))
                self.audio_output.setVolume(1.0)
                self.player.play()
        except Exception as exc:
            import logging
            logging.debug("Preview synthesis failed: %s", exc)

    # ── Recording ──

    def _toggle_recording(self) -> None:
        if self.recording_process is None:
            temp_dir = self.project_root / "temp"
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_file = temp_dir / "recorded_reference.mp3"
            cmd = [
                "ffmpeg", "-y", "-f", "pulse", "-i", "default",
                "-ac", "1", "-ar", "44100", "-codec:a", "libmp3lame", "-b:a", "192k",
                str(temp_file),
            ]
            try:
                self.recording_process = subprocess.Popen(
                    cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                self.record_button.setText("Stop (Recording...)")
                self.record_button.setStyleSheet(recording_style(self._theme))
                self._append_log("Microphone recording started...")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to start recording: {e}")
        else:
            try:
                self.recording_process.stdin.write(b"q\n")
                self.recording_process.stdin.flush()
                self.recording_process.wait(timeout=5)
            except Exception:
                try:
                    self.recording_process.terminate()
                    self.recording_process.wait(timeout=2)
                except Exception:
                    self.recording_process.kill()
            self.recording_process = None
            self.record_button.setText("Record")
            self.record_button.setStyleSheet("")
            temp_file = self.project_root / "temp" / "recorded_reference.mp3"
            if temp_file.exists():
                self.clone_page.profile_reference_audio.setText(str(temp_file))
                if not self.voice_profile_name.text().strip():
                    self.voice_profile_name.setText("Recorded Voice")

    # ── Speaker detection flow ──

    def _video_key(self, video_path: Path) -> str:
        try:
            return str(video_path.resolve())
        except OSError:
            return str(video_path)

    def _is_per_person_mode(self, voice_mode: str) -> bool:
        return voice_mode in {"per_person", "per_person_auto"}

    def _persistent_cache_dir(self, settings=None):
        if settings is not None and not settings.enable_persistent_cache:
            return None
        cache_dir = self.project_root / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def _url_import_cookies_file(self) -> Path | None:
        text = self.settings_page.url_import_cookies_file.text().strip()
        if text:
            return Path(text).expanduser()
        return env_path("VIDEO_IMPORT_COOKIES_FILE")

    def _progress_with_events(self, stage: str):
        def update(value: int) -> None:
            self._set_progress(stage, value)
            QApplication.processEvents()
        return update

    def _set_speaker_detection_busy(self, busy: bool) -> None:
        self.detect_speakers_button.setText("Detecting speakers..." if busy else "Detect Speakers / Map Voices")
        self.speakers_page.detect_button.setText("Detecting speakers..." if busy else "Detect Speakers / Map Voices")
        self.start_button.setEnabled(not busy and self.worker is None)
        self.cancel_button.setEnabled(busy or self.worker is not None)
        self.detect_speakers_button.setEnabled(not busy and self.voice_gender.currentData() == "per_person")

    def _start_speaker_detection_flow(self, settings, after_complete):
        if self.speaker_detection_thread is not None:
            QMessageBox.information(self, "Speaker detection", "Already running.")
            return
        self.speaker_voice_mappings = {}
        self.diarization_turns = {}
        self._speaker_detection_videos = settings.input_videos or [settings.input_video]
        self._speaker_detection_index = 0
        self._speaker_detection_settings = settings
        self._speaker_detection_after_complete = after_complete
        self._speaker_detection_continue_after_thread = False
        self._speaker_detection_final_success = None
        self._set_speaker_detection_busy(True)
        self._run_next_speaker_detection()

    def _run_next_speaker_detection(self) -> None:
        settings = self._speaker_detection_settings
        if settings is None:
            self._finish_speaker_detection(False)
            return
        if self._speaker_detection_index >= len(self._speaker_detection_videos):
            self._finish_speaker_detection(True)
            return
        video_path = self._speaker_detection_videos[self._speaker_detection_index]
        self.speaker_detection_thread = QThread()
        self.speaker_detection_worker = SpeakerDetectionWorker(video_path, settings, self.project_root)
        self.speaker_detection_worker.moveToThread(self.speaker_detection_thread)
        self.speaker_detection_thread.started.connect(self.speaker_detection_worker.run)
        self.speaker_detection_worker.signals.log.connect(self._append_log)
        self.speaker_detection_worker.signals.progress.connect(self._set_progress)
        self.speaker_detection_worker.signals.completed.connect(self._speaker_detection_completed)
        self.speaker_detection_worker.signals.failed.connect(self._speaker_detection_failed)
        self.speaker_detection_thread.finished.connect(self.speaker_detection_worker.deleteLater)
        self.speaker_detection_thread.finished.connect(self.speaker_detection_thread.deleteLater)
        self.speaker_detection_thread.finished.connect(self._speaker_detection_thread_finished)
        self.speaker_detection_thread.start()

    def _speaker_detection_completed(self, video_path, work_dir, turns) -> None:
        settings = self._speaker_detection_settings
        if settings is None:
            self._speaker_detection_final_success = False
        else:
            mapped = self._save_detected_speaker_mapping(video_path, work_dir, turns, settings)
            if mapped:
                self._speaker_detection_index += 1
                if self._speaker_detection_index < len(self._speaker_detection_videos):
                    self._speaker_detection_continue_after_thread = True
                else:
                    self._speaker_detection_final_success = True
            else:
                self._speaker_detection_final_success = False
        if self.speaker_detection_thread is not None:
            self.speaker_detection_thread.quit()

    def _speaker_detection_failed(self, video_path, work_dir, message) -> None:
        settings = self._speaker_detection_settings
        if settings is not None and not settings.keep_temp:
            remove_tree(work_dir)
        answer = QMessageBox.question(
            self, "Speaker detection failed",
            f"Failed for {video_path.name}:\n{message}\n\nContinue with Auto mode?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if answer == QMessageBox.StandardButton.Yes:
            auto_index = self.voice_gender.findData("auto")
            if auto_index >= 0:
                self.voice_gender.setCurrentIndex(auto_index)
            self._speaker_detection_final_success = True
        else:
            self._speaker_detection_final_success = False
        if self.speaker_detection_thread is not None:
            self.speaker_detection_thread.quit()

    def _speaker_detection_thread_finished(self) -> None:
        self.speaker_detection_thread = None
        self.speaker_detection_worker = None
        if self._speaker_detection_continue_after_thread:
            self._speaker_detection_continue_after_thread = False
            self._run_next_speaker_detection()
            return
        if self._speaker_detection_final_success is not None:
            self._finish_speaker_detection(self._speaker_detection_final_success)

    def _finish_speaker_detection(self, success: bool) -> None:
        after_complete = self._speaker_detection_after_complete
        self._speaker_detection_videos = []
        self._speaker_detection_index = 0
        self._speaker_detection_settings = None
        self._speaker_detection_after_complete = None
        self._speaker_detection_continue_after_thread = False
        self._speaker_detection_final_success = None
        self._set_speaker_detection_busy(False)
        if after_complete is not None:
            after_complete(success)

    def _has_per_person_voice_maps(self, settings) -> bool:
        videos = settings.input_videos or [settings.input_video]
        for video_path in videos:
            video_key = self._video_key(video_path)
            if video_key not in self.speaker_voice_mappings or video_key not in self.diarization_turns:
                return False
        return True

    def _detect_and_map_speakers(self) -> None:
        settings = self._settings_from_ui()
        error = self._validate_settings(settings)
        if error:
            QMessageBox.warning(self, "Invalid settings", error)
            return
        self.logs_page.reset_progress()
        self._start_speaker_detection_flow(
            settings, lambda success: self._speaker_detection_button_finished(success),
        )

    def _speaker_detection_button_finished(self, success: bool) -> None:
        if success and self.voice_gender.currentData() == "per_person":
            QMessageBox.information(self, "Speaker mapping ready", "Speaker voice mappings have been saved.")

    def _save_detected_speaker_mapping(self, video_path, work_dir, turns, settings):
        speaker_ids = speaker_ids_from_turns(turns)
        dialog = SpeakerMappingDialog(
            video_path.name, speaker_ids, work_dir,
            settings.min_reference_seconds, settings.enable_audio_cleanup,
            self._persistent_cache_dir(settings), self.voice_profiles,
            self._preview_speaker_voice, self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            if not settings.keep_temp:
                remove_tree(work_dir)
            return False
        video_key = self._video_key(video_path)
        self.speaker_voice_mappings[video_key] = dialog.mappings()
        self.diarization_turns[video_key] = [turn.to_dict() for turn in turns]
        self._append_log(f"Mapped {len(speaker_ids)} speaker(s) for {video_path.name}")
        if not settings.keep_temp:
            remove_tree(work_dir)
        return True

    def _preview_speaker_voice(self, speaker_id: str, mapping: dict[str, str]) -> None:
        if not MULTIMEDIA_AVAILABLE:
            return
        reference = mapping.get("cleaned_reference_audio_path") or mapping.get("reference_audio_path")
        if not reference or not Path(reference).exists():
            QMessageBox.warning(self, "Preview Voice", "Select and clean a valid reference first.")
            return
        command_template = self.rvc_command.text().strip()
        if not command_template or "{reference}" not in command_template:
            QMessageBox.warning(self, "Preview Voice", "Command must contain {reference}.")
            return
        preview_dir = self.project_root / "temp" / "speaker_previews"
        preview_dir.mkdir(parents=True, exist_ok=True)
        safe_id = "".join(ch if ch.isalnum() else "_" for ch in speaker_id)
        base_mp3 = preview_dir / f"{safe_id}_base.mp3"
        cloned_wav = preview_dir / f"{safe_id}_preview.wav"
        test_text = "សួស្តី នេះគឺជាសំឡេងសាកល្បង។"
        rate_str = f"{'+' if self.speech_rate.value() >= 0 else ''}{self.speech_rate.value()}%"
        pitch_str = f"{'+' if self.pitch_hz.value() >= 0 else ''}{self.pitch_hz.value()}Hz"
        edge_tts_bin = self.project_root / ".venv" / "bin" / "edge-tts"
        base_voice, _ = self._voice_selection(self.voice_female, FEMALE_EDGE_VOICES[0])
        try:
            subprocess.run(
                [str(edge_tts_bin if edge_tts_bin.exists() else "edge-tts"),
                 "--voice", base_voice, "--text", test_text,
                 "--rate", rate_str, "--pitch", pitch_str, "--write-media", str(base_mp3)],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            rendered = command_template.format(
                input=str(base_mp3), output=str(cloned_wav), model="", index="", reference=str(reference),
            )
            self._run_clone_command(rendered)
        except Exception as exc:
            QMessageBox.warning(self, "Preview Voice", f"Failed: {exc}")
            return
        if cloned_wav.exists():
            self.player.setSource(QUrl.fromLocalFile(str(cloned_wav)))
            self.audio_output.setVolume(1.0)
            self.player.play()

    # ── Settings from UI ──

    def _settings_from_ui(self) -> PipelineSettings:
        input_videos = self.selected_input_videos if self.selected_input_videos else []
        if not input_videos:
            files = self.import_page.file_drop.files()
            input_videos = files if files else []
        input_video = input_videos[0] if input_videos else Path(".")

        output_dir = Path(self.import_page.output_folder.text()).expanduser()
        clone_workflow = self.clone_page.clone_workflow.currentData()
        rvc_ref = None
        if clone_workflow != "gender_profiles" and self.rvc_reference_audio.text().strip():
            rvc_ref = Path(self.rvc_reference_audio.text()).expanduser()
        glossary_text = self.translate_page.glossary_path.text().strip()
        glossary = Path(glossary_text).expanduser() if glossary_text else None
        review_text = self.translate_page.review_json_path.text().strip()
        review_json = Path(review_text).expanduser() if review_text else None

        voice_female, voice_female_ref = self._voice_selection(self.voice_female, FEMALE_EDGE_VOICES[0])
        voice_male, voice_male_ref = self._voice_selection(self.voice_male, MALE_EDGE_VOICES[0])
        if clone_workflow == "gender_profiles":
            voice_female_ref = voice_female_ref or self._generated_profile_reference_from_combo(
                self.voice_female, "female"
            )
            voice_male_ref = voice_male_ref or self._generated_profile_reference_from_combo(
                self.voice_male, "male"
            )

        vg = self.voice_page
        tp = self.translate_page
        cp = self.clone_page
        ap = self.audio_page
        ep = self.export_page
        sp = self.settings_page

        return PipelineSettings(
            input_video=input_video,
            input_videos=input_videos,
            output_dir=output_dir,
            source_language=self.import_page.source_language.currentData(),
            voice_gender=self.voice_gender.currentData(),
            tts_provider=str(vg.tts_provider.currentData() or "edge"),
            voice_female=voice_female,
            voice_male=voice_male,
            speech_rate=self.speech_rate.value(),
            pitch_hz=self.pitch_hz.value(),
            emotion_strength=float(vg.emotion_strength.value() / 100.0),
            whisper_model=sp.whisper_model.currentText(),
            device=sp.device.currentData(),
            voice_female_reference_path=voice_female_ref,
            voice_male_reference_path=voice_male_ref,
            keep_temp=sp.keep_temp_check.isChecked(),
            rvc_enabled=False,
            rvc_reference_audio_path=rvc_ref,
            rvc_clone_gender="all" if clone_workflow == "gender_profiles" else cp.rvc_clone_gender.currentData(),
            rvc_command_template=self.rvc_command.text().strip(),
            clone_workflow=clone_workflow,
            speaker_voice_mappings=self.speaker_voice_mappings.copy(),
            diarization_turns=self.diarization_turns.copy(),
            enable_audio_cleanup=ap.audio_cleanup_check.isChecked(),
            enable_final_mastering=ap.final_mastering_check.isChecked(),
            alignment_mode=sp.alignment_mode.currentData() or "natural",
            enable_persistent_cache=sp.persistent_cache_check.isChecked(),
            auto_speaker_references=self.voice_gender.currentData() == "per_person_auto",
            preserve_bgm=ap.preserve_bgm_check.isChecked(),
            preset=sp.preset.currentData(),
            transcript_review_mode=tp.review_mode.currentData(),
            khmer_style=tp.khmer_style.currentData(),
            glossary_path=glossary,
            review_json_path=review_json,
            save_review_json=ep.save_review_json_check.isChecked(),
            export_dubbed_audio=ep.export_audio_check.isChecked(),
            export_original_transcript=ep.export_original_check.isChecked(),
            export_raw_khmer=ep.export_raw_khmer_check.isChecked(),
            export_improved_khmer=ep.export_improved_khmer_check.isChecked(),
            export_subtitles=ep.export_srt_check.isChecked(),
            export_quality_report=ep.export_quality_check.isChecked(),
            voice_volume=float(ap.voice_volume.value()),
            bgm_volume=float(ap.bgm_volume.value()),
            content_style=tp.content_style.currentData(),
            publish_target=ap.publish_target.currentData(),
            custom_lufs=float(ap.custom_lufs.value()),
            enable_bgm_ducking=ap.bgm_ducking_check.isChecked(),
            duck_depth_db=float(ap.duck_depth.value()),
            enable_per_speaker_prosody=ap.per_speaker_prosody_check.isChecked(),
            enable_clone_verification=cp.clone_verification_check.isChecked(),
            generate_script_only=self._is_script_only,
            burn_subtitles=ep.burn_subtitles_check.isChecked(),
            subtitle_language=ep.subtitle_language.currentData(),
            subtitle_font_size=ep.subtitle_font_size.value(),
            subtitle_font_name=ep.subtitle_font_name.currentText(),
            subtitle_color=ep.subtitle_color.currentText(),
            subtitle_bg_opacity=float(ep.subtitle_bg_opacity.value()),
            overlay_text=ep.overlay_text.text().strip(),
            overlay_image_path=Path(ep.overlay_image_path.text()) if ep.overlay_image_path.text().strip() else None,
            overlay_position=ep.overlay_text_position_picker.selected,
            overlay_text_position=ep.overlay_text_position_picker.selected,
            overlay_image_position=ep.overlay_image_position_picker.selected,
            overlay_opacity=float(ep.overlay_opacity.value()),
            clone_backend=cp.clone_backend.currentData(),
            emotion_aware_clone=False,
            emotion_clone_mode=cp.emotion_mode.currentData(),
            translation_backend=tp.translation_backend.currentData(),
            ai_skip_review=tp.ai_skip_review.isChecked(),
            narration_style=tp.narration_style.currentData(),
            end_screen_enabled=ep.end_screen_enabled.isChecked(),
            end_screen_text=ep.end_screen_text.text().strip(),
            end_screen_image_path=Path(ep.end_screen_image_path.text()) if ep.end_screen_image_path.text().strip() else None,
            end_screen_duration=float(ep.end_screen_duration.value()),
            end_screen_bg_color=ep.end_screen_bg_color.currentData(),
            sponsor_cards=[c.to_dict() for c in self.sponsor_page.get_sponsor_cards()],
            footer_overlay_enabled=self.sponsor_page.get_footer_config().enabled,
            footer_overlay_config=self.sponsor_page.get_footer_config().to_dict(),
        )

    def _validate_settings(self, settings) -> str | None:
        from licensing.client import LicenseClient
        license_result = LicenseClient().validate()
        if not license_result.valid:
            return license_result.message
        if not settings.output_dir.exists():
            return "Select an existing output folder."
        videos = settings.input_videos or [settings.input_video]
        for video_path in videos:
            if not video_path.exists():
                return f"Missing video: {video_path.name}"
            try:
                if not has_audio_stream(video_path):
                    return f"'{video_path.name}' has no audio track."
            except Exception as e:
                return f"Failed to check '{video_path.name}': {e}"
        if getattr(settings, "tts_provider", "edge") == "gemini":
            from modules.gemini_tts_engine import resolve_gemini_api_keys

            if not resolve_gemini_api_keys():
                return "Gemini TTS requires GEMINI_API_KEY, GEMINI_API_KEY_FALLBACK, or GEMINI_API_KEYS."
            return None
        clone_workflow = getattr(settings, "clone_workflow", "")
        if clone_workflow == "gender_profiles":
            settings.voice_female_reference_path = (
                settings.voice_female_reference_path
                or self._generated_profile_reference_from_combo(self.voice_female, "female")
            )
            settings.voice_male_reference_path = (
                settings.voice_male_reference_path
                or self._generated_profile_reference_from_combo(self.voice_male, "male")
            )
            missing = []
            if settings.voice_female_reference_path is None:
                missing.append("female")
            if settings.voice_male_reference_path is None:
                missing.append("male")
            if missing:
                return (
                    "Select generated female and male profiles in Voice & TTS Settings "
                    f"before using Male/Female generated profiles. Missing: {', '.join(missing)}."
                )
        per_person_mode = self._is_per_person_mode(settings.voice_gender)
        generated_profile_clone = _uses_generated_profile_clone(settings)
        if per_person_mode or generated_profile_clone:
            if settings.clone_backend in INTERNAL_PER_PERSON_CLONE_BACKENDS:
                backend_error = internal_per_person_clone_backend_error(settings)
                if backend_error:
                    return backend_error
        if per_person_mode:
            if settings.clone_backend in INTERNAL_PER_PERSON_CLONE_BACKENDS:
                pass
            elif not settings.rvc_command_template:
                return "Per-person voice cloning requires a clone command template."
            elif "{reference}" not in settings.rvc_command_template:
                return "Per-person cloning requires {reference} in the command."
        elif generated_profile_clone and settings.clone_backend not in INTERNAL_PER_PERSON_CLONE_BACKENDS:
            if not settings.rvc_command_template:
                return "Generated voice profiles require a clone command template or a built-in clone backend."
            if "{reference}" not in settings.rvc_command_template:
                return "Generated voice profiles require {reference} in the clone command."
        internal_clone_backend = (
            (per_person_mode or generated_profile_clone)
            and settings.clone_backend in INTERNAL_PER_PERSON_CLONE_BACKENDS
        )
        if (
            settings.rvc_enabled
            and not settings.rvc_command_template
            and not internal_clone_backend
        ):
            return "RVC is enabled; enter the external command template."
        return None

    # ── Pipeline execution ──

    def _start(self, script_only: bool = False) -> None:
        self._is_script_only = script_only
        settings = self._settings_from_ui()
        error = self._validate_settings(settings)
        if error:
            QMessageBox.warning(self, "Invalid settings", error)
            return
        self.logs_page.reset_progress()
        self.log_console.clear()
        self._navigate_to("logs")
        self.sidebar.select("logs")

        if settings.voice_gender == "per_person":
            self.start_button.setEnabled(False)
            self.generate_script_button.setEnabled(False)
            if not self._has_per_person_voice_maps(settings):
                self._start_speaker_detection_flow(settings, self._continue_start_after_speaker_detection)
                return
            settings = self._settings_from_ui()
            error = self._validate_settings(settings)
            if error:
                self.start_button.setEnabled(True)
                self.generate_script_button.setEnabled(True)
                QMessageBox.warning(self, "Invalid settings", error)
                return
        self._start_pipeline(settings)

    def _continue_start_after_speaker_detection(self, success: bool) -> None:
        if not success:
            self.start_button.setEnabled(True)
            self.generate_script_button.setEnabled(True)
            return
        settings = self._settings_from_ui()
        error = self._validate_settings(settings)
        if error:
            self.start_button.setEnabled(True)
            self.generate_script_button.setEnabled(True)
            QMessageBox.warning(self, "Invalid settings", error)
            return
        self._start_pipeline(settings)

    def _start_pipeline(self, settings) -> None:
        self._append_log("Starting dubbing job")
        thread = QThread()
        worker = PipelineWorker(settings, self.project_root)
        self._attach_pipeline_worker(thread, worker)
        self.start_button.setEnabled(False)
        self.generate_script_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.cancel_button.show()
        self.open_video_button.setEnabled(False)
        self.open_video_button.hide()
        self.last_output_video = None
        self.thread.start()

    def _attach_pipeline_worker(self, thread: QThread, worker: PipelineWorker) -> None:
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.signals.log.connect(self._append_log)
        worker.signals.progress.connect(self._set_progress)
        worker.signals.draft_updated.connect(self._refresh_draft_queue)
        worker.signals.finished.connect(self._finished)
        worker.signals.failed.connect(self._failed)
        worker.signals.finished.connect(thread.quit)
        worker.signals.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._thread_finished)
        self.thread = thread
        self.worker = worker

    def _cancel(self) -> None:
        self.cancel_button.setEnabled(False)
        if self.url_import_worker:
            self._append_log("Cancelling URL import...")
            self.url_import_worker.cancel()
        elif self.speaker_detection_worker:
            self._append_log("Pausing — progress saved. Resume from Sessions page.")
            self.speaker_detection_worker.cancel()
        elif self.worker:
            self._append_log("Pausing — progress saved. Resume from Sessions page.")
            self.worker.cancel()

    def _append_log(self, message: str) -> None:
        self.logs_page.append_log(message)
        if message.startswith("PROCESSING VIDEO "):
            payload = message[len("PROCESSING VIDEO "):].split(":", 1)[0].strip()
            self.status_bar.set_batch(f"Processing {payload}")
        elif message.startswith("PROCESSING DRAFT "):
            payload = message[len("PROCESSING DRAFT "):].split(":", 1)[0].strip()
            self.status_bar.set_batch(f"Processing {payload}")
        elif message.startswith("PROCESSING "):
            self.status_bar.set_batch(message.strip("=").strip())

    def _set_progress(self, stage: str, value: int) -> None:
        self.logs_page.set_progress(stage, value)
        stage_labels = dict(STAGES)
        label = stage_labels.get(stage, stage)
        self.status_bar.set_stage(label, value)

    def _start_url_import(self, urls: list[str], auto_start: bool = False) -> None:
        if self.url_import_thread is not None:
            QMessageBox.information(self, "URL import", "A URL import is already running.")
            return
        queue_running = self.worker is not None and self.worker.draft_queue_path is not None
        if self.thread is not None and not queue_running:
            QMessageBox.warning(self, "URL import", "Wait for the current pipeline run to finish first.")
            return
        if not urls:
            QMessageBox.warning(self, "URL import", "Paste at least one supported video URL first.")
            return

        self._url_import_auto_start = bool(auto_start)

        prefix = self.import_page.import_name_prefix.text().strip()
        if not prefix:
            prefix, ok = QInputDialog.getText(
                self,
                "Import name prefix",
                "Enter the base name for imported files:",
                text="Good",
            )
            if not ok or not prefix.strip():
                QMessageBox.warning(self, "URL import", "Import canceled because no name prefix was provided.")
                return
            prefix = prefix.strip()
            self.import_page.import_name_prefix.setText(prefix)

        cookies_file = self._url_import_cookies_file()
        mode = "import and start dubbing" if self._url_import_auto_start else "import only"
        self._append_log(f"Starting URL import ({mode}) for {len(urls)} URL(s)")
        self.url_import_thread = QThread()
        self.url_import_worker = VideoImportWorker(
            urls,
            self.project_root,
            cookies_file=cookies_file,
        )
        self.url_import_worker.moveToThread(self.url_import_thread)
        self.url_import_thread.started.connect(self.url_import_worker.run)
        self.url_import_worker.signals.log.connect(self._append_log)
        self.url_import_worker.signals.progress.connect(self._set_url_import_progress)
        self.url_import_worker.signals.finished.connect(self._url_import_finished)
        self.url_import_worker.signals.failed.connect(self._url_import_failed)
        self.url_import_worker.signals.finished.connect(self.url_import_thread.quit)
        self.url_import_worker.signals.failed.connect(self.url_import_thread.quit)
        self.url_import_thread.finished.connect(self.url_import_worker.deleteLater)
        self.url_import_thread.finished.connect(self.url_import_thread.deleteLater)
        self.url_import_thread.finished.connect(self._url_import_thread_finished)
        self.import_page.set_url_import_busy(True)
        self.start_button.setEnabled(False)
        self.generate_script_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.cancel_button.show()
        self.url_import_worker.name_prefix = prefix
        self.url_import_thread.start()

    def _set_url_import_progress(self, url: str, value: int) -> None:
        label = "Importing video URLs"
        self.status_bar.set_stage(label, value)
        if value >= 100:
            self._append_log(f"URL import complete: {url}")

    def _url_import_finished(self, imported: list[Path], failures: list[tuple[str, str]]) -> None:
        existing = self.import_page.file_drop.files()
        merged = existing + [path for path in imported if path not in existing]
        if merged:
            self.import_page.file_drop.set_files(merged)
        if imported:
            self.import_page.clear_url_input()

        auto_start = self._url_import_auto_start
        queue_running = self.worker is not None and self.worker.draft_queue_path is not None
        if imported and auto_start:
            if queue_running:
                settings = self._settings_from_ui()
                self._reload_draft_queue()
                self.draft_queue.add_jobs(
                    settings,
                    imported,
                    self._source_urls_for_videos(imported),
                )
                self._append_log(f"Added {len(imported)} imported video(s) to the running Draft Queue.")
                self._refresh_draft_queue()
            else:
                self._reload_draft_queue()
                if self.draft_queue.has_runnable():
                    settings = self._settings_from_ui()
                    self.draft_queue.add_jobs(
                        settings,
                        imported,
                        self._source_urls_for_videos(imported),
                    )
                    self._append_log(f"Added {len(imported)} imported video(s) to the Draft Queue.")
                    self._refresh_draft_queue()
                    self._start_draft_queue()
                else:
                    settings = self._settings_from_ui()
                    error = self._validate_settings(settings)
                    if error:
                        self._append_log(
                            "Imported videos are ready, but pipeline settings are invalid so auto-start was skipped."
                        )
                    else:
                        self._append_log("Imported videos are ready. Starting auto-processing...")
                        self._start_pipeline(settings)
        elif imported:
            self._append_log(
                "Imported videos are ready. Preview them, then use Start Dubbing or Add to Draft Queue."
            )
            if imported[0].is_file():
                self.import_page.video_preview.set_video(imported[0])

        if failures:
            details = "\n".join(f"- {url}: {message}" for url, message in failures)
            QMessageBox.warning(
                self,
                "URL import",
                f"Imported {len(imported)} video(s), but some URLs failed:\n{details}",
            )
        elif imported:
            if auto_start:
                QMessageBox.information(self, "URL import", f"Imported {len(imported)} video(s).")
            else:
                QMessageBox.information(
                    self,
                    "URL import",
                    f"Imported {len(imported)} video(s). They are ready to preview or queue — dubbing was not started.",
                )
        else:
            QMessageBox.warning(self, "URL import", "No videos were imported.")

    def _url_import_failed(self, message: str) -> None:
        self._append_log(f"URL import failed: {message}")
        QMessageBox.warning(self, "URL import", message)

    def _url_import_thread_finished(self) -> None:
        self.url_import_thread = None
        self.url_import_worker = None
        self.import_page.set_url_import_busy(False)
        self.start_button.setEnabled(self.worker is None and self.speaker_detection_thread is None)
        self.generate_script_button.setEnabled(self.worker is None and self.speaker_detection_thread is None)
        if self.worker is None and self.speaker_detection_thread is None:
            self.cancel_button.setEnabled(False)
            self.cancel_button.hide()

    def _finished(self, output_path: str) -> None:
        self._append_log(f"Completed: {output_path}" if output_path else "Completed")
        if output_path:
            self.last_output_video = Path(output_path)
            self.open_video_button.setEnabled(True)
            self.open_video_button.show()
        self.status_bar.set_stage("Done!", 100)
        self._refresh_draft_queue()
        self.sessions_page.refresh()
        self.editor_page.refresh_sessions()
        if output_path:
            QMessageBox.information(self, "Finished", f"Dubbed video created:\n{output_path}")
        else:
            QMessageBox.information(self, "Finished", "Draft Queue finished.")

    def _failed(self, message: str) -> None:
        self._append_log(f"Failed: {message}")
        self.status_bar.set_stage("Failed", 0)
        self._refresh_draft_queue()
        self.sessions_page.refresh()
        self.editor_page.refresh_sessions()
        if "cancel" in message.lower() or "pause" in message.lower():
            QMessageBox.information(
                self, "Paused",
                "Processing paused. Go to Sessions page to resume where you left off.",
            )
        else:
            QMessageBox.critical(
                self, "Processing failed",
                f"{message}\n\nYou can resume from the Sessions page.",
            )

    def _thread_finished(self) -> None:
        self.start_button.setEnabled(True)
        self.generate_script_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.cancel_button.hide()
        self.status_bar.set_batch("")
        self.thread = None
        self.worker = None
        self._refresh_draft_queue()

    # ── Session resume / re-dub ──

    def _resume_session(self, work_dir: str) -> None:
        from core.session import DubbingSession
        try:
            session = DubbingSession.load(Path(work_dir))
        except Exception as exc:
            QMessageBox.critical(self, "Resume Failed", str(exc))
            return
        if self.thread is not None:
            QMessageBox.warning(self, "Busy", "A pipeline is already running.")
            return
        self._append_log(f"\nResuming session for {session.video_name}...")
        self._navigate_to("logs")
        self.sidebar.select("logs")
        self.start_button.setEnabled(False)
        self.generate_script_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.cancel_button.show()
        self.open_video_button.setEnabled(False)
        self.open_video_button.hide()
        self.last_output_video = None

        worker = PipelineWorker(session.settings, self.project_root, resume_session=session)
        thread = QThread()
        self._attach_pipeline_worker(thread, worker)
        thread.start()

    def _open_editor_for_session(self, work_dir: str) -> None:
        self._navigate_to("editor")
        self.sidebar.select("editor")
        self.editor_page.load_session(work_dir)

    def _preview_editor_segment(self, work_dir: str, segment_index: int, text: str) -> None:
        from core.session import DubbingSession
        if self.preview_thread is not None:
            QMessageBox.information(self, "Preview Segment", "A segment preview is already running.")
            self.editor_page.preview_finished()
            return
        try:
            session = DubbingSession.load(Path(work_dir))
        except Exception as exc:
            self.editor_page.preview_finished()
            QMessageBox.critical(self, "Preview Failed", str(exc))
            return

        self._append_log(f"Previewing segment {segment_index + 1}...")
        worker = PreviewSegmentWorker(session, segment_index, text)
        thread = QThread()
        worker.moveToThread(thread)
        worker.signals.finished.connect(self._preview_editor_finished)
        worker.signals.failed.connect(self._preview_editor_failed)
        thread.started.connect(worker.run)
        thread.finished.connect(self._preview_thread_finished)
        self.preview_thread = thread
        self.preview_worker = worker
        thread.start()

    def _preview_editor_finished(self, audio_path: str) -> None:
        self._append_log(f"Preview ready: {audio_path}")
        if MULTIMEDIA_AVAILABLE and self.player is not None and self.audio_output is not None:
            self.player.setSource(QUrl.fromLocalFile(audio_path))
            self.audio_output.setVolume(1.0)
            self.player.play()
        else:
            QDesktopServices.openUrl(QUrl.fromLocalFile(audio_path))
        self.editor_page.preview_finished()
        if self.preview_thread is not None:
            self.preview_thread.quit()

    def _preview_editor_failed(self, message: str) -> None:
        self._append_log(message)
        self.editor_page.preview_finished()
        QMessageBox.warning(self, "Preview Failed", message)
        if self.preview_thread is not None:
            self.preview_thread.quit()

    def _preview_thread_finished(self) -> None:
        self.preview_thread = None
        self.preview_worker = None

    def _start_redub(self, work_dir: str, edits: dict) -> None:
        from core.session import DubbingSession
        if self.thread is not None:
            QMessageBox.warning(self, "Busy", "A pipeline is already running.")
            return
        try:
            session = DubbingSession.load(Path(work_dir))
        except Exception as exc:
            QMessageBox.critical(self, "Re-dub Failed", str(exc))
            return
        self._append_log(f"\nRe-dubbing {len(edits)} segment(s)...")
        self._navigate_to("logs")
        self.sidebar.select("logs")
        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.cancel_button.show()

        worker = RedubWorker(session, edits, self.project_root)
        thread = QThread()
        worker.moveToThread(thread)
        worker.signals.log.connect(self._append_log)
        worker.signals.progress.connect(self._set_progress)
        worker.signals.finished.connect(self._finished)
        worker.signals.failed.connect(self._failed)
        thread.started.connect(worker.run)
        thread.finished.connect(self._thread_finished)
        self.thread = thread
        self.worker = worker
        thread.start()

    def _open_finished_video(self) -> None:
        if self.last_output_video and self.last_output_video.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.last_output_video)))
        else:
            QMessageBox.warning(self, "Error", "Finished video not found.")

    def _check_setup(self) -> None:
        if self.setup_check_thread is not None:
            return
        settings = self._settings_from_ui()
        self._append_log("Running setup check...")
        self.start_button.setEnabled(False)
        self.setup_check_button.setEnabled(False)
        self.setup_check_button.setText("Checking...")
        self.setup_check_thread = QThread()
        self.setup_check_worker = SetupCheckWorker(settings, self.project_root)
        self.setup_check_worker.moveToThread(self.setup_check_thread)
        self.setup_check_thread.started.connect(self.setup_check_worker.run)
        self.setup_check_worker.signals.log.connect(self._append_log)
        self.setup_check_worker.signals.finished.connect(self._setup_check_finished)
        self.setup_check_worker.signals.finished.connect(self.setup_check_thread.quit)
        self.setup_check_thread.finished.connect(self.setup_check_worker.deleteLater)
        self.setup_check_thread.finished.connect(self.setup_check_thread.deleteLater)
        self.setup_check_thread.finished.connect(self._setup_check_thread_finished)
        self.setup_check_thread.start()

    def _save_and_test_gemini_key(self) -> None:
        from config.user_secrets import save_user_secret
        from modules.gemini_key_validator import validate_gemini_api_key

        key = self.settings_page.gemini_api_key.text().strip()
        self.settings_page.test_gemini_button.setEnabled(False)
        self.settings_page.gemini_key_status.setText("Testing the key with Gemini...")
        QApplication.processEvents()
        valid, message = validate_gemini_api_key(key)
        self.settings_page.test_gemini_button.setEnabled(True)
        self.settings_page.gemini_key_status.setText(message)
        if valid:
            save_user_secret("GEMINI_API_KEY", key)
            os.environ["GEMINI_API_KEY"] = key
            self.settings_page.gemini_api_key.setReadOnly(True)
            self.settings_page.change_gemini_key_button.setText("Change Key")
            self.voice_page._update_provider_status()
            QMessageBox.information(self, "Gemini API", message)
        else:
            QMessageBox.warning(self, "Gemini API", message)

    def _show_gemini_setup_notice(self) -> None:
        QMessageBox.information(
            self,
            "Add Your Gemini API Key",
            "Before using Gemini AI or Gemini TTS, open Settings, add your Gemini API key, "
            "then click Save & Test Gemini Key. The app will ask Gemini directly to verify that it works.",
        )

    def _activate_license(self) -> None:
        from licensing.client import LicenseClient
        client = LicenseClient()
        if not client.required:
            message = "Set LICENSE_SERVER_URL in the distributed app before accepting paid activations."
            self.settings_page.license_status.setText(message)
            QMessageBox.warning(self, "License", message)
            return
        result = client.activate(self.settings_page.license_key.text())
        if result.valid:
            self.settings_page.license_key.setReadOnly(True)
            self.settings_page.change_license_key_button.setText("Change Key")
            self.settings_page.update_license_display()
            QMessageBox.information(self, "License", f"{result.message}\nExpires: {result.expires_at}")
        else:
            self.settings_page.license_status.setText(result.message)
            QMessageBox.warning(self, "License", result.message)

    def _open_purchase_wizard(self) -> None:
        from config.user_secrets import load_user_secrets
        from gui.dialogs.onboarding import AccessOnboardingDialog

        dialog = AccessOnboardingDialog(parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.settings_page.update_license_display()
            self.settings_page.license_key.setReadOnly(True)
            self.settings_page.change_license_key_button.setText("Change Key")
            self.settings_page.gemini_api_key.setText(os.getenv("GEMINI_API_KEY", ""))
            self.settings_page.gemini_key_status.setText("Gemini key saved and verified.")

    def _setup_check_finished(self, success: bool) -> None:
        if success:
            self._append_log("Setup check passed.")
            QMessageBox.information(self, "Setup Check", "Passed.")
        else:
            self._append_log("Setup check found errors.")
            QMessageBox.warning(self, "Setup Check", "Errors found. Check logs.")

    def _setup_check_thread_finished(self) -> None:
        self.setup_check_thread = None
        self.setup_check_worker = None
        self.start_button.setEnabled(self.worker is None)
        self.setup_check_button.setEnabled(True)
        self.setup_check_button.setText("Check Setup (ffmpeg, models, etc.)")

    # ── Settings persistence ──

    def _save_settings(self) -> None:
        settings_path = self.project_root / "settings.json"
        try:
            config: dict = {}
            for page in (
                self.import_page, self.voice_page, self.translate_page,
                self.clone_page, self.audio_page, self.export_page,
                self.settings_page,
            ):
                config.update(page.save_state())
            config["advanced_mode"] = self.sidebar.is_advanced()
            config["sponsor_cards"] = [c.to_dict() for c in self.sponsor_page.get_sponsor_cards()]
            config["footer_overlay"] = self.sponsor_page.get_footer_config().to_dict()
            settings_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        except Exception as exc:
            import logging
            logging.warning("Failed to save settings: %s", exc)

    def _load_settings(self) -> None:
        settings_path = self.project_root / "settings.json"
        if not settings_path.exists():
            return
        try:
            config = json.loads(settings_path.read_text(encoding="utf-8"))
            for page in (
                self.import_page, self.voice_page, self.translate_page,
                self.clone_page, self.audio_page, self.export_page,
                self.settings_page,
            ):
                page.load_state(config)
            if config.get("advanced_mode", False):
                self.sidebar.set_advanced(True)
            from modules.sponsor_card import SponsorCardConfig
            from modules.footer_overlay import FooterOverlayConfig
            sponsor_cards_data = config.get("sponsor_cards", [])
            if sponsor_cards_data:
                self.sponsor_page.set_sponsor_cards(
                    [SponsorCardConfig.from_dict(c) for c in sponsor_cards_data]
                )
            footer_data = config.get("footer_overlay", {})
            if footer_data:
                self.sponsor_page.set_footer_config(FooterOverlayConfig.from_dict(footer_data))
        except Exception as exc:
            import logging
            logging.warning("Failed to load settings: %s", exc)

    def closeEvent(self, event) -> None:
        if self.speaker_detection_worker is not None:
            self.speaker_detection_worker.cancel()
        if self.speaker_detection_thread is not None:
            self.speaker_detection_thread.quit()
            self.speaker_detection_thread.wait(3000)
        if self.recording_process is not None:
            try:
                self.recording_process.terminate()
                self.recording_process.wait()
            except Exception:
                pass
        self._save_settings()
        event.accept()


def run_app(project_root: Path, keep_temp: bool = False) -> int:
    load_project_env(project_root)
    QLocale.setDefault(QLocale(QLocale.Language.English, QLocale.Country.UnitedStates))
    app = QApplication([])
    from gui.dialogs.onboarding import AccessOnboardingDialog, validate_saved_startup_access

    access_ready, access_message = validate_saved_startup_access()
    if not access_ready:
        onboarding = AccessOnboardingDialog(access_message)
        if onboarding.exec() != QDialog.DialogCode.Accepted:
            return 0
    window = AppWindow(project_root=project_root, keep_temp=keep_temp)
    window.show()
    return app.exec()
