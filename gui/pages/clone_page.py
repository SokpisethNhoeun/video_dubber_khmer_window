from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

FILE_DIALOG_OPTIONS = QFileDialog.Option(0)


class ClonePage(QWidget):
    workflow_changed = pyqtSignal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._tts_only_preferred = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        header = QLabel("Voice Cloning")
        header.setObjectName("PageHeader")
        layout.addWidget(header)

        desc = QLabel(
            "This page controls the optional clone stage that runs after Khmer TTS. "
            "Saved voice creation is separate from choosing which voices the video uses."
        )
        desc.setObjectName("PageDesc")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        clone_section = QLabel("Clone video output")
        clone_section.setObjectName("SectionHeader")
        layout.addWidget(clone_section)

        self.clone_workflow = QComboBox()
        self.clone_workflow.addItem("Auto per-person clone (recommended)", "auto_per_person")
        self.clone_workflow.addItem("Male/Female generated profiles after TTS", "gender_profiles")
        self.clone_workflow.addItem("Single reference clone for all speakers", "single_reference")
        layout.addWidget(self.clone_workflow)

        self.workflow_help = QLabel("")
        self.workflow_help.setObjectName("HintLabel")
        self.workflow_help.setWordWrap(True)
        layout.addWidget(self.workflow_help)

        self.clone_quality_status = QLabel("")
        self.clone_quality_status.setObjectName("HintLabel")
        self.clone_quality_status.setWordWrap(True)
        layout.addWidget(self.clone_quality_status)

        self.rvc_enabled = QCheckBox("Run clone stage after TTS")
        self.rvc_enabled.setChecked(False)
        layout.addWidget(self.rvc_enabled)

        self.tts_only_hint = QLabel("")
        self.tts_only_hint.setObjectName("HintLabel")
        self.tts_only_hint.setWordWrap(True)
        layout.addWidget(self.tts_only_hint)

        clone_form = QFormLayout()
        clone_form.setSpacing(12)

        self.clone_backend = QComboBox()
        self.clone_backend.addItem("Qwen3-TTS 1.7B (best clone + emotion)", "qwen3")
        self.clone_backend.addItem("CosyVoice 2 (emotional voice clone)", "cosyvoice")
        self.clone_backend.addItem("XTTS-v2 (direct voice clone)", "xtts")
        self.clone_backend.addItem("OpenVoice (timbre transfer)", "openvoice")
        clone_form.addRow("Clone Backend", self.clone_backend)

        ref_row = QHBoxLayout()
        self.rvc_reference_audio = QLineEdit()
        self.rvc_reference_audio.setPlaceholderText("Reference audio for single-reference cloning")
        self.reference_audio_button = QPushButton("Audio")
        self.reference_audio_button.setObjectName("CompactButton")
        self.reference_audio_button.clicked.connect(self._browse_reference)
        ref_row.addWidget(self.rvc_reference_audio, 1)
        ref_row.addWidget(self.reference_audio_button)
        clone_form.addRow("Single reference", ref_row)

        self.rvc_clone_gender = QComboBox()
        self.rvc_clone_gender.addItem("All speakers", "all")
        self.rvc_clone_gender.addItem("Female only", "female")
        self.rvc_clone_gender.addItem("Male only", "male")
        clone_form.addRow("Clone gender", self.rvc_clone_gender)

        self.clone_verification_check = QCheckBox("Verify clone quality (ECAPA similarity)")
        self.clone_verification_check.setChecked(True)
        clone_form.addRow("", self.clone_verification_check)

        self.emotion_aware_check = QCheckBox("Emotion-aware cloning (preserve source emotion & style)")
        self.emotion_aware_check.setToolTip(
            "Automatically analyzes each segment for emotion, energy, and pacing.\n"
            "Uses source audio clips plus detected speaking style during cloning.\n"
            "Qwen3-TTS, XTTS-v2, and CosyVoice 2 only."
        )
        self.emotion_aware_check.setChecked(True)
        clone_form.addRow("", self.emotion_aware_check)

        self.emotion_mode = QComboBox()
        self.emotion_mode.addItem("Auto (reference + detected emotion)", "auto")
        self.emotion_mode.addItem("Reference-based (source audio clip only)", "reference")
        self.emotion_mode.addItem("Instruction-based (emotion prompts only)", "instruction")
        auto_index = self.emotion_mode.findData("auto")
        if auto_index >= 0:
            self.emotion_mode.setCurrentIndex(auto_index)
        clone_form.addRow("Emotion mode", self.emotion_mode)

        self.rvc_command = QLineEdit()
        self.rvc_command.setPlaceholderText("External clone command")
        clone_form.addRow("Command", self.rvc_command)

        self.clone_command_status = QLabel("")
        clone_form.addRow("", self.clone_command_status)

        layout.addLayout(clone_form)

        profile_section = QLabel("Voice profiles")
        profile_section.setObjectName("SectionHeader")
        layout.addWidget(profile_section)

        profile_desc = QLabel(
            "Create, import, test, or delete saved voices. To use male/female generated voices in a video, "
            "select them on the Voices page."
        )
        profile_desc.setObjectName("HintLabel")
        profile_desc.setWordWrap(True)
        layout.addWidget(profile_desc)

        profile_form = QFormLayout()
        profile_form.setSpacing(12)

        profile_ref_row = QHBoxLayout()
        self.profile_reference_audio = QLineEdit()
        self.profile_reference_audio.setPlaceholderText("MP3/WAV used to create a saved voice profile")
        self.profile_reference_audio_button = QPushButton("Audio")
        self.profile_reference_audio_button.setObjectName("CompactButton")
        self.profile_reference_audio_button.clicked.connect(self._browse_profile_reference)
        self.record_button = QPushButton("Record")
        self.record_button.setObjectName("CompactButton")
        profile_ref_row.addWidget(self.profile_reference_audio, 1)
        profile_ref_row.addWidget(self.profile_reference_audio_button)
        profile_ref_row.addWidget(self.record_button)
        profile_form.addRow("Profile audio", profile_ref_row)

        profile_row = QHBoxLayout()
        self.voice_profile_name = QLineEdit()
        self.voice_profile_name.setPlaceholderText("Name this voice")
        self.generate_voice_button = QPushButton("Generate Voice")
        self.generate_voice_button.setObjectName("CompactButton")
        self.import_voices_button = QPushButton("Import Voices")
        self.import_voices_button.setObjectName("CompactButton")
        profile_row.addWidget(self.voice_profile_name, 1)
        profile_row.addWidget(self.generate_voice_button)
        profile_row.addWidget(self.import_voices_button)
        profile_form.addRow("Voice name", profile_row)

        self.voice_profile_gender = QComboBox()
        self.voice_profile_gender.addItem("Female", "female")
        self.voice_profile_gender.addItem("Male", "male")
        profile_form.addRow("Voice type", self.voice_profile_gender)

        saved_row = QHBoxLayout()
        self.saved_voice_profiles = QComboBox()
        self.saved_voice_profiles.addItem("Custom reference audio", "")
        self.test_saved_button = QPushButton("Test")
        self.test_saved_button.setObjectName("CompactButton")
        self.delete_saved_button = QPushButton("Delete")
        self.delete_saved_button.setObjectName("CancelButton")
        self.delete_saved_button.setFixedWidth(70)
        saved_row.addWidget(self.saved_voice_profiles, 1)
        saved_row.addWidget(self.test_saved_button)
        saved_row.addWidget(self.delete_saved_button)
        profile_form.addRow("Saved voice", saved_row)

        layout.addLayout(profile_form)
        layout.addStretch(1)

        self._clone_widgets = [
            self.clone_backend,
            self.rvc_reference_audio,
            self.reference_audio_button,
            self.rvc_clone_gender,
            self.clone_verification_check,
            self.emotion_aware_check,
            self.emotion_mode,
            self.rvc_command,
        ]
        self.rvc_enabled.toggled.connect(self._toggle_fields)
        self.clone_backend.currentIndexChanged.connect(self._on_backend_changed)
        self.emotion_aware_check.toggled.connect(self._on_emotion_aware_changed)
        self.clone_workflow.currentIndexChanged.connect(self._on_workflow_changed)
        self._toggle_fields(self.rvc_enabled.isChecked())
        self._on_backend_changed()
        self._on_workflow_changed()

    def _toggle_fields(self, enabled: bool) -> None:
        for widget in self._clone_widgets:
            widget.setEnabled(enabled)
        self._update_workflow_controls()
        self._update_compact_mode()
        self._update_workflow_help()

    def _update_workflow_controls(self) -> None:
        gender_profiles = self.clone_workflow.currentData() == "gender_profiles"
        enabled = self.rvc_enabled.isChecked()
        for widget in (
            self.rvc_reference_audio,
            self.reference_audio_button,
            self.rvc_clone_gender,
        ):
            widget.setEnabled(enabled and not gender_profiles)
        if gender_profiles:
            self.rvc_reference_audio.setPlaceholderText("Female and male generated voices are selected on the Voices page")
        else:
            self.rvc_reference_audio.setPlaceholderText("Reference audio for single-reference cloning")

    def set_tts_only_preferred(self, enabled: bool) -> None:
        self._tts_only_preferred = enabled
        self._update_compact_mode()
        self._update_workflow_help()

    def _update_compact_mode(self) -> None:
        compact = self._tts_only_preferred and not self.rvc_enabled.isChecked()
        for widget in (
            self.clone_workflow,
            self.workflow_help,
            self.clone_quality_status,
        ):
            widget.setVisible(not compact)
        self.tts_only_hint.setVisible(compact)
        if compact:
            self.tts_only_hint.setText(
                "Cloning is optional and currently OFF. Turn it on only if you want Stage 6 voice conversion."
            )
        else:
            self.tts_only_hint.setText("")

    def _on_backend_changed(self) -> None:
        backend = self.clone_backend.currentData()
        supports_emotion = backend in ("xtts", "cosyvoice", "qwen3")
        self.emotion_aware_check.setVisible(supports_emotion)
        self._on_emotion_aware_changed()

    def _on_emotion_aware_changed(self) -> None:
        backend = self.clone_backend.currentData()
        show_mode = backend == "cosyvoice" and self.emotion_aware_check.isChecked()
        self.emotion_mode.setVisible(show_mode)

    def _on_workflow_changed(self) -> None:
        self._update_workflow_controls()
        self._update_workflow_help()
        self.workflow_changed.emit(str(self.clone_workflow.currentData() or ""))

    def _update_workflow_help(self) -> None:
        workflow = self.clone_workflow.currentData()
        help_text = {
            "auto_per_person": (
                "Detects speakers and builds a separate source reference for each person."
            ),
            "gender_profiles": (
                "Uses the Female Voice and Male Voice selected on the Voices page. "
                "The single reference and saved voice controls below are not used for this workflow."
            ),
            "single_reference": (
                "Uses one reference audio for all selected speakers. This is separate from male/female generated profiles."
            ),
        }.get(workflow, "")
        prefix = "Cloning is ON. " if self.rvc_enabled.isChecked() else "Cloning is OFF. "
        self.workflow_help.setText(prefix + help_text)

    def save_state(self) -> dict:
        return {
            "clone_workflow": self.clone_workflow.currentData(),
            "rvc_enabled": self.rvc_enabled.isChecked(),
            "clone_backend": self.clone_backend.currentText(),
            "rvc_clone_gender": self.rvc_clone_gender.currentText(),
            "clone_verification": self.clone_verification_check.isChecked(),
            "emotion_aware_clone": self.emotion_aware_check.isChecked(),
            "emotion_clone_mode": self.emotion_mode.currentText(),
        }

    def load_state(self, config: dict) -> None:
        workflow_index = self.clone_workflow.findData(config.get("clone_workflow", "auto_per_person"))
        if workflow_index >= 0:
            self.clone_workflow.setCurrentIndex(workflow_index)
        self.rvc_enabled.setChecked(config.get("rvc_enabled", self.rvc_enabled.isChecked()))
        self.clone_backend.setCurrentText(config.get("clone_backend", ""))
        self.rvc_clone_gender.setCurrentText(config.get("rvc_clone_gender", ""))
        self.clone_verification_check.setChecked(config.get("clone_verification", True))
        self.emotion_aware_check.setChecked(config.get("emotion_aware_clone", True))
        mode_text = config.get("emotion_clone_mode", "")
        mode_index = self.emotion_mode.findText(mode_text)
        if mode_index < 0:
            mode_index = self.emotion_mode.findData("auto")
        if mode_index >= 0:
            self.emotion_mode.setCurrentIndex(mode_index)

    def _browse_reference(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select single-reference clone audio", "",
            "Audio files (*.mp3 *.wav);;All files (*)",
            options=FILE_DIALOG_OPTIONS,
        )
        if path:
            self.rvc_reference_audio.setText(path)

    def _browse_profile_reference(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select voice profile source audio", "",
            "Audio files (*.mp3 *.wav);;All files (*)",
            options=FILE_DIALOG_OPTIONS,
        )
        if path:
            self.profile_reference_audio.setText(path)
