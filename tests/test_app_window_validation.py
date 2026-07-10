from __future__ import annotations

from types import SimpleNamespace

from conftest import make_pipeline_settings

import gui.app_window as app_window_module
from core.draft_queue import DraftQueue, STATUS_FAILED
from core.context import PipelineSettings
from gui.app_window import AppWindow, clone_setup_status
from modules.voice_profiles import VoiceProfile


class _FakeSignal:
    def __init__(self) -> None:
        self.callbacks = []

    def connect(self, callback) -> None:
        self.callbacks.append(callback)


class _FakeThread:
    def __init__(self) -> None:
        self.started = _FakeSignal()
        self.finished = _FakeSignal()

    def quit(self) -> None:
        pass

    def deleteLater(self) -> None:  # noqa: N802
        pass


class _FakeWorkerSignals:
    def __init__(self) -> None:
        self.log = _FakeSignal()
        self.progress = _FakeSignal()
        self.draft_updated = _FakeSignal()
        self.finished = _FakeSignal()
        self.failed = _FakeSignal()


class _FakeWorker:
    def __init__(self) -> None:
        self.signals = _FakeWorkerSignals()
        self.moved_to = None

    def moveToThread(self, thread) -> None:  # noqa: N802
        self.moved_to = thread

    def run(self) -> None:
        pass

    def deleteLater(self) -> None:  # noqa: N802
        pass


def test_pipeline_worker_attachment_releases_thread_on_finish_or_fail():
    window = AppWindow.__new__(AppWindow)
    thread = _FakeThread()
    worker = _FakeWorker()

    window._attach_pipeline_worker(thread, worker)

    assert worker.moved_to is thread
    assert worker.run in thread.started.callbacks
    assert window._refresh_draft_queue in worker.signals.draft_updated.callbacks
    assert thread.quit in worker.signals.finished.callbacks
    assert thread.quit in worker.signals.failed.callbacks
    assert worker.deleteLater in thread.finished.callbacks
    assert thread.deleteLater in thread.finished.callbacks
    assert window._thread_finished in thread.finished.callbacks
    assert window.thread is thread
    assert window.worker is worker


def test_pipeline_settings_default_does_not_save_review_json(tmp_path):
    settings = PipelineSettings(
        input_video=tmp_path / "video.mp4",
        output_dir=tmp_path,
        source_language="en",
        voice_gender="female",
        voice_female="km-KH-SreymomNeural",
        voice_male="km-KH-PisethNeural",
        speech_rate=0,
        pitch_hz=0,
        whisper_model="medium",
        device="cpu",
    )

    assert settings.save_review_json is False
    assert settings.emotion_aware_clone is True


def test_settings_from_ui_preserves_selected_gemini_tts_provider(monkeypatch, tmp_path):
    from PyQt6.QtWidgets import QApplication

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    video = tmp_path / "video.mp4"
    video.write_bytes(b"mp4")

    window = AppWindow(project_root=tmp_path, keep_temp=False)
    index = window.voice_page.tts_provider.findData("gemini")
    window.voice_page.tts_provider.setCurrentIndex(index)
    window.import_page.file_drop.set_files([video])
    window.import_page.output_folder.setText(str(tmp_path))

    settings = window._settings_from_ui()

    assert settings.tts_provider == "gemini"
    window.close()
    app.processEvents()


def test_start_queue_applies_current_tts_provider_to_queued_drafts(tmp_path):
    video_a = tmp_path / "a.mp4"
    video_b = tmp_path / "b.mp4"
    video_a.write_bytes(b"a")
    video_b.write_bytes(b"b")
    settings = make_pipeline_settings(tmp_path, input_video=video_a, tts_provider="edge")
    queue = DraftQueue(tmp_path / "draft_queue.json")
    jobs = queue.add_jobs(settings, [video_a, video_b])
    queue.mark_failed(jobs[1].draft_id, "failed earlier")

    logs: list[str] = []
    window = AppWindow.__new__(AppWindow)
    window.draft_queue = DraftQueue.load(queue.path)
    window.voice_page = SimpleNamespace(
        tts_provider=SimpleNamespace(currentData=lambda: "gemini")
    )
    window._append_log = logs.append

    window._apply_current_tts_provider_to_queued_drafts()
    loaded = DraftQueue.load(queue.path)

    assert loaded.jobs[0].settings.tts_provider == "gemini"
    assert loaded.jobs[1].status == STATUS_FAILED
    assert loaded.jobs[1].settings.tts_provider == "edge"
    assert logs == ["Updated queued draft(s) to use gemini TTS."]


def test_cosyvoice_per_person_accepts_empty_command_when_python_is_configured(
    monkeypatch, tmp_path,
):
    python_bin = tmp_path / "python"
    python_bin.write_text("#!/bin/sh\n")
    python_bin.chmod(0o755)
    monkeypatch.setenv("COSYVOICE_PYTHON", str(python_bin))
    monkeypatch.setattr(app_window_module, "has_audio_stream", lambda _path: True)
    settings = make_pipeline_settings(
        tmp_path,
        output_dir=tmp_path,
        voice_gender="per_person_auto",
        clone_backend="cosyvoice",
        rvc_enabled=True,
        rvc_command_template="",
    )

    window = AppWindow.__new__(AppWindow)

    assert window._validate_settings(settings) is None


def test_xtts_per_person_accepts_empty_single_reference_when_python_is_configured(
    monkeypatch, tmp_path,
):
    python_bin = tmp_path / "python"
    python_bin.write_text("#!/bin/sh\n")
    python_bin.chmod(0o755)
    monkeypatch.setenv("OPENVOICE_PYTHON", str(python_bin))
    monkeypatch.setattr(app_window_module, "has_audio_stream", lambda _path: True)
    settings = make_pipeline_settings(
        tmp_path,
        output_dir=tmp_path,
        voice_gender="per_person_auto",
        clone_backend="xtts",
        rvc_enabled=True,
        rvc_reference_audio_path=None,
        rvc_command_template="",
        clone_workflow="auto_per_person",
    )

    window = AppWindow.__new__(AppWindow)

    assert window._validate_settings(settings) is None


def test_qwen3_per_person_accepts_empty_command_when_python_is_configured(
    monkeypatch, tmp_path,
):
    python_bin = tmp_path / "python"
    python_bin.write_text("#!/bin/sh\n")
    python_bin.chmod(0o755)
    monkeypatch.setenv("QWEN3_TTS_PYTHON", str(python_bin))
    monkeypatch.setattr(app_window_module, "has_audio_stream", lambda _path: True)
    monkeypatch.setattr(
        app_window_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="/usr/bin/sox\n", stderr=""),
    )
    settings = make_pipeline_settings(
        tmp_path,
        output_dir=tmp_path,
        voice_gender="per_person_auto",
        clone_backend="qwen3",
        rvc_enabled=True,
        rvc_reference_audio_path=None,
        rvc_command_template="",
        clone_workflow="auto_per_person",
    )

    window = AppWindow.__new__(AppWindow)

    assert window._validate_settings(settings) is None


def test_qwen3_per_person_reports_python_env_when_missing(monkeypatch, tmp_path):
    monkeypatch.delenv("QWEN3_TTS_PYTHON", raising=False)
    monkeypatch.setattr(app_window_module, "has_audio_stream", lambda _path: True)
    settings = make_pipeline_settings(
        tmp_path,
        output_dir=tmp_path,
        voice_gender="per_person_auto",
        clone_backend="qwen3",
        rvc_enabled=True,
        rvc_command_template="",
    )

    window = AppWindow.__new__(AppWindow)

    assert window._validate_settings(settings) == "Qwen3-TTS 1.7B requires QWEN3_TTS_PYTHON to be set."


def test_qwen3_generated_profiles_require_sox_in_clone_environment(monkeypatch, tmp_path):
    python_bin = tmp_path / "python"
    python_bin.write_text("#!/bin/sh\n")
    python_bin.chmod(0o755)
    male_reference = tmp_path / "male.wav"
    male_reference.write_bytes(b"fake")
    monkeypatch.setenv("QWEN3_TTS_PYTHON", str(python_bin))
    monkeypatch.setattr(app_window_module, "has_audio_stream", lambda _path: True)
    monkeypatch.setattr(
        app_window_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    settings = make_pipeline_settings(
        tmp_path,
        output_dir=tmp_path,
        voice_gender="auto",
        clone_backend="qwen3",
        rvc_enabled=False,
        voice_male_reference_path=male_reference,
        rvc_command_template="",
    )

    window = AppWindow.__new__(AppWindow)

    assert "requires SoX" in window._validate_settings(settings)


def test_gemini_tts_requires_api_key(monkeypatch, tmp_path):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(app_window_module, "has_audio_stream", lambda _path: True)
    settings = make_pipeline_settings(
        tmp_path,
        output_dir=tmp_path,
        tts_provider="gemini",
    )

    window = AppWindow.__new__(AppWindow)

    assert "Gemini TTS requires GEMINI_API_KEY" in window._validate_settings(settings)


def test_gemini_tts_accepts_fallback_api_key(monkeypatch, tmp_path):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY_FALLBACK", "fallback-key")
    monkeypatch.setattr(app_window_module, "has_audio_stream", lambda _path: True)
    settings = make_pipeline_settings(
        tmp_path,
        output_dir=tmp_path,
        tts_provider="gemini",
    )

    window = AppWindow.__new__(AppWindow)

    assert window._validate_settings(settings) is None


def test_gemini_per_person_does_not_require_clone_backend(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("COSYVOICE_PYTHON", raising=False)
    monkeypatch.setattr(app_window_module, "has_audio_stream", lambda _path: True)
    settings = make_pipeline_settings(
        tmp_path,
        output_dir=tmp_path,
        voice_gender="per_person_auto",
        clone_backend="cosyvoice",
        rvc_enabled=True,
        rvc_command_template="",
        tts_provider="gemini",
    )

    window = AppWindow.__new__(AppWindow)

    assert window._validate_settings(settings) is None


def test_cosyvoice_per_person_reports_python_env_instead_of_command_template(
    monkeypatch, tmp_path,
):
    monkeypatch.delenv("COSYVOICE_PYTHON", raising=False)
    monkeypatch.setattr(app_window_module, "has_audio_stream", lambda _path: True)
    settings = make_pipeline_settings(
        tmp_path,
        output_dir=tmp_path,
        voice_gender="per_person_auto",
        clone_backend="cosyvoice",
        rvc_enabled=True,
        rvc_command_template="",
    )

    window = AppWindow.__new__(AppWindow)

    assert window._validate_settings(settings) == "CosyVoice 2 requires COSYVOICE_PYTHON to be set."


def test_single_reference_status_warns_about_same_voice(tmp_path):
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"fake")
    settings = make_pipeline_settings(
        tmp_path,
        voice_gender="auto",
        rvc_enabled=True,
        rvc_reference_audio_path=reference,
        clone_workflow="single_reference",
    )

    status = clone_setup_status(settings)

    assert "single reference clone runs after TTS" in status
    assert "male and female voices may sound similar" in status


def test_clone_setup_status_explains_tts_only_mode_when_clone_disabled(tmp_path):
    settings = make_pipeline_settings(
        tmp_path,
        voice_gender="auto",
        rvc_enabled=False,
    )

    status = clone_setup_status(settings)

    assert "Clone stage is OFF" in status
    assert "no post-TTS voice conversion" in status


def test_auto_per_person_status_reports_emotion_enabled(tmp_path):
    settings = make_pipeline_settings(
        tmp_path,
        voice_gender="per_person_auto",
        rvc_enabled=True,
        clone_backend="cosyvoice",
        clone_workflow="auto_per_person",
        emotion_aware_clone=True,
    )

    status = clone_setup_status(settings)

    assert "Auto per-person clone" in status
    assert "Emotion matching is ON" in status


def test_auto_per_person_status_warns_when_backend_cannot_use_emotion(tmp_path):
    settings = make_pipeline_settings(
        tmp_path,
        voice_gender="per_person_auto",
        rvc_enabled=True,
        clone_backend="openvoice",
        clone_workflow="auto_per_person",
        emotion_aware_clone=True,
    )

    status = clone_setup_status(settings)

    assert "Emotion matching needs Qwen3-TTS, CosyVoice 2, or XTTS-v2" in status


def test_gender_profiles_status_requires_both_profiles(tmp_path):
    female_reference = tmp_path / "female.wav"
    female_reference.write_bytes(b"fake")
    settings = make_pipeline_settings(
        tmp_path,
        voice_gender="auto",
        rvc_enabled=True,
        voice_female_reference_path=female_reference,
        voice_male_reference_path=None,
        clone_workflow="gender_profiles",
    )

    status = clone_setup_status(settings)

    assert "select generated female and male profiles" in status
    assert "Missing: male" in status


def test_gender_profiles_validation_requires_generated_references(monkeypatch, tmp_path):
    monkeypatch.setattr(app_window_module, "has_audio_stream", lambda _path: True)
    settings = make_pipeline_settings(
        tmp_path,
        output_dir=tmp_path,
        voice_gender="auto",
        rvc_enabled=True,
        clone_backend="qwen3",
        rvc_command_template=(
            '/tmp/python -m modules.openvoice_voice_clone --input "{input}" '
            '--output "{output}" --reference "{reference}"'
        ),
        clone_workflow="gender_profiles",
        voice_female_reference_path=None,
        voice_male_reference_path=None,
    )

    window = AppWindow.__new__(AppWindow)

    error = window._validate_settings(settings)

    assert "Select generated female and male profiles" in error
    assert "Missing: female, male" in error


def test_voice_selection_resolves_generated_profile_label(tmp_path):
    female_reference = tmp_path / "fvoice1" / "reference.wav"
    male_reference = tmp_path / "mvoice2" / "reference.wav"
    female_reference.parent.mkdir()
    male_reference.parent.mkdir()
    female_reference.write_bytes(b"fake")
    male_reference.write_bytes(b"fake")
    window = AppWindow.__new__(AppWindow)
    window.voice_profiles = [
        VoiceProfile("Fvoice1", "fvoice1", "female", female_reference, female_reference, "", 12.0, "ok"),
        VoiceProfile("Mvoice2", "mvoice2", "male", male_reference, male_reference, "", 12.0, "ok"),
    ]
    female_combo = SimpleNamespace(
        currentData=lambda: {"kind": "edge", "voice": "km-KH-SreymomNeural"},
        currentText=lambda: "Fvoice1 (generated)",
    )
    male_combo = SimpleNamespace(
        currentData=lambda: {"kind": "edge", "voice": "km-KH-PisethNeural"},
        currentText=lambda: "Mvoice2 (generated)",
    )

    female_voice, female_ref = window._voice_selection(female_combo, "km-KH-SreymomNeural")
    male_voice, male_ref = window._voice_selection(male_combo, "km-KH-PisethNeural")

    assert female_voice == "km-KH-SreymomNeural"
    assert female_ref == female_reference
    assert male_voice == "km-KH-PisethNeural"
    assert male_ref == male_reference
    assert window._has_generated_profile_selection(female_combo)
    assert window._has_generated_profile_selection(male_combo)


def test_generated_profile_status_wins_over_clone_switch_off(tmp_path):
    male_reference = tmp_path / "male.wav"
    male_reference.write_bytes(b"fake")
    settings = make_pipeline_settings(
        tmp_path,
        voice_gender="auto",
        rvc_enabled=False,
        voice_male_reference_path=male_reference,
        clone_workflow="single_reference",
    )

    status = clone_setup_status(settings)

    assert "select generated female and male profiles" in status
    assert "Missing: female" in status
