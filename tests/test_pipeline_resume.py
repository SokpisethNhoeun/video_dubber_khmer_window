from __future__ import annotations

import base64
from pathlib import Path

import pytest

from conftest import make_pipeline_settings
from core import pipeline as pipeline_mod
from core.context import PipelineContext, PipelineSettings, Segment
from core.pipeline import DubbingPipeline
from core.session import STATUS_COMPLETED, STATUS_FAILED, DubbingSession


def _stub_stages(monkeypatch: pytest.MonkeyPatch, calls: list[str]):
    """Replace heavy stage functions with recording stubs that produce the
    minimal files/objects the pipeline needs."""

    def fake_extract(video, audio_wav, progress, cancel_event):
        calls.append("extract_audio")
        Path(audio_wav).write_bytes(b"wav")
        return 10.0

    def fake_transcribe(audio_wav, lang, model, device, duration, progress, log, cancel):
        calls.append("transcription")
        return [
            Segment(index=0, start=0.0, end=3.0, text="hello"),
            Segment(index=1, start=3.0, end=6.0, text="world"),
        ]

    def fake_translate(segments, lang, device, progress, log, cancel):
        calls.append("translation")
        for seg in segments:
            seg.translated_text = f"khmer-{seg.index}"
        return segments

    def fake_review(segments, *args, **kwargs):
        calls.append("transcript_review")
        return segments

    def fake_tts(segments, *args, work_dir=None, **kwargs):
        calls.append("tts")
        target = args[3] if len(args) > 3 else kwargs.get("work_dir")
        for seg in segments:
            path = Path(target) / f"tts_{seg.index}.mp3"
            path.write_bytes(b"mp3")
            seg.tts_path = path
        return segments

    def fake_clone(segments, *args, **kwargs):
        calls.append("voice_clone")
        return segments

    def fake_align(segments, final_audio, work_dir, duration, *args, **kwargs):
        calls.append("alignment")
        Path(final_audio).write_bytes(b"final")

    def fake_mux(video, audio, output, progress, cancel):
        calls.append("muxing")
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_bytes(b"mp4")

    def fake_exports(*args, **kwargs):
        return []

    monkeypatch.setattr(pipeline_mod, "extract_audio", fake_extract)
    monkeypatch.setattr(pipeline_mod, "transcribe_audio", fake_transcribe)
    monkeypatch.setattr(pipeline_mod, "translate_segments", fake_translate)
    monkeypatch.setattr(pipeline_mod, "review_segments", fake_review)
    monkeypatch.setattr(pipeline_mod, "synthesize_tts", fake_tts)
    monkeypatch.setattr(pipeline_mod, "optional_voice_clone", fake_clone)
    monkeypatch.setattr(pipeline_mod, "align_audio_segments", fake_align)
    monkeypatch.setattr(pipeline_mod, "mux_video", fake_mux)
    monkeypatch.setattr(pipeline_mod, "export_pipeline_outputs", fake_exports)


def _build(tmp_path: Path, session: DubbingSession | None = None):
    settings = (
        session.settings
        if session
        else make_pipeline_settings(tmp_path, translation_backend="nllb")
    )
    work_dir = session.work_dir if session else tmp_path / "temp" / "job_test"
    context = PipelineContext(settings=settings, work_dir=work_dir)
    if session is None:
        session = DubbingSession(work_dir=work_dir, settings=settings)
    return DubbingPipeline(context, session), session


def test_full_run_checkpoints_and_completes(tmp_path: Path, monkeypatch) -> None:
    calls: list[str] = []
    _stub_stages(monkeypatch, calls)
    pipe, session = _build(tmp_path)

    output = pipe.run()
    assert output.exists()
    loaded = DubbingSession.load(session.work_dir)
    assert loaded.status == STATUS_COMPLETED
    for stage in ["extract_audio", "transcription", "translation", "transcript_review",
                  "gender_detection", "tts", "voice_clone", "alignment"]:
        assert loaded.is_complete(stage), stage
    # Work dir must survive so the segment editor can reuse it.
    assert session.work_dir.exists()
    assert len(loaded.segments) == 2
    assert loaded.get_artifact("output_video") == output


def test_failure_persists_session_and_keeps_work_dir(tmp_path: Path, monkeypatch) -> None:
    calls: list[str] = []
    _stub_stages(monkeypatch, calls)

    def boom(*args, **kwargs):
        raise RuntimeError("tts exploded")

    monkeypatch.setattr(pipeline_mod, "synthesize_tts", boom)
    pipe, session = _build(tmp_path)

    with pytest.raises(RuntimeError, match="tts exploded"):
        pipe.run()

    assert session.work_dir.exists()
    loaded = DubbingSession.load(session.work_dir)
    assert loaded.status == STATUS_FAILED
    assert loaded.failed_stage == "tts"
    assert "tts exploded" in loaded.error
    assert loaded.is_complete("translation")
    assert not loaded.is_complete("tts")


def test_resume_skips_completed_stages(tmp_path: Path, monkeypatch) -> None:
    calls: list[str] = []
    _stub_stages(monkeypatch, calls)

    def boom(*args, **kwargs):
        raise RuntimeError("tts exploded")

    monkeypatch.setattr(pipeline_mod, "synthesize_tts", boom)
    pipe, session = _build(tmp_path)
    with pytest.raises(RuntimeError):
        pipe.run()

    # Second run: restore stubs and resume from the saved session.
    calls.clear()
    _stub_stages(monkeypatch, calls)
    resumed = DubbingSession.load(session.work_dir)
    pipe2, _ = _build(tmp_path, session=resumed)

    output = pipe2.run()
    assert output.exists()
    # Stages before TTS must NOT re-run.
    assert "extract_audio" not in calls
    assert "transcription" not in calls
    assert "translation" not in calls
    assert "transcript_review" not in calls
    # Remaining stages run.
    assert calls == ["tts", "alignment", "muxing"]
    assert DubbingSession.load(session.work_dir).status == STATUS_COMPLETED


def test_resume_fails_cleanly_when_audio_missing(tmp_path: Path, monkeypatch) -> None:
    calls: list[str] = []
    _stub_stages(monkeypatch, calls)

    def boom(*args, **kwargs):
        raise RuntimeError("tts exploded")

    monkeypatch.setattr(pipeline_mod, "synthesize_tts", boom)
    pipe, session = _build(tmp_path)
    with pytest.raises(RuntimeError):
        pipe.run()

    (session.work_dir / "source_mono_16k.wav").unlink()
    resumed = DubbingSession.load(session.work_dir)
    pipe2, _ = _build(tmp_path, session=resumed)
    with pytest.raises(RuntimeError, match="Cannot resume"):
        pipe2.run()


def test_no_session_keeps_legacy_cleanup(tmp_path: Path, monkeypatch) -> None:
    calls: list[str] = []
    _stub_stages(monkeypatch, calls)
    settings = make_pipeline_settings(tmp_path, translation_backend="nllb")
    work_dir = tmp_path / "temp" / "job_legacy"
    context = PipelineContext(settings=settings, work_dir=work_dir)
    pipe = DubbingPipeline(context)

    output = pipe.run()
    assert output.exists()
    # Without a session the old behavior applies: temp dir removed.
    assert not work_dir.exists()


def test_pipeline_uses_gemini_tts_before_alignment(tmp_path: Path, monkeypatch) -> None:
    calls: list[str] = []
    settings = make_pipeline_settings(
        tmp_path,
        tts_provider="gemini",
        voice_gender="female",
        output_dir=tmp_path / "out",
        translation_backend="nllb",
    )
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = tmp_path / "temp" / "job_gemini"
    session = DubbingSession(work_dir=work_dir, settings=settings)
    context = PipelineContext(settings=settings, work_dir=work_dir)
    pipe = DubbingPipeline(context, session)

    def fake_extract(video, audio_wav, progress, cancel_event):
        calls.append("extract_audio")
        Path(audio_wav).parent.mkdir(parents=True, exist_ok=True)
        Path(audio_wav).write_bytes(b"wav")
        return 2.0

    def fake_transcribe(audio_wav, lang, model, device, duration, progress, log, cancel):
        calls.append("transcription")
        return [
            Segment(index=0, start=0.0, end=1.0, text="hello"),
            Segment(index=1, start=1.0, end=2.0, text="world"),
        ]

    def fake_translate(segments, lang, device, progress, log, cancel):
        calls.append("translation")
        for segment in segments:
            segment.translated_text = f"សួស្តី {segment.index}"
        return segments

    def fake_review(segments, *args, **kwargs):
        calls.append("transcript_review")
        return segments

    def fake_gemini(*_args, **_kwargs):
        calls.append("gemini_tts")
        pcm = b"\x00\x00" * 240
        return {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "inlineData": {
                                    "data": base64.b64encode(pcm).decode("ascii"),
                                }
                            }
                        ]
                    }
                }
            ]
        }

    def fake_align(segments, final_audio, work_dir, duration, *args, **kwargs):
        calls.append("alignment")
        assert {segment.tts_group_id for segment in segments} == {"gemini_chunk_000", "gemini_chunk_001"}
        assert all(segment.tts_path and segment.tts_path.exists() for segment in segments)
        Path(final_audio).write_bytes(b"final")

    def fake_mux(video, audio, output, progress, cancel):
        calls.append("muxing")
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_bytes(b"mp4")

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_TTS_MAX_REQUESTS", "2")
    monkeypatch.setattr("modules.gemini_tts_engine.call_gemini_tts", fake_gemini)
    monkeypatch.setattr(pipeline_mod, "extract_audio", fake_extract)
    monkeypatch.setattr(pipeline_mod, "transcribe_audio", fake_transcribe)
    monkeypatch.setattr(pipeline_mod, "translate_segments", fake_translate)
    monkeypatch.setattr(pipeline_mod, "review_segments", fake_review)
    monkeypatch.setattr(pipeline_mod, "align_audio_segments", fake_align)
    monkeypatch.setattr(pipeline_mod, "mux_video", fake_mux)
    monkeypatch.setattr(pipeline_mod, "export_pipeline_outputs", lambda *args, **kwargs: [])

    output = pipe.run()

    assert output.exists()
    assert "gemini_tts" in calls
    assert "voice_clone" not in calls
