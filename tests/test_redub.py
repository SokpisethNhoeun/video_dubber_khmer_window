from __future__ import annotations

from pathlib import Path

import pytest

from conftest import make_pipeline_settings
from core import pipeline as pipeline_mod
from core.context import PipelineContext, PipelineSettings, Segment
from core.pipeline import DubbingPipeline
from core.session import DubbingSession


def _completed_session(tmp_path: Path) -> DubbingSession:
    work_dir = tmp_path / "temp" / "job_done"
    work_dir.mkdir(parents=True)
    settings = make_pipeline_settings(tmp_path)
    segments = []
    for i in range(3):
        tts = work_dir / f"tts_{i}.mp3"
        tts.write_bytes(b"mp3")
        segments.append(
            Segment(
                index=i,
                start=float(i * 2),
                end=float(i * 2 + 2),
                text=f"line {i}",
                translated_text=f"khmer {i}",
                tts_path=tts,
            )
        )
    session = DubbingSession(work_dir=work_dir, settings=settings, segments=segments)
    session.duration = 6.0
    for stage in ["extract_audio", "transcription", "speaker_detection", "translation",
                  "transcript_review", "gender_detection", "tts", "voice_clone", "alignment"]:
        session.mark_stage_complete(stage)
    session.mark_completed()
    session.save()
    return session


def test_redub_regenerates_only_edited_segments(tmp_path: Path, monkeypatch) -> None:
    session = _completed_session(tmp_path)
    session.segments[1].user_edited_text = "កែសម្រួល"

    tts_indices: list[int] = []
    align_counts: list[int] = []

    def fake_tts(segments, *args, **kwargs):
        for seg in segments:
            tts_indices.append(seg.index)
            path = session.work_dir / f"tts_{seg.index}_v2.mp3"
            path.write_bytes(b"mp3v2")
            seg.tts_path = path
        return segments

    def fake_align(segments, final_audio, work_dir, duration, *args, **kwargs):
        align_counts.append(len(segments))
        Path(final_audio).write_bytes(b"final")

    def fake_mux(video, audio, output, progress, cancel):
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_bytes(b"mp4")

    monkeypatch.setattr(pipeline_mod, "synthesize_tts", fake_tts)
    monkeypatch.setattr(pipeline_mod, "align_audio_segments", fake_align)
    monkeypatch.setattr(pipeline_mod, "mux_video", fake_mux)
    monkeypatch.setattr(pipeline_mod, "export_pipeline_outputs", lambda *a, **kw: [])

    context = PipelineContext(settings=session.settings, work_dir=session.work_dir)
    pipe = DubbingPipeline(context, session)
    output = pipe.redub_segments(session, [1])

    assert output.exists()
    # Only the edited segment was re-synthesized.
    assert tts_indices == [1]
    # Alignment rebuilds the whole 3-segment timeline.
    assert align_counts == [3]
    # Untouched segments keep their original audio.
    assert session.segments[0].tts_path == session.work_dir / "tts_0.mp3"
    assert session.segments[1].tts_path == session.work_dir / "tts_1_v2.mp3"
    # Session updated on disk.
    loaded = DubbingSession.load(session.work_dir)
    assert loaded.get_artifact("output_video") == output
    assert loaded.segments[1].user_edited_text == "កែសម្រួល"


def test_redub_raises_on_unknown_indices(tmp_path: Path, monkeypatch) -> None:
    session = _completed_session(tmp_path)
    context = PipelineContext(settings=session.settings, work_dir=session.work_dir)
    pipe = DubbingPipeline(context, session)
    with pytest.raises(ValueError, match="No matching segments"):
        pipe.redub_segments(session, [99])
