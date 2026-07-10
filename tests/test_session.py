from __future__ import annotations

from pathlib import Path

from conftest import make_pipeline_settings
from core.context import PipelineSettings, Segment
from core.session import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    DubbingSession,
    delete_session,
    list_sessions,
    prune_sessions,
    segment_from_dict,
    segment_to_dict,
    settings_from_dict,
    settings_to_dict,
)


def _make_settings(tmp_path: Path) -> PipelineSettings:
    return make_pipeline_settings(
        tmp_path,
        source_language="english",
        glossary_path=tmp_path / "glossary.csv",
        input_videos=[tmp_path / "a.mp4", tmp_path / "b.mp4"],
    )


def _make_session(tmp_path: Path) -> DubbingSession:
    work_dir = tmp_path / "temp" / "job_abc123"
    return DubbingSession(work_dir=work_dir, settings=_make_settings(tmp_path))


def test_settings_round_trip(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    restored = settings_from_dict(settings_to_dict(settings))
    assert restored == settings
    assert isinstance(restored.input_video, Path)
    assert restored.glossary_path == tmp_path / "glossary.csv"
    assert restored.input_videos == [tmp_path / "a.mp4", tmp_path / "b.mp4"]
    assert restored.voice_female_reference_path is None


def test_settings_from_legacy_overlay_position_populates_new_positions(tmp_path: Path) -> None:
    data = settings_to_dict(_make_settings(tmp_path))
    data["overlay_position"] = "top_left"
    data.pop("overlay_text_position", None)
    data.pop("overlay_image_position", None)

    restored = settings_from_dict(data)

    assert restored.overlay_position == "top_left"
    assert restored.overlay_text_position == "top_left"
    assert restored.overlay_image_position == "top_left"


def test_segment_round_trip_with_khmer_and_paths(tmp_path: Path) -> None:
    work_dir = tmp_path / "job"
    seg = Segment(
        index=3,
        start=1.5,
        end=4.25,
        text="Hello world",
        translated_text="សួស្តី​ពិភពលោក",
        user_edited_text="សួស្តី",
        speaker_id="spk_1",
        speaker_label="Host",
        tts_path=work_dir / "tts" / "00003.mp3",
        cloned_path=None,
    )
    data = segment_to_dict(seg, work_dir)
    # Paths inside work_dir stored relative so the temp folder can move.
    assert data["tts_path"] == str(Path("tts") / "00003.mp3")
    restored = segment_from_dict(data, work_dir)
    assert restored == seg
    assert restored.tts_text == "សួស្តី"


def test_segment_path_outside_work_dir_stays_absolute(tmp_path: Path) -> None:
    work_dir = tmp_path / "job"
    external = tmp_path / "elsewhere" / "clip.wav"
    seg = Segment(index=0, start=0.0, end=1.0, text="x", cloned_path=external)
    data = segment_to_dict(seg, work_dir)
    assert data["cloned_path"] == str(external)
    assert segment_from_dict(data, work_dir).cloned_path == external


def test_session_save_load_round_trip(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    session.segments = [
        Segment(index=0, start=0.0, end=2.0, text="a", translated_text="ក"),
        Segment(index=1, start=2.0, end=4.0, text="b", tts_path=session.work_dir / "tts" / "1.mp3"),
    ]
    session.duration = 12.5
    session.segment_genders = {0: "female", 1: "male"}
    session.speaker_mappings = {"spk_1": {"label": "Host"}}
    session.set_artifact("audio_wav", session.work_dir / "source_mono_16k.wav")
    session.mark_stage_complete("extract_audio")
    session.mark_stage_complete("transcription")
    session.mark_stage_complete("transcription")  # idempotent
    session.save()

    loaded = DubbingSession.load(session.work_dir)
    assert loaded.session_id == "job_abc123"
    assert loaded.completed_stages == ["extract_audio", "transcription"]
    assert loaded.duration == 12.5
    assert loaded.segment_genders == {0: "female", 1: "male"}
    assert loaded.speaker_mappings == {"spk_1": {"label": "Host"}}
    assert loaded.get_artifact("audio_wav") == session.work_dir / "source_mono_16k.wav"
    assert loaded.segments == session.segments
    assert loaded.settings == session.settings
    assert loaded.is_complete("transcription")
    assert not loaded.is_complete("tts")


def test_session_status_transitions(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    session.mark_failed("tts", "boom")
    session.save()
    loaded = DubbingSession.load(session.work_dir)
    assert loaded.status == STATUS_FAILED
    assert loaded.failed_stage == "tts"
    assert loaded.error == "boom"

    loaded.mark_completed()
    assert loaded.status == STATUS_COMPLETED
    assert loaded.failed_stage == ""


def test_list_sessions_skips_corrupt_json(tmp_path: Path) -> None:
    temp_dir = tmp_path / "temp"
    good = DubbingSession(work_dir=temp_dir / "job_good", settings=_make_settings(tmp_path))
    good.save()
    corrupt_dir = temp_dir / "job_bad"
    corrupt_dir.mkdir(parents=True)
    (corrupt_dir / "session.json").write_text("{not json", encoding="utf-8")

    summaries = list_sessions(temp_dir)
    assert [s.session_id for s in summaries] == ["job_good"]
    assert summaries[0].video_name == "video.mp4"


def test_delete_and_prune_sessions(tmp_path: Path) -> None:
    temp_dir = tmp_path / "temp"
    sessions = []
    for i in range(4):
        s = DubbingSession(work_dir=temp_dir / f"job_{i}", settings=_make_settings(tmp_path))
        s.mark_completed()
        s.updated_at = f"2026-07-0{i + 1}T00:00:00"
        s.work_dir.mkdir(parents=True, exist_ok=True)
        s.path.write_text(
            __import__("json").dumps(s.to_dict(), ensure_ascii=False), encoding="utf-8"
        )
        sessions.append(s)
    failed = DubbingSession(work_dir=temp_dir / "job_failed", settings=_make_settings(tmp_path))
    failed.mark_failed("tts", "x")
    failed.save()

    removed = prune_sessions(temp_dir, keep=2)
    assert removed == 2
    remaining = list_sessions(temp_dir)
    # Newest 2 completed + the failed one survive.
    ids = {s.session_id for s in remaining}
    assert ids == {"job_2", "job_3", "job_failed"}

    delete_session(temp_dir / "job_failed")
    assert not (temp_dir / "job_failed").exists()
