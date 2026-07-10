from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from core.context import Segment
from modules.diarizer import SpeakerTurn, assign_speakers_to_segments


def _segment(index: int = 0) -> Segment:
    return Segment(index=index, start=0.0, end=2.0, text="hello")


def _turn() -> SpeakerTurn:
    return SpeakerTurn(start=0.0, end=2.0, speaker_id="speaker_1")


def _fake_verification_module(similarity: float, calls: list[str]) -> SimpleNamespace:
    return SimpleNamespace(
        get_segment_embedding=lambda *_args, **_kwargs: calls.append("segment") or object(),
        get_file_embedding=lambda *_args, **_kwargs: calls.append("file") or object(),
        compute_similarity=lambda *_args, **_kwargs: calls.append("similarity") or similarity,
    )


def test_auto_reference_mapping_keeps_diarization_assignment_without_verification(
    tmp_path: Path,
    monkeypatch,
) -> None:
    audio_path = tmp_path / "source.wav"
    reference_path = tmp_path / "reference.wav"
    audio_path.write_bytes(b"")
    reference_path.write_bytes(b"")
    calls: list[str] = []
    monkeypatch.setitem(sys.modules, "modules.speaker_verification", _fake_verification_module(0.1, calls))

    [assigned] = assign_speakers_to_segments(
        [_segment()],
        [_turn()],
        {
            "speaker_1": {
                "label": "Speaker 1",
                "reference_audio_path": str(reference_path),
                "reference_status": "auto",
                "auto_reference": "true",
            }
        },
        None,
        audio_path,
    )

    assert assigned.speaker_id == "speaker_1"
    assert assigned.speaker_label == "Speaker 1"
    assert calls == []


def test_manual_reference_low_confidence_keeps_diarization_assignment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    audio_path = tmp_path / "source.wav"
    reference_path = tmp_path / "reference.wav"
    audio_path.write_bytes(b"")
    reference_path.write_bytes(b"")
    calls: list[str] = []
    monkeypatch.setitem(sys.modules, "modules.speaker_verification", _fake_verification_module(0.32, calls))

    [assigned] = assign_speakers_to_segments(
        [_segment()],
        [_turn()],
        {"speaker_1": {"label": "Manual Speaker", "reference_audio_path": str(reference_path)}},
        None,
        audio_path,
    )

    assert assigned.speaker_id == "speaker_1"
    assert assigned.speaker_label == "Manual Speaker"
    assert calls == ["segment", "file", "similarity"]


def test_manual_reference_obvious_mismatch_rejects_assignment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    audio_path = tmp_path / "source.wav"
    reference_path = tmp_path / "reference.wav"
    audio_path.write_bytes(b"")
    reference_path.write_bytes(b"")
    calls: list[str] = []
    monkeypatch.setitem(sys.modules, "modules.speaker_verification", _fake_verification_module(0.15, calls))

    [assigned] = assign_speakers_to_segments(
        [_segment()],
        [_turn()],
        {"speaker_1": {"label": "Manual Speaker", "reference_audio_path": str(reference_path)}},
        None,
        audio_path,
    )

    assert assigned.speaker_id is None
    assert assigned.speaker_label is None
    assert calls == ["segment", "file", "similarity"]
