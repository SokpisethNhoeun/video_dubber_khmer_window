from __future__ import annotations

import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import write_silence_wav as _write_silence_wav
from modules import speaker_verification


def _fake_torchaudio():
    """Return a fake torchaudio module that reads real WAV headers."""
    def _info(path):
        with wave.open(str(path), "rb") as w:
            return SimpleNamespace(
                num_frames=w.getnframes(),
                sample_rate=w.getframerate(),
            )
    return SimpleNamespace(info=_info)


def _patch_embeddings(monkeypatch: pytest.MonkeyPatch, similarity: float) -> None:
    monkeypatch.setattr(
        speaker_verification, "_torchaudio", _fake_torchaudio
    )
    monkeypatch.setattr(
        speaker_verification,
        "get_file_embedding",
        lambda path: SimpleNamespace(path=str(path)),
    )
    monkeypatch.setattr(
        speaker_verification, "compute_similarity", lambda a, b: similarity
    )


def test_verify_cloned_segments_returns_mean_similarity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference = _write_silence_wav(tmp_path / "ref.wav", 5.0)
    cloned = [
        _write_silence_wav(tmp_path / f"c{i}.wav", 2.0) for i in range(3)
    ]
    _patch_embeddings(monkeypatch, similarity=0.82)

    mean_sim, count = speaker_verification.verify_cloned_segments(
        "spk_1", reference, cloned
    )
    assert count == 3
    assert mean_sim == pytest.approx(0.82)


def test_verify_cloned_segments_skips_too_short_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference = _write_silence_wav(tmp_path / "ref.wav", 5.0)
    long_clip = _write_silence_wav(tmp_path / "long.wav", 3.0)
    tiny_clip = _write_silence_wav(tmp_path / "tiny.wav", 0.3)
    _patch_embeddings(monkeypatch, similarity=0.6)

    mean_sim, count = speaker_verification.verify_cloned_segments(
        "spk_1", reference, [long_clip, tiny_clip]
    )
    # Tiny clip is below CLONE_MIN_SAMPLE_SECONDS and must be discarded so
    # we don't judge the clone off half a phoneme.
    assert count == 1
    assert mean_sim == pytest.approx(0.6)


def test_verify_cloned_segments_returns_zero_when_nothing_usable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference = _write_silence_wav(tmp_path / "ref.wav", 5.0)
    # All clips shorter than the minimum sample duration.
    tiny_clips = [_write_silence_wav(tmp_path / f"t{i}.wav", 0.3) for i in range(2)]
    _patch_embeddings(monkeypatch, similarity=0.9)

    mean_sim, count = speaker_verification.verify_cloned_segments(
        "spk_1", reference, tiny_clips
    )
    # Caller should treat (0.0, 0) as "verification skipped", not "failed".
    assert count == 0
    assert mean_sim == 0.0


def test_pipeline_verify_wipes_cloned_path_on_low_similarity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the clone doesn't sound like the reference, the pipeline must
    revert those segments to base TTS by clearing cloned_path."""
    from core import pipeline as pipeline_mod
    from core.pipeline import DubbingPipeline

    reference = _write_silence_wav(tmp_path / "ref.wav", 5.0)
    clone_a = _write_silence_wav(tmp_path / "clone.wav", 2.0)

    segments = [SimpleNamespace(speaker_id="spk_1", cloned_path=clone_a, index=0)]
    speaker_mappings = {
        "spk_1": {
            "label": "Speaker 1",
            "reference_audio_path": str(reference),
            "cleaned_reference_audio_path": str(reference),
        }
    }
    context = SimpleNamespace(
        emit_log=lambda m: None,
        quality_report=SimpleNamespace(
            speaker_quality=[{"speaker": "Speaker 1", "tier": "good", "score": 90}],
            voice_clone_failures=[],
        ),
    )

    monkeypatch.setattr(pipeline_mod, "verify_cloned_segments", lambda *a, **kw: (0.30, 1))
    pipe = DubbingPipeline.__new__(DubbingPipeline)
    pipe.context = context
    pipe._verify_clone_similarity(segments, speaker_mappings)

    assert segments[0].cloned_path is None
    assert context.quality_report.speaker_quality[0]["clone_verdict"] == "low"
    assert len(context.quality_report.voice_clone_failures) == 1


def test_pipeline_verify_keeps_cloned_path_on_strong_similarity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from core import pipeline as pipeline_mod
    from core.pipeline import DubbingPipeline

    reference = _write_silence_wav(tmp_path / "ref.wav", 5.0)
    clone_a = _write_silence_wav(tmp_path / "clone.wav", 2.0)
    segments = [SimpleNamespace(speaker_id="spk_1", cloned_path=clone_a, index=0)]
    speaker_mappings = {
        "spk_1": {
            "label": "Speaker 1",
            "reference_audio_path": str(reference),
            "cleaned_reference_audio_path": str(reference),
        }
    }
    context = SimpleNamespace(
        emit_log=lambda m: None,
        quality_report=SimpleNamespace(
            speaker_quality=[{"speaker": "Speaker 1", "tier": "good", "score": 90}],
            voice_clone_failures=[],
        ),
    )

    monkeypatch.setattr(pipeline_mod, "verify_cloned_segments", lambda *a, **kw: (0.75, 2))
    pipe = DubbingPipeline.__new__(DubbingPipeline)
    pipe.context = context
    pipe._verify_clone_similarity(segments, speaker_mappings)

    assert segments[0].cloned_path == clone_a
    entry = context.quality_report.speaker_quality[0]
    assert entry["clone_verdict"] == "strong"
    assert entry["clone_similarity"] == 0.75
    assert context.quality_report.voice_clone_failures == []


def test_verify_cloned_segments_caps_sample_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference = _write_silence_wav(tmp_path / "ref.wav", 5.0)
    many_clones = [
        _write_silence_wav(tmp_path / f"c{i}.wav", 2.0)
        for i in range(speaker_verification.CLONE_MAX_SAMPLES_PER_SPEAKER + 3)
    ]
    calls: list[str] = []

    def counted_embedding(path):
        calls.append(str(path))
        return SimpleNamespace(path=str(path))

    monkeypatch.setattr(speaker_verification, "_torchaudio", _fake_torchaudio)
    monkeypatch.setattr(speaker_verification, "get_file_embedding", counted_embedding)
    monkeypatch.setattr(speaker_verification, "compute_similarity", lambda a, b: 0.7)

    mean_sim, count = speaker_verification.verify_cloned_segments(
        "spk_1", reference, many_clones
    )
    # One embed for the reference + one per sampled clone.
    assert count == speaker_verification.CLONE_MAX_SAMPLES_PER_SPEAKER
    assert len(calls) == speaker_verification.CLONE_MAX_SAMPLES_PER_SPEAKER + 1
    assert mean_sim == pytest.approx(0.7)


def test_voice_clone_summary_reports_fallback_segments(tmp_path: Path) -> None:
    from core.context import Segment
    from modules.voice_cloner import _log_clone_result_summary

    cloned_path = tmp_path / "clone.wav"
    cloned_path.write_bytes(b"wav")
    segments = [
        Segment(index=0, start=0, end=1, text="a", cloned_path=cloned_path),
        Segment(index=1, start=1, end=2, text="b"),
    ]
    logs: list[str] = []

    _log_clone_result_summary(segments, True, logs.append)

    assert logs == [
        "Voice clone result: 1/2 segment(s) cloned; 1 segment(s) use default Khmer TTS fallback"
    ]
