from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np
import pytest

from conftest import speech_like_signal, write_wav
from modules.reference_quality import assess_reference, _score_and_tier


SR = 16000


def _write_wav(path: Path, samples: np.ndarray, sr: int = SR) -> Path:
    return write_wav(path, samples, sr)


def _speech_like(seconds: float, snr_db: float = 25.0) -> np.ndarray:
    return speech_like_signal(seconds, sr=SR, snr_db=snr_db, f0=140.0)


def test_score_and_tier_flags_short_and_noisy():
    score, tier, reasons = _score_and_tier(
        duration_seconds=3.0,
        voiced_ratio=0.2,
        snr_db=3.0,
        clipping_ratio=0.02,
        music_harmonicity=0.9,
    )
    assert tier == "bad"
    assert score < 40.0
    # Every failure mode should surface a reason string so the user can act.
    assert any("short" in r for r in reasons)
    assert any("silence" in r for r in reasons)
    assert any("SNR" in r for r in reasons)
    assert any("clipping" in r for r in reasons)
    assert any("music" in r for r in reasons)


def test_score_and_tier_accepts_healthy_reference():
    score, tier, reasons = _score_and_tier(
        duration_seconds=20.0,
        voiced_ratio=0.7,
        snr_db=25.0,
        clipping_ratio=0.0,
        music_harmonicity=0.1,
    )
    assert tier == "good"
    assert score >= 90.0
    assert reasons == []


def test_assess_reference_missing_file_returns_bad(tmp_path: Path):
    quality = assess_reference(tmp_path / "does_not_exist.wav")
    assert quality.tier == "bad"
    assert quality.score == 0.0
    assert "missing" in quality.reasons[0]


def test_assess_reference_clean_speech_scores_at_least_weak(tmp_path: Path):
    wav_path = _write_wav(tmp_path / "clean.wav", _speech_like(20.0, snr_db=25.0))
    quality = assess_reference(wav_path)
    assert quality.duration_seconds == pytest.approx(20.0, abs=0.1)
    # Synthetic speech is imperfect but should clearly beat the "bad" threshold.
    assert quality.tier in {"good", "weak"}, quality.reasons
    assert quality.snr_db > 5.0


def test_assess_reference_short_clip_is_penalized(tmp_path: Path):
    wav_path = _write_wav(tmp_path / "short.wav", _speech_like(3.0, snr_db=25.0))
    quality = assess_reference(wav_path)
    assert quality.duration_seconds == pytest.approx(3.0, abs=0.1)
    # A 3s clip is not usable for cloning even if clean; expect at best "weak".
    assert quality.tier in {"weak", "bad"}
    assert any("short" in r for r in quality.reasons)


def test_to_dict_shape_is_json_serializable(tmp_path: Path):
    import json
    wav_path = _write_wav(tmp_path / "clip.wav", _speech_like(12.0, snr_db=20.0))
    quality = assess_reference(wav_path)
    payload = quality.to_dict()
    # Must be JSON-round-trippable to land in quality_report.json cleanly.
    encoded = json.dumps(payload)
    restored = json.loads(encoded)
    assert set(restored.keys()) >= {
        "path", "tier", "score", "duration_seconds", "voiced_ratio",
        "snr_db", "peak_dbfs", "clipping_ratio", "music_harmonicity", "reasons",
    }
