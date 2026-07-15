from __future__ import annotations

import math
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.context import PipelineSettings, Segment


# ---------------------------------------------------------------------------
# WAV file helpers
# ---------------------------------------------------------------------------

def write_wav(path: Path, samples: np.ndarray, sr: int = 16000) -> Path:
    """Write a mono 16-bit WAV file from float samples in [-1, 1].

    Used by tests that need real WAV headers (e.g. torchaudio.info, ffprobe).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sr)
        wav.writeframes(pcm.tobytes())
    return path


def write_silence_wav(path: Path, seconds: float, sr: int = 16000) -> Path:
    """Write a silent WAV of a given duration.

    Useful when tests only need a valid WAV with the right duration,
    not meaningful audio content.
    """
    return write_wav(path, np.zeros(int(seconds * sr), dtype=np.float32), sr=sr)


def speech_like_signal(
    seconds: float,
    sr: int = 16000,
    snr_db: float | None = None,
    f0: float = 140.0,
) -> np.ndarray:
    """Synthesise bursty harmonic tone that mimics voiced speech.

    Parameters
    ----------
    seconds : float
        Duration in seconds.
    sr : int
        Sample rate (default 16000).
    snr_db : float | None
        If given, add Gaussian noise at this signal-to-noise ratio.
        When *None*, no noise is added.
    f0 : float
        Fundamental frequency for the harmonic bursts (default 140 Hz).
    """
    n = int(seconds * sr)
    t = np.arange(n) / sr
    signal = np.zeros(n, dtype=np.float32)
    for start in np.arange(0.0, seconds, 0.35):
        end = start + 0.22
        if end > seconds:
            break
        i0 = int(start * sr)
        i1 = int(end * sr)
        burst_t = t[i0:i1] - start
        burst = (
            0.6 * np.sin(2 * math.pi * f0 * burst_t)
            + 0.3 * np.sin(2 * math.pi * 2 * f0 * burst_t)
            + 0.1 * np.sin(2 * math.pi * 3 * f0 * burst_t)
        ).astype(np.float32)
        envelope = np.hanning(len(burst)).astype(np.float32)
        signal[i0:i1] += burst * envelope * 0.7

    if snr_db is not None:
        signal_rms = float(np.sqrt(np.mean(signal ** 2) + 1e-9))
        noise_rms = signal_rms / (10 ** (snr_db / 20))
        noise = np.random.default_rng(0).normal(0.0, noise_rms, size=n).astype(np.float32)
        signal = signal + noise

    return signal


# ---------------------------------------------------------------------------
# PipelineSettings factory
# ---------------------------------------------------------------------------

def make_pipeline_settings(tmp_path: Path, **overrides) -> PipelineSettings:
    """Build a ``PipelineSettings`` with sensible test defaults.

    Any keyword argument is forwarded to the dataclass constructor,
    overriding the default value.
    """
    video = tmp_path / "video.mp4"
    if not video.exists():
        video.write_bytes(b"fake")

    defaults = dict(
        input_video=video,
        output_dir=tmp_path / "out",
        source_language="en",
        voice_gender="female",
        voice_female="km-KH-SreymomNeural",
        voice_male="km-KH-PisethNeural",
        speech_rate=0,
        pitch_hz=0,
        whisper_model="medium",
        device="cpu",
        preserve_bgm=False,
        enable_final_mastering=False,
        enable_persistent_cache=False,
        export_dubbed_audio=False,
        export_original_transcript=False,
        export_raw_khmer=False,
        export_improved_khmer=False,
        export_subtitles=False,
        export_quality_report=False,
        save_review_json=False,
        burn_subtitles=False,
    )
    defaults.update(overrides)
    return PipelineSettings(**defaults)


@pytest.fixture(autouse=True)
def mock_license_validation(monkeypatch):
    """Automatically mock LicenseClient.validate to return a valid license result during tests."""
    from licensing.client import LicenseClient, LicenseResult
    monkeypatch.setattr(
        LicenseClient,
        "validate",
        lambda self: LicenseResult(valid=True, message="Valid license (mocked for tests)", plan="pro")
    )

