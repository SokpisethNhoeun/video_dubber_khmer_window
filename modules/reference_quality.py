from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np


QualityTier = Literal["good", "weak", "bad"]


# Thresholds tuned for voice-clone reference audio (16 kHz mono expected).
# References cleaner than these produce noticeably better clones.
MIN_DURATION_SECONDS = 8.0
GOOD_DURATION_SECONDS = 15.0
MIN_VOICED_RATIO = 0.35
GOOD_VOICED_RATIO = 0.60
MIN_SNR_DB = 8.0
GOOD_SNR_DB = 18.0
CLIPPING_RATIO_BAD = 0.005  # >0.5% samples clipped => bad
MUSIC_HARMONICITY_BAD = 0.70  # sustained-tonal energy ratio suggesting music bed
FRAME_MS = 25
HOP_MS = 10


@dataclass
class ReferenceQuality:
    """Per-reference audio quality assessment used to gate voice cloning."""

    path: Path
    tier: QualityTier
    score: float  # 0..100 composite
    duration_seconds: float
    voiced_ratio: float
    snr_db: float
    peak_dbfs: float
    clipping_ratio: float
    music_harmonicity: float
    reasons: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.tier == "good"

    @property
    def clone_recommended(self) -> bool:
        # "weak" references still clone but with a warning; "bad" should be skipped/replaced.
        return self.tier in {"good", "weak"}

    def to_dict(self) -> dict:
        return {
            "path": str(self.path),
            "tier": self.tier,
            "score": round(self.score, 1),
            "duration_seconds": round(self.duration_seconds, 2),
            "voiced_ratio": round(self.voiced_ratio, 3),
            "snr_db": round(self.snr_db, 1),
            "peak_dbfs": round(self.peak_dbfs, 1),
            "clipping_ratio": round(self.clipping_ratio, 4),
            "music_harmonicity": round(self.music_harmonicity, 3),
            "reasons": self.reasons,
        }


def _load_mono_16k(path: Path) -> tuple[np.ndarray, int]:
    """Load audio as mono float32 at 16 kHz. Uses torchaudio (already a project dep)."""
    import torch
    import torchaudio

    signal, sr = torchaudio.load(str(path))
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)
    if sr != 16000:
        signal = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(signal)
        sr = 16000
    return signal.squeeze(0).cpu().numpy().astype(np.float32), sr


def _frame_energy(samples: np.ndarray, sr: int) -> np.ndarray:
    frame_len = int(sr * FRAME_MS / 1000)
    hop_len = int(sr * HOP_MS / 1000)
    if len(samples) < frame_len:
        return np.array([np.sqrt(np.mean(samples**2) + 1e-12)])
    n_frames = 1 + (len(samples) - frame_len) // hop_len
    energies = np.empty(n_frames, dtype=np.float32)
    for i in range(n_frames):
        start = i * hop_len
        frame = samples[start : start + frame_len]
        energies[i] = np.sqrt(np.mean(frame**2) + 1e-12)
    return energies


def _voiced_ratio_and_snr(samples: np.ndarray, sr: int) -> tuple[float, float]:
    """Energy-based voice-activity split. Returns (voiced_fraction, SNR_dB).

    Not a phoneme-accurate VAD, but robust enough to catch silence-heavy or
    noise-dominated references without adding a webrtcvad dependency.
    """
    energies = _frame_energy(samples, sr)
    if len(energies) == 0:
        return 0.0, 0.0

    # Adaptive threshold: noise floor from lowest 20th percentile,
    # speech is anything 8 dB above it.
    energies_db = 20 * np.log10(energies + 1e-9)
    noise_floor_db = float(np.percentile(energies_db, 20))
    speech_thr_db = noise_floor_db + 8.0
    voiced_mask = energies_db > speech_thr_db
    voiced_ratio = float(np.mean(voiced_mask))

    voiced_energy = energies[voiced_mask]
    unvoiced_energy = energies[~voiced_mask]
    if voiced_energy.size == 0 or unvoiced_energy.size == 0:
        return voiced_ratio, 0.0
    signal_rms = float(np.sqrt(np.mean(voiced_energy**2) + 1e-12))
    noise_rms = float(np.sqrt(np.mean(unvoiced_energy**2) + 1e-12))
    snr_db = 20 * np.log10((signal_rms + 1e-9) / (noise_rms + 1e-9))
    return voiced_ratio, float(snr_db)


def _clipping_ratio(samples: np.ndarray) -> tuple[float, float]:
    """Return (clipped_sample_fraction, peak_dbfs)."""
    if samples.size == 0:
        return 0.0, -120.0
    peak = float(np.max(np.abs(samples)))
    peak_dbfs = 20 * np.log10(peak + 1e-9)
    # Consider anything within 0.1 dB of full-scale as clipped.
    clipped = np.sum(np.abs(samples) >= 0.988)
    return float(clipped) / samples.size, float(peak_dbfs)


def _music_harmonicity(samples: np.ndarray, sr: int) -> float:
    """Rough music/BGM presence score using spectral flatness stability.

    Music beds tend to have sustained tonal energy in the mid-band with
    low spectral flatness over long windows. Pure speech has bursty
    formants with higher variance in flatness. Returns 0..1 where higher
    ~= more music-like.
    """
    if len(samples) < sr:  # need at least 1s
        return 0.0

    win = int(sr * 0.5)  # 500 ms windows
    hop = int(sr * 0.25)
    n = 1 + (len(samples) - win) // hop
    if n < 4:
        return 0.0

    flatnesses = np.empty(n, dtype=np.float32)
    hann = np.hanning(win).astype(np.float32)
    for i in range(n):
        start = i * hop
        frame = samples[start : start + win] * hann
        spec = np.abs(np.fft.rfft(frame)) + 1e-9
        # Geometric mean / arithmetic mean -> spectral flatness in [0, 1]
        log_mean = np.mean(np.log(spec))
        arith_mean = np.mean(spec)
        flatnesses[i] = float(np.exp(log_mean) / (arith_mean + 1e-9))

    # Low variance in flatness across the clip suggests sustained tonal content.
    flatness_stability = 1.0 - float(np.clip(np.std(flatnesses) * 10.0, 0.0, 1.0))
    # Combine with average flatness (music tends to have moderate flatness, ~0.2-0.5).
    avg_flatness = float(np.mean(flatnesses))
    music_score = flatness_stability * (1.0 - abs(avg_flatness - 0.35) * 1.5)
    return float(np.clip(music_score, 0.0, 1.0))


def _score_and_tier(
    duration_seconds: float,
    voiced_ratio: float,
    snr_db: float,
    clipping_ratio: float,
    music_harmonicity: float,
) -> tuple[float, QualityTier, list[str]]:
    reasons: list[str] = []
    score = 100.0

    if duration_seconds < MIN_DURATION_SECONDS:
        score -= 40.0
        reasons.append(f"too short ({duration_seconds:.1f}s; want {GOOD_DURATION_SECONDS:.0f}s+)")
    elif duration_seconds < GOOD_DURATION_SECONDS:
        score -= 15.0 * (GOOD_DURATION_SECONDS - duration_seconds) / (GOOD_DURATION_SECONDS - MIN_DURATION_SECONDS)

    if voiced_ratio < MIN_VOICED_RATIO:
        score -= 25.0
        reasons.append(f"mostly silence ({voiced_ratio*100:.0f}% voiced)")
    elif voiced_ratio < GOOD_VOICED_RATIO:
        score -= 10.0

    if snr_db < MIN_SNR_DB:
        score -= 25.0
        reasons.append(f"low SNR ({snr_db:.1f} dB)")
    elif snr_db < GOOD_SNR_DB:
        score -= 10.0 * (GOOD_SNR_DB - snr_db) / (GOOD_SNR_DB - MIN_SNR_DB)

    if clipping_ratio > CLIPPING_RATIO_BAD:
        score -= 20.0
        reasons.append(f"clipping ({clipping_ratio*100:.2f}% of samples)")

    if music_harmonicity > MUSIC_HARMONICITY_BAD:
        score -= 20.0
        reasons.append(f"music/BGM likely present (harmonicity {music_harmonicity:.2f})")

    score = max(0.0, min(100.0, score))
    if score >= 70.0:
        tier: QualityTier = "good"
    elif score >= 45.0:
        tier = "weak"
    else:
        tier = "bad"
    return score, tier, reasons


def assess_reference(path: Path) -> ReferenceQuality:
    """Compute a full quality assessment for a reference audio file.

    Never raises for a readable audio file; instead returns a "bad" tier with
    reasons so the caller can decide (skip clone, warn user, request replacement).
    """
    resolved = Path(path).expanduser()
    if not resolved.exists():
        return ReferenceQuality(
            path=resolved,
            tier="bad",
            score=0.0,
            duration_seconds=0.0,
            voiced_ratio=0.0,
            snr_db=0.0,
            peak_dbfs=-120.0,
            clipping_ratio=0.0,
            music_harmonicity=0.0,
            reasons=["file missing"],
        )

    try:
        samples, sr = _load_mono_16k(resolved)
    except Exception as exc:
        return ReferenceQuality(
            path=resolved,
            tier="bad",
            score=0.0,
            duration_seconds=0.0,
            voiced_ratio=0.0,
            snr_db=0.0,
            peak_dbfs=-120.0,
            clipping_ratio=0.0,
            music_harmonicity=0.0,
            reasons=[f"could not decode audio: {exc}"],
        )

    duration = float(len(samples)) / float(sr) if sr else 0.0
    voiced_ratio, snr_db = _voiced_ratio_and_snr(samples, sr)
    clip_ratio, peak_dbfs = _clipping_ratio(samples)
    music = _music_harmonicity(samples, sr)
    score, tier, reasons = _score_and_tier(duration, voiced_ratio, snr_db, clip_ratio, music)

    return ReferenceQuality(
        path=resolved,
        tier=tier,
        score=score,
        duration_seconds=duration,
        voiced_ratio=voiced_ratio,
        snr_db=snr_db,
        peak_dbfs=peak_dbfs,
        clipping_ratio=clip_ratio,
        music_harmonicity=music,
        reasons=reasons,
    )
