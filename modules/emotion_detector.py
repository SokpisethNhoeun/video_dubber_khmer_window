from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from modules.emotion_reference import EmotionClip

NEUTRAL_INSTRUCT = "Speak naturally with moderate energy in a clear neutral tone"
CONFIDENCE_THRESHOLD = 0.52
MIN_MARGIN = 0.08

EMOTION_INSTRUCT: dict[str, str] = {
    "happy": "Speak happily with a warm upbeat tone and natural smiles in the voice",
    "sad": "Speak softly with a subdued melancholic tone and gentle sadness",
    "angry": "Speak firmly with intensity, tension, and controlled anger",
    "excited": "Speak excitedly with high energy, enthusiasm, and lively expression",
    "calm": "Speak calmly and peacefully with a relaxed steady tone",
    "surprise": "Speak with sudden surprise, raised energy, and expressive astonishment",
    "whisper": "Speak in a soft whisper with low volume and an intimate breathy tone",
    "shout": "Speak loudly with strong projection, urgency, and commanding presence",
    "neutral": NEUTRAL_INSTRUCT,
}


@dataclass(frozen=True)
class EmotionAnalysis:
    label: str
    instruct_text: str
    confidence: float
    energy: float
    pacing_offset_pct: int
    pitch_offset_hz: int
    is_neutral_fallback: bool


def _load_mono(clip_path: Path, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf
        samples, sr = sf.read(str(clip_path), dtype="float32")
    except Exception:
        import subprocess
        import struct

        proc = subprocess.run(
            ["ffmpeg", "-i", str(clip_path), "-f", "s16le", "-ac", "1", "-ar", str(target_sr), "-"],
            capture_output=True,
        )
        raw = proc.stdout
        n = len(raw) // 2
        samples = np.array(struct.unpack(f"<{n}h", raw[:n * 2]), dtype=np.float32) / 32768.0
        sr = target_sr

    if samples.ndim > 1:
        samples = samples.mean(axis=1)

    if sr != target_sr and len(samples) > 1:
        ratio = target_sr / sr
        indices = np.arange(0, len(samples), 1.0 / ratio).astype(int)
        indices = indices[indices < len(samples)]
        samples = samples[indices]

    return samples, target_sr


def _frame_features(samples: np.ndarray, sr: int) -> dict[str, float]:
    frame_size = max(1, int(0.025 * sr))
    hop = max(1, frame_size // 2)
    n_frames = max(1, (len(samples) - frame_size) // hop)

    energies: list[float] = []
    zcrs: list[float] = []
    for i in range(n_frames):
        start = i * hop
        frame = samples[start : start + frame_size]
        if len(frame) < 2:
            continue
        energies.append(float(np.sqrt(np.mean(frame**2))))
        zcrs.append(float(np.mean(np.abs(np.diff(np.sign(frame))) > 0)))

    rms = float(np.sqrt(np.mean(samples**2)))
    peak = float(np.max(np.abs(samples))) if len(samples) else 0.0
    zcr = float(np.mean(zcrs)) if zcrs else 0.0
    energy_std = float(np.std(energies)) if energies else 0.0
    energy_mean = float(np.mean(energies)) if energies else rms
    crest = peak / max(rms, 1e-6)
    spike_ratio = (max(energies) / max(energy_mean, 1e-6)) if energies else 1.0

    # Lightweight pitch estimate from autocorrelation on voiced frames.
    pitch_hz = 0.0
    if rms > 0.01 and len(samples) >= sr // 10:
        segment = samples[: min(len(samples), sr)]
        corr = np.correlate(segment, segment, mode="full")
        corr = corr[len(corr) // 2 :]
        min_lag = int(sr / 400)
        max_lag = min(int(sr / 70), len(corr) - 1)
        if max_lag > min_lag:
            search = corr[min_lag:max_lag]
            lag = int(np.argmax(search)) + min_lag
            if corr[lag] > 0.2 * corr[0]:
                pitch_hz = float(sr / lag)

    return {
        "rms": rms,
        "peak": peak,
        "zcr": zcr,
        "energy_std": energy_std,
        "crest": crest,
        "spike_ratio": spike_ratio,
        "pitch_hz": pitch_hz,
    }


def _score_emotions(features: dict[str, float]) -> dict[str, float]:
    rms = features["rms"]
    zcr = features["zcr"]
    energy_std = features["energy_std"]
    crest = features["crest"]
    spike_ratio = features["spike_ratio"]
    pitch = features["pitch_hz"]

    scores = {
        "whisper": 0.0,
        "shout": 0.0,
        "excited": 0.0,
        "angry": 0.0,
        "happy": 0.0,
        "sad": 0.0,
        "calm": 0.0,
        "surprise": 0.0,
        "neutral": 0.35,
    }

    peak = features["peak"]
    if rms < 0.025 and zcr < 0.07:
        scores["whisper"] += 1.2 - rms * 20.0
    if rms > 0.14 or peak > 0.75:
        scores["shout"] += min(1.5, rms * 6.0 + (crest - 2.0) * 0.25)
    if rms > 0.10 and energy_std > 0.03 and zcr > 0.06:
        scores["excited"] += min(1.4, rms * 4.0 + energy_std * 8.0)
    if rms > 0.09 and energy_std > 0.035 and crest > 2.5:
        scores["angry"] += min(1.3, energy_std * 10.0 + (crest - 2.0) * 0.3)
    if 0.05 < rms < 0.12 and pitch > 180:
        scores["happy"] += min(1.2, (pitch - 150) / 120.0 + energy_std * 5.0)
    if rms < 0.06 and energy_std < 0.025 and pitch < 170:
        scores["sad"] += min(1.2, (0.06 - rms) * 12.0 + (170 - max(pitch, 80)) / 120.0)
    if rms < 0.07 and zcr < 0.05 and energy_std < 0.03:
        scores["calm"] += min(1.2, (0.07 - rms) * 8.0 + (0.05 - zcr) * 6.0)
    if spike_ratio > 2.2 and energy_std > 0.03:
        scores["surprise"] += min(1.3, (spike_ratio - 2.0) * 0.5 + energy_std * 6.0)

    return scores


def _prosody_offsets(label: str) -> tuple[int, int]:
    mapping = {
        "excited": (10, 8),
        "shout": (12, 10),
        "surprise": (8, 12),
        "angry": (6, 6),
        "happy": (5, 6),
        "sad": (-6, -4),
        "calm": (-5, -2),
        "whisper": (-8, -6),
        "neutral": (0, 0),
    }
    return mapping.get(label, (0, 0))


QWEN3_EMOTION_SAMPLING: dict[str, dict[str, float]] = {
    "neutral": {"temperature": 0.85, "top_p": 0.95, "repetition_penalty": 1.05},
    "happy": {"temperature": 0.95, "top_p": 0.98, "repetition_penalty": 1.03},
    "excited": {"temperature": 1.0, "top_p": 1.0, "repetition_penalty": 1.02},
    "angry": {"temperature": 0.92, "top_p": 0.92, "repetition_penalty": 1.08},
    "sad": {"temperature": 0.82, "top_p": 0.88, "repetition_penalty": 1.06},
    "calm": {"temperature": 0.78, "top_p": 0.90, "repetition_penalty": 1.05},
    "whisper": {"temperature": 0.75, "top_p": 0.85, "repetition_penalty": 1.04},
    "shout": {"temperature": 0.98, "top_p": 0.95, "repetition_penalty": 1.03},
    "surprise": {"temperature": 0.96, "top_p": 0.97, "repetition_penalty": 1.03},
}


def qwen3_sampling_for_emotion(analysis: EmotionAnalysis | None) -> dict[str, float]:
    """Return Qwen3-TTS sampling overrides tuned to detected emotion."""
    if analysis is None or analysis.is_neutral_fallback:
        return dict(QWEN3_EMOTION_SAMPLING["neutral"])
    base = dict(QWEN3_EMOTION_SAMPLING.get(analysis.label, QWEN3_EMOTION_SAMPLING["neutral"]))
    energy_boost = min(0.12, analysis.energy * 0.08)
    if analysis.label in {"excited", "shout", "surprise", "happy", "angry"}:
        base["temperature"] = min(1.05, base["temperature"] + energy_boost)
    elif analysis.label in {"calm", "sad", "whisper"}:
        base["temperature"] = max(0.65, base["temperature"] - energy_boost)
    return base


def apply_emotion_to_clone_item(
    item: dict,
    clip: EmotionClip | None,
    emotion_mode: str,
    analysis: EmotionAnalysis | None,
    source_text: str = "",
) -> None:
    """Attach per-segment emotion reference and optional instruct text for cloning."""
    if clip is None or not clip.usable or not clip.clip_path.exists():
        return

    item["reference_path"] = str(clip.clip_path)
    item["emotion_reference_path"] = str(clip.clip_path)
    if source_text.strip():
        item["emotion_ref_text"] = source_text.strip()
    if analysis is not None:
        item["emotion_label"] = analysis.label
        item["emotion_confidence"] = f"{analysis.confidence:.2f}"
        item.update(qwen3_sampling_for_emotion(analysis))

    mode = (emotion_mode or "auto").strip().lower()
    if mode == "reference":
        return

    instruct = None
    if mode == "instruction":
        instruct = (analysis.instruct_text if analysis else None) or NEUTRAL_INSTRUCT
    elif mode == "auto" and analysis is not None and not analysis.is_neutral_fallback:
        instruct = analysis.instruct_text

    if instruct:
        item["instruct_text"] = instruct


def analyze_segment_emotion(clip_path: Path) -> EmotionAnalysis:
    """Analyze a source speech clip for emotion, pacing, and speaking style."""
    try:
        samples, sr = _load_mono(clip_path)
    except Exception:
        return EmotionAnalysis(
            label="neutral",
            instruct_text=NEUTRAL_INSTRUCT,
            confidence=0.0,
            energy=0.0,
            pacing_offset_pct=0,
            pitch_offset_hz=0,
            is_neutral_fallback=True,
        )

    if len(samples) < sr * 0.1:
        return EmotionAnalysis(
            label="neutral",
            instruct_text=NEUTRAL_INSTRUCT,
            confidence=0.0,
            energy=0.0,
            pacing_offset_pct=0,
            pitch_offset_hz=0,
            is_neutral_fallback=True,
        )

    features = _frame_features(samples, sr)
    scores = _score_emotions(features)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_label, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = best_score - second_score

    confidence = min(1.0, max(0.0, best_score * 0.45 + margin * 0.8))
    use_neutral = best_label == "neutral" or confidence < CONFIDENCE_THRESHOLD or margin < MIN_MARGIN
    label = "neutral" if use_neutral else best_label
    pacing, pitch = _prosody_offsets(label)
    energy = min(1.0, features["rms"] / 0.2)

    return EmotionAnalysis(
        label=label,
        instruct_text=EMOTION_INSTRUCT[label],
        confidence=confidence if not use_neutral else min(confidence, 0.4),
        energy=energy,
        pacing_offset_pct=pacing,
        pitch_offset_hz=pitch,
        is_neutral_fallback=use_neutral,
    )


def detect_segment_emotion(clip_path: Path) -> str:
    """Backward-compatible helper returning CosyVoice instruct text only."""
    return analyze_segment_emotion(clip_path).instruct_text


def analyze_emotion_clips(clips: dict[int, EmotionClip]) -> dict[int, EmotionAnalysis]:
    analyses: dict[int, EmotionAnalysis] = {}
    for index, clip in clips.items():
        if clip.usable and clip.clip_path.exists():
            analyses[index] = analyze_segment_emotion(clip.clip_path)
        else:
            analyses[index] = EmotionAnalysis(
                label="neutral",
                instruct_text=NEUTRAL_INSTRUCT,
                confidence=0.0,
                energy=0.0,
                pacing_offset_pct=0,
                pitch_offset_hz=0,
                is_neutral_fallback=True,
            )
    return analyses
