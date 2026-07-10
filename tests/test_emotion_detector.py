from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import write_wav
from modules.emotion_detector import (
    NEUTRAL_INSTRUCT,
    EmotionAnalysis,
    analyze_segment_emotion,
    apply_emotion_to_clone_item,
    detect_segment_emotion,
)
from modules.emotion_reference import EmotionClip


def _write_wav(path: Path, samples: np.ndarray, sr: int = 16000) -> None:
    write_wav(path, samples, sr)


class TestDetectSegmentEmotion:
    def test_high_energy_returns_excited_or_shout(self, tmp_path):
        sr = 16000
        t = np.linspace(0, 2, sr * 2, endpoint=False)
        samples = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.random.randn(len(t))
        samples = np.clip(samples, -1, 1).astype(np.float32)
        wav = tmp_path / "loud.wav"
        _write_wav(wav, samples, sr)

        analysis = analyze_segment_emotion(wav)
        result = detect_segment_emotion(wav)
        assert analysis.label in {"excited", "shout", "angry", "happy", "surprise", "neutral"}
        assert result

    def test_low_energy_returns_calm_or_whisper(self, tmp_path):
        sr = 16000
        t = np.linspace(0, 2, sr * 2, endpoint=False)
        samples = 0.005 * np.sin(2 * np.pi * 200 * t)
        samples = samples.astype(np.float32)
        wav = tmp_path / "quiet.wav"
        _write_wav(wav, samples, sr)

        analysis = analyze_segment_emotion(wav)
        assert analysis.label in {"calm", "whisper", "sad", "neutral"}

    def test_moderate_energy_can_fallback_to_neutral(self, tmp_path):
        sr = 16000
        t = np.linspace(0, 2, sr * 2, endpoint=False)
        samples = 0.08 * np.sin(2 * np.pi * 300 * t)
        samples = samples.astype(np.float32)
        wav = tmp_path / "moderate.wav"
        _write_wav(wav, samples, sr)

        analysis = analyze_segment_emotion(wav)
        assert isinstance(analysis, EmotionAnalysis)
        assert analysis.instruct_text

    def test_nonexistent_file_returns_neutral_fallback(self):
        analysis = analyze_segment_emotion(Path("/nonexistent/audio.wav"))
        assert analysis.is_neutral_fallback
        assert analysis.instruct_text == NEUTRAL_INSTRUCT

    def test_very_short_audio_returns_neutral_fallback(self, tmp_path):
        sr = 16000
        samples = np.zeros(100, dtype=np.float32)
        wav = tmp_path / "tiny.wav"
        _write_wav(wav, samples, sr)

        analysis = analyze_segment_emotion(wav)
        assert analysis.is_neutral_fallback


def test_apply_emotion_auto_mode_adds_instruct_when_confident(tmp_path):
    clip_path = tmp_path / "clip.wav"
    _write_wav(clip_path, np.full(16000, 0.2, dtype=np.float32))
    clip = EmotionClip(
        segment_index=0,
        clip_path=clip_path,
        duration=1.0,
        snr_db=20.0,
        usable=True,
        fallback_reason="",
    )
    analysis = EmotionAnalysis(
        label="excited",
        instruct_text="Speak excitedly",
        confidence=0.9,
        energy=0.8,
        pacing_offset_pct=10,
        pitch_offset_hz=8,
        is_neutral_fallback=False,
    )
    item: dict = {}
    apply_emotion_to_clone_item(item, clip, "auto", analysis, source_text="你好世界")
    assert item["reference_path"] == str(clip_path)
    assert item["emotion_ref_text"] == "你好世界"
    assert item["instruct_text"] == "Speak excitedly"
    assert item["temperature"] > 0.9


def test_apply_emotion_auto_mode_skips_instruct_when_neutral(tmp_path):
    clip_path = tmp_path / "clip.wav"
    clip_path.write_bytes(b"wav")
    clip = EmotionClip(
        segment_index=0,
        clip_path=clip_path,
        duration=1.0,
        snr_db=20.0,
        usable=True,
        fallback_reason="",
    )
    analysis = EmotionAnalysis(
        label="neutral",
        instruct_text=NEUTRAL_INSTRUCT,
        confidence=0.2,
        energy=0.1,
        pacing_offset_pct=0,
        pitch_offset_hz=0,
        is_neutral_fallback=True,
    )
    item: dict = {}
    apply_emotion_to_clone_item(item, clip, "auto", analysis)
    assert "instruct_text" not in item
