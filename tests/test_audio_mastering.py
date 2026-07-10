from __future__ import annotations

import math
import re
import shutil
import subprocess
import wave
from pathlib import Path
from threading import Event

import numpy as np
import pytest

from conftest import write_wav
from modules.audio_quality import (
    PUBLISH_TARGETS,
    master_final_audio,
    resolve_publish_target,
)
from modules.bgm_separator import mix_audio_ducked


SR = 44100
FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None
requires_ffmpeg = pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg not on PATH")


def _write_wav(path: Path, samples: np.ndarray, sr: int = SR) -> Path:
    return write_wav(path, samples, sr)


def _speech_like(seconds: float, sr: int = SR) -> np.ndarray:
    """Bursty harmonic signal that plays well with ffmpeg's LUFS analyzer."""
    n = int(seconds * sr)
    t = np.arange(n) / sr
    signal = np.zeros(n, dtype=np.float32)
    for start in np.arange(0.0, seconds, 0.35):
        end = start + 0.22
        if end > seconds:
            break
        i0 = int(start * sr)
        i1 = int(end * sr)
        bt = t[i0:i1] - start
        burst = (
            0.6 * np.sin(2 * math.pi * 180 * bt)
            + 0.3 * np.sin(2 * math.pi * 360 * bt)
        ).astype(np.float32)
        env = np.hanning(len(burst)).astype(np.float32)
        signal[i0:i1] += burst * env * 0.5
    return signal


def _measured_lufs(path: Path) -> float | None:
    """Run ffmpeg's loudnorm analyzer and pull the Integrated LUFS number
    out of the summary text. Returns None if we can't parse it."""
    out = subprocess.run(
        [
            "ffmpeg", "-i", str(path),
            "-af", "loudnorm=print_format=summary",
            "-f", "null", "-",
        ],
        capture_output=True, text=True,
    )
    m = re.search(r"Input Integrated:\s*(-?[\d.]+)", out.stderr)
    return float(m.group(1)) if m else None


def test_publish_targets_have_expected_presets():
    # Renaming or removing a preset without updating the wizard would leave
    # the user with a silently-broken dropdown, so this is worth locking.
    for key in ("youtube", "tiktok", "instagram", "broadcast"):
        assert key in PUBLISH_TARGETS
        lufs, tp = PUBLISH_TARGETS[key]
        assert -20.0 <= lufs <= -8.0
        assert -3.0 <= tp <= 0.0


def test_resolve_publish_target_youtube_default():
    lufs, tp = resolve_publish_target("youtube")
    assert lufs == -14.0
    assert tp == -1.5


def test_resolve_publish_target_falls_back_on_unknown_name():
    # We deliberately don't raise on an unknown preset so a bad settings.json
    # can't crash the pipeline. YouTube is the sane fallback.
    lufs, tp = resolve_publish_target("something_new")
    assert (lufs, tp) == PUBLISH_TARGETS["youtube"]


def test_resolve_publish_target_custom_uses_override():
    lufs, tp = resolve_publish_target("custom", custom_lufs=-10.0)
    assert lufs == -10.0


@requires_ffmpeg
def test_master_final_audio_hits_youtube_target(tmp_path: Path):
    src = _write_wav(tmp_path / "in.wav", _speech_like(8.0))
    out = tmp_path / "yt.wav"
    master_final_audio(src, out, 8.0, Event(), target_lufs=-14.0, true_peak_dbtp=-1.5)
    assert out.exists()
    lufs = _measured_lufs(out)
    # ffmpeg's single-pass loudnorm converges within ~1 LUFS of the target;
    # tighter than that requires two-pass and isn't worth the runtime cost.
    assert lufs is not None and abs(lufs - (-14.0)) < 1.5, f"LUFS {lufs}"


@requires_ffmpeg
def test_master_final_audio_hits_tiktok_target(tmp_path: Path):
    src = _write_wav(tmp_path / "in.wav", _speech_like(8.0))
    out = tmp_path / "tt.wav"
    master_final_audio(src, out, 8.0, Event(), target_lufs=-12.0, true_peak_dbtp=-1.5)
    lufs = _measured_lufs(out)
    assert lufs is not None and abs(lufs - (-12.0)) < 1.5, f"LUFS {lufs}"


@requires_ffmpeg
def test_mix_audio_ducked_produces_valid_output(tmp_path: Path):
    vocals = _write_wav(tmp_path / "voc.wav", _speech_like(3.0) * 0.7)
    # Steady tone as a BGM stand-in; the sidechain compressor should
    # attenuate it during the vocal bursts.
    n = int(3.0 * SR)
    t = np.arange(n) / SR
    bgm_signal = 0.5 * np.sin(2 * math.pi * 220 * t).astype(np.float32)
    bgm = _write_wav(tmp_path / "bgm.wav", bgm_signal)
    out = tmp_path / "mixed.wav"

    mix_audio_ducked(vocals, bgm, out, Event(), duck_depth_db=8.0)
    assert out.exists()
    assert out.stat().st_size > 1000
