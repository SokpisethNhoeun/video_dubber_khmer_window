from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def read_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, always_2d=False, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        raise ValueError(f"Audio file is empty: {path}")
    return audio, sample_rate


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio), dtype=np.float64) + 1e-12))


def average_spectrum(audio: np.ndarray, frame_size: int = 2048, hop: int = 512) -> np.ndarray:
    window = np.hanning(frame_size).astype(np.float32)
    total = np.zeros(frame_size // 2 + 1, dtype=np.float64)
    count = 0
    global_rms = rms(audio)
    for start in range(0, max(1, audio.size - frame_size), hop):
        frame = audio[start : start + frame_size]
        if frame.size < frame_size:
            break
        if rms(frame) < global_rms * 0.2:
            continue
        spectrum = np.fft.rfft(frame * window)
        total += np.abs(spectrum)
        count += 1
    if count == 0:
        return np.ones(frame_size // 2 + 1, dtype=np.float32)
    return np.asarray(total / count, dtype=np.float32)


def smooth(values: np.ndarray, width: int = 17) -> np.ndarray:
    if width <= 1:
        return values
    kernel = np.ones(width, dtype=np.float32) / width
    padded = np.pad(values, (width // 2, width // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def match_spectrum(input_audio: np.ndarray, reference_audio: np.ndarray) -> np.ndarray:
    frame_size = 2048
    hop = 512
    window = np.hanning(frame_size).astype(np.float32)
    input_profile = average_spectrum(input_audio, frame_size, hop)
    reference_profile = average_spectrum(reference_audio, frame_size, hop)
    gain = reference_profile / np.maximum(input_profile, 1e-5)
    gain = smooth(np.clip(gain, 0.45, 2.2))

    output = np.zeros(input_audio.size + frame_size, dtype=np.float32)
    weights = np.zeros_like(output)
    for start in range(0, max(1, input_audio.size), hop):
        frame = input_audio[start : start + frame_size]
        if frame.size == 0:
            break
        if frame.size < frame_size:
            frame = np.pad(frame, (0, frame_size - frame.size))
        spectrum = np.fft.rfft(frame * window)
        shaped = np.fft.irfft(spectrum * gain, n=frame_size).astype(np.float32)
        output[start : start + frame_size] += shaped * window
        weights[start : start + frame_size] += window * window

    output = output[: input_audio.size]
    weights = weights[: input_audio.size]
    active = weights > 1e-6
    output[active] /= weights[active]
    output[~active] = input_audio[~active]
    return output


def match_loudness(input_audio: np.ndarray, reference_audio: np.ndarray) -> np.ndarray:
    target_rms = rms(reference_audio)
    source_rms = rms(input_audio)
    if source_rms < 1e-6:
        return input_audio
    gain = max(0.35, min(3.0, target_rms / source_rms))
    audio = input_audio * gain
    drive = 1.15
    audio = np.tanh(audio * drive) / np.tanh(drive)
    peak = float(np.max(np.abs(audio)) + 1e-9)
    if peak > 0.96:
        audio = audio * (0.96 / peak)
    return audio.astype(np.float32)


def post_clone_match(cloned_path: Path, reference_path: Path, output_path: Path | None = None) -> Path:
    """Apply spectrum + loudness matching to a cloned audio file against its reference.
    Writes result to output_path (defaults to overwriting cloned_path in-place).
    """
    if output_path is None:
        output_path = cloned_path

    cloned_audio, sr = read_mono(cloned_path)
    ref_audio, ref_sr = read_mono(reference_path)

    if ref_sr != sr:
        import subprocess
        import tempfile
        tmp = Path(tempfile.mktemp(suffix=".wav"))
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-i", str(reference_path), "-ar", str(sr), "-ac", "1", str(tmp)],
            check=True,
        )
        ref_audio, _ = read_mono(tmp)
        tmp.unlink(missing_ok=True)

    shaped = match_spectrum(cloned_audio, ref_audio)
    final = match_loudness(shaped, ref_audio)
    sf.write(str(output_path), final, sr, subtype="PCM_16")
    return output_path
