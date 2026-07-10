from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from modules.audio_matching import match_loudness, match_spectrum, read_mono, rms


DEFAULT_SAMPLE_RATE = 24000


class ReferenceCloneError(RuntimeError):
    pass


def _run(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise ReferenceCloneError(detail or f"Command failed: {' '.join(command)}")


def _ensure_ffmpeg() -> None:
    if not shutil.which("ffmpeg"):
        raise ReferenceCloneError("ffmpeg is required for the built-in reference voice clone tool.")


def _decode_audio(source: Path, target: Path, sample_rate: int) -> None:
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            str(target),
        ]
    )


def _atempo_chain(factor: float) -> str:
    factors: list[float] = []
    remaining = factor
    while remaining < 0.5:
        factors.append(0.5)
        remaining /= 0.5
    while remaining > 2.0:
        factors.append(2.0)
        remaining /= 2.0
    factors.append(remaining)
    return ",".join(f"atempo={item:.6f}" for item in factors)


def _pitch_shift(source: Path, target: Path, sample_rate: int, ratio: float) -> None:
    ratio = max(0.72, min(1.38, ratio))
    if abs(ratio - 1.0) < 0.03:
        shutil.copy2(source, target)
        return
    tempo = _atempo_chain(1.0 / ratio)
    filters = f"asetrate={sample_rate * ratio:.3f},aresample={sample_rate},{tempo}"
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-af",
            filters,
            "-ar",
            str(sample_rate),
            str(target),
        ]
    )


def _read_mono(path: Path) -> tuple[np.ndarray, int]:
    return read_mono(path)


def _rms(audio: np.ndarray) -> float:
    return rms(audio)


def _median_f0(audio: np.ndarray, sample_rate: int) -> float | None:
    if audio.size < sample_rate // 5 or _rms(audio) < 1e-5:
        return None

    frame_size = 2048
    hop = 512
    min_f0 = 65.0
    max_f0 = 420.0
    min_lag = max(1, int(sample_rate / max_f0))
    max_lag = min(frame_size - 1, int(sample_rate / min_f0))
    window = np.hanning(frame_size).astype(np.float32)
    global_rms = _rms(audio)
    estimates: list[float] = []

    for start in range(0, max(1, audio.size - frame_size), hop):
        frame = audio[start : start + frame_size]
        if frame.size < frame_size:
            break
        if _rms(frame) < global_rms * 0.35:
            continue
        frame = (frame - np.mean(frame)) * window
        corr = np.correlate(frame, frame, mode="full")[frame_size - 1 :]
        if corr[0] <= 1e-9:
            continue
        search = corr[min_lag:max_lag]
        if search.size == 0:
            continue
        peak_index = int(np.argmax(search)) + min_lag
        confidence = corr[peak_index] / corr[0]
        if confidence < 0.28:
            continue
        estimates.append(sample_rate / peak_index)

    if not estimates:
        return None
    return float(np.median(np.asarray(estimates, dtype=np.float32)))


def _match_spectrum(input_audio: np.ndarray, reference_audio: np.ndarray) -> np.ndarray:
    return match_spectrum(input_audio, reference_audio)


def _match_loudness(input_audio: np.ndarray, reference_audio: np.ndarray) -> np.ndarray:
    return match_loudness(input_audio, reference_audio)


def clone_reference_voice(input_path: Path, output_path: Path, reference_path: Path, sample_rate: int) -> dict[str, float | str]:
    _ensure_ffmpeg()
    input_path = input_path.expanduser()
    output_path = output_path.expanduser()
    reference_path = reference_path.expanduser()
    if not input_path.exists():
        raise FileNotFoundError(f"Input audio does not exist: {input_path}")
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference audio does not exist: {reference_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="reference_voice_clone_") as temp_name:
        temp_dir = Path(temp_name)
        decoded_input = temp_dir / "input.wav"
        decoded_reference = temp_dir / "reference.wav"
        pitched_input = temp_dir / "pitched.wav"
        shaped_output = temp_dir / "shaped.wav"

        _decode_audio(input_path, decoded_input, sample_rate)
        _decode_audio(reference_path, decoded_reference, sample_rate)
        input_audio, input_rate = _read_mono(decoded_input)
        reference_audio, reference_rate = _read_mono(decoded_reference)
        if input_rate != sample_rate or reference_rate != sample_rate:
            raise ReferenceCloneError("Internal sample-rate conversion failed.")

        input_f0 = _median_f0(input_audio, sample_rate)
        reference_f0 = _median_f0(reference_audio, sample_rate)
        pitch_ratio = 1.0
        if input_f0 and reference_f0:
            pitch_ratio = reference_f0 / input_f0

        _pitch_shift(decoded_input, pitched_input, sample_rate, pitch_ratio)
        pitched_audio, _ = _read_mono(pitched_input)
        shaped_audio = _match_spectrum(pitched_audio, reference_audio)
        final_audio = _match_loudness(shaped_audio, reference_audio)
        sf.write(shaped_output, final_audio, sample_rate, subtype="PCM_16")
        _run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(shaped_output),
                "-ar",
                str(sample_rate),
                "-ac",
                "1",
                str(output_path),
            ]
        )

    return {
        "input_f0": round(input_f0 or 0.0, 2),
        "reference_f0": round(reference_f0 or 0.0, 2),
        "pitch_ratio": round(max(0.72, min(1.38, pitch_ratio)), 4),
        "backend": "local-dsp-reference-match",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Built-in local reference voice matching tool.")
    parser.add_argument("--input", required=True, type=Path, help="Input synthesized speech audio.")
    parser.add_argument("--output", required=True, type=Path, help="Output WAV path.")
    parser.add_argument("--reference", required=True, type=Path, help="Reference speaker MP3/WAV.")
    parser.add_argument("--sample-rate", default=DEFAULT_SAMPLE_RATE, type=int)
    args = parser.parse_args()

    stats = clone_reference_voice(args.input, args.output, args.reference, args.sample_rate)
    print(
        "reference voice match complete: "
        f"backend={stats['backend']} "
        f"input_f0={stats['input_f0']}Hz "
        f"reference_f0={stats['reference_f0']}Hz "
        f"pitch_ratio={stats['pitch_ratio']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
