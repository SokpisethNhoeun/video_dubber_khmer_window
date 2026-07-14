from __future__ import annotations

import math
import re
import shutil
import subprocess
import time
from pathlib import Path
from threading import Event
from typing import Callable, TypedDict

from core.context import CancellationError, Segment
from config.runtime import executable_for, windows_creation_flags


ProgressCallback = Callable[[int], None]
LogCallback = Callable[[str], None]


class FFmpegError(RuntimeError):
    """Raised when ffmpeg or ffprobe exits unsuccessfully."""


class FFmpegMissingError(FFmpegError):
    """Raised when ffmpeg or ffprobe cannot be found in PATH."""


class AudioFitResult(TypedDict):
    generated_duration: float
    target_duration: float
    speed_used: float
    trim_required: bool
    trim_duration: float
    adjusted_duration: float


class AlignmentUnit(TypedDict):
    index: int
    start: float
    end: float
    duration: float
    speech_path: Path
    speaker: str
    label: str
    count: int


def ensure_ffmpeg() -> None:
    if not executable_for("ffmpeg"):
        raise FFmpegMissingError("ffmpeg is not installed or is not in PATH")
    if not executable_for("ffprobe"):
        raise FFmpegMissingError("ffprobe is not installed or is not in PATH")


def _run_checked(command: list[str], cancel_event: Event | None = None) -> None:
    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        creationflags=windows_creation_flags(),
    )
    try:
        while process.poll() is None:
            if cancel_event and cancel_event.is_set():
                raise CancellationError("Processing cancelled by user")
            time.sleep(0.1)
    except Exception:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise
    if process.returncode != 0:
        raise FFmpegError(f"Command failed: {' '.join(command)}")


def ffprobe_duration(media_path: Path) -> float:
    ensure_ffmpeg()
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(media_path),
    ]
    result = subprocess.run(
        command, capture_output=True, text=True, encoding="utf-8", errors="replace",
        check=False, creationflags=windows_creation_flags(),
    )
    if result.returncode != 0:
        raise FFmpegError(result.stderr.strip() or f"Could not inspect duration for {media_path}")
    raw = result.stdout.strip()
    if not raw or raw.upper() == "N/A":
        return 0.0
    try:
        return max(0.01, float(raw))
    except ValueError as exc:
        raise FFmpegError(f"Invalid duration reported for {media_path}") from exc


def has_audio_stream(media_path: Path) -> bool:
    ensure_ffmpeg()
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=codec_type",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(media_path),
    ]
    result = subprocess.run(
        command, capture_output=True, text=True, encoding="utf-8", errors="replace",
        check=False, creationflags=windows_creation_flags(),
    )
    if result.returncode != 0:
        return False
    return "audio" in result.stdout.strip()


def _parse_ffmpeg_time(line: str) -> float | None:
    match = re.search(r"time=(\d+):(\d+):(\d+(?:\.\d+)?)", line)
    if not match:
        return None
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _extract_ffmpeg_error(stderr: str) -> str:
    """Extract meaningful error lines from ffmpeg stderr, skipping version/config header."""
    meaningful = []
    for line in stderr.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("ffmpeg version", "built with", "configuration:", "lib")):
            continue
        if re.match(r"^\s*(lib\w+|Hyper|Copyright)", stripped):
            continue
        meaningful.append(stripped)
    if not meaningful:
        return ""
    return "\n".join(meaningful[-20:])


def run_ffmpeg_progress(
    command: list[str],
    total_duration: float,
    progress_cb: ProgressCallback | None,
    cancel_event: Event | None,
) -> None:
    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=windows_creation_flags(),
    )
    try:
        last_progress = -1
        assert process.stderr is not None
        stderr_lines = []
        while True:
            if cancel_event and cancel_event.is_set():
                raise CancellationError("Processing cancelled by user")

            line = process.stderr.readline()
            if line:
                stderr_lines.append(line)
                current_time = _parse_ffmpeg_time(line)
                if current_time is not None and total_duration > 0:
                    progress = min(99, int((current_time / total_duration) * 100))
                    if progress != last_progress and progress_cb:
                        progress_cb(progress)
                    last_progress = progress

            if process.poll() is not None:
                break
    except Exception:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise

    remaining_stderr = process.stderr.read()
    if remaining_stderr:
        stderr_lines.append(remaining_stderr)

    full_stderr = "".join(stderr_lines).strip()
    if process.returncode != 0:
        error_lines = _extract_ffmpeg_error(full_stderr)
        raise FFmpegError(error_lines or full_stderr or f"ffmpeg failed: {' '.join(command)}")
    if progress_cb:
        progress_cb(100)


def extract_audio(
    input_video: Path,
    output_wav: Path,
    progress_cb: ProgressCallback | None,
    cancel_event: Event,
) -> float:
    ensure_ffmpeg()
    if not input_video.exists():
        raise FileNotFoundError(f"Input video does not exist: {input_video}")
    duration = ffprobe_duration(input_video)
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_video),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(output_wav),
    ]
    run_ffmpeg_progress(command, duration, progress_cb, cancel_event)
    return duration


def convert_to_wav(input_audio: Path, output_wav: Path, cancel_event: Event) -> None:
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_audio),
        "-ac",
        "1",
        "-ar",
        "44100",
        "-c:a",
        "pcm_s16le",
        str(output_wav),
    ]
    _run_checked(command, cancel_event)


def trim_audio_segment(input_wav: Path, output_wav: Path, start: float, duration: float, cancel_event: Event) -> None:
    ensure_ffmpeg()
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{max(0.0, start):.3f}",
        "-i",
        str(input_wav),
        "-t",
        f"{max(0.01, duration):.3f}",
        "-ac",
        "1",
        "-ar",
        "44100",
        "-c:a",
        "pcm_s16le",
        str(output_wav),
    ]
    _run_checked(command, cancel_event)


def make_silence(duration: float, output_wav: Path, cancel_event: Event) -> None:
    duration = max(0.01, duration)
    command = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "anullsrc=channel_layout=mono:sample_rate=44100",
        "-t",
        f"{duration:.3f}",
        "-c:a",
        "pcm_s16le",
        str(output_wav),
    ]
    _run_checked(command, cancel_event)


def _atempo_chain(speed: float) -> str:
    speed = max(0.5, min(2.5, speed))
    parts: list[float] = []
    remaining = speed
    while remaining > 2.0:
        parts.append(2.0)
        remaining /= 2.0
    while remaining < 0.5:
        parts.append(0.5)
        remaining /= 0.5
    parts.append(remaining)
    return ",".join(f"atempo={part:.5f}" for part in parts)


def adjust_audio_speed(input_wav: Path, output_wav: Path, speed: float, cancel_event: Event) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_wav),
        "-filter:a",
        _atempo_chain(speed),
        "-ac",
        "1",
        "-ar",
        "44100",
        "-c:a",
        "pcm_s16le",
        str(output_wav),
    ]
    _run_checked(command, cancel_event)


def fit_audio_to_duration(
    input_wav: Path,
    output_wav: Path,
    target_duration: float,
    max_speed: float,
    cancel_event: Event,
    normalize_voice: bool = True,
) -> AudioFitResult:
    """Write a mono 44.1kHz WAV that is exactly target_duration seconds long.

    Speeds up to max_speed (capped at 1.8x) to fit. If still longer after
    speed-up, trims the tail to maintain lip sync with the video timeline.
    Pads with silence if shorter.
    """
    target_duration = max(0.01, target_duration)
    max_speed = max(1.0, min(1.8, max_speed))
    generated_duration = ffprobe_duration(input_wav)
    required_speed = generated_duration / target_duration
    speed_used = min(max_speed, required_speed) if required_speed > 1.0 else 1.0
    adjusted_duration = generated_duration / speed_used
    trim_duration = max(0.0, adjusted_duration - target_duration)
    trim_required = trim_duration > 0.01

    filters = []
    if abs(speed_used - 1.0) > 0.001:
        filters.append(_atempo_chain(speed_used))
    if normalize_voice:
        filters.extend([
            "highpass=f=60",
            "dynaudnorm=f=150:g=12:p=0.95:m=8",
            "alimiter=limit=0.95",
        ])
    filters.extend(["apad", f"atrim=0:{target_duration:.3f}"])

    output_wav.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_wav),
        "-af",
        ",".join(filters),
        "-ac",
        "1",
        "-ar",
        "44100",
        "-c:a",
        "pcm_s16le",
        str(output_wav),
    ]
    _run_checked(command, cancel_event)

    return {
        "generated_duration": generated_duration,
        "target_duration": target_duration,
        "speed_used": speed_used,
        "trim_required": trim_required,
        "trim_duration": trim_duration,
        "adjusted_duration": adjusted_duration,
    }


def shorten_excessive_pauses(input_wav: Path, output_wav: Path, cancel_event: Event) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_wav),
        "-af",
        (
            "silenceremove=start_periods=1:start_duration=0.05:start_threshold=-50dB:"
            "stop_periods=-1:stop_duration=0.45:stop_threshold=-50dB"
        ),
        "-ac",
        "1",
        "-ar",
        "44100",
        "-c:a",
        "pcm_s16le",
        str(output_wav),
    ]
    _run_checked(command, cancel_event)


def _alignment_max_speed(mode: str) -> float:
    if mode == "strict":
        return 1.8
    if mode == "energetic":
        return 1.72
    return 1.6


def concat_wavs(wav_files: list[Path], output_wav: Path, concat_file: Path, cancel_event: Event) -> None:
    with concat_file.open("w", encoding="utf-8") as handle:
        for wav_file in wav_files:
            escaped = str(wav_file.resolve()).replace("\\", "\\\\").replace("'", "\\'")
            handle.write(f"file '{escaped}'\n")
    command = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_file),
        "-ac",
        "1",
        "-ar",
        "44100",
        "-c:a",
        "pcm_s16le",
        str(output_wav),
    ]
    _run_checked(command, cancel_event)


def _alignment_units(segments: list[Segment]) -> list[AlignmentUnit]:
    units: list[AlignmentUnit] = []
    active_segments = [segment for segment in segments if segment.enabled]
    index = 0
    while index < len(active_segments):
        segment = active_segments[index]
        group_id = getattr(segment, "tts_group_id", "")
        group = [segment]
        if group_id:
            cursor = index + 1
            while cursor < len(active_segments) and getattr(active_segments[cursor], "tts_group_id", "") == group_id:
                group.append(active_segments[cursor])
                cursor += 1
            index = cursor
        else:
            index += 1

        first = group[0]
        last = group[-1]
        start = first.start
        end = last.end
        units.append(
            {
                "index": first.index,
                "start": start,
                "end": end,
                "duration": max(0.01, end - start),
                "speech_path": first.speech_path,
                "speaker": first.speaker_label or first.speaker_id or "",
                "label": (
                    f"Gemini chunk {first.tts_group_id}"
                    if group_id and len(group) > 1
                    else f"segment {first.index + 1}"
                ),
                "count": len(group),
            }
        )
    return units


def align_audio_segments(
    segments: list[Segment],
    output_wav: Path,
    work_dir: Path,
    total_duration: float,
    progress_cb: ProgressCallback | None,
    log_cb: LogCallback | None,
    cancel_event: Event,
    mode: str = "natural",
    quality_report=None,
    shorten_pauses: bool = True,
) -> Path:
    ensure_ffmpeg()
    units = _alignment_units(segments)
    if not units:
        raise ValueError("No segments available for alignment")
    if mode not in {"natural", "strict", "energetic"}:
        raise ValueError(f"Unsupported alignment mode: {mode}")

    align_dir = work_dir / "alignment"
    align_dir.mkdir(parents=True, exist_ok=True)
    pieces: list[Path] = []
    cursor = 0.0
    max_speed = _alignment_max_speed(mode)

    for position, unit in enumerate(units, start=1):
        if cancel_event.is_set():
            raise CancellationError("Processing cancelled by user")

        unit_index = int(unit["index"])
        raw_wav = align_dir / f"{unit_index:05d}_raw.wav"
        convert_to_wav(unit["speech_path"], raw_wav, cancel_event)

        generated_duration = ffprobe_duration(raw_wav)
        target_duration = float(unit["duration"])

        fit_input_wav = raw_wav
        required_speed = generated_duration / target_duration

        if shorten_pauses and required_speed > max_speed:
            compact_wav = align_dir / f"{unit_index:05d}_compact.wav"
            try:
                shorten_excessive_pauses(raw_wav, compact_wav, cancel_event)
                compact_duration = ffprobe_duration(compact_wav)
                if compact_duration < generated_duration:
                    fit_input_wav = compact_wav
                    generated_duration = compact_duration
                    required_speed = generated_duration / target_duration
                    if log_cb:
                        log_cb(
                            f"  {unit['label']}: shortened pauses before speed-up "
                            f"({compact_duration:.2f}s for {target_duration:.2f}s slot)"
                        )
            except Exception as exc:
                if log_cb:
                    log_cb(f"  Warning: could not shorten pauses for {unit['label']}: {exc}")

        aligned_wav = align_dir / f"{unit_index:05d}_aligned.wav"
        fit_result = fit_audio_to_duration(
            fit_input_wav,
            aligned_wav,
            target_duration,
            max_speed,
            cancel_event,
        )
        speed_used = float(fit_result["speed_used"])
        trim_required = bool(fit_result["trim_required"])
        trim_duration = float(fit_result["trim_duration"])
        adjusted_duration = float(fit_result["adjusted_duration"])

        if quality_report is not None and trim_required:
            quality_report.long_segments.append(
                {
                    "index": unit_index + 1,
                    "speaker": unit["speaker"],
                    "generated_duration": round(generated_duration, 3),
                    "target_duration": round(target_duration, 3),
                    "adjusted_duration": round(adjusted_duration, 3),
                    "speed": round(speed_used, 3),
                    "trim_required": True,
                    "trim_duration": round(trim_duration, 3),
                }
            )

        if log_cb:
            trim_message = f", trimmed {trim_duration:.2f}s" if trim_required else ", no trim"
            log_cb(
                f"  {mode.capitalize()} sync {unit['label']} {position}/{len(units)}: "
                f"generated {generated_duration:.2f}s, slot {target_duration:.2f}s, "
                f"speed {speed_used:.2f}x{trim_message}"
            )
            if trim_required:
                log_cb(
                    f"  Timing warning: {unit['label']} is still too long after "
                    f"{speed_used:.2f}x speed; shorten this Khmer line if the cut sounds abrupt"
                )

        if quality_report is not None:
            quality_report.timing_segments.append(
                {
                    "index": unit_index + 1,
                    "speaker": unit["speaker"],
                    "generated_duration": round(generated_duration, 3),
                    "original_segment_duration": round(target_duration, 3),
                    "speed_adjustment": round(speed_used, 3),
                    "trim_required": trim_required,
                    "trim_duration": round(trim_duration, 3),
                    "still_runs_long": False,
                }
            )

        if unit["start"] > cursor:
            silence = align_dir / f"{unit_index:05d}_gap.wav"
            make_silence(float(unit["start"]) - cursor, silence, cancel_event)
            pieces.append(silence)
            cursor = float(unit["start"])

        pieces.append(aligned_wav)
        cursor = float(unit["end"])

        if progress_cb:
            progress_cb(math.floor((position / len(units)) * 90))

    if total_duration > cursor:
        final_silence = align_dir / "final_gap.wav"
        make_silence(total_duration - cursor, final_silence, cancel_event)
        pieces.append(final_silence)

    concat_wavs(pieces, output_wav, align_dir / "concat.txt", cancel_event)
    if progress_cb:
        progress_cb(100)
    return output_wav


def mux_video(
    input_video: Path,
    audio_wav: Path,
    output_video: Path,
    progress_cb: ProgressCallback | None,
    cancel_event: Event,
) -> Path:
    ensure_ffmpeg()
    output_video.parent.mkdir(parents=True, exist_ok=True)
    duration = ffprobe_duration(input_video)
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_video),
        "-i",
        str(audio_wav),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        str(output_video),
    ]
    run_ffmpeg_progress(command, duration, progress_cb, cancel_event)
    return output_video


def safe_output_path(input_video: Path, output_dir: Path) -> Path:
    stem = input_video.stem
    output = output_dir / f"{stem}_khmer_dubbed.mp4"
    if not output.exists():
        return output
    for number in range(1, 1000):
        candidate = output_dir / f"{stem}_khmer_dubbed_{number}.mp4"
        if not candidate.exists():
            return candidate
    raise FileExistsError("Could not create a unique output filename")


def remove_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
