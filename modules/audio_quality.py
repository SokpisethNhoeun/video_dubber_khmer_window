from __future__ import annotations

import hashlib
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Thread
from core.context import CancellationError
from config.runtime import windows_creation_flags
from modules.audio_utils import FFmpegError, ensure_ffmpeg, ffprobe_duration


SUPPORTED_REFERENCE_EXTENSIONS = {".mp3", ".wav"}


@dataclass
class ReferenceValidation:
    path: Path
    exists: bool
    supported: bool
    duration: float = 0.0
    peak_db: float | None = None
    rms_db: float | None = None
    dc_offset: float | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.exists and self.supported and not any(
            warning.startswith(("missing", "unsupported", "silent")) for warning in self.warnings
        )

    @property
    def status(self) -> str:
        if not self.exists:
            return "missing"
        if not self.supported:
            return "unsupported"
        if not self.warnings:
            return "ok"
        return "warning: " + "; ".join(self.warnings)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def media_cache_key(path: Path) -> str:
    try:
        stat = path.stat()
        seed = f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
    except OSError:
        seed = str(path)
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _run_capture(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, check=False)


def _audio_stats(path: Path, cancel_event: Event | None = None) -> tuple[float | None, float | None, float | None]:
    ensure_ffmpeg()
    command = [
        "ffmpeg",
        "-hide_banner",
        "-i",
        str(path),
        "-af",
        "astats=metadata=1:reset=0",
        "-f",
        "null",
        "-",
    ]
    text = _run_interruptible(command, cancel_event)

    peak_db = None
    rms_db = None
    dc_offset = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if "Peak level dB:" in line:
            peak_db = _parse_stat_float(line)
        elif "RMS level dB:" in line:
            rms_db = _parse_stat_float(line)
        elif "DC offset:" in line:
            dc_offset = _parse_stat_float(line)
    return peak_db, rms_db, dc_offset


def _parse_stat_float(line: str) -> float | None:
    try:
        return float(line.rsplit(":", 1)[1].strip())
    except (IndexError, ValueError):
        return None


def validate_reference_audio(path: Path, min_seconds: float = 10.0, cancel_event: Event | None = None) -> ReferenceValidation:
    resolved = path.expanduser()
    exists = resolved.exists()
    supported = resolved.suffix.lower() in SUPPORTED_REFERENCE_EXTENSIONS
    validation = ReferenceValidation(path=resolved, exists=exists, supported=supported)

    if not exists:
        validation.warnings.append("missing file")
        return validation
    if not supported:
        validation.warnings.append("unsupported format; use MP3 or WAV")
        return validation

    try:
        validation.duration = ffprobe_duration(resolved)
        validation.peak_db, validation.rms_db, validation.dc_offset = _audio_stats(resolved, cancel_event)
    except Exception as exc:
        validation.warnings.append(f"analysis failed: {exc}")
        return validation

    if validation.duration < min_seconds:
        validation.warnings.append(f"short reference ({validation.duration:.1f}s; recommend {min_seconds:.0f}s+)")
    if validation.rms_db is not None and validation.rms_db <= -55:
        validation.warnings.append("silent or near-silent audio")
    if validation.peak_db is not None and validation.peak_db >= -0.2:
        validation.warnings.append("clipped or too loud")
    if validation.dc_offset is not None and abs(validation.dc_offset) >= 0.02:
        validation.warnings.append("noticeable DC offset")
    if validation.peak_db is not None and validation.rms_db is not None:
        crest = validation.peak_db - validation.rms_db
        if validation.rms_db > -35 and crest < 6:
            validation.warnings.append("possibly noisy or over-compressed")
    return validation


def prepare_reference_audio(
    input_audio: Path,
    work_dir: Path,
    min_seconds: float,
    cancel_event: Event,
    persistent_cache_dir: Path | None = None,
    cache_hits: dict[str, int] | None = None,
) -> tuple[Path | None, ReferenceValidation]:
    validation = validate_reference_audio(input_audio, min_seconds, cancel_event=cancel_event)
    if not validation.exists or not validation.supported:
        return None, validation

    source_hash = file_hash(validation.path)
    job_dir = work_dir / "cleaned_references"
    job_dir.mkdir(parents=True, exist_ok=True)
    job_output = job_dir / f"{source_hash}.wav"

    cached_output = None
    if persistent_cache_dir is not None:
        cached_dir = persistent_cache_dir / "references"
        cached_dir.mkdir(parents=True, exist_ok=True)
        cached_output = cached_dir / f"{source_hash}.wav"
        if cached_output.exists():
            if cache_hits is not None:
                cache_hits["references"] = cache_hits.get("references", 0) + 1
            shutil.copy2(cached_output, job_output)
            return job_output, validation

    output = cached_output or job_output
    _clean_audio(validation.path, output, cancel_event)
    if output != job_output:
        shutil.copy2(output, job_output)
    return job_output, validation


def _clean_audio(input_audio: Path, output_wav: Path, cancel_event: Event | None = None) -> None:
    ensure_ffmpeg()
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    filters = (
        "highpass=f=20,"
        "silenceremove=start_periods=1:start_duration=0.15:start_threshold=-45dB:"
        "stop_periods=1:stop_duration=0.25:stop_threshold=-45dB,"
        "loudnorm=I=-16:TP=-1.5:LRA=11,"
        "alimiter=limit=0.95"
    )
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_audio),
        "-af",
        filters,
        "-ac",
        "1",
        "-ar",
        "44100",
        "-c:a",
        "pcm_s16le",
        str(output_wav),
    ]
    _run_interruptible(command, cancel_event)


# Publish-target presets: integrated LUFS and true-peak in dBTP.
# YouTube normalizes uploads to ~-14 LUFS; delivering closer to target avoids
# post-upload volume swings. TikTok's normalization is less predictable so we
# aim slightly louder — audible over phone speakers without triggering peaks.
PUBLISH_TARGETS: dict[str, tuple[float, float]] = {
    "youtube": (-14.0, -1.5),
    "tiktok": (-12.0, -1.5),
    "instagram": (-14.0, -1.5),
    "broadcast": (-16.0, -1.0),
}


def resolve_publish_target(name: str, custom_lufs: float | None = None) -> tuple[float, float]:
    """Return (integrated_lufs, true_peak_dbtp) for a publish preset. If the
    name is unknown, fall back to YouTube; a custom LUFS override wins when
    ``name`` is 'custom' or the preset is missing."""
    if name == "custom" and custom_lufs is not None:
        return float(custom_lufs), -1.5
    return PUBLISH_TARGETS.get(name, PUBLISH_TARGETS["youtube"])


def master_final_audio(
    input_wav: Path,
    output_wav: Path,
    duration: float,
    cancel_event: Event,
    target_lufs: float = -14.0,
    true_peak_dbtp: float = -1.5,
) -> Path:
    """Master the final dubbed audio to a specific integrated LUFS.

    The previous fixed -16 LUFS undershot YouTube's -14 LUFS target so uploads
    got quietly turned down by YouTube's normalizer. Making this configurable
    lets us hit the exact target per platform.
    """
    ensure_ffmpeg()
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    filters = (
        f"loudnorm=I={target_lufs:.1f}:TP={true_peak_dbtp:.1f}:LRA=11,"
        f"alimiter=limit=0.95,"
        f"apad,atrim=0:{duration:.3f}"
    )
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_wav),
        "-af",
        filters,
        "-ac",
        "1",
        "-ar",
        "44100",
        "-c:a",
        "pcm_s16le",
        str(output_wav),
    ]
    _run_interruptible(command, cancel_event)
    return output_wav


def _run_interruptible(command: list[str], cancel_event: Event | None) -> str:
    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=windows_creation_flags(),
    )
    stderr: list[str] = []

    def read_stderr() -> None:
        if process.stderr:
            for line in process.stderr:
                stderr.append(line)

    reader = Thread(target=read_stderr, name="ffmpeg-stderr", daemon=True)
    reader.start()
    try:
        while process.poll() is None:
            if cancel_event and cancel_event.is_set():
                raise CancellationError("Processing cancelled by user")
            import time
            time.sleep(0.1)
    except Exception:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise
    finally:
        reader.join(timeout=5)
    if process.returncode != 0:
        raise FFmpegError("".join(stderr).strip() or f"Command failed: {' '.join(command)}")
    return "".join(stderr)
