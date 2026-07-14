from __future__ import annotations
import shutil
import subprocess
import sys
import time
from pathlib import Path
from threading import Event, Thread

from config.runtime import windows_creation_flags

def install_demucs_if_needed(log_cb = None) -> None:
    try:
        import demucs  # noqa: F401
    except ImportError:
        if getattr(sys, "frozen", False):
            raise RuntimeError(
                "Background music preservation is unavailable because Demucs is missing from "
                "this installation. Reinstall the latest Khmer Video Dubber setup package."
            )
        raise RuntimeError(
            "Demucs is required for background music preservation but is not installed. "
            "Install it with: python -m pip install 'demucs>=4,<5'"
        )

def _is_cuda_error(stderr_text: str) -> bool:
    """Check if a Demucs stderr contains a CUDA-related failure."""
    lower = stderr_text.lower()
    return "cuda" in lower or "cusparse" in lower or "cublas" in lower or "gpu" in lower


def _run_demucs(
    cmd: list[str],
    cancel_event: Event,
    log_cb=None,
) -> None:
    """Run a Demucs subprocess, raising on failure or cancellation."""
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=windows_creation_flags(),
    )
    stderr_lines: list[str] = []

    def read_stderr() -> None:
        if process.stderr:
            for line in process.stderr:
                stderr_lines.append(line)

    stderr_reader = Thread(target=read_stderr, name="demucs-stderr", daemon=True)
    stderr_reader.start()
    try:
        while process.poll() is None:
            if cancel_event.is_set():
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise RuntimeError("Processing cancelled by user")
            time.sleep(0.1)
    except RuntimeError:
        raise
    except Exception:
        process.terminate()
        process.wait()
        raise
    finally:
        stderr_reader.join(timeout=5)

    if process.returncode != 0:
        stderr = "".join(stderr_lines)
        raise RuntimeError(f"Demucs separation failed: {stderr}")


def separate_vocals_demucs(
    input_wav: Path,
    output_dir: Path,
    cancel_event: Event,
    device: str = "cuda",
    log_cb = None
) -> tuple[Path, Path]:
    """
    Separates vocals and background track (no_vocals) from input_wav using HTDemucs.
    Returns (vocals_wav, no_vocals_wav)

    If CUDA fails, automatically retries on CPU.
    """
    install_demucs_if_needed(log_cb)

    base_cmd = [
        sys.executable, "-m", "demucs.separate",
        "--two-stems=vocals",
        "-o", str(output_dir),
    ]

    # --- First attempt: preferred device ---
    use_device = "cuda" if "cuda" in device.lower() or "gpu" in device.lower() else "cpu"
    cmd = base_cmd + ["--device", use_device, str(input_wav)]
    if log_cb:
        log_cb(f"Running Demucs source separation (Vocals vs Background Music) on {use_device}...")

    try:
        _run_demucs(cmd, cancel_event, log_cb)
    except RuntimeError as exc:
        stderr_text = str(exc)
        if use_device == "cuda" and _is_cuda_error(stderr_text):
            if log_cb:
                log_cb("CUDA Demucs failed; falling back to CPU (this will be slower)...")
            # Clean up any partial output before retrying
            model_name = "htdemucs"
            partial_dir = output_dir / model_name / input_wav.stem
            if partial_dir.exists():
                shutil.rmtree(partial_dir, ignore_errors=True)
            cmd_cpu = base_cmd + ["--device", "cpu", str(input_wav)]
            _run_demucs(cmd_cpu, cancel_event, log_cb)
        else:
            raise

    model_name = "htdemucs"
    separated_dir = output_dir / model_name / input_wav.stem
    vocals_file = separated_dir / "vocals.wav"
    no_vocals_file = separated_dir / "no_vocals.wav"

    if not vocals_file.exists() or not no_vocals_file.exists():
        raise FileNotFoundError(f"Demucs output files not found in {separated_dir}")

    return vocals_file, no_vocals_file

def mix_audio_ducked(
    vocals_wav: Path,
    bgm_wav: Path,
    output_wav: Path,
    cancel_event: Event,
    vocals_volume: float = 1.0,
    bgm_volume: float = 0.85,
    duck_depth_db: float = 8.0,
    attack_ms: int = 80,
    release_ms: int = 400,
) -> Path:
    """Mix vocals + BGM with sidechain compression on the BGM so it auto-
    dips while Khmer voice is playing.

    ``duck_depth_db`` translates to the compressor ratio: deeper duck = higher
    ratio. This is the biggest single fix for the "TTS floats on top of the
    music" impression that flat amix leaves behind.
    """
    output_wav.parent.mkdir(parents=True, exist_ok=True)

    # Map a desired duck depth (dB) to a compressor ratio. ~8 dB ducking
    # sounds natural for voiceover; deeper values start pumping.
    ratio = max(2.0, min(20.0, duck_depth_db / 1.0))
    filters = (
        f"[0:a]volume={vocals_volume},aformat=sample_rates=44100:channel_layouts=stereo[voc];"
        f"[1:a]volume={bgm_volume},aformat=sample_rates=44100:channel_layouts=stereo[bgm_pre];"
        f"[bgm_pre][voc]sidechaincompress="
        f"threshold=0.05:ratio={ratio:.1f}:attack={attack_ms}:release={release_ms}:"
        f"makeup=1:knee=2.5[bgm_ducked];"
        f"[voc][bgm_ducked]amix=inputs=2:duration=first:dropout_transition=2[a]"
    )
    command = [
        "ffmpeg",
        "-y",
        "-i", str(vocals_wav),
        "-i", str(bgm_wav),
        "-filter_complex", filters,
        "-map", "[a]",
        "-c:a", "pcm_s16le",
        "-ar", "44100",
        str(output_wav),
    ]

    from modules.audio_utils import ensure_ffmpeg
    ensure_ffmpeg()

    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        while process.poll() is None:
            if cancel_event.is_set():
                process.terminate()
                raise RuntimeError("Processing cancelled by user")
            import time
            time.sleep(0.1)
    except Exception:
        process.terminate()
        process.wait()
        raise

    if process.returncode != 0:
        raise RuntimeError("Sidechain ducking mix failed.")
    return output_wav


def mix_audio_tracks(
    vocals_wav: Path,
    bgm_wav: Path,
    output_wav: Path,
    cancel_event: Event,
    vocals_volume: float = 1.0,
    bgm_volume: float = 0.8
) -> Path:
    """
    Mixes vocals and background tracks using ffmpeg amix filter.
    bgm_volume defaults to 0.8 to make sure the dubbed voice is clearly heard.
    """
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    
    filters = (
        f"[0:a]volume={vocals_volume}[v];"
        f"[1:a]volume={bgm_volume}[b];"
        f"[v][b]amix=inputs=2:duration=first:dropout_transition=2[a]"
    )
    
    command = [
        "ffmpeg",
        "-y",
        "-i", str(vocals_wav),
        "-i", str(bgm_wav),
        "-filter_complex", filters,
        "-map", "[a]",
        "-c:a", "pcm_s16le",
        "-ar", "44100",
        str(output_wav)
    ]
    
    from modules.audio_utils import ensure_ffmpeg
    ensure_ffmpeg()
    
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        while process.poll() is None:
            if cancel_event.is_set():
                process.terminate()
                raise RuntimeError("Processing cancelled by user")
            import time
            time.sleep(0.1)
    except Exception:
        process.terminate()
        process.wait()
        raise
        
    if process.returncode != 0:
        raise RuntimeError("Failed to mix dubbed audio with background music track.")
        
    return output_wav
