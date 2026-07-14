from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from platformdirs import user_data_dir


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def resource_root() -> Path:
    """Read-only application resources (PyInstaller _internal when frozen)."""
    bundle = getattr(sys, "_MEIPASS", "")
    return Path(bundle).resolve() if bundle else Path(__file__).resolve().parents[1]


def install_root() -> Path:
    """Directory containing the installed executable, or the source root."""
    return Path(sys.executable).resolve().parent if is_frozen() else Path(__file__).resolve().parents[1]


def working_root() -> Path:
    """Writable project state root; never write under Program Files when frozen."""
    if not is_frozen():
        return install_root()
    path = Path(user_data_dir("khmer-video-dubber", appauthor=False))
    path.mkdir(parents=True, exist_ok=True)
    return path


def bundled_binary(name: str) -> Path | None:
    suffix = ".exe" if os.name == "nt" else ""
    filename = name if name.lower().endswith(suffix) else f"{name}{suffix}"
    candidates = (
        install_root() / "bin" / filename,
        resource_root() / "bin" / filename,
    )
    return next((path for path in candidates if path.is_file()), None)


def configure_bundled_tools() -> None:
    """Make bundled FFmpeg/FFprobe discoverable by all existing subprocess calls."""
    ffmpeg = bundled_binary("ffmpeg")
    ffprobe = bundled_binary("ffprobe")
    if not ffmpeg or not ffprobe:
        return
    bin_dir = str(ffmpeg.parent)
    current = os.environ.get("PATH", "")
    if bin_dir not in current.split(os.pathsep):
        os.environ["PATH"] = bin_dir + os.pathsep + current


def executable_for(name: str) -> str | None:
    bundled = bundled_binary(name)
    return str(bundled) if bundled else shutil.which(name)


def windows_creation_flags() -> int:
    """Prevent child processes from opening terminal windows in the GUI app."""
    if os.name != "nt":
        return 0
    return int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
