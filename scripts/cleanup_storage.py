#!/usr/bin/env python3
"""Report and optionally remove local cache/temp folders.

The default mode is a dry run. Use --yes with explicit options to delete.
Downloaded model weights and virtualenvs are not included in --all-safe because
they are expensive to recreate and may be required for offline use.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CLEAN_TARGETS = {
    "temp": [ROOT / "temp"],
    "cache": [ROOT / "cache"],
    "uv_cache": [ROOT / ".uv-cache"],
    "pytest_cache": [ROOT / ".pytest_cache"],
    "pycache": [
        path
        for path in ROOT.rglob("__pycache__")
        if not {".venv", ".uv-cache", ".openvoice-python", "third_party"} & set(path.parts)
    ],
}

SAFE_GROUP = ("temp", "cache", "uv_cache", "pytest_cache", "pycache")


def format_size(size: int) -> str:
    units = ("B", "KB", "MB", "GB", "TB")
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def path_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file() or path.is_symlink():
        return path.lstat().st_size

    total = 0
    for child in path.rglob("*"):
        try:
            if child.is_file() or child.is_symlink():
                total += child.lstat().st_size
        except OSError:
            continue
    return total


def selected_targets(args: argparse.Namespace) -> list[Path]:
    selected: list[str] = []
    if args.all_safe:
        selected.extend(SAFE_GROUP)
    for key in CLEAN_TARGETS:
        if getattr(args, key):
            selected.append(key)

    if not selected:
        selected.extend(SAFE_GROUP)

    paths: list[Path] = []
    seen: set[Path] = set()
    for key in selected:
        for path in CLEAN_TARGETS[key]:
            if path not in seen:
                seen.add(path)
                paths.append(path)
    return paths


def remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Clean local generated files and caches from this project."
    )
    parser.add_argument("--yes", action="store_true", help="delete selected paths")
    parser.add_argument(
        "--all-safe",
        action="store_true",
        help="select temp, cache, uv cache, pytest cache, and __pycache__ folders",
    )
    parser.add_argument("--temp", action="store_true", help="select temp/")
    parser.add_argument("--cache", action="store_true", help="select cache/")
    parser.add_argument("--uv-cache", dest="uv_cache", action="store_true")
    parser.add_argument("--pytest-cache", dest="pytest_cache", action="store_true")
    parser.add_argument("--pycache", action="store_true", help="select __pycache__")
    args = parser.parse_args()

    targets = selected_targets(args)
    total = 0

    print("Mode:", "delete" if args.yes else "dry-run")
    for path in targets:
        size = path_size(path)
        total += size
        rel = path.relative_to(ROOT)
        status = "exists" if path.exists() else "missing"
        print(f"{format_size(size):>10}  {status:7}  {rel}")

    print(f"{format_size(total):>10}  selected total")
    print("Note: hardlinked files can make actual freed disk space smaller.")

    if not args.yes:
        print("No files deleted. Re-run with --yes to delete selected paths.")
        return 0

    for path in targets:
        remove_path(path)
    print("Cleanup complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
