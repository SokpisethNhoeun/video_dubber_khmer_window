from __future__ import annotations

import os
import shutil
import time
from collections import deque
from pathlib import Path
from threading import Event
from typing import Callable

import requests
from huggingface_hub import HfApi

from config.models import FASTER_WHISPER_MODEL_PREFIX
from config.paths import whisper_cache_dir


ProgressCallback = Callable[[str, int, int, float, float | None], None]


class DownloadPaused(Exception):
    pass


class DownloadCancelled(Exception):
    pass


class HuggingFaceModelDownloadManager:
    """Range-resumable Hugging Face downloader that writes a valid HF cache snapshot."""

    def __init__(self, repo_id: str, cache_dir: Path, chunk_size: int = 1024 * 256) -> None:
        self.repo_id = repo_id
        self.cache_dir = Path(cache_dir)
        self.chunk_size = chunk_size
        self._pause = Event()
        self._cancel = Event()
        self._parts: set[Path] = set()

    def pause(self) -> None:
        self._pause.set()

    def resume(self) -> None:
        self._pause.clear()

    def cancel(self) -> None:
        self._cancel.set()
        for part in tuple(self._parts):
            part.unlink(missing_ok=True)

    def _files(self) -> list[tuple[str, int]]:
        info = HfApi().model_info(self.repo_id, files_metadata=True)
        return [(item.rfilename, int(item.size or 0)) for item in info.siblings]

    def download(self, progress: ProgressCallback | None = None) -> Path:
        self._cancel.clear()
        files = self._files()
        total = sum(size for _, size in files)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(self.cache_dir).free
        required = max(total + 512 * 1024 * 1024, int(total * 1.1))
        if total and free < required:
            raise RuntimeError(
                f"Not enough disk space. This model needs about {required / 1024**3:.1f} GB "
                f"including download overhead, but only {free / 1024**3:.1f} GB is available."
            )
        repo_dir = self.cache_dir / f"models--{self.repo_id.replace('/', '--')}"
        blobs = repo_dir / "blobs"
        blobs.mkdir(parents=True, exist_ok=True)
        downloaded_before = 0
        commit = "main"

        for filename, expected_size in files:
            if self._cancel.is_set():
                raise DownloadCancelled()
            url = f"https://huggingface.co/{self.repo_id}/resolve/main/{filename}"
            part = blobs / (filename.replace("/", "--") + ".part")
            self._parts.add(part)
            offset = part.stat().st_size if part.exists() else 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            with requests.get(url, headers=headers, stream=True, timeout=(10, 30), allow_redirects=True) as response:
                response.raise_for_status()
                if offset and response.status_code != 206:
                    offset = 0
                    part.unlink(missing_ok=True)
                commit = response.headers.get("X-Repo-Commit", commit)
                etag = response.headers.get("ETag", "").strip('"') or filename.replace("/", "--")
                final_blob = blobs / etag
                if final_blob.exists() and (not expected_size or final_blob.stat().st_size == expected_size):
                    downloaded_before += expected_size
                    self._link_snapshot(repo_dir, commit, filename, final_blob)
                    continue
                samples: deque[tuple[float, int]] = deque()
                mode = "ab" if offset else "wb"
                current = offset
                with part.open(mode) as handle:
                    for chunk in response.iter_content(self.chunk_size):
                        if self._cancel.is_set():
                            handle.close()
                            part.unlink(missing_ok=True)
                            raise DownloadCancelled()
                        if self._pause.is_set():
                            raise DownloadPaused()
                        if not chunk:
                            continue
                        handle.write(chunk)
                        current += len(chunk)
                        now = time.monotonic()
                        samples.append((now, current))
                        while len(samples) > 1 and now - samples[0][0] > 3:
                            samples.popleft()
                        speed = 0.0
                        if len(samples) > 1 and samples[-1][0] > samples[0][0]:
                            speed = (samples[-1][1] - samples[0][1]) / (samples[-1][0] - samples[0][0])
                        done = downloaded_before + current
                        eta = (total - done) / speed if speed > 0 else None
                        if progress:
                            progress(filename, done, total, speed, eta)
                os.replace(part, final_blob)
                if expected_size and final_blob.stat().st_size != expected_size:
                    final_blob.unlink(missing_ok=True)
                    raise RuntimeError(
                        f"Integrity check failed for {filename}: expected {expected_size} bytes."
                    )
                self._parts.discard(part)
                self._link_snapshot(repo_dir, commit, filename, final_blob)
                downloaded_before += expected_size or current

        refs = repo_dir / "refs"
        refs.mkdir(exist_ok=True)
        (refs / "main").write_text(commit, encoding="utf-8")
        return repo_dir / "snapshots" / commit

    @staticmethod
    def _link_snapshot(repo_dir: Path, commit: str, filename: str, blob: Path) -> None:
        target = repo_dir / "snapshots" / commit / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.unlink(missing_ok=True)
        try:
            target.symlink_to(os.path.relpath(blob, target.parent))
        except OSError:
            import shutil
            shutil.copy2(blob, target)


class ModelDownloadManager(HuggingFaceModelDownloadManager):
    def __init__(self, model_name: str, cache_dir: Path | None = None, chunk_size: int = 1024 * 256) -> None:
        self.model_name = model_name
        super().__init__(
            f"{FASTER_WHISPER_MODEL_PREFIX}-{model_name}",
            Path(cache_dir or whisper_cache_dir()),
            chunk_size,
        )
