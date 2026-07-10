from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from threading import Event
from typing import Callable

from modules.audio_utils import has_audio_stream, remove_tree
from modules.video_import.base import DownloadResult, VideoImportProvider
from modules.video_import.errors import DownloadCancelledError, UnsupportedUrlError, VideoImportError
from modules.video_import.registry import resolve_provider

LogCallback = Callable[[str], None]
ProgressCallback = Callable[[str, int], None]
VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".flv"}


class VideoImportService:
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir

    def import_video(
        self,
        url: str,
        *,
        cookies_file: Path | None = None,
        cancel_event: Event | None = None,
        log: LogCallback | None = None,
        progress: ProgressCallback | None = None,
        preferred_stem: str | None = None,
    ) -> Path:
        provider = resolve_provider(url)
        if provider is None:
            raise UnsupportedUrlError(f"Unsupported video URL: {url}")

        normalized = provider.normalize_url(url)
        content_id = provider.content_id(normalized) or self._fallback_key(normalized)
        dest_dir = self.cache_dir / provider.name / content_id
        safe_stem = self._safe_filename_stem(preferred_stem)
        cached = self._cached_video(dest_dir)
        if cached is not None:
            if cached.stem == "source":
                cached = self._rename_video(cached, safe_stem)
            self._write_metadata(dest_dir, provider, normalized, cached)
            if log is not None:
                log(f"Using cached {provider.display_name} video for {normalized}")
            if progress is not None:
                progress(normalized, 100)
            return cached

        if cookies_file is not None and not cookies_file.exists():
            raise VideoImportError(f"Cookies file not found: {cookies_file}")

        if log is not None:
            log(f"Importing {provider.display_name} video from {normalized}")

        def on_progress(progress_url: str, status: dict) -> None:
            if cancel_event and cancel_event.is_set():
                raise DownloadCancelledError("URL import cancelled by user.")
            if progress is None:
                return
            progress(progress_url, self._progress_percent(status))

        download_result = provider.download(
            normalized,
            dest_dir,
            cookies_file=cookies_file,
            cancel_event=cancel_event,
            progress_hook=on_progress,
        )
        source_title = None
        if isinstance(download_result, DownloadResult):
            downloaded = download_result.path
            source_title = download_result.source_title
        else:
            downloaded = download_result
        if not has_audio_stream(downloaded):
            raise VideoImportError(f"Downloaded video has no audio track: {downloaded.name}")
        khmer_title = self._translate_title_to_khmer(source_title)
        downloaded = self._rename_video(downloaded, self._safe_filename_stem(khmer_title) or safe_stem)
        self._write_metadata(dest_dir, provider, normalized, downloaded, source_title, khmer_title)
        if progress is not None:
            progress(normalized, 100)
        return downloaded

    @staticmethod
    def source_url_for_video(video_path: Path) -> str:
        metadata_path = Path(video_path).parent / "import.json"
        if not metadata_path.exists():
            return ""
        try:
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            return ""
        return str(data.get("source_url") or data.get("normalized_url") or "").strip()

    def cleanup_video_cache(self, video_path: Path) -> bool:
        """Remove the URL-import cache folder for a processed imported video."""
        video_path = Path(video_path)
        import_dir = video_path.parent
        metadata_path = import_dir / "import.json"
        if not metadata_path.exists():
            return False
        if import_dir.resolve() == self.cache_dir.resolve():
            return False
        try:
            import_dir.resolve().relative_to(self.cache_dir.resolve())
        except ValueError:
            return False
        remove_tree(import_dir)
        return True

    @staticmethod
    def _write_metadata(
        dest_dir: Path,
        provider: VideoImportProvider,
        normalized_url: str,
        video_path: Path,
        source_title: str | None = None,
        khmer_title: str | None = None,
    ) -> None:
        dest_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "provider": provider.name,
            "provider_name": provider.display_name,
            "source_url": normalized_url,
            "normalized_url": normalized_url,
            "video_path": str(video_path),
        }
        if source_title:
            metadata["source_title"] = source_title
        if khmer_title:
            metadata["khmer_title"] = khmer_title
        (dest_dir / "import.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @staticmethod
    def _translate_title_to_khmer(source_title: str | None) -> str | None:
        if not source_title:
            return None
        try:
            from deep_translator import GoogleTranslator

            translated = GoogleTranslator(source="auto", target="km").translate(source_title)
        except Exception:
            return None
        clean = str(translated or "").strip()
        return clean or None

    @staticmethod
    def _cached_video(dest_dir: Path) -> Path | None:
        if not dest_dir.exists():
            return None
        candidates = sorted(p for p in dest_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES)
        for candidate in reversed(candidates):
            if has_audio_stream(candidate):
                return candidate
        return None

    @staticmethod
    def _safe_filename_stem(stem: str | None) -> str | None:
        clean = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", (stem or "").strip())
        clean = re.sub(r"\s+", " ", clean).strip(" ._-")
        if len(clean) > 120:
            clean = clean[:120].rstrip(" ._-")
        return clean or None

    @staticmethod
    def _rename_video(video_path: Path, preferred_stem: str | None) -> Path:
        if not preferred_stem:
            return video_path
        suffix = video_path.suffix or ".mp4"
        if video_path.stem == preferred_stem:
            return video_path

        target = video_path.with_name(f"{preferred_stem}{suffix}")
        if target.exists() and target != video_path:
            for index in range(2, 1000):
                candidate = video_path.with_name(f"{preferred_stem}_{index}{suffix}")
                if not candidate.exists():
                    target = candidate
                    break
            else:
                raise VideoImportError(f"Could not create a unique import filename for {preferred_stem}")

        video_path.rename(target)
        return target

    @staticmethod
    def _fallback_key(url: str) -> str:
        return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _progress_percent(status: dict) -> int:
        state = status.get("status")
        if state == "finished":
            return 100
        downloaded = status.get("downloaded_bytes") or 0
        total = status.get("total_bytes") or status.get("total_bytes_estimate") or 0
        if not total:
            return 0
        return max(0, min(99, int(downloaded * 100 / total)))
