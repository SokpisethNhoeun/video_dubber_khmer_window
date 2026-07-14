from __future__ import annotations

import hashlib
from pathlib import Path
from threading import Event
from urllib.parse import urlsplit, urlunsplit

from modules.video_import.base import DownloadResult, ProgressHook, VideoImportProvider
from modules.video_import.errors import DownloadCancelledError, VideoImportError


class GenericYtDlpProvider(VideoImportProvider):
    name = "generic"
    display_name = "URL Download (Generic)"

    def can_handle(self, url: str) -> bool:
        url_strip = url.strip()
        return url_strip.startswith(("http://", "https://"))

    def normalize_url(self, url: str) -> str:
        parts = urlsplit(url.strip())
        scheme = parts.scheme or "https"
        host = parts.netloc.lower()
        return urlunsplit((scheme, host, parts.path, parts.query, ""))

    def content_id(self, url: str) -> str | None:
        normalized = self.normalize_url(url)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

    def download(
        self,
        url: str,
        dest_dir: Path,
        *,
        cookies_file: Path | None = None,
        cancel_event: Event | None = None,
        progress_hook: ProgressHook | None = None,
    ) -> DownloadResult:
        try:
            from yt_dlp import YoutubeDL
            from yt_dlp.utils import DownloadError
        except Exception as exc:  # pragma: no cover
            raise VideoImportError("yt-dlp is not installed. Install it to import URL videos.") from exc

        normalized = self.normalize_url(url)
        dest_dir.mkdir(parents=True, exist_ok=True)

        def on_progress(status: dict) -> None:
            if cancel_event and cancel_event.is_set():
                raise DownloadCancelledError("URL import cancelled by user.")
            if progress_hook is not None:
                progress_hook(normalized, status)

        # Prioritize formats without watermarks (e.g. TikTok/Douyin unwatermarked raw streams)
        # using standard yt-dlp format sorting.
        ydl_opts = {
            "format": "bestvideo*+bestaudio/best",
            "merge_output_format": "mp4",
            "outtmpl": str(dest_dir / "source.%(ext)s"),
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "restrictfilenames": False,
            "progress_hooks": [on_progress],
        }
        if cookies_file is not None:
            ydl_opts["cookiefile"] = str(cookies_file)

        try:
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(normalized, download=True)
        except DownloadCancelledError:
            raise
        except DownloadError as exc:
            raise VideoImportError(str(exc)) from exc
        except Exception as exc:
            raise VideoImportError(f"Failed to download video: {exc}") from exc

        downloaded = self._resolve_downloaded_path(dest_dir, info)
        if downloaded is None or not downloaded.exists():
            raise VideoImportError(f"Downloaded file not found for {normalized}")
        return DownloadResult(downloaded, self._source_title(info))

    @staticmethod
    def _resolve_downloaded_path(dest_dir: Path, info: dict) -> Path | None:
        filename = info.get("_filename")
        if filename:
            path = Path(filename)
            if path.exists():
                return path

        requested_downloads = info.get("requested_downloads") or []
        for item in requested_downloads:
            filepath = item.get("filepath")
            if filepath and Path(filepath).exists():
                return Path(filepath)

        candidates = sorted(dest_dir.glob("source.*"))
        return candidates[-1] if candidates else None

    @staticmethod
    def _source_title(info: dict) -> str | None:
        title = str(info.get("title") or "").strip()
        return title or None
