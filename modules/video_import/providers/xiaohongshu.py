from __future__ import annotations

import hashlib
import re
from pathlib import Path
from threading import Event
from urllib.parse import urlsplit, urlunsplit

from modules.video_import.base import DownloadResult, ProgressHook, VideoImportProvider
from modules.video_import.errors import DownloadCancelledError, VideoImportError

_XHS_HOSTS = {
    "xiaohongshu.com",
    "www.xiaohongshu.com",
    "rednote.com",
    "www.rednote.com",
    "xhslink.com",
    "www.xhslink.com",
}
_CONTENT_ID_RE = re.compile(r"/(?:explore|discovery/item)/([0-9a-z]+)", re.IGNORECASE)


class XiaohongshuProvider(VideoImportProvider):
    name = "xiaohongshu"
    display_name = "Xiaohongshu / RedNote"

    def can_handle(self, url: str) -> bool:
        host = urlsplit(url.strip()).netloc.lower()
        return host in _XHS_HOSTS

    def normalize_url(self, url: str) -> str:
        parts = urlsplit(url.strip())
        scheme = parts.scheme or "https"
        host = parts.netloc.lower()
        return urlunsplit((scheme, host, parts.path, parts.query, ""))

    def content_id(self, url: str) -> str | None:
        normalized = self.normalize_url(url)
        match = _CONTENT_ID_RE.search(urlsplit(normalized).path)
        if match:
            return match.group(1).lower()
        return None

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
        except Exception as exc:  # pragma: no cover - exercised indirectly via setup/use errors
            raise VideoImportError("yt-dlp is not installed. Install it to import URL videos.") from exc

        normalized = self.normalize_url(url)
        dest_dir.mkdir(parents=True, exist_ok=True)

        def on_progress(status: dict) -> None:
            if cancel_event and cancel_event.is_set():
                raise DownloadCancelledError("URL import cancelled by user.")
            if progress_hook is not None:
                progress_hook(normalized, status)

        ydl_opts = {
            "format": "direct/bestvideo*+bestaudio/best",
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
            raise VideoImportError(f"Failed to download {self.display_name} video: {exc}") from exc

        downloaded = self._resolve_downloaded_path(dest_dir, info)
        if downloaded is None or not downloaded.exists():
            raise VideoImportError(f"Downloaded file not found for {normalized}")
        return DownloadResult(downloaded, self._source_title(info))

    @staticmethod
    def cache_key(url: str) -> str:
        return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]

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
