from __future__ import annotations


class VideoImportError(RuntimeError):
    """Base error for URL-based video imports."""


class UnsupportedUrlError(VideoImportError):
    """Raised when no provider recognizes a URL."""


class DownloadCancelledError(VideoImportError):
    """Raised when the user cancels a download."""
