from __future__ import annotations

import re

from modules.video_import.base import VideoImportProvider
from modules.video_import.providers import XiaohongshuProvider, GenericYtDlpProvider

_URL_RE = re.compile(r"https?://[^\s<>\"]+", re.IGNORECASE)
_PROVIDERS: list[VideoImportProvider] = [
    XiaohongshuProvider(),
    GenericYtDlpProvider(),
]


def providers() -> list[VideoImportProvider]:
    return list(_PROVIDERS)


def resolve_provider(url: str) -> VideoImportProvider | None:
    for provider in _PROVIDERS:
        if provider.can_handle(url):
            return provider
    return None


def extract_urls(text: str) -> list[str]:
    urls = []
    seen: set[str] = set()
    for match in _URL_RE.findall(text or ""):
        trimmed = match.rstrip(".,;)")
        if trimmed not in seen:
            urls.append(trimmed)
            seen.add(trimmed)
    return urls
