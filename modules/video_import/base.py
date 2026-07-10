from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Callable


ProgressHook = Callable[[str, dict], None]


@dataclass(frozen=True)
class DownloadResult:
    path: Path
    source_title: str | None = None


class VideoImportProvider(ABC):
    name: str
    display_name: str

    @abstractmethod
    def can_handle(self, url: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def normalize_url(self, url: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def content_id(self, url: str) -> str | None:
        raise NotImplementedError

    @abstractmethod
    def download(
        self,
        url: str,
        dest_dir: Path,
        *,
        cookies_file: Path | None = None,
        cancel_event: Event | None = None,
        progress_hook: ProgressHook | None = None,
    ) -> Path | DownloadResult:
        raise NotImplementedError
