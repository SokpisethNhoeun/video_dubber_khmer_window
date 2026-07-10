from __future__ import annotations

from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Callable


class PipelineLogger:
    def __init__(self, log_file: Path, gui_callback: Callable[[str], None] | None = None) -> None:
        self.log_file = log_file
        self.gui_callback = gui_callback
        self._lock = Lock()
        try:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass

    def __call__(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] {message}"
        with self._lock:
            try:
                with self.log_file.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
            except OSError:
                # Silently fail or write to stderr if log file is unwritable
                pass
        if self.gui_callback:
            try:
                self.gui_callback(line)
            except Exception:
                # Prevent GUI callback failures from crashing the main pipeline thread
                pass
