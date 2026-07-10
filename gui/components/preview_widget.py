from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Callable

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class _PreviewWorker(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, render_fn: Callable[[], Path], parent=None) -> None:
        super().__init__(parent)
        self._render_fn = render_fn

    def run(self) -> None:
        try:
            path = self._render_fn()
            self.finished.emit(str(path))
        except Exception as e:
            self.error.emit(str(e))


class OverlayPreviewWidget(QWidget):
    """Reusable preview widget that displays a rendered PNG frame."""

    def __init__(self, width: int = 480, height: int = 270, parent=None) -> None:
        super().__init__(parent)
        self._preview_width = width
        self._preview_height = height
        self._worker: _PreviewWorker | None = None
        self._temp_dir = tempfile.mkdtemp(prefix="overlay_preview_")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self._image_label = QLabel()
        self._image_label.setFixedSize(width, height)
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setStyleSheet(
            "QLabel { background: #1e1e2e; border: 1px solid #333; border-radius: 6px; }"
        )
        self._image_label.setText("No preview")
        layout.addWidget(self._image_label)

        btn_row = QHBoxLayout()
        self._refresh_btn = QPushButton("Refresh Preview")
        self._refresh_btn.setObjectName("CompactButton")
        self._refresh_btn.clicked.connect(self._on_refresh_clicked)
        btn_row.addWidget(self._refresh_btn)

        self._status_label = QLabel("")
        self._status_label.setObjectName("HintLabel")
        btn_row.addWidget(self._status_label)
        btn_row.addStretch()

        layout.addLayout(btn_row)

        self._render_fn: Callable[[], Path] | None = None

    def set_render_function(self, fn: Callable[[], Path]) -> None:
        self._render_fn = fn

    def get_temp_path(self, name: str = "preview.png") -> Path:
        return Path(self._temp_dir) / name

    def refresh(self) -> None:
        self._on_refresh_clicked()

    def _on_refresh_clicked(self) -> None:
        if not self._render_fn:
            self._status_label.setText("No render function set")
            return

        if self._worker and self._worker.isRunning():
            return

        self._status_label.setText("Rendering...")
        self._refresh_btn.setEnabled(False)

        self._worker = _PreviewWorker(self._render_fn, self)
        self._worker.finished.connect(self._on_render_done)
        self._worker.error.connect(self._on_render_error)
        self._worker.start()

    def _on_render_done(self, path: str) -> None:
        self._refresh_btn.setEnabled(True)
        self._status_label.setText("")

        pixmap = QPixmap(path)
        if not pixmap.isNull():
            scaled = pixmap.scaled(
                self._preview_width, self._preview_height,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._image_label.setPixmap(scaled)
        else:
            self._image_label.setText("Failed to load preview")

    def _on_render_error(self, error: str) -> None:
        self._refresh_btn.setEnabled(True)
        self._status_label.setText(f"Error: {error[:60]}")
        self._image_label.setText("Preview failed")
