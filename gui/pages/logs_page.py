from __future__ import annotations

from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from gui.dialogs.transcript_review import ZoomableTextEdit
from gui.components.progress_panel import ProgressPanel


class LogsPage(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        header_row = QHBoxLayout()
        header = QLabel("Pipeline Monitor & Logs")
        header.setObjectName("PageHeader")
        header_row.addWidget(header)
        header_row.addStretch(1)

        self.clear_button = QPushButton("Clear Log")
        self.clear_button.setObjectName("CompactButton")
        self.clear_button.clicked.connect(self._clear_log)
        header_row.addWidget(self.clear_button)
        layout.addLayout(header_row)

        self.progress_panel = ProgressPanel()
        layout.addWidget(self.progress_panel)

        log_header = QHBoxLayout()
        log_label = QLabel("Execution Log")
        log_label.setObjectName("SectionTitle")
        log_header.addWidget(log_label)
        log_header.addStretch(1)
        hint = QLabel("Ctrl + wheel to zoom")
        hint.setObjectName("HintLabel")
        log_header.addWidget(hint)
        layout.addLayout(log_header)

        self.log_console = ZoomableTextEdit()
        layout.addWidget(self.log_console, 1)

    def append_log(self, message: str) -> None:
        self.log_console.append(message)

    def _clear_log(self) -> None:
        self.log_console.clear()

    def set_progress(self, stage: str, value: int) -> None:
        self.progress_panel.set_progress(stage, value)

    def reset_progress(self) -> None:
        self.progress_panel.reset()
