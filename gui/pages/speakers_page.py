from __future__ import annotations

from PyQt6.QtWidgets import (
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class SpeakersPage(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        header = QLabel("Speaker Detection")
        header.setObjectName("PageHeader")
        layout.addWidget(header)

        desc = QLabel(
            "Detect speakers in your video and map each one to a voice.\n"
            "This is only needed for multi-speaker videos."
        )
        desc.setObjectName("PageDesc")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        self.detect_button = QPushButton("Detect Speakers / Map Voices")
        self.detect_button.setObjectName("SecondaryButton")
        self.detect_button.setFixedHeight(44)
        layout.addWidget(self.detect_button)

        self.status_display = QTextEdit()
        self.status_display.setReadOnly(True)
        self.status_display.setObjectName("LogConsole")
        self.status_display.setPlaceholderText(
            "Speaker detection results will appear here.\n\n"
            "Tip: Select 'Auto per-person clone' or 'Per-person manual' in the Voice page first."
        )
        layout.addWidget(self.status_display, 1)

    def set_status(self, text: str) -> None:
        self.status_display.setPlainText(text)

    def append_status(self, text: str) -> None:
        self.status_display.append(text)
