from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget


class WizardNavBar(QWidget):
    """Back / Next navigation bar for a stepped wizard.

    Emits back_clicked / next_clicked. The Next button auto-relabels to
    "Finish" on the last step; callers can override via set_next_label.
    """

    back_clicked = pyqtSignal()
    next_clicked = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("WizardNavBar")

        row = QHBoxLayout(self)
        row.setContentsMargins(4, 6, 4, 6)
        row.setSpacing(10)

        self.hint = QLabel("")
        self.hint.setObjectName("Hint")
        self.hint.setWordWrap(True)

        self.back_button = QPushButton("◀  Back")
        self.back_button.setObjectName("SecondaryButton")
        self.back_button.clicked.connect(self.back_clicked.emit)

        self.next_button = QPushButton("Next  ▶")
        self.next_button.setObjectName("StartButton")
        self.next_button.clicked.connect(self.next_clicked.emit)

        row.addWidget(self.hint, 1)
        row.addWidget(self.back_button)
        row.addWidget(self.next_button)

    def set_state(self, current: int, total: int) -> None:
        self.back_button.setEnabled(current > 0)
        is_last = current >= total - 1
        self.next_button.setText("Finish ✓" if is_last else "Next  ▶")

    def set_next_enabled(self, enabled: bool, reason: str = "") -> None:
        self.next_button.setEnabled(enabled)
        self.hint.setText(reason if not enabled else "")

    def set_next_label(self, label: str) -> None:
        self.next_button.setText(label)
