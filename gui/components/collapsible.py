from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QPushButton, QVBoxLayout, QWidget


class CollapsibleSection(QWidget):
    """A section with a toggle header and a hidden/shown body widget."""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("CollapsibleSection")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)

        self.toggle = QPushButton(f"▶  {title}")
        self.toggle.setObjectName("CollapsibleToggle")
        self.toggle.setCheckable(True)
        self.toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self.toggle.clicked.connect(self._on_toggled)
        self._title = title

        self.body = QWidget()
        self.body.setObjectName("CollapsibleBody")
        self.body_layout = QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(6, 6, 6, 6)
        self.body_layout.setSpacing(8)
        self.body.hide()

        outer.addWidget(self.toggle)
        outer.addWidget(self.body)

    def add_widget(self, widget: QWidget) -> None:
        self.body_layout.addWidget(widget)

    def set_expanded(self, expanded: bool) -> None:
        self.toggle.setChecked(expanded)
        self._apply(expanded)

    def _on_toggled(self, checked: bool) -> None:
        self._apply(checked)

    def _apply(self, expanded: bool) -> None:
        self.body.setVisible(expanded)
        arrow = "▼" if expanded else "▶"
        self.toggle.setText(f"{arrow}  {self._title}")
