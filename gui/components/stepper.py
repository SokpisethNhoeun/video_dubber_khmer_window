from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget


class _StepDot(QWidget):
    clicked = pyqtSignal()

    def __init__(self, index: int, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("StepperItem")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.badge = QLabel(str(index + 1))
        self.badge.setObjectName("StepperBadge")
        self.badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.badge.setFixedSize(30, 30)

        self.title = QLabel(title)
        self.title.setObjectName("StepperTitle")
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self.badge, 0, Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self.title, 0, Qt.AlignmentFlag.AlignHCenter)

    def set_state(self, state: str) -> None:
        # state: "current" | "done" | "pending"
        self.badge.setProperty("state", state)
        self.title.setProperty("state", state)
        # Force re-polish so the property selector matches.
        for w in (self.badge, self.title):
            w.style().unpolish(w)
            w.style().polish(w)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            self.clicked.emit()
            return
        super().keyPressEvent(event)


class _StepConnector(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("StepperConnector")
        self.setFixedHeight(2)
        self.setMinimumWidth(30)

    def set_active(self, active: bool) -> None:
        self.setProperty("active", "true" if active else "false")
        self.style().unpolish(self)
        self.style().polish(self)


class Stepper(QWidget):
    """Horizontal step indicator. Emits step_clicked(int) when a step is clicked."""

    step_clicked = pyqtSignal(int)

    def __init__(self, titles: list[str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Stepper")
        self._current = 0
        self._completed: set[int] = set()
        self._dots: list[_StepDot] = []
        self._connectors: list[_StepConnector] = []
        self._row = QHBoxLayout(self)
        self._row.setContentsMargins(12, 10, 12, 10)
        self._row.setSpacing(6)
        self.set_titles(titles)

    def set_titles(self, titles: list[str]) -> None:
        # Clear existing dots/connectors.
        while self._row.count():
            item = self._row.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        self._dots = []
        self._connectors = []

        for i, title in enumerate(titles):
            dot = _StepDot(i, title)
            dot.clicked.connect(lambda idx=i: self.step_clicked.emit(idx))
            self._dots.append(dot)
            self._row.addWidget(dot, 0)
            if i < len(titles) - 1:
                conn = _StepConnector()
                self._connectors.append(conn)
                self._row.addWidget(conn, 1)

        if self._dots:
            self.set_current(min(self._current, len(self._dots) - 1))

    def set_current(self, index: int) -> None:
        self._current = max(0, min(index, len(self._dots) - 1))
        self._render()

    def set_completed(self, indices: set[int]) -> None:
        """Mark steps as actually completed, independent of which page is showing.

        Without this, a dot's "done" state would just mean "the user has
        scrolled past it", which is misleading once navigation is free-form
        (Back/Next/step-click can all jump to any page regardless of whether
        earlier steps were finished).
        """
        self._completed = set(indices)
        self._render()

    def _render(self) -> None:
        for i, dot in enumerate(self._dots):
            if i == self._current:
                dot.set_state("current")
            elif i in self._completed:
                dot.set_state("done")
            else:
                dot.set_state("pending")
        for i, conn in enumerate(self._connectors):
            conn.set_active(i < self._current)

    def current(self) -> int:
        return self._current
