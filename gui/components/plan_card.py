from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QKeyEvent, QMouseEvent
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget


class PlanCard(QWidget):
    """Selectable subscription plan card: name, price, feature bullets, optional badge."""

    clicked = pyqtSignal(str)

    def __init__(
        self,
        plan_id: str,
        name: str,
        price: str,
        features: list[str],
        *,
        recommended: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.plan_id = plan_id
        self.setObjectName("Card")
        self.setProperty("hoverable", "true")
        self.setProperty("selected", "false")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(6)

        if recommended:
            badge = QLabel("BEST VALUE")
            badge.setObjectName("CardBadge")
            layout.addWidget(badge, 0, Qt.AlignmentFlag.AlignLeft)

        name_label = QLabel(name)
        name_label.setStyleSheet("font-size: 15px; font-weight: 700;")
        layout.addWidget(name_label)

        price_label = QLabel(price)
        price_label.setStyleSheet("font-size: 20px; font-weight: 800;")
        layout.addWidget(price_label)

        for feature in features:
            feature_label = QLabel(f"✓  {feature}")
            feature_label.setObjectName("PageDesc")
            feature_label.setWordWrap(True)
            layout.addWidget(feature_label)

        layout.addStretch(1)

        self.selection_label = QLabel("○  Select this plan")
        self.selection_label.setObjectName("CardSelection")
        self.selection_label.setStyleSheet("font-weight: 700;")
        layout.addWidget(self.selection_label)

        self.setAccessibleName(f"{name} subscription plan")
        self.setAccessibleDescription(f"{price}. Select this plan to continue to payment.")

    def set_selected(self, selected: bool) -> None:
        self.setProperty("selected", "true" if selected else "false")
        self.selection_label.setText("✓  Selected" if selected else "○  Select this plan")
        self.style().unpolish(self)
        self.style().polish(self)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.plan_id)
        super().mousePressEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            self.clicked.emit(self.plan_id)
            event.accept()
            return
        super().keyPressEvent(event)
