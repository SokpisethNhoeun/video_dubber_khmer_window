from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QMouseEvent
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

    def set_selected(self, selected: bool) -> None:
        self.setProperty("selected", "true" if selected else "false")
        self.style().unpolish(self)
        self.style().polish(self)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.plan_id)
        super().mousePressEvent(event)
