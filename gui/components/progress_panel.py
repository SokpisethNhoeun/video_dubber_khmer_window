from __future__ import annotations

from PyQt6.QtWidgets import QLabel, QProgressBar, QVBoxLayout, QWidget

from config.models import STAGES


class ProgressPanel(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("ProgressPanel")
        self._bars: dict[str, QProgressBar] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        title = QLabel("Pipeline Progress")
        title.setObjectName("SectionTitle")
        layout.addWidget(title)

        for stage_key, stage_label in STAGES:
            label = QLabel(stage_label)
            label.setObjectName("ProgressLabel")
            layout.addWidget(label)

            bar = QProgressBar()
            bar.setFixedHeight(12)
            bar.setTextVisible(True)
            bar.setValue(0)
            bar.setFormat("%p%")
            self._bars[stage_key] = bar
            layout.addWidget(bar)

        layout.addStretch(1)

    def set_progress(self, stage: str, value: int) -> None:
        if stage in self._bars:
            self._bars[stage].setValue(max(0, min(100, value)))

    def reset(self) -> None:
        for bar in self._bars.values():
            bar.setValue(0)

    @property
    def bars(self) -> dict[str, QProgressBar]:
        return self._bars
