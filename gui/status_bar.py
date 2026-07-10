from __future__ import annotations

import subprocess

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QProgressBar, QWidget


def _query_gpu() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            if len(parts) >= 3:
                util = parts[0].strip()
                mem_used = parts[1].strip()
                mem_total = parts[2].strip()
                return f"GPU {util}%  |  VRAM {mem_used}/{mem_total} MB"
    except Exception:
        pass
    return ""


class _Separator(QFrame):
    def __init__(self) -> None:
        super().__init__()
        self.setFrameShape(QFrame.Shape.VLine)
        self.setObjectName("StatusSeparator")
        self.setFixedHeight(18)


class StatusBar(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("StatusBar")
        self.setFixedHeight(36)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 0, 16, 0)
        layout.setSpacing(12)

        # Progress section
        progress_icon = QLabel("⏳")
        progress_icon.setObjectName("StatusIcon")
        layout.addWidget(progress_icon)

        self.stage_label = QLabel("Ready")
        self.stage_label.setObjectName("StatusLabel")
        layout.addWidget(self.stage_label)

        self.overall_progress = QProgressBar()
        self.overall_progress.setFixedHeight(14)
        self.overall_progress.setFixedWidth(240)
        self.overall_progress.setTextVisible(True)
        self.overall_progress.setValue(0)
        layout.addWidget(self.overall_progress)

        layout.addWidget(_Separator())

        # GPU section
        gpu_icon = QLabel("🖥")
        gpu_icon.setObjectName("StatusIcon")
        layout.addWidget(gpu_icon)

        self.gpu_label = QLabel("GPU —")
        self.gpu_label.setObjectName("StatusLabel")
        self.gpu_label.setMinimumWidth(180)
        layout.addWidget(self.gpu_label)

        layout.addWidget(_Separator())

        # Queue / batch section
        queue_icon = QLabel("📋")
        queue_icon.setObjectName("StatusIcon")
        layout.addWidget(queue_icon)

        self.batch_label = QLabel("Queue: idle")
        self.batch_label.setObjectName("StatusLabel")
        layout.addWidget(self.batch_label)

        layout.addStretch(1)

        layout.addWidget(_Separator())

        # Notification section
        self.notification_label = QLabel("")
        self.notification_label.setObjectName("NotificationLabel")
        layout.addWidget(self.notification_label)

        # GPU polling timer
        self._gpu_timer = QTimer(self)
        self._gpu_timer.timeout.connect(self._refresh_gpu)
        self._gpu_timer.start(3000)
        self._refresh_gpu()

    def _refresh_gpu(self) -> None:
        info = _query_gpu()
        self.gpu_label.setText(info if info else "GPU —")

    def set_stage(self, stage: str, progress: int) -> None:
        self.stage_label.setText(stage)
        self.overall_progress.setValue(progress)

    def set_batch(self, text: str) -> None:
        if text:
            self.batch_label.setText(f"Queue: {text}")
        else:
            self.batch_label.setText("Queue: idle")

    def set_notification(self, text: str) -> None:
        self.notification_label.setText(text)

    def reset(self) -> None:
        self.stage_label.setText("Ready")
        self.overall_progress.setValue(0)
        self.batch_label.setText("Queue: idle")
        self.notification_label.setText("")
