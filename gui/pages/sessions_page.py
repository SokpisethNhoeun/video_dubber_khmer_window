from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.session import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    DubbingSession,
    delete_session,
    list_sessions,
    prune_sessions,
)


_STATUS_LABELS = {
    "running": "Running",
    "failed": "Failed",
    "cancelled": "Cancelled",
    "completed": "Completed",
}


class SessionsPage(QWidget):
    resume_requested = pyqtSignal(str)  # work_dir path
    edit_requested = pyqtSignal(str)    # work_dir path
    delete_requested = pyqtSignal(str)  # work_dir path

    def __init__(self, project_root: Path, parent=None) -> None:
        super().__init__(parent)
        self._project_root = project_root
        self._summaries: list = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)

        header = QLabel("Sessions")
        header.setObjectName("PageHeader")
        layout.addWidget(header)

        desc = QLabel(
            "Previous dubbing runs are saved here. Resume a failed run or open "
            "a completed session in the Segment Editor to fine-tune individual lines."
        )
        desc.setObjectName("PageDesc")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Video", "Status", "Failed At", "Segments", "Last Updated"])
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        h = self.table.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        h.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        h.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        h.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        h.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.table, 1)

        btn_row = QHBoxLayout()
        self.resume_button = QPushButton("Resume")
        self.resume_button.setObjectName("PrimaryButton")
        self.resume_button.setToolTip("Continue from the failed/cancelled stage")
        self.resume_button.clicked.connect(self._on_resume)
        btn_row.addWidget(self.resume_button)

        self.edit_button = QPushButton("Open in Editor")
        self.edit_button.setObjectName("SecondaryButton")
        self.edit_button.setToolTip("Edit segment text and re-dub")
        self.edit_button.clicked.connect(self._on_edit)
        btn_row.addWidget(self.edit_button)

        btn_row.addStretch(1)

        self.delete_button = QPushButton("Delete")
        self.delete_button.setObjectName("DangerButton")
        self.delete_button.clicked.connect(self._on_delete)
        btn_row.addWidget(self.delete_button)

        self.prune_button = QPushButton("Prune Old")
        self.prune_button.setObjectName("CompactButton")
        self.prune_button.setToolTip("Keep only the 5 most recent completed sessions")
        self.prune_button.clicked.connect(self._on_prune)
        btn_row.addWidget(self.prune_button)

        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.setObjectName("CompactButton")
        self.refresh_button.clicked.connect(self.refresh)
        btn_row.addWidget(self.refresh_button)

        layout.addLayout(btn_row)

    def refresh(self) -> None:
        temp_dir = self._project_root / "temp"
        self._summaries = list_sessions(temp_dir)
        self.table.setRowCount(len(self._summaries))
        for row, s in enumerate(self._summaries):
            self.table.setItem(row, 0, QTableWidgetItem(s.video_name))
            status_item = QTableWidgetItem(_STATUS_LABELS.get(s.status, s.status))
            self.table.setItem(row, 1, status_item)
            self.table.setItem(row, 2, QTableWidgetItem(s.failed_stage or "—"))
            self.table.setItem(row, 3, QTableWidgetItem(str(s.segment_count)))
            self.table.setItem(row, 4, QTableWidgetItem(s.updated_at[:16].replace("T", " ")))

    def _selected_work_dir(self) -> str | None:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return None
        idx = rows[0].row()
        if idx >= len(self._summaries):
            return None
        return str(self._summaries[idx].work_dir)

    def _on_resume(self) -> None:
        wd = self._selected_work_dir()
        if wd:
            self.resume_requested.emit(wd)

    def _on_edit(self) -> None:
        wd = self._selected_work_dir()
        if wd:
            self.edit_requested.emit(wd)

    def _on_delete(self) -> None:
        wd = self._selected_work_dir()
        if not wd:
            return
        reply = QMessageBox.question(
            self, "Delete Session",
            "Delete this session and all its temporary files?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            delete_session(Path(wd))
            self.refresh()

    def _on_prune(self) -> None:
        removed = prune_sessions(self._project_root / "temp", keep=5)
        self.refresh()
        if removed:
            QMessageBox.information(self, "Pruned", f"Removed {removed} old completed session(s).")
