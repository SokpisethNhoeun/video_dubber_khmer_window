from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
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

from core.session import DubbingSession, list_sessions


class EditorPage(QWidget):
    redub_requested = pyqtSignal(str, object)  # work_dir, {index: text}
    preview_requested = pyqtSignal(str, int, str)  # work_dir, segment_index, text

    def __init__(self, project_root: Path, parent=None) -> None:
        super().__init__(parent)
        self._project_root = project_root
        self._session: DubbingSession | None = None
        self._dirty: set[int] = set()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)

        header = QLabel("Segment Editor")
        header.setObjectName("PageHeader")
        layout.addWidget(header)

        desc = QLabel(
            "Edit the Khmer text for any segment, then click Re-dub to regenerate "
            "only the changed segments without re-running the full pipeline."
        )
        desc.setObjectName("PageDesc")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Session selector
        sel_row = QHBoxLayout()
        sel_row.addWidget(QLabel("Session:"))
        self.session_combo = QComboBox()
        self.session_combo.setMinimumWidth(300)
        sel_row.addWidget(self.session_combo, 1)
        self.load_button = QPushButton("Load")
        self.load_button.setObjectName("CompactButton")
        self.load_button.clicked.connect(self._on_load)
        sel_row.addWidget(self.load_button)
        layout.addLayout(sel_row)

        # Segment table
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["#", "Start", "End", "Speaker", "Original", "Khmer (editable)"])
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.verticalHeader().setVisible(False)
        self.table.itemSelectionChanged.connect(self._update_preview_button)
        h = self.table.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        h.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        h.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        h.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        h.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        h.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        self.table.cellChanged.connect(self._on_cell_changed)
        layout.addWidget(self.table, 1)

        # Action row
        btn_row = QHBoxLayout()
        self.dirty_label = QLabel("No edits")
        self.dirty_label.setObjectName("HintLabel")
        btn_row.addWidget(self.dirty_label)
        btn_row.addStretch(1)

        self.preview_button = QPushButton("Preview Segment")
        self.preview_button.setObjectName("SecondaryButton")
        self.preview_button.setEnabled(False)
        self.preview_button.clicked.connect(self._on_preview)
        btn_row.addWidget(self.preview_button)

        self.redub_button = QPushButton("Re-dub Edited Segments")
        self.redub_button.setObjectName("PrimaryButton")
        self.redub_button.setEnabled(False)
        self.redub_button.clicked.connect(self._on_redub)
        btn_row.addWidget(self.redub_button)
        layout.addLayout(btn_row)

    def refresh_sessions(self) -> None:
        self.session_combo.clear()
        temp_dir = self._project_root / "temp"
        summaries = list_sessions(temp_dir)
        for s in summaries:
            if "translation" in s.completed_stages:
                label = f"{s.video_name} — {s.status} ({s.segment_count} segs)"
                self.session_combo.addItem(label, str(s.work_dir))

    def load_session(self, work_dir: str) -> None:
        try:
            self._session = DubbingSession.load(Path(work_dir))
        except Exception as exc:
            QMessageBox.warning(self, "Load Failed", str(exc))
            return
        self._dirty.clear()
        self._populate_table()
        self._update_dirty_label()
        self._update_preview_button()

        idx = self.session_combo.findData(work_dir)
        if idx >= 0:
            self.session_combo.setCurrentIndex(idx)

    def _on_load(self) -> None:
        wd = self.session_combo.currentData()
        if wd:
            self.load_session(wd)

    def _populate_table(self) -> None:
        self.table.blockSignals(True)
        segments = self._session.segments if self._session else []
        self.table.setRowCount(len(segments))
        for row, seg in enumerate(segments):
            self.table.setItem(row, 0, self._ro_item(str(seg.index + 1)))
            self.table.setItem(row, 1, self._ro_item(self._fmt_time(seg.start)))
            self.table.setItem(row, 2, self._ro_item(self._fmt_time(seg.end)))
            self.table.setItem(row, 3, self._ro_item(seg.speaker_label or seg.speaker_id or "—"))
            self.table.setItem(row, 4, self._ro_item(seg.text or ""))
            khmer_item = QTableWidgetItem(seg.tts_text or "")
            khmer_item.setFlags(khmer_item.flags() | Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 5, khmer_item)
        self.table.blockSignals(False)
        self._update_preview_button()

    def _on_cell_changed(self, row: int, col: int) -> None:
        if col != 5 or self._session is None:
            return
        seg = self._session.segments[row]
        new_text = (self.table.item(row, 5).text() or "").strip()
        original = seg.tts_text or ""
        if new_text != original:
            self._dirty.add(seg.index)
        else:
            self._dirty.discard(seg.index)
        self._update_dirty_label()
        self._update_preview_button()

    def _update_dirty_label(self) -> None:
        n = len(self._dirty)
        if n == 0:
            self.dirty_label.setText("No edits")
            self.redub_button.setEnabled(False)
        else:
            self.dirty_label.setText(f"{n} segment(s) edited")
            self.redub_button.setEnabled(True)

    def _on_redub(self) -> None:
        if not self._session or not self._dirty:
            return
        edits: dict[int, str] = {}
        for row in range(self.table.rowCount()):
            seg = self._session.segments[row]
            if seg.index in self._dirty:
                text = (self.table.item(row, 5).text() or "").strip()
                if not text:
                    QMessageBox.warning(
                        self, "Empty Text",
                        f"Segment {seg.index + 1} has empty Khmer text. Please fill it in.",
                    )
                    return
                edits[seg.index] = text
        self.redub_button.setEnabled(False)
        self.redub_requested.emit(str(self._session.work_dir), edits)

    def _selected_row(self) -> int:
        rows = sorted({index.row() for index in self.table.selectionModel().selectedRows()})
        if rows:
            return rows[0]
        return self.table.currentRow()

    def _update_preview_button(self) -> None:
        enabled = False
        if self._session is not None:
            row = self._selected_row()
            if 0 <= row < self.table.rowCount():
                item = self.table.item(row, 5)
                enabled = bool(item and item.text().strip())
        self.preview_button.setEnabled(enabled)

    def _on_preview(self) -> None:
        if not self._session:
            return
        row = self._selected_row()
        if row < 0 or row >= self.table.rowCount():
            QMessageBox.information(self, "Preview Segment", "Select one segment to preview.")
            return
        text = (self.table.item(row, 5).text() or "").strip()
        if not text:
            QMessageBox.warning(self, "Preview Segment", "The selected segment has no Khmer text.")
            return
        segment = self._session.segments[row]
        self.preview_button.setEnabled(False)
        self.preview_button.setText("Previewing...")
        self.preview_requested.emit(str(self._session.work_dir), segment.index, text)

    def preview_finished(self) -> None:
        self.preview_button.setText("Preview Segment")
        self._update_preview_button()

    def _ro_item(self, text: str) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        return item

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        m, s = divmod(seconds, 60)
        return f"{int(m):02d}:{s:05.2f}"
