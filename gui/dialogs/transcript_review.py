from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
)


class ZoomableTextEdit(QTextEdit):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setReadOnly(True)
        self.setObjectName("LogConsole")

    def wheelEvent(self, event) -> None:
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            if delta > 0:
                self.zoomIn(1)
            elif delta < 0:
                self.zoomOut(1)
            event.accept()
        else:
            super().wheelEvent(event)

    def keyPressEvent(self, event) -> None:
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            key = event.key()
            if key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
                self.zoomIn(1)
                event.accept()
            elif key == Qt.Key.Key_Minus:
                self.zoomOut(1)
                event.accept()
            else:
                super().keyPressEvent(event)
        else:
            super().keyPressEvent(event)


class TranscriptReviewDialog(QDialog):
    COLUMNS = ["", "Enabled", "Time", "Speaker", "Source", "Raw Khmer", "Improved Khmer", "Notes"]

    def __init__(self, review_path: Path, preview_callback: Callable[[str], None] | None = None, parent=None) -> None:
        super().__init__(parent)
        self.review_path = review_path
        self.preview_callback = preview_callback
        self.payload: dict = {}
        self.setWindowTitle(f"Transcript Review - {review_path.name}")
        self.resize(1280, 650)

        layout = QVBoxLayout(self)
        self.table = QTableWidget()
        self.table.setColumnCount(len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        layout.addWidget(self.table, 1)

        actions = QHBoxLayout()
        preview_button = QPushButton("Preview Segment")
        merge_button = QPushButton("Merge Selected")
        save_button = QPushButton("Save")
        close_button = QPushButton("Close")
        preview_button.clicked.connect(self._preview_selected)
        merge_button.clicked.connect(self._merge_selected)
        save_button.clicked.connect(self._save)
        close_button.clicked.connect(self.accept)
        actions.addWidget(preview_button)
        actions.addWidget(merge_button)
        actions.addStretch(1)
        actions.addWidget(save_button)
        actions.addWidget(close_button)
        layout.addLayout(actions)

        self._load()

    def _load(self) -> None:
        self.payload = json.loads(self.review_path.read_text(encoding="utf-8"))
        segments = self.payload.get("segments", [])
        self.table.setRowCount(len(segments))
        for row, segment in enumerate(segments):
            start = float(segment.get("start", 0) or 0)
            end = float(segment.get("end", 0) or 0)

            play_btn = QPushButton("Play")
            play_btn.setFixedWidth(50)
            play_btn.clicked.connect(lambda _, r=row: self._play_row(r))
            self.table.setCellWidget(row, 0, play_btn)

            enabled_item = QTableWidgetItem()
            enabled_item.setCheckState(
                Qt.CheckState.Checked if segment.get("enabled", True) else Qt.CheckState.Unchecked
            )
            self.table.setItem(row, 1, enabled_item)
            self.table.setItem(row, 2, QTableWidgetItem(f"{start:.2f} - {end:.2f}"))
            self.table.setItem(row, 3, QTableWidgetItem(str(segment.get("speaker", ""))))
            self.table.setItem(row, 4, QTableWidgetItem(str(segment.get("source_text", ""))))
            self.table.setItem(row, 5, QTableWidgetItem(str(segment.get("raw_khmer_text", ""))))
            self.table.setItem(row, 6, QTableWidgetItem(str(segment.get("user_edited_text") or segment.get("improved_khmer_text", ""))))
            self.table.setItem(row, 7, QTableWidgetItem(str(segment.get("review_notes", ""))))
        self.table.setColumnWidth(0, 55)
        self.table.resizeColumnsToContents()
        self.table.setColumnWidth(0, 55)

    def _play_row(self, row: int) -> None:
        text_item = self.table.item(row, 6)
        text = text_item.text().strip() if text_item else ""
        if not text:
            text_item = self.table.item(row, 5)
            text = text_item.text().strip() if text_item else ""
        if not text:
            return
        if self.preview_callback:
            self.preview_callback(text)

    def _selected_rows(self) -> list[int]:
        return sorted({index.row() for index in self.table.selectionModel().selectedRows()})

    def _merge_selected(self) -> None:
        rows = self._selected_rows()
        if len(rows) < 2:
            QMessageBox.information(self, "Merge Selected", "Select at least two rows to merge.")
            return
        first = rows[0]
        for column in (4, 5, 6):
            merged = " ".join(
                (self.table.item(row, column).text().strip() if self.table.item(row, column) else "")
                for row in rows
            ).strip()
            self.table.setItem(first, column, QTableWidgetItem(merged))
        for row in rows[1:]:
            item = self.table.item(row, 1)
            if item is not None:
                item.setCheckState(Qt.CheckState.Unchecked)
            note = self.table.item(row, 7).text() if self.table.item(row, 7) else ""
            self.table.setItem(row, 7, QTableWidgetItem((note + " merged into previous").strip()))

    def _preview_selected(self) -> None:
        rows = self._selected_rows()
        if not rows:
            QMessageBox.information(self, "Preview Segment", "Select one segment to preview.")
            return
        text_item = self.table.item(rows[0], 6)
        text = text_item.text().strip() if text_item else ""
        if not text:
            QMessageBox.warning(self, "Preview Segment", "The selected segment has no Khmer text.")
            return
        if self.preview_callback:
            self.preview_callback(text)

    def _save(self) -> None:
        segments = self.payload.get("segments", [])
        for row, segment in enumerate(segments):
            enabled_item = self.table.item(row, 1)
            source_item = self.table.item(row, 4)
            raw_item = self.table.item(row, 5)
            improved_item = self.table.item(row, 6)
            notes_item = self.table.item(row, 7)
            segment["enabled"] = enabled_item.checkState() == Qt.CheckState.Checked if enabled_item else True
            segment["source_text"] = source_item.text().strip() if source_item else ""
            segment["raw_khmer_text"] = raw_item.text().strip() if raw_item else ""
            segment["user_edited_text"] = improved_item.text().strip() if improved_item else ""
            segment["review_notes"] = notes_item.text().strip() if notes_item else ""
        self.review_path.write_text(json.dumps(self.payload, indent=2, ensure_ascii=False), encoding="utf-8")
        QMessageBox.information(self, "Transcript Review", "Review JSON saved.")
