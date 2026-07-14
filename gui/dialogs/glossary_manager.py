from __future__ import annotations

from pathlib import Path
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)


class GlossaryManagerDialog(QDialog):
    COLUMNS = ["Source Word / Phrase", "Khmer Translation"]

    def __init__(self, glossary_path: Path, parent=None) -> None:
        super().__init__(parent)
        self.glossary_path = glossary_path
        self.setWindowTitle(f"Glossary Manager - {self.glossary_path.name}")
        self.resize(600, 450)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        desc = QLabel(
            "Add terms here to keep their translation consistent. "
            "For example: <b>Google</b> = <b>ហ្គូហ្គល</b>"
        )
        desc.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(desc)

        self.table = QTableWidget()
        self.table.setColumnCount(len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table, 1)

        # Toolbar under table
        toolbar = QHBoxLayout()
        add_btn = QPushButton("➕ Add Term")
        add_btn.setObjectName("SecondaryButton")
        add_btn.clicked.connect(self._add_row)
        
        del_btn = QPushButton("🗑️ Delete Selected")
        del_btn.setObjectName("SecondaryButton")
        del_btn.clicked.connect(self._delete_selected)
        
        toolbar.addWidget(add_btn)
        toolbar.addWidget(del_btn)
        toolbar.addStretch()
        layout.addLayout(toolbar)

        # Action Buttons (Save/Cancel)
        actions = QHBoxLayout()
        save_btn = QPushButton("💾 Save Changes")
        save_btn.setObjectName("PrimaryButton")
        save_btn.clicked.connect(self._save)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        
        actions.addStretch()
        actions.addWidget(cancel_btn)
        actions.addWidget(save_btn)
        layout.addLayout(actions)

        self._load_glossary_file()

    def _load_glossary_file(self) -> None:
        self.table.setRowCount(0)
        if not self.glossary_path.exists():
            return

        try:
            terms = []
            for raw_line in self.glossary_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    source, target = line.split("=", 1)
                elif "," in line:
                    source, target = line.split(",", 1)
                else:
                    continue
                source = source.strip()
                target = target.strip()
                if source and target:
                    terms.append((source, target))

            for source, target in terms:
                row = self.table.rowCount()
                self.table.insertRow(row)
                self.table.setItem(row, 0, QTableWidgetItem(source))
                self.table.setItem(row, 1, QTableWidgetItem(target))
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", f"Failed to load glossary file: {exc}")

    def _add_row(self) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(""))
        self.table.setItem(row, 1, QTableWidgetItem(""))
        self.table.setCurrentCell(row, 0)
        item = self.table.item(row, 0)
        if item is not None:
            self.table.editItem(item)

    def _delete_selected(self) -> None:
        current_row = self.table.currentRow()
        if current_row < 0:
            QMessageBox.warning(self, "No Selection", "Please select a row to delete.")
            return
        self.table.removeRow(current_row)

    def _save(self) -> None:
        lines = [
            "# Auto-generated translation glossary",
            "# Format: source = target",
            "",
        ]
        for r in range(self.table.rowCount()):
            src_item = self.table.item(r, 0)
            tgt_item = self.table.item(r, 1)
            
            src = src_item.text().strip() if src_item else ""
            tgt = tgt_item.text().strip() if tgt_item else ""
            
            if src and tgt:
                lines.append(f"{src} = {tgt}")

        try:
            self.glossary_path.parent.mkdir(parents=True, exist_ok=True)
            self.glossary_path.write_text("\n".join(lines), encoding="utf-8")
            self.accept()
        except Exception as exc:
            QMessageBox.critical(self, "Save Error", f"Failed to save glossary file: {exc}")
