from __future__ import annotations

from pathlib import Path
from PyQt6.QtWidgets import QApplication

from gui.dialogs.glossary_manager import GlossaryManagerDialog


def test_glossary_manager_loads_and_saves_terms(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])

    glossary_file = tmp_path / "glossary.txt"
    glossary_file.write_text(
        "# Initial comments\n"
        "Google = ហ្គូហ្គល\n"
        "Apple, អេផល\n"
        "\n",
        encoding="utf-8"
    )

    dialog = GlossaryManagerDialog(glossary_file)

    # Verify loaded terms
    assert dialog.table.rowCount() == 2
    assert dialog.table.item(0, 0).text() == "Google"
    assert dialog.table.item(0, 1).text() == "ហ្គូហ្គល"
    assert dialog.table.item(1, 0).text() == "Apple"
    assert dialog.table.item(1, 1).text() == "អេផល"

    # Add a term
    dialog._add_row()
    assert dialog.table.rowCount() == 3
    dialog.table.item(2, 0).setText("Microsoft")
    dialog.table.item(2, 1).setText("ម៉ៃក្រូសូហ្វ")

    # Save
    dialog._save()

    # Verify saved content
    content = glossary_file.read_text(encoding="utf-8")
    assert "Google = ហ្គូហ្គល" in content
    assert "Apple = អេផល" in content
    assert "Microsoft = ម៉ៃក្រូសូហ្វ" in content

    # Delete selected
    dialog.table.setCurrentCell(1, 0)
    dialog._delete_selected()
    assert dialog.table.rowCount() == 2

    dialog._save()
    content_deleted = glossary_file.read_text(encoding="utf-8")
    assert "Apple = អេផល" not in content_deleted
    assert "Google = ហ្គូហ្គល" in content_deleted

    dialog.close()
    assert app is not None
