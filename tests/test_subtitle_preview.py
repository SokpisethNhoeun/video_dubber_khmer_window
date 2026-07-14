from __future__ import annotations

from PyQt6.QtWidgets import QApplication
from gui.pages.export_page import SubtitlePreviewWidget


def test_subtitle_preview_widget_style_updates(monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])

    widget = SubtitlePreviewWidget()
    assert widget._font_name == "Noto Sans Khmer"
    assert widget._font_size == 24
    assert widget._color == "white"
    assert widget._bg_opacity == 0.0

    widget.update_style("Kantumruy Pro", 30, "yellow", 0.5)

    assert widget._font_name == "Kantumruy Pro"
    assert widget._font_size == 30
    assert widget._color == "yellow"
    assert widget._bg_opacity == 0.5

    widget.close()
    assert app is not None
