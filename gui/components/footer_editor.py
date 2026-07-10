from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from gui.components.preview_widget import OverlayPreviewWidget
from modules.footer_overlay import FooterOverlayConfig
from modules.overlay_preview import render_footer_preview


class FooterOverlayEditor(QWidget):
    """Widget to configure footer/banner overlay with preview."""

    config_changed = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._video_path: Path | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        form = QFormLayout()
        form.setSpacing(8)

        self.enabled_check = QCheckBox("Enable footer banner")
        self.enabled_check.toggled.connect(self._on_enabled_changed)
        form.addRow(self.enabled_check)

        self.style_combo = QComboBox()
        self.style_combo.addItem("Fixed (static)", "fixed")
        self.style_combo.addItem("Marquee (scrolling)", "marquee")
        self.style_combo.addItem("Circular (rotation)", "circular")
        self.style_combo.currentIndexChanged.connect(self._on_style_changed)
        form.addRow("Style", self.style_combo)

        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText("Banner text — supports ខ្មែរ and English")
        form.addRow("Text", self.text_input)

        # Circular rotation text list
        self._texts_label = QLabel("Rotation texts:")
        form.addRow(self._texts_label)

        self.texts_list = QListWidget()
        self.texts_list.setMaximumHeight(100)
        form.addRow(self.texts_list)

        texts_btn_row = QHBoxLayout()
        self._add_text_btn = QPushButton("+ Add")
        self._add_text_btn.setObjectName("CompactButton")
        self._add_text_btn.clicked.connect(self._add_rotation_text)
        self._remove_text_btn = QPushButton("- Remove")
        self._remove_text_btn.setObjectName("CompactButton")
        self._remove_text_btn.clicked.connect(self._remove_rotation_text)
        texts_btn_row.addWidget(self._add_text_btn)
        texts_btn_row.addWidget(self._remove_text_btn)
        texts_btn_row.addStretch()
        form.addRow("", texts_btn_row)

        self.position_combo = QComboBox()
        self.position_combo.addItem("Bottom", "bottom")
        self.position_combo.addItem("Top", "top")
        form.addRow("Position", self.position_combo)

        self.bg_color = QComboBox()
        self.bg_color.addItem("Transparent (no bg)", "transparent")
        self.bg_color.addItem("Black", "black")
        self.bg_color.addItem("White", "white")
        self.bg_color.addItem("Blue", "0x1a237e")
        self.bg_color.addItem("Red", "0xb71c1c")
        self.bg_color.addItem("Green", "0x1b5e20")
        self.bg_color.setEditable(True)
        self.bg_color.lineEdit().setPlaceholderText("Or type hex e.g. 0xFF5500")
        form.addRow("Background", self.bg_color)

        self.text_color = QComboBox()
        self.text_color.addItem("White", "white")
        self.text_color.addItem("Black", "black")
        self.text_color.addItem("Yellow", "yellow")
        self.text_color.addItem("Red", "red")
        self.text_color.addItem("Cyan", "cyan")
        self.text_color.setEditable(True)
        self.text_color.lineEdit().setPlaceholderText("Or type hex e.g. 0xFF5500")
        form.addRow("Text color", self.text_color)

        self.opacity = QDoubleSpinBox()
        self.opacity.setRange(0.0, 1.0)
        self.opacity.setValue(0.7)
        self.opacity.setSingleStep(0.1)
        form.addRow("BG Opacity", self.opacity)

        self.scroll_speed = QSpinBox()
        self.scroll_speed.setRange(50, 500)
        self.scroll_speed.setValue(150)
        self.scroll_speed.setSuffix(" px/s")
        form.addRow("Scroll speed", self.scroll_speed)

        self.rotation_interval = QDoubleSpinBox()
        self.rotation_interval.setRange(1.0, 30.0)
        self.rotation_interval.setValue(5.0)
        self.rotation_interval.setSingleStep(0.5)
        self.rotation_interval.setSuffix(" s")
        form.addRow("Rotation interval", self.rotation_interval)

        self.font_size = QSpinBox()
        self.font_size.setRange(0, 96)
        self.font_size.setValue(0)
        self.font_size.setSpecialValueText("Auto")
        form.addRow("Font size", self.font_size)

        layout.addLayout(form)

        self._preview = OverlayPreviewWidget(400, 225, self)
        self._preview.set_render_function(self._render_preview)
        layout.addWidget(self._preview)

        self._on_style_changed(0)
        self._on_enabled_changed(False)

    def set_video_path(self, path: Path | None) -> None:
        self._video_path = path

    def _on_enabled_changed(self, enabled: bool) -> None:
        self.style_combo.setEnabled(enabled)
        self.text_input.setEnabled(enabled)
        self.texts_list.setEnabled(enabled)
        self._add_text_btn.setEnabled(enabled)
        self._remove_text_btn.setEnabled(enabled)
        self.position_combo.setEnabled(enabled)
        self.bg_color.setEnabled(enabled)
        self.text_color.setEnabled(enabled)
        self.opacity.setEnabled(enabled)
        self.scroll_speed.setEnabled(enabled)
        self.rotation_interval.setEnabled(enabled)
        self.font_size.setEnabled(enabled)

    def _on_style_changed(self, _index: int) -> None:
        style = self.style_combo.currentData()
        is_circular = style == "circular"
        is_marquee = style == "marquee"

        self.text_input.setVisible(not is_circular)
        self._texts_label.setVisible(is_circular)
        self.texts_list.setVisible(is_circular)
        self._add_text_btn.setVisible(is_circular)
        self._remove_text_btn.setVisible(is_circular)
        self.scroll_speed.setVisible(is_marquee)
        self.rotation_interval.setVisible(is_circular)

    def _add_rotation_text(self) -> None:
        text = self.text_input.text().strip()
        if not text:
            text = f"Sponsor #{self.texts_list.count() + 1}"
        self.texts_list.addItem(text)

    def _remove_rotation_text(self) -> None:
        row = self.texts_list.currentRow()
        if row >= 0:
            self.texts_list.takeItem(row)

    def get_config(self) -> FooterOverlayConfig:
        texts = [self.texts_list.item(i).text() for i in range(self.texts_list.count())]
        bg = self.bg_color.currentData() or self.bg_color.currentText().strip() or "black"
        tc = self.text_color.currentData() or self.text_color.currentText().strip() or "white"
        return FooterOverlayConfig(
            enabled=self.enabled_check.isChecked(),
            style=self.style_combo.currentData(),
            text=self.text_input.text(),
            texts=texts,
            position=self.position_combo.currentData(),
            bg_color=bg,
            text_color=tc,
            opacity=self.opacity.value(),
            scroll_speed=self.scroll_speed.value(),
            rotation_interval=self.rotation_interval.value(),
            font_size=self.font_size.value(),
        )

    def set_config(self, config: FooterOverlayConfig) -> None:
        self.enabled_check.setChecked(config.enabled)

        idx = self.style_combo.findData(config.style)
        if idx >= 0:
            self.style_combo.setCurrentIndex(idx)

        self.text_input.setText(config.text)

        self.texts_list.clear()
        for text in config.texts:
            self.texts_list.addItem(text)

        idx = self.position_combo.findData(config.position)
        if idx >= 0:
            self.position_combo.setCurrentIndex(idx)

        idx = self.bg_color.findData(config.bg_color)
        if idx >= 0:
            self.bg_color.setCurrentIndex(idx)
        else:
            self.bg_color.setCurrentText(config.bg_color)

        idx = self.text_color.findData(config.text_color)
        if idx >= 0:
            self.text_color.setCurrentIndex(idx)
        else:
            self.text_color.setCurrentText(config.text_color)

        self.opacity.setValue(config.opacity)
        self.scroll_speed.setValue(config.scroll_speed)
        self.rotation_interval.setValue(config.rotation_interval)
        self.font_size.setValue(config.font_size)

        self._on_enabled_changed(config.enabled)
        self._on_style_changed(0)

    def _render_preview(self) -> Path:
        config = self.get_config()
        output = self._preview.get_temp_path("footer_preview.png")
        return render_footer_preview(config, output, self._video_path)
