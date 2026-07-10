from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from gui.components.preview_widget import OverlayPreviewWidget
from modules.overlay_preview import render_card_preview
from modules.sponsor_card import SponsorCardConfig

FILE_DIALOG_OPTIONS = QFileDialog.Option(0)


class SponsorCardEditor(QWidget):
    """Widget to configure a single sponsor card with preview."""

    remove_requested = pyqtSignal(object)
    config_changed = pyqtSignal()

    def __init__(self, index: int = 0, parent=None) -> None:
        super().__init__(parent)
        self._index = index

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        header_row = QHBoxLayout()
        self._title = QLabel(f"Card #{index + 1}")
        self._title.setStyleSheet("font-weight: bold;")
        header_row.addWidget(self._title)
        header_row.addStretch()

        remove_btn = QPushButton("Remove")
        remove_btn.setObjectName("DangerButton")
        remove_btn.clicked.connect(lambda: self.remove_requested.emit(self))
        header_row.addWidget(remove_btn)
        layout.addLayout(header_row)

        form = QFormLayout()
        form.setSpacing(8)

        self.card_type = QComboBox()
        self.card_type.addItem("Text", "text")
        self.card_type.addItem("Image", "image")
        self.card_type.currentIndexChanged.connect(self._on_type_changed)
        form.addRow("Type", self.card_type)

        self.position = QComboBox()
        self.position.addItem("Front (before video)", "front")
        self.position.addItem("Center (middle)", "center")
        self.position.addItem("End (after video)", "end")
        form.addRow("Position", self.position)

        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText("Sponsor text — supports ខ្មែរ and English")
        form.addRow("Text", self.text_input)

        img_row = QHBoxLayout()
        self.image_path = QLineEdit()
        self.image_path.setPlaceholderText("Path to sponsor image")
        img_browse = QPushButton("Browse")
        img_browse.setObjectName("CompactButton")
        img_browse.clicked.connect(self._browse_image)
        img_row.addWidget(self.image_path, 1)
        img_row.addWidget(img_browse)
        form.addRow("Image", img_row)

        self.duration = QDoubleSpinBox()
        self.duration.setRange(1.0, 15.0)
        self.duration.setValue(3.0)
        self.duration.setSingleStep(0.5)
        self.duration.setSuffix(" s")
        form.addRow("Duration", self.duration)

        self.bg_color = QComboBox()
        self.bg_color.addItem("Black", "black")
        self.bg_color.addItem("White", "white")
        self.bg_color.addItem("Blue", "0x1a237e")
        self.bg_color.addItem("Red", "0xb71c1c")
        self.bg_color.addItem("Green", "0x1b5e20")
        form.addRow("Background", self.bg_color)

        self.text_color = QComboBox()
        self.text_color.addItem("White", "white")
        self.text_color.addItem("Black", "black")
        self.text_color.addItem("Yellow", "yellow")
        form.addRow("Text color", self.text_color)

        self.font_size = QSpinBox()
        self.font_size.setRange(0, 120)
        self.font_size.setValue(0)
        self.font_size.setSpecialValueText("Auto")
        form.addRow("Font size", self.font_size)

        layout.addLayout(form)

        self._preview = OverlayPreviewWidget(320, 180, self)
        self._preview.set_render_function(self._render_preview)
        layout.addWidget(self._preview)

        self.setStyleSheet(
            "SponsorCardEditor { border: 1px solid #333; border-radius: 8px; padding: 8px; }"
        )

    def _on_type_changed(self, _index: int) -> None:
        is_text = self.card_type.currentData() == "text"
        self.text_input.setEnabled(is_text)
        self.bg_color.setEnabled(is_text)
        self.text_color.setEnabled(is_text)
        self.font_size.setEnabled(is_text)
        self.image_path.setEnabled(not is_text)

    def _browse_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select sponsor image", "",
            "Images (*.png *.jpg *.jpeg *.svg *.webp);;All files (*)",
            options=FILE_DIALOG_OPTIONS,
        )
        if path:
            self.image_path.setText(path)

    def get_config(self) -> SponsorCardConfig:
        return SponsorCardConfig(
            card_type=self.card_type.currentData(),
            position=self.position.currentData(),
            text=self.text_input.text(),
            image_path=self.image_path.text(),
            duration=self.duration.value(),
            bg_color=self.bg_color.currentData(),
            text_color=self.text_color.currentData(),
            font_size=self.font_size.value(),
        )

    def set_config(self, config: SponsorCardConfig) -> None:
        idx = self.card_type.findData(config.card_type)
        if idx >= 0:
            self.card_type.setCurrentIndex(idx)

        idx = self.position.findData(config.position)
        if idx >= 0:
            self.position.setCurrentIndex(idx)

        self.text_input.setText(config.text)
        self.image_path.setText(config.image_path)
        self.duration.setValue(config.duration)

        idx = self.bg_color.findData(config.bg_color)
        if idx >= 0:
            self.bg_color.setCurrentIndex(idx)

        idx = self.text_color.findData(config.text_color)
        if idx >= 0:
            self.text_color.setCurrentIndex(idx)

        self.font_size.setValue(config.font_size)
        self._on_type_changed(0)

    def _render_preview(self) -> Path:
        config = self.get_config()
        output = self._preview.get_temp_path(f"card_{self._index}.png")
        return render_card_preview(config, output)

    def set_index(self, index: int) -> None:
        self._index = index
        self._title.setText(f"Card #{index + 1}")
