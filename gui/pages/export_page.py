from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QFont, QPen
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

FILE_DIALOG_OPTIONS = QFileDialog.Option(0)

POSITIONS = ["top_left", "top_right", "center", "bottom_left", "bottom_right"]


class OverlayPositionPicker(QWidget):
    """Visual 16:9 preview where user clicks to choose overlay position."""

    position_changed = pyqtSignal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._selected = "bottom_right"
        self.setFixedSize(220, 124)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    @property
    def selected(self) -> str:
        return self._selected

    @selected.setter
    def selected(self, value: str) -> None:
        if value in POSITIONS:
            self._selected = value
            self.update()
            self.position_changed.emit(value)

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        # Video frame background
        p.setBrush(QColor("#1e1e2e"))
        p.setPen(QPen(QColor("#444"), 1))
        p.drawRoundedRect(0, 0, w - 1, h - 1, 6, 6)

        # Draw grid lines (light)
        p.setPen(QPen(QColor("#333"), 1, Qt.PenStyle.DotLine))
        p.drawLine(w // 2, 0, w // 2, h)
        p.drawLine(0, h // 2, w, h // 2)

        # Position zones
        margin = 12
        dot_size = 28
        zones = {
            "top_left": (margin, margin),
            "top_right": (w - margin - dot_size, margin),
            "center": ((w - dot_size) // 2, (h - dot_size) // 2),
            "bottom_left": (margin, h - margin - dot_size),
            "bottom_right": (w - margin - dot_size, h - margin - dot_size),
        }

        font = QFont()
        font.setPixelSize(10)
        p.setFont(font)

        for pos, (x, y) in zones.items():
            is_active = pos == self._selected
            if is_active:
                p.setBrush(QColor("#7c3aed"))
                p.setPen(QPen(QColor("#a78bfa"), 2))
            else:
                p.setBrush(QColor("#2a2a3e"))
                p.setPen(QPen(QColor("#555"), 1))
            p.drawRoundedRect(x, y, dot_size, dot_size, 4, 4)

            # Label
            label = pos.replace("_", "\n").title() if pos != "center" else "C"
            p.setPen(QColor("#ddd") if is_active else QColor("#888"))
            p.drawText(x, y, dot_size, dot_size, Qt.AlignmentFlag.AlignCenter, label)

        p.end()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        x, y = event.position().x(), event.position().y()
        w, h = self.width(), self.height()
        # Determine which zone was clicked
        col = "left" if x < w / 3 else ("right" if x > 2 * w / 3 else "center")
        row = "top" if y < h / 3 else ("bottom" if y > 2 * h / 3 else "center")

        if row == "center" and col == "center":
            self.selected = "center"
        elif row == "center":
            self.selected = f"bottom_{col}"
        elif col == "center":
            self.selected = f"{row}_right"
        else:
            self.selected = f"{row}_{col}"


class ExportPage(QWidget):
    save_defaults_requested = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        header_row = QHBoxLayout()
        header = QLabel("Export Options")
        header.setObjectName("PageHeader")
        header_row.addWidget(header)
        header_row.addStretch(1)

        self.save_defaults_button = QPushButton("💾 Save as Default")
        self.save_defaults_button.setObjectName("SecondaryButton")
        self.save_defaults_button.setToolTip("Save current export settings as your default for new sessions")
        self.save_defaults_button.clicked.connect(self._on_save_defaults)
        header_row.addWidget(self.save_defaults_button)
        layout.addLayout(header_row)

        desc = QLabel("Choose which files to export alongside the dubbed video.")
        desc.setObjectName("PageDesc")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        form = QFormLayout()
        form.setSpacing(10)

        self.save_review_json_check = QCheckBox("Save review JSON")
        self.save_review_json_check.setChecked(False)
        form.addRow(self.save_review_json_check)

        self.edit_review_button = QPushButton("Edit Review JSON")
        self.edit_review_button.setObjectName("CompactButton")
        form.addRow("", self.edit_review_button)

        self.export_audio_check = QCheckBox("Export dubbed audio (WAV)")
        self.export_audio_check.setChecked(True)
        form.addRow(self.export_audio_check)

        self.export_original_check = QCheckBox("Export original transcript")
        self.export_original_check.setChecked(True)
        form.addRow(self.export_original_check)

        self.export_raw_khmer_check = QCheckBox("Export raw Khmer text")
        self.export_raw_khmer_check.setChecked(True)
        form.addRow(self.export_raw_khmer_check)

        self.export_improved_khmer_check = QCheckBox("Export improved Khmer text")
        self.export_improved_khmer_check.setChecked(True)
        form.addRow(self.export_improved_khmer_check)

        self.export_srt_check = QCheckBox("Export subtitles (SRT)")
        self.export_srt_check.setChecked(True)
        form.addRow(self.export_srt_check)

        self.export_quality_check = QCheckBox("Export quality report")
        self.export_quality_check.setChecked(True)
        form.addRow(self.export_quality_check)

        sep = QLabel("")
        form.addRow(sep)

        subtitle_header = QLabel("Subtitle Burn-in")
        subtitle_header.setObjectName("SectionTitle")
        form.addRow(subtitle_header)

        self.burn_subtitles_check = QCheckBox("Burn subtitles into video")
        form.addRow(self.burn_subtitles_check)

        self.subtitle_language = QComboBox()
        self.subtitle_language.addItem("Khmer", "khmer")
        self.subtitle_language.addItem("Source language", "source")
        self.subtitle_language.addItem("Both", "both")
        form.addRow("Subtitle language", self.subtitle_language)

        self.subtitle_font_size = QSpinBox()
        self.subtitle_font_size.setRange(12, 60)
        self.subtitle_font_size.setValue(24)
        form.addRow("Font size", self.subtitle_font_size)

        sep2 = QLabel("")
        form.addRow(sep2)

        overlay_header = QLabel("Video Overlay")
        overlay_header.setObjectName("SectionTitle")
        form.addRow(overlay_header)

        self.overlay_text = QLineEdit()
        self.overlay_text.setPlaceholderText("Optional text overlay")
        form.addRow("Overlay text", self.overlay_text)

        img_row = QHBoxLayout()
        self.overlay_image_path = QLineEdit()
        self.overlay_image_path.setPlaceholderText("Optional image overlay")
        img_browse = QPushButton("Browse")
        img_browse.setObjectName("CompactButton")
        img_browse.clicked.connect(self._browse_overlay_image)
        img_row.addWidget(self.overlay_image_path, 1)
        img_row.addWidget(img_browse)
        form.addRow("Overlay image", img_row)

        self.overlay_text_position_picker = OverlayPositionPicker()
        text_pos_row, self._text_position_label = self._build_position_row(
            self.overlay_text_position_picker,
            "Click to set text position:",
        )
        form.addRow("Text position", text_pos_row)

        self.overlay_image_position_picker = OverlayPositionPicker()
        image_pos_row, self._image_position_label = self._build_position_row(
            self.overlay_image_position_picker,
            "Click to set image position:",
        )
        form.addRow("Image position", image_pos_row)

        self.overlay_opacity = QDoubleSpinBox()
        self.overlay_opacity.setRange(0.0, 1.0)
        self.overlay_opacity.setValue(0.7)
        self.overlay_opacity.setSingleStep(0.1)
        form.addRow("Opacity", self.overlay_opacity)

        sep3 = QLabel("")
        form.addRow(sep3)

        endscreen_header = QLabel("End Screen (3s Card)")
        endscreen_header.setObjectName("SectionTitle")
        form.addRow(endscreen_header)

        self.end_screen_enabled = QCheckBox("Add end screen to video")
        form.addRow(self.end_screen_enabled)

        self.end_screen_text = QLineEdit()
        self.end_screen_text.setPlaceholderText("Text to show (e.g. Subscribe! ចុចSubscribe)")
        form.addRow("End text", self.end_screen_text)

        end_img_row = QHBoxLayout()
        self.end_screen_image_path = QLineEdit()
        self.end_screen_image_path.setPlaceholderText("Or use an image instead of text")
        end_img_browse = QPushButton("Browse")
        end_img_browse.setObjectName("CompactButton")
        end_img_browse.clicked.connect(self._browse_end_image)
        end_img_row.addWidget(self.end_screen_image_path, 1)
        end_img_row.addWidget(end_img_browse)
        form.addRow("End image", end_img_row)

        self.end_screen_bg_color = QComboBox()
        self.end_screen_bg_color.addItem("Black", "black")
        self.end_screen_bg_color.addItem("White", "white")
        form.addRow("Background", self.end_screen_bg_color)

        self.end_screen_duration = QDoubleSpinBox()
        self.end_screen_duration.setRange(1.0, 10.0)
        self.end_screen_duration.setValue(3.0)
        self.end_screen_duration.setSingleStep(0.5)
        self.end_screen_duration.setSuffix(" s")
        form.addRow("Duration", self.end_screen_duration)

        layout.addLayout(form)
        layout.addStretch(1)

    def save_state(self) -> dict:
        return {
            "save_review_json": self.save_review_json_check.isChecked(),
            "export_audio": self.export_audio_check.isChecked(),
            "export_original": self.export_original_check.isChecked(),
            "export_raw_khmer": self.export_raw_khmer_check.isChecked(),
            "export_improved_khmer": self.export_improved_khmer_check.isChecked(),
            "export_srt": self.export_srt_check.isChecked(),
            "export_quality": self.export_quality_check.isChecked(),
            "burn_subtitles": self.burn_subtitles_check.isChecked(),
            "subtitle_language": self.subtitle_language.currentText(),
            "subtitle_font_size": self.subtitle_font_size.value(),
            "overlay_text": self.overlay_text.text().strip(),
            "overlay_image_path": self.overlay_image_path.text().strip(),
            "overlay_position": self.overlay_text_position_picker.selected,
            "overlay_text_position": self.overlay_text_position_picker.selected,
            "overlay_image_position": self.overlay_image_position_picker.selected,
            "overlay_opacity": self.overlay_opacity.value(),
            "end_screen_enabled": self.end_screen_enabled.isChecked(),
            "end_screen_text": self.end_screen_text.text().strip(),
            "end_screen_image_path": self.end_screen_image_path.text().strip(),
            "end_screen_bg_color": self.end_screen_bg_color.currentData(),
            "end_screen_duration": self.end_screen_duration.value(),
        }

    def load_state(self, config: dict) -> None:
        self.save_review_json_check.setChecked(config.get("save_review_json", False))
        self.export_audio_check.setChecked(config.get("export_audio", True))
        self.export_original_check.setChecked(config.get("export_original", True))
        self.export_raw_khmer_check.setChecked(config.get("export_raw_khmer", True))
        self.export_improved_khmer_check.setChecked(config.get("export_improved_khmer", True))
        self.export_srt_check.setChecked(config.get("export_srt", True))
        self.export_quality_check.setChecked(config.get("export_quality", True))
        self.burn_subtitles_check.setChecked(config.get("burn_subtitles", False))
        self.subtitle_language.setCurrentText(config.get("subtitle_language", ""))
        self.subtitle_font_size.setValue(config.get("subtitle_font_size", 24))
        self.overlay_text.setText(config.get("overlay_text", ""))
        self.overlay_image_path.setText(config.get("overlay_image_path", ""))
        legacy_position = config.get("overlay_position", "bottom_right")
        self.overlay_text_position_picker.selected = config.get("overlay_text_position", legacy_position)
        self.overlay_image_position_picker.selected = config.get("overlay_image_position", legacy_position)
        self.overlay_opacity.setValue(config.get("overlay_opacity", 0.7))
        self.end_screen_enabled.setChecked(config.get("end_screen_enabled", False))
        self.end_screen_text.setText(config.get("end_screen_text", ""))
        self.end_screen_image_path.setText(config.get("end_screen_image_path", ""))
        bg_idx = self.end_screen_bg_color.findData(config.get("end_screen_bg_color", "black"))
        if bg_idx >= 0:
            self.end_screen_bg_color.setCurrentIndex(bg_idx)
        self.end_screen_duration.setValue(config.get("end_screen_duration", 3.0))

    def _on_save_defaults(self) -> None:
        self.save_defaults_requested.emit()
        QMessageBox.information(self, "Defaults Saved", "Your export settings have been saved as the default.")

    def _build_position_row(self, picker: OverlayPositionPicker, label_text: str) -> tuple[QHBoxLayout, QLabel]:
        position_label = QLabel(picker.selected.replace("_", " ").title())
        position_label.setObjectName("HintLabel")
        picker.position_changed.connect(
            lambda pos, target=position_label: target.setText(pos.replace("_", " ").title())
        )
        row = QHBoxLayout()
        row.addWidget(picker)
        col = QVBoxLayout()
        col.addWidget(QLabel(label_text))
        col.addWidget(position_label)
        col.addStretch()
        row.addLayout(col)
        return row, position_label

    def _browse_overlay_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select overlay image", "",
            "Images (*.png *.jpg *.jpeg *.svg);;All files (*)",
            options=FILE_DIALOG_OPTIONS,
        )
        if path:
            self.overlay_image_path.setText(path)

    def _browse_end_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select end screen image", "",
            "Images (*.png *.jpg *.jpeg *.svg);;All files (*)",
            options=FILE_DIALOG_OPTIONS,
        )
        if path:
            self.end_screen_image_path.setText(path)
