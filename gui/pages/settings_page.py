from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from config.models import WHISPER_MODELS


class SettingsPage(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        header = QLabel("Settings")
        header.setObjectName("PageHeader")
        layout.addWidget(header)

        desc = QLabel("General pipeline settings — device, model, caching, and presets.")
        desc.setObjectName("PageDesc")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        layout.addWidget(self._section_title("License"))
        license_form = QFormLayout()
        license_form.setSpacing(12)
        self.license_key = QLineEdit()
        self.license_key.setPlaceholderText("KVD-XXXXXX-XXXXXX-XXXXXX-XXXXXX")
        license_form.addRow("Subscription key", self.license_key)
        self.activate_license_button = QPushButton("Activate on This Device")
        self.activate_license_button.setObjectName("SecondaryButton")
        self.license_status = QLabel("$11.99/month · $59.99/6 months · $99.99/year")
        self.license_status.setObjectName("HintLabel")
        self.license_status.setWordWrap(True)
        license_form.addRow("", self.activate_license_button)
        license_form.addRow("", self.license_status)
        self.open_purchase_wizard_button = QPushButton("Buy a New License...")
        self.open_purchase_wizard_button.setObjectName("SecondaryButton")
        license_form.addRow("", self.open_purchase_wizard_button)
        layout.addLayout(license_form)

        layout.addWidget(self._section_title("AI Keys"))
        ai_form = QFormLayout()
        ai_form.setSpacing(12)
        self.gemini_api_key = QLineEdit()
        self.gemini_api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.gemini_api_key.setPlaceholderText("Your own Gemini API key")
        ai_form.addRow("Gemini API key", self.gemini_api_key)
        self.test_gemini_button = QPushButton("Save & Test Gemini Key")
        self.test_gemini_button.setObjectName("SecondaryButton")
        self.gemini_key_status = QLabel("A valid Gemini key is required for Gemini features.")
        self.gemini_key_status.setObjectName("HintLabel")
        self.gemini_key_status.setWordWrap(True)
        ai_form.addRow("", self.test_gemini_button)
        ai_form.addRow("", self.gemini_key_status)
        layout.addLayout(ai_form)

        layout.addWidget(self._section_title("Pipeline"))
        pipeline_form = QFormLayout()
        pipeline_form.setSpacing(12)
        self.preset = QComboBox()
        self.preset.addItem("Best Quality", "best")
        self.preset.addItem("Balanced (recommended)", "balanced")
        self.preset.addItem("Fast Draft", "fast")
        pipeline_form.addRow("Preset", self.preset)

        self.whisper_model = QComboBox()
        for model in WHISPER_MODELS:
            self.whisper_model.addItem(model, model)
        self.whisper_model.setCurrentText("medium")
        pipeline_form.addRow("Whisper model", self.whisper_model)

        self.device = QComboBox()
        self.device.addItem("Automatic (GPU if available)", "auto")
        self.device.addItem("CUDA (GPU)", "cuda")
        self.device.addItem("CPU", "cpu")
        pipeline_form.addRow("Device", self.device)

        from config.device import resolve_compute_device
        _, device_message = resolve_compute_device("auto")
        self.device_status = QLabel(device_message)
        self.device_status.setObjectName("HintLabel")
        self.device_status.setWordWrap(True)
        pipeline_form.addRow("", self.device_status)

        self.alignment_mode = QComboBox()
        self.alignment_mode.addItem("Natural (recommended)", "natural")
        self.alignment_mode.addItem("Strict", "strict")
        self.alignment_mode.addItem("Energetic", "energetic")
        pipeline_form.addRow("Alignment", self.alignment_mode)

        self.persistent_cache_check = QCheckBox("Use persistent cache")
        self.persistent_cache_check.setChecked(True)
        pipeline_form.addRow(self.persistent_cache_check)

        self.keep_temp_check = QCheckBox("Keep temporary files")
        pipeline_form.addRow(self.keep_temp_check)
        layout.addLayout(pipeline_form)

        layout.addWidget(self._section_title("Import"))
        import_form = QFormLayout()
        import_form.setSpacing(12)
        cookies_row_widget = QWidget()
        cookies_row = QHBoxLayout(cookies_row_widget)
        cookies_row.setContentsMargins(0, 0, 0, 0)
        self.url_import_cookies_file = QLineEdit()
        self.url_import_cookies_file.setPlaceholderText("Optional Netscape cookies.txt file")
        cookies_browse = QPushButton("Browse")
        cookies_browse.setObjectName("SecondaryButton")
        cookies_browse.clicked.connect(self._browse_url_import_cookies_file)
        cookies_row.addWidget(self.url_import_cookies_file, 1)
        cookies_row.addWidget(cookies_browse)
        import_form.addRow("URL import cookies file", cookies_row_widget)
        layout.addLayout(import_form)

        layout.addWidget(self._section_title("App"))
        app_form = QFormLayout()
        app_form.setSpacing(12)
        self.setup_check_button = QPushButton("Check Setup (ffmpeg, models, etc.)")
        self.setup_check_button.setObjectName("SecondaryButton")
        app_form.addRow(self.setup_check_button)

        self.theme_button = QPushButton("Toggle Theme")
        self.theme_button.setObjectName("SecondaryButton")
        app_form.addRow(self.theme_button)
        layout.addLayout(app_form)

        layout.addStretch(1)

    @staticmethod
    def _section_title(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("SectionTitle")
        return label

    def save_state(self) -> dict:
        return {
            "whisper_model": self.whisper_model.currentText(),
            "device": self.device.currentData() or "auto",
            "preset": self.preset.currentText(),
            "alignment_mode": self.alignment_mode.currentData(),
            "keep_temp": self.keep_temp_check.isChecked(),
            "persistent_cache": self.persistent_cache_check.isChecked(),
            "url_import_cookies_file": self.url_import_cookies_file.text().strip(),
        }

    def load_state(self, config: dict) -> None:
        self.whisper_model.setCurrentText(config.get("whisper_model", "medium"))
        device_val = config.get("device", "auto")
        idx = self.device.findData(device_val)
        if idx < 0:
            idx = self.device.findText(device_val)
        if idx >= 0:
            self.device.setCurrentIndex(idx)
        self.preset.setCurrentText(config.get("preset", "Balanced (recommended)"))
        alignment_val = config.get("alignment_mode", "natural")
        alignment_index = self.alignment_mode.findData(alignment_val)
        if alignment_index >= 0:
            self.alignment_mode.setCurrentIndex(alignment_index)
        self.keep_temp_check.setChecked(config.get("keep_temp", False))
        self.persistent_cache_check.setChecked(config.get("persistent_cache", True))
        self.url_import_cookies_file.setText(config.get("url_import_cookies_file", ""))

    def _browse_url_import_cookies_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select cookies file",
            str(Path.home()),
            "Text files (*.txt);;All files (*)",
        )
        if path:
            self.url_import_cookies_file.setText(path)
