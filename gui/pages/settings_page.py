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
from config import paths as model_paths


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

        license_row_widget = QWidget()
        license_row = QHBoxLayout(license_row_widget)
        license_row.setContentsMargins(0, 0, 0, 0)
        self.license_key = QLineEdit()
        self.license_key.setPlaceholderText("KVD-XXXXXX-XXXXXX-XXXXXX-XXXXXX")
        self.license_key.setReadOnly(True)
        self.change_license_key_button = QPushButton("Change Key")
        self.change_license_key_button.setObjectName("SecondaryButton")
        self.change_license_key_button.clicked.connect(self._toggle_license_key_edit)
        license_row.addWidget(self.license_key, 1)
        license_row.addWidget(self.change_license_key_button)

        license_form.addRow("Subscription key", license_row_widget)
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

        gemini_row_widget = QWidget()
        gemini_row = QHBoxLayout(gemini_row_widget)
        gemini_row.setContentsMargins(0, 0, 0, 0)
        self.gemini_api_key = QLineEdit()
        self.gemini_api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.gemini_api_key.setPlaceholderText("Your own Gemini API key")
        self.gemini_api_key.setReadOnly(True)
        self.change_gemini_key_button = QPushButton("Change Key")
        self.change_gemini_key_button.setObjectName("SecondaryButton")
        self.change_gemini_key_button.clicked.connect(self._toggle_gemini_key_edit)
        gemini_row.addWidget(self.gemini_api_key, 1)
        gemini_row.addWidget(self.change_gemini_key_button)

        ai_form.addRow("Gemini API key", gemini_row_widget)
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
        self.refresh_installed_models()
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
            "whisper_model": self.whisper_model.currentData() or "",
            "device": self.device.currentData() or "auto",
            "preset": self.preset.currentText(),
            "alignment_mode": self.alignment_mode.currentData(),
            "keep_temp": self.keep_temp_check.isChecked(),
            "persistent_cache": self.persistent_cache_check.isChecked(),
            "url_import_cookies_file": self.url_import_cookies_file.text().strip(),
        }

    def load_state(self, config: dict) -> None:
        whisper_index = self.whisper_model.findData(config.get("whisper_model", "medium"))
        if whisper_index >= 0:
            self.whisper_model.setCurrentIndex(whisper_index)
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

    def refresh_installed_models(self, preferred: str | None = None) -> None:
        current = preferred or self.whisper_model.currentData() or "medium"
        installed_models = model_paths.installed_whisper_models(WHISPER_MODELS)
        self.whisper_model.clear()
        for model in installed_models:
            self.whisper_model.addItem(model, model)
        self.whisper_model.setEnabled(bool(installed_models))
        if not installed_models:
            self.whisper_model.addItem("No model installed — open Downloads", "")
            return
        index = self.whisper_model.findData(current)
        self.whisper_model.setCurrentIndex(index if index >= 0 else 0)

    def _browse_url_import_cookies_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select cookies file",
            str(Path.home()),
            "Text files (*.txt);;All files (*)",
        )
        if path:
            self.url_import_cookies_file.setText(path)

    def _toggle_license_key_edit(self) -> None:
        if self.license_key.isReadOnly():
            self.license_key.setReadOnly(False)
            self.change_license_key_button.setText("Lock Key")
            self.license_key.setFocus()
            self.license_key.selectAll()
        else:
            self.license_key.setReadOnly(True)
            self.change_license_key_button.setText("Change Key")

    def _toggle_gemini_key_edit(self) -> None:
        if self.gemini_api_key.isReadOnly():
            self.gemini_api_key.setReadOnly(False)
            self.change_gemini_key_button.setText("Lock Key")
            self.gemini_api_key.setFocus()
            self.gemini_api_key.selectAll()
        else:
            self.gemini_api_key.setReadOnly(True)
            self.change_gemini_key_button.setText("Change Key")

    def update_license_display(self) -> None:
        from config.user_secrets import load_user_secrets
        sec = load_user_secrets()
        key = sec.get("LICENSE_KEY", "")
        self.license_key.setText(key)

        start = sec.get("LICENSE_ACTIVATED_AT", "")
        end = sec.get("LICENSE_EXPIRES_AT", "")

        if key:
            if start or end:
                from datetime import datetime
                def fmt(iso):
                    if not iso: return "N/A"
                    try: return datetime.fromisoformat(iso.replace("Z", "+00:00")).strftime("%Y-%m-%d")
                    except: return iso[:10]
                self.license_status.setText(f"Active Subscription Key: {fmt(start)} to {fmt(end)}")
            else:
                self.license_status.setText("License key saved. Click activate to verify dates.")
        else:
            self.license_status.setText("No active subscription key. $11.99/month · $59.99/6 months · $99.99/year")
