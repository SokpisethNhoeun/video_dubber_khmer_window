from __future__ import annotations

from PyQt6.QtWidgets import QApplication
from gui.pages.settings_page import SettingsPage
from config.user_secrets import save_user_secret


def test_settings_page_keys_readonly_and_toggles(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])

    monkeypatch.setattr("config.user_secrets.secrets_path", lambda: tmp_path / "secrets.json")

    page = SettingsPage()

    # 1. Assert default states (read-only)
    assert page.license_key.isReadOnly()
    assert page.gemini_api_key.isReadOnly()
    assert page.change_license_key_button.text() == "Change Key"
    assert page.change_gemini_key_button.text() == "Change Key"

    # 2. Toggle license key
    page._toggle_license_key_edit()
    assert not page.license_key.isReadOnly()
    assert page.change_license_key_button.text() == "Lock Key"

    page._toggle_license_key_edit()
    assert page.license_key.isReadOnly()
    assert page.change_license_key_button.text() == "Change Key"

    # 3. Toggle Gemini API key
    page._toggle_gemini_key_edit()
    assert not page.gemini_api_key.isReadOnly()
    assert page.change_gemini_key_button.text() == "Lock Key"

    page._toggle_gemini_key_edit()
    assert page.gemini_api_key.isReadOnly()
    assert page.change_gemini_key_button.text() == "Change Key"

    # 4. Test update_license_display
    save_user_secret("LICENSE_KEY", "KVD-TEST-KEY")
    save_user_secret("LICENSE_ACTIVATED_AT", "2026-01-01T00:00:00Z")
    save_user_secret("LICENSE_EXPIRES_AT", "2027-01-01T00:00:00Z")

    page.update_license_display()
    assert page.license_key.text() == "KVD-TEST-KEY"
    assert "2026-01-01" in page.license_status.text()
    assert "2027-01-01" in page.license_status.text()

    page.close()
    assert app is not None


def test_settings_lists_only_downloaded_whisper_models(monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr("config.paths.installed_whisper_models", lambda _models: ["tiny", "small"])

    page = SettingsPage()

    assert [page.whisper_model.itemData(i) for i in range(page.whisper_model.count())] == [
        "tiny", "small"
    ]
    assert page.whisper_model.currentData() == "tiny"
    monkeypatch.setattr("config.paths.installed_whisper_models", lambda _models: ["tiny", "small", "medium"])
    page.refresh_installed_models("medium")
    assert page.whisper_model.currentData() == "medium"
    page.close()
    assert app is not None


def test_settings_disables_model_selector_when_none_are_downloaded(monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr("config.paths.installed_whisper_models", lambda _models: [])

    page = SettingsPage()

    assert page.whisper_model.isEnabled() is False
    assert page.whisper_model.currentData() == ""
    page.close()
    assert app is not None
