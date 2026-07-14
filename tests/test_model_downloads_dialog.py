from PyQt6.QtWidgets import QApplication

from config.env import resolve_review_api_credentials
from gui.dialogs.model_downloads import ModelDownloadsDialog


def test_transcript_review_key_defaults_to_google_ai_studio(monkeypatch) -> None:
    monkeypatch.setenv("TRANSCRIPT_REVIEW_API_KEY", "customer-key")
    monkeypatch.delenv("TRANSCRIPT_REVIEW_API_BASE_URL", raising=False)
    monkeypatch.delenv("TRANSCRIPT_REVIEW_MODEL", raising=False)

    key, base_url, model = resolve_review_api_credentials()

    assert key == "customer-key"
    assert base_url == "https://generativelanguage.googleapis.com/v1beta/openai"
    assert model == "gemini-3.1-flash-lite"


def test_download_dialog_lists_managed_models(monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])

    dialog = ModelDownloadsDialog()

    assert set(dialog._rows) == {
        "tiny", "base", "small", "medium", "large-v3", "nllb", "qwen3", "cosyvoice"
    }
    assert dialog.windowTitle() == "Model Downloads"
    dialog.close()
    assert app is not None
