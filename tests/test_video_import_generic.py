from __future__ import annotations

from pathlib import Path

from modules.video_import.base import DownloadResult
from modules.video_import.providers.generic import GenericYtDlpProvider
from modules.video_import.registry import resolve_provider


def test_generic_provider_can_handle_http_urls() -> None:
    provider = GenericYtDlpProvider()
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    assert provider.can_handle(url)
    assert provider.normalize_url(url) == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    
    content_id = provider.content_id(url)
    assert len(content_id) == 16
    # Should be deterministic
    assert provider.content_id(url) == content_id


def test_registry_resolves_generic_provider_as_fallback() -> None:
    yt_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    xhs_url = "https://www.xiaohongshu.com/discovery/item/6A4B2573000000001C027DF7"

    xhs_provider = resolve_provider(xhs_url)
    assert xhs_provider is not None
    assert xhs_provider.name == "xiaohongshu"

    generic_provider = resolve_provider(yt_url)
    assert generic_provider is not None
    assert generic_provider.name == "generic"


def test_generic_provider_uses_expected_yt_dlp_options(monkeypatch, tmp_path: Path) -> None:
    provider = GenericYtDlpProvider()
    captured: dict = {}
    target = tmp_path / "source.mp4"

    class FakeYoutubeDL:
        def __init__(self, opts):
            captured["opts"] = opts

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def extract_info(self, url, download=True):
            captured["url"] = url
            captured["download"] = download
            target.write_bytes(b"mp4")
            return {"_filename": str(target), "title": "Generic video title"}

    monkeypatch.setitem(__import__("sys").modules, "yt_dlp", type("FakeModule", (), {"YoutubeDL": FakeYoutubeDL})())
    monkeypatch.setitem(
        __import__("sys").modules,
        "yt_dlp.utils",
        type("FakeUtils", (), {"DownloadError": RuntimeError})(),
    )

    result = provider.download(
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        tmp_path,
        cookies_file=tmp_path / "cookies.txt",
    )

    assert isinstance(result, DownloadResult)
    assert result.path == target
    assert result.source_title == "Generic video title"
    assert captured["url"] == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    assert captured["download"] is True
    # Verify format selection is set to standard high-quality streams to avoid watermarks/logos
    assert captured["opts"]["format"] == "bestvideo*+bestaudio/best"
    assert captured["opts"]["merge_output_format"] == "mp4"
    assert captured["opts"]["cookiefile"] == str(tmp_path / "cookies.txt")
    assert captured["opts"]["outtmpl"] == str(tmp_path / "source.%(ext)s")
