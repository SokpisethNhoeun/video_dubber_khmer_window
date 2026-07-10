from __future__ import annotations

from pathlib import Path

from modules.video_import.base import DownloadResult
from modules.video_import.providers.xiaohongshu import XiaohongshuProvider


def test_xiaohongshu_provider_extracts_content_id_and_normalizes_url() -> None:
    provider = XiaohongshuProvider()
    url = "https://www.xiaohongshu.com/discovery/item/6A4B2573000000001C027DF7?xsec_token=abc#frag"

    assert provider.can_handle(url)
    assert provider.normalize_url(url) == (
        "https://www.xiaohongshu.com/discovery/item/6A4B2573000000001C027DF7?xsec_token=abc"
    )
    assert provider.content_id(url) == "6a4b2573000000001c027df7"


def test_xiaohongshu_provider_accepts_rednote_and_short_links() -> None:
    provider = XiaohongshuProvider()

    assert provider.can_handle("https://www.rednote.com/explore/69ce30d3000000002100791c")
    assert provider.can_handle("https://xhslink.com/a/AbCdEf")


def test_xiaohongshu_provider_uses_expected_yt_dlp_options(monkeypatch, tmp_path: Path) -> None:
    provider = XiaohongshuProvider()
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
            return {"_filename": str(target), "title": "Source video title"}

    monkeypatch.setitem(__import__("sys").modules, "yt_dlp", type("FakeModule", (), {"YoutubeDL": FakeYoutubeDL})())
    monkeypatch.setitem(
        __import__("sys").modules,
        "yt_dlp.utils",
        type("FakeUtils", (), {"DownloadError": RuntimeError})(),
    )

    result = provider.download(
        "https://www.rednote.com/explore/69ce30d3000000002100791c",
        tmp_path,
        cookies_file=tmp_path / "cookies.txt",
    )

    assert isinstance(result, DownloadResult)
    assert result.path == target
    assert result.source_title == "Source video title"
    assert captured["url"] == "https://www.rednote.com/explore/69ce30d3000000002100791c"
    assert captured["download"] is True
    assert captured["opts"]["format"] == "direct/bestvideo*+bestaudio/best"
    assert captured["opts"]["merge_output_format"] == "mp4"
    assert captured["opts"]["cookiefile"] == str(tmp_path / "cookies.txt")
    assert captured["opts"]["outtmpl"] == str(tmp_path / "source.%(ext)s")
