from __future__ import annotations

from modules.video_import.registry import extract_urls, resolve_provider


def test_extract_urls_handles_multiline_text_and_deduplicates() -> None:
    text = """
    https://www.xiaohongshu.com/discovery/item/6a4b2573000000001c027df7
    https://www.rednote.com/explore/69ce30d3000000002100791c,
    https://www.rednote.com/explore/69ce30d3000000002100791c
    """

    assert extract_urls(text) == [
        "https://www.xiaohongshu.com/discovery/item/6a4b2573000000001c027df7",
        "https://www.rednote.com/explore/69ce30d3000000002100791c",
    ]


def test_resolve_provider_recognizes_xiaohongshu_and_rednote() -> None:
    provider = resolve_provider("https://www.xiaohongshu.com/discovery/item/6a4b2573000000001c027df7")
    assert provider is not None
    assert provider.name == "xiaohongshu"

    provider = resolve_provider("https://www.rednote.com/explore/69ce30d3000000002100791c")
    assert provider is not None
    assert provider.name == "xiaohongshu"


def test_resolve_provider_returns_none_for_unsupported_url() -> None:
    assert resolve_provider("ftp://example.com/video/123") is None
