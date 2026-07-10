from __future__ import annotations

from threading import Event
from unittest.mock import MagicMock, patch

from core.context import Segment
from modules.google_translator import translate_segments_google


def _make_segments(texts: list[str]) -> list[Segment]:
    return [Segment(index=i, start=float(i), end=float(i + 1), text=t) for i, t in enumerate(texts)]


def test_khmer_source_copies_text():
    segments = _make_segments(["សួស្តី", "អរគុណ"])
    result = translate_segments_google(segments, "km", None, None, Event())
    assert result[0].translated_text == "សួស្តី"
    assert result[1].raw_khmer_text == "អរគុណ"


def test_empty_segments_raises():
    import pytest
    with pytest.raises(ValueError, match="No segments"):
        translate_segments_google([], "zh", None, None, Event())


def test_unsupported_language_raises():
    import pytest
    segments = _make_segments(["hello"])
    with pytest.raises(ValueError, match="Unsupported"):
        translate_segments_google(segments, "xx", None, None, Event())


@patch("deep_translator.GoogleTranslator")
def test_translates_via_google(mock_cls):
    mock_translator = MagicMock()
    mock_translator.translate.side_effect = ["ខ្មែរ១", "ខ្មែរ២"]
    mock_cls.return_value = mock_translator

    segments = _make_segments(["hello", "world"])
    result = translate_segments_google(segments, "en", None, None, Event())

    assert result[0].translated_text == "ខ្មែរ១"
    assert result[1].translated_text == "ខ្មែរ២"
    assert mock_translator.translate.call_count == 2
