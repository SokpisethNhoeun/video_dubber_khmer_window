from __future__ import annotations

from threading import Event

from core.context import Segment
from modules.tts_engine import repair_segments_for_tts, synthesize_tts


def test_repair_segments_disables_empty_source_segment():
    segment = Segment(index=0, start=0.0, end=1.0, text="", enabled=True)
    repair_segments_for_tts([segment], "zh", log_cb=None, cancel_event=Event())
    assert segment.enabled is False
    assert "no source or Khmer text" in segment.review_notes


def test_repair_segments_syncs_existing_khmer_fields():
    segment = Segment(
        index=0,
        start=0.0,
        end=1.0,
        text="hello",
        translated_text="សួស្តី",
        enabled=True,
    )
    repair_segments_for_tts([segment], "zh", log_cb=None, cancel_event=Event())
    assert segment.tts_text == "សួស្តី"


def test_repair_segments_uses_google_translate_for_small_nllb_recovery(monkeypatch):
    segment = Segment(index=0, start=0.0, end=1.0, text="hello", enabled=True)

    class FakeTranslator:
        def translate(self, text: str) -> str:
            assert text == "hello"
            return "សួស្តី"

    monkeypatch.setattr("deep_translator.GoogleTranslator", lambda **kwargs: FakeTranslator())

    repair_segments_for_tts(
        [segment],
        "zh",
        log_cb=None,
        cancel_event=Event(),
        translation_backend="google",
    )
    assert segment.tts_text == "សួស្តី"
    assert "recovered Khmer via Google Translate" in segment.review_notes


def test_repair_segments_uses_ai_for_bulk_missing(monkeypatch):
    segments = [
        Segment(index=i, start=float(i), end=float(i) + 1.0, text=f"line {i}", enabled=True)
        for i in range(6)
    ]
    calls: list[str] = []

    def fake_recover(batch, source_language, progress_cb, log_cb, cancel_event, **kwargs):
        calls.append("recover")
        for segment in batch:
            segment.translated_text = f"ខ្មែរ {segment.index}"
            segment.raw_khmer_text = segment.translated_text
            segment.improved_khmer_text = segment.translated_text
        return batch

    monkeypatch.setattr("modules.ai_translator.recover_missing_khmer_with_ai", fake_recover)

    repair_segments_for_tts(
        segments,
        "zh",
        log_cb=None,
        cancel_event=Event(),
        translation_backend="google",
    )

    assert calls == ["recover"]
    assert segments[0].tts_text == "ខ្មែរ 0"


def test_repair_segments_uses_ai_for_ai_backend_even_for_one_segment(monkeypatch):
    segment = Segment(index=0, start=0.0, end=1.0, text="hello", enabled=True)
    called = {"value": False}

    def fake_recover(batch, source_language, progress_cb, log_cb, cancel_event, **kwargs):
        called["value"] = True
        segment.translated_text = "សួស្តី"
        segment.raw_khmer_text = "សួស្តី"
        segment.improved_khmer_text = "សួស្តី"
        return batch

    monkeypatch.setattr("modules.ai_translator.recover_missing_khmer_with_ai", fake_recover)

    repair_segments_for_tts(
        [segment],
        "zh",
        log_cb=None,
        cancel_event=Event(),
        translation_backend="ai",
    )

    assert called["value"] is True
    assert segment.tts_text == "សួស្តី"


def test_synthesize_tts_explains_missing_enabled_khmer_rows(tmp_path):
    segment = Segment(index=0, start=0.0, end=1.0, text="", enabled=True)

    try:
        synthesize_tts(
            [segment],
            "female",
            0,
            0,
            tmp_path,
            None,
            None,
            Event(),
            source_language="zh",
        )
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected synthesize_tts to reject a session without Khmer TTS rows")

    assert "no enabled Khmer transcript rows" in message
    assert "rerun the video from translation/review" in message
