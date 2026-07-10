from __future__ import annotations

import time
from threading import Event
from typing import Callable

from core.context import CancellationError, Segment

ProgressCallback = Callable[[int], None]
LogCallback = Callable[[str], None]

LANGUAGE_MAP = {
    "zh": "zh-CN",
    "en": "en",
    "km": "km",
}

MAX_RETRIES = 3
RETRY_DELAY = 2.0
BATCH_SIZE = 10


def translate_segments_google(
    segments: list[Segment],
    source_language: str,
    progress_cb: ProgressCallback | None,
    log_cb: LogCallback | None,
    cancel_event: Event,
) -> list[Segment]:
    if not segments:
        raise ValueError("No segments available for translation")

    if source_language == "km":
        for index, segment in enumerate(segments, start=1):
            segment.translated_text = segment.text
            segment.raw_khmer_text = segment.text
            segment.improved_khmer_text = segment.text
            if progress_cb:
                progress_cb(int((index / len(segments)) * 100))
        if log_cb:
            log_cb("Source language is Khmer; translation stage copied original text")
        return segments

    src_code = LANGUAGE_MAP.get(source_language)
    if not src_code:
        raise ValueError(f"Unsupported source language for Google Translate: {source_language}")

    from deep_translator import GoogleTranslator

    translator = GoogleTranslator(source=src_code, target="km")

    if log_cb:
        log_cb(f"Using Google Translate: {src_code} → km ({len(segments)} segments)")

    translated_count = 0
    for start in range(0, len(segments), BATCH_SIZE):
        if cancel_event.is_set():
            raise CancellationError("Processing cancelled by user")

        batch = segments[start : start + BATCH_SIZE]
        for segment in batch:
            if cancel_event.is_set():
                raise CancellationError("Processing cancelled by user")

            text = segment.text.strip()
            if not text:
                segment.translated_text = ""
                segment.raw_khmer_text = ""
                segment.improved_khmer_text = ""
                translated_count += 1
                continue

            result = _translate_with_retry(translator, text, log_cb)
            segment.translated_text = result
            segment.raw_khmer_text = result
            segment.improved_khmer_text = result
            if log_cb:
                log_cb(f'  Translated: "{text}" -> "{result}"')

            translated_count += 1
            if progress_cb:
                progress_cb(int((translated_count / len(segments)) * 100))

    return segments


def _translate_with_retry(
    translator, text: str, log_cb: LogCallback | None
) -> str:
    for attempt in range(MAX_RETRIES):
        try:
            return translator.translate(text)
        except Exception as exc:
            if attempt < MAX_RETRIES - 1:
                if log_cb:
                    log_cb(f"  Google Translate retry {attempt + 1}: {exc}")
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                raise RuntimeError(f"Google Translate failed after {MAX_RETRIES} retries: {exc}") from exc
    return text
