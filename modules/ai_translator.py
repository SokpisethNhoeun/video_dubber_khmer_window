from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from threading import Event
from typing import Callable

from config.env import resolve_review_api_credentials
from core.context import CancellationError, Segment
from modules.glossary_builder import extract_glossary_terms

ProgressCallback = Callable[[int], None]
LogCallback = Callable[[str], None]

BATCH_SIZE = 40
CONTEXT_WINDOW = 10
MAX_RETRIES = 3
RETRY_DELAY = 2.0
SUMMARY_MAX_CHARS = 700

LANGUAGE_LABELS = {
    "zh": "Chinese",
    "en": "English",
    "km": "Khmer",
}

CONTENT_STYLE_INSTRUCTIONS = {
    "casual_vlog": (
        "Casual YouTube/TikTok vlog tone. First person, informal spoken Khmer. "
        "Contractions and everyday phrasing welcome."
    ),
    "educational": (
        "Educational / explainer. Address viewer directly, clear sentences, "
        "preserve technical terms consistently."
    ),
    "reaction": (
        "Reaction / commentary. Preserve emotional beats, keep it punchy, "
        "short sentences over long ones."
    ),
    "movie_dialogue": (
        "Movie / drama dialogue. Preserve character voice, slang, insults, "
        "profanity, threats, and emotional intensity faithfully in natural "
        "spoken Khmer. Do not make dialogue polite unless the original is polite."
    ),
    "documentary": (
        "Documentary narration. Measured, authoritative, slightly formal but "
        "still clear spoken Khmer."
    ),
    "tutorial": (
        "Tutorial / how-to. Step-by-step clarity, address viewer as 'you', "
        "practical everyday Khmer."
    ),
    "news": (
        "News reporting. Neutral, factual, clear pronunciation-friendly Khmer."
    ),
    "generic": (
        "Match the original tone as closely as possible."
    ),
}

KHMER_STYLE_LABELS = {
    "natural": "Natural conversational Khmer",
    "simple": "Simple, easy-to-understand Khmer",
    "formal": "Formal Khmer",
}


def _build_system_prompt(
    source_language: str,
    content_style: str,
    khmer_style: str,
    glossary_terms: list[str],
    user_glossary: dict[str, str],
) -> str:
    source_label = LANGUAGE_LABELS.get(source_language, source_language)
    content_instruction = CONTENT_STYLE_INSTRUCTIONS.get(
        content_style, CONTENT_STYLE_INSTRUCTIONS["casual_vlog"]
    )
    style_label = KHMER_STYLE_LABELS.get(khmer_style, khmer_style)

    glossary_section = ""
    if glossary_terms or user_glossary:
        entries = []
        for term in glossary_terms:
            entries.append(f"  - {term}")
        for source, target in user_glossary.items():
            entries.append(f"  - {source} → {target}")
        glossary_section = (
            "==================================================\n"
            "GLOSSARY (use these exact translations consistently)\n"
            "==================================================\n\n"
            + "\n".join(entries) + "\n\n"
        )

    return f"""==================================================
PRIMARY TASK
==================================================

Translate every input segment FROM {source_label} TO Khmer.
Reply with a single JSON object matching the OUTPUT SCHEMA below.
Do NOT reply in {source_label}. Do NOT write prose. Do NOT explain. Do NOT summarize.

CRITICAL LANGUAGE REQUIREMENT:
- Every "text" value must contain Khmer Unicode characters: ក ខ គ ច ជ ញ ត ន ប ម យ រ ល វ ស ហ ...
- English output is invalid.
- Romanized Khmer output is invalid.
- Copying the {source_label} source text is invalid.
- If the source line is a name or title, transliterate it into Khmer script.

==================================================
OUTPUT SCHEMA (this is the ONLY valid response shape)
==================================================

{{"segments":[{{"index":0,"text":"<Khmer translation of input segment 0>"}},{{"index":1,"text":"<Khmer translation of input segment 1>"}}]}}

Rules:
- The top-level MUST be a JSON object with a "segments" array. Never reply with a bare array.
- Output exactly one entry per input segment. Preserve every "index" exactly. Preserve input order.
- Every "text" value MUST contain Khmer script. Never {source_label}. Never English. Never romanization.
- No markdown, no code fences, no comments, no trailing text.

==================================================
LANGUAGE
==================================================

- Register: {style_label}
- Natural spoken Khmer. Sounds like a native narrator. Not word-for-word. Not literary.

==================================================
TIMING
==================================================

- Each input segment includes a "duration" in seconds.
- Keep the Khmer translation short enough to be spoken comfortably within that duration.
- Prefer concise wording over literal translation. Compress when necessary. Never add filler words.

==================================================
MEANING
==================================================

- Preserve the original meaning. Do not add or omit facts.
- Compress wording instead of removing meaning when tight on time.
- This can be movie/dialogue content. If the source contains profanity, insults,
  crude slang, sexual references, violence, or other mature language, translate
  it faithfully into natural Khmer. Do not censor, sanitize, moralize, soften,
  replace with [censored], or refuse because of bad words.
- Keep the same intensity as the source, but do not add new insults, slurs, or
  stronger meaning that was not present in the original.

==================================================
CONSISTENCY
==================================================

- Use the same Khmer spelling for every recurring name, place, brand, or technical term.
- If the glossary below provides a translation, use it exactly.
- Once you transliterate a new name, use that spelling in every later segment.

==================================================
STYLE
==================================================

{content_instruction}

{glossary_section}==================================================
FINAL REMINDER
==================================================

Reply with the JSON object only. Top-level "segments" array. Khmer text values.
No prose. No explanation. No {source_label}. No English. No romanization. No markdown fences.
"""

def _build_script_summary(segments: list[Segment]) -> str:
    full_text = " ".join(s.text.strip() for s in segments if s.text.strip())
    if len(full_text) > SUMMARY_MAX_CHARS:
        return full_text[:SUMMARY_MAX_CHARS] + "..."
    return full_text


def _build_batch_payload(
    batch: list[Segment],
    previous_translations: list[dict],
    script_summary: str,
    batch_index: int,
    total_batches: int,
) -> list[dict]:
    user_data = {
        "batch": f"{batch_index + 1}/{total_batches}",
        "script_summary": script_summary,
    }
    if previous_translations:
        user_data["previous_context"] = previous_translations

    user_data["segments"] = [
        {
            "index": s.index,
            "duration": round(s.end - s.start, 3),
            "speaker": s.speaker_label or s.speaker_id or "",
            "text": s.text,
        }
        for s in batch
    ]

    return [{"role": "user", "content": json.dumps(user_data, ensure_ascii=False)}]


def _build_khmer_retry_message(
    batch: list[Segment],
    non_khmer_indices: list[int],
    attempt: int,
) -> dict:
    return {
        "role": "user",
        "content": json.dumps(
            {
                "retry": attempt + 1,
                "previous_response_rejected": True,
                "reason": (
                    "The previous response was not Khmer. It used English, "
                    "romanization, or copied the source language."
                ),
                "invalid_indices": non_khmer_indices[:10],
                "required_output": (
                    "Return Khmer Unicode script only in each text field. "
                    "Every text value must contain Khmer characters in the "
                    "Unicode range U+1780-U+17FF."
                ),
                "schema": {
                    "segments": [
                        {"index": batch[0].index if batch else 0, "text": "ខ្មែរ..."}
                    ]
                },
                "segments": [
                    {
                        "index": s.index,
                        "duration": round(s.end - s.start, 3),
                        "speaker": s.speaker_label or s.speaker_id or "",
                        "text": s.text,
                    }
                    for s in batch
                ],
            },
            ensure_ascii=False,
        ),
    }


def _load_glossary(glossary_path: Path | None) -> dict[str, str]:
    if glossary_path is None or not glossary_path.exists():
        return {}
    glossary: dict[str, str] = {}
    for raw_line in glossary_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            source, target = line.split("=", 1)
        elif "," in line:
            source, target = line.split(",", 1)
        else:
            continue
        source = source.strip()
        target = target.strip()
        if source and target:
            glossary[source] = target
    return glossary


class _ContentFilterBlocked(Exception):
    """Model refused to generate content (safety filter, empty completion)."""


_BLOCKED = object()


def _strip_code_fence(text: str) -> str:
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    start = 1
    end = len(lines)
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip() == "```":
            end = i
            break
    return "\n".join(lines[start:end])


def _parse_response(raw: str) -> dict:
    text = _strip_code_fence(raw.strip())
    body = json.loads(text)

    try:
        choice = body["choices"][0]
        finish_reason = choice.get("finish_reason", "")
        resp_body = choice["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return body

    resp_text = (resp_body or "").strip()
    if not resp_text:
        model = body.get("model", "?")
        raise _ContentFilterBlocked(
            f"model {model!r} returned empty content (finish_reason={finish_reason!r})"
        )

    return json.loads(_strip_code_fence(resp_text))


def _call_api(
    system_prompt: str,
    messages: list[dict],
    cancel_event: Event,
    log_cb: LogCallback | None,
) -> dict | object | None:
    api_key, base_url, model = resolve_review_api_credentials()
    if not api_key:
        raise RuntimeError(
            "No API key found. Set TRANSCRIPT_REVIEW_API_KEY, OPENAI_API_KEY, or "
            "GEMINI_API_KEY environment variable."
        )
    timeout = float(os.getenv("TRANSCRIPT_REVIEW_TIMEOUT", "120"))

    payload = {
        "model": model,
        "temperature": 0.3,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            *messages,
        ],
    }
    if os.getenv("TRANSCRIPT_REVIEW_JSON_MODE", "1") != "0":
        payload["response_format"] = {"type": "json_object"}

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    if cancel_event.is_set():
        raise CancellationError("Processing cancelled by user")

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except Exception as e:
        error_msg = e.read().decode("utf-8") if hasattr(e, "read") else str(e)
        if log_cb:
            log_cb(f"AI translation API error: {error_msg}")
        return None

    if cancel_event.is_set():
        raise CancellationError("Processing cancelled by user")

    try:
        return _parse_response(raw)
    except _ContentFilterBlocked as e:
        if log_cb:
            log_cb(
                f"AI translation blocked by model safety filter: {e}. "
                "Retrying the same request will not help — switch to a different "
                "model in your OmniRouter combo (a non-Gemini fallback), or "
                "reduce BATCH_SIZE if the block is content-length related."
            )
        return _BLOCKED
    except Exception as e:
        if log_cb:
            content_preview = ""
            try:
                content = json.loads(raw)["choices"][0]["message"]["content"] or ""
                content_preview = f"\nModel content (first 400 chars): {content[:400]!r}"
            except Exception:
                pass
            log_cb(
                f"Failed to parse AI translation response: {e}"
                f"{content_preview}\nRaw envelope: {raw[:300]}..."
            )
        return None


_TRANSLATION_TEXT_KEYS = (
    "text",
    "translation",
    "translated_text",
    "khmer",
    "khmer_text",
    "target",
    "target_text",
)


def _looks_like_segment(item: object) -> bool:
    return (
        isinstance(item, dict)
        and "index" in item
        and any(key in item for key in _TRANSLATION_TEXT_KEYS)
    )


def _segment_translation_text(item: dict) -> str:
    for key in _TRANSLATION_TEXT_KEYS:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _contains_khmer(text: str) -> bool:
    return any("\u1780" <= char <= "\u17ff" for char in text)


def _requires_khmer_retry(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if _contains_khmer(stripped):
        return False
    # Pure numbers/punctuation can appear in valid short segments, but Latin
    # or other alphabetic output means the model translated to the wrong
    # language and should be retried instead of accepted silently.
    return any(char.isalpha() for char in stripped)


def _extract_segments_payload(result: object) -> list | None:
    if isinstance(result, dict):
        segments = result.get("segments")
        if isinstance(segments, list):
            return segments
        if all(_looks_like_segment(v) for v in result.values()) and result:
            return list(result.values())
        return None
    if isinstance(result, list) and result and all(_looks_like_segment(x) for x in result):
        return result
    return None


def _translate_batch_with_retry(
    system_prompt: str,
    messages: list[dict],
    batch: list[Segment],
    cancel_event: Event,
    log_cb: LogCallback | None,
    allow_review_recovery: bool = False,
) -> dict[int, str]:
    retry_messages = list(messages)
    max_attempts = 1 if allow_review_recovery else MAX_RETRIES
    for attempt in range(max_attempts):
        if cancel_event.is_set():
            raise CancellationError("Processing cancelled by user")

        result = _call_api(system_prompt, retry_messages, cancel_event, log_cb)
        if result is _BLOCKED:
            return {}
        if result is None:
            if attempt < max_attempts - 1:
                if log_cb:
                    log_cb(f"  AI translation retry {attempt + 1}/{max_attempts}")
                time.sleep(RETRY_DELAY * (attempt + 1))
                continue
            return {}

        segments_payload = _extract_segments_payload(result)
        if segments_payload is None:
            if log_cb:
                shape = type(result).__name__
                preview = repr(result)[:200]
                log_cb(
                    f"  AI translation returned unexpected shape ({shape}): {preview}"
                )
            if attempt < max_attempts - 1:
                if log_cb:
                    log_cb(f"  AI translation retry {attempt + 1}/{max_attempts}")
                time.sleep(RETRY_DELAY * (attempt + 1))
                continue
            return {}

        by_index: dict[int, str] = {}
        non_khmer: list[int] = []
        for item in segments_payload:
            if isinstance(item, dict) and str(item.get("index", "")).isdigit():
                idx = int(item["index"])
                text = _segment_translation_text(item)
                if text:
                    if _requires_khmer_retry(text):
                        non_khmer.append(idx)
                    else:
                        by_index[idx] = text

        expected = {s.index for s in batch}
        missing = expected - set(by_index.keys())
        if non_khmer and log_cb:
            log_cb(
                f"  AI translation returned non-Khmer text for "
                f"{len(non_khmer)} segment(s) (indices {non_khmer[:5]})"
            )
        if not missing:
            return by_index

        if allow_review_recovery:
            if log_cb:
                log_cb(
                    f"  AI translation missing {len(missing)} segments "
                    f"(indices {sorted(missing)[:5]}), deferring to AI review"
                )
            return by_index

        if attempt < max_attempts - 1:
            if log_cb:
                log_cb(
                    f"  AI translation missing {len(missing)} segments "
                    f"(indices {sorted(missing)[:5]}), retrying..."
                )
            if non_khmer:
                retry_messages = messages + [
                    _build_khmer_retry_message(batch, non_khmer, attempt)
                ]
            time.sleep(RETRY_DELAY * (attempt + 1))
        else:
            if log_cb:
                log_cb(
                    f"  AI translation: {len(missing)} segments still missing "
                    f"after {max_attempts} retries"
                )

    return by_index


def translate_segments_ai(
    segments: list[Segment],
    source_language: str,
    progress_cb: ProgressCallback | None,
    log_cb: LogCallback | None,
    cancel_event: Event,
    content_style: str = "casual_vlog",
    khmer_style: str = "simple",
    glossary_path: Path | None = None,
    allow_review_recovery: bool = False,
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

    source_label = LANGUAGE_LABELS.get(source_language, source_language)
    if log_cb:
        log_cb(
            f"Using AI Translation: {source_label} → Khmer "
            f"({len(segments)} segments, batch size {BATCH_SIZE})"
        )

    source_texts = [s.text for s in segments if s.text.strip()]
    glossary_terms = [t.source for t in extract_glossary_terms(source_texts)]
    user_glossary = _load_glossary(glossary_path)

    system_prompt = _build_system_prompt(
        source_language, content_style, khmer_style, glossary_terms, user_glossary
    )
    script_summary = _build_script_summary(segments)

    total_batches = (len(segments) + BATCH_SIZE - 1) // BATCH_SIZE
    translated_count = 0
    khmer_count = 0
    deferred_to_review_count = 0
    previous_translations: list[dict] = []

    for batch_idx in range(total_batches):
        if cancel_event.is_set():
            raise CancellationError("Processing cancelled by user")

        start = batch_idx * BATCH_SIZE
        batch = segments[start : start + BATCH_SIZE]

        if log_cb:
            log_cb(
                f"  Translating batch {batch_idx + 1}/{total_batches} "
                f"(segments {batch[0].index}-{batch[-1].index})"
            )

        messages = _build_batch_payload(
            batch, previous_translations, script_summary,
            batch_idx, total_batches,
        )

        by_index = _translate_batch_with_retry(
            system_prompt,
            messages,
            batch,
            cancel_event,
            log_cb,
            allow_review_recovery=allow_review_recovery
            and os.getenv("AI_TRANSLATION_ALLOW_SOURCE_FALLBACK", "0") != "1",
        )

        missing_indices = [segment.index for segment in batch if not by_index.get(segment.index)]
        allow_source_fallback = os.getenv("AI_TRANSLATION_ALLOW_SOURCE_FALLBACK", "0") == "1"
        if missing_indices and not allow_source_fallback and not allow_review_recovery:
            raise RuntimeError(
                "AI translation did not return valid Khmer for "
                f"{len(missing_indices)} segment(s) in batch {batch_idx + 1}/{total_batches} "
                f"(indices {missing_indices[:8]}). Stopping before TTS so source-language "
                "text is not synthesized with Khmer voices. Try a stronger translation model "
                "or enable AI transcript review to recover missing lines."
            )
        if missing_indices and allow_review_recovery and not allow_source_fallback and log_cb:
            log_cb(
                "  AI translation deferred "
                f"{len(missing_indices)} missing segment(s) to AI review "
                f"(indices {missing_indices[:8]})"
            )

        for segment in batch:
            text = by_index.get(segment.index, "")
            if text:
                segment.translated_text = text
                segment.raw_khmer_text = text
                segment.improved_khmer_text = text
                segment.review_notes = ""
                khmer_count += 1
            else:
                if allow_review_recovery and not allow_source_fallback:
                    segment.translated_text = ""
                    segment.raw_khmer_text = ""
                    segment.improved_khmer_text = ""
                    segment.review_notes = (
                        "AI translation missing; recover from source during AI review"
                    )
                    deferred_to_review_count += 1
                else:
                    segment.translated_text = segment.text
                    segment.raw_khmer_text = segment.text
                    segment.improved_khmer_text = segment.text
                if log_cb:
                    log_cb(
                        f"  Warning: segment {segment.index} not translated by AI, "
                        + (
                            "deferred to AI review"
                            if allow_review_recovery and not allow_source_fallback
                            else "kept original text"
                        )
                    )

            translated_count += 1
            if progress_cb:
                progress_cb(int((translated_count / len(segments)) * 100))

        new_context = [
            {"index": s.index, "source": s.text, "translation": s.translated_text}
            for s in batch
        ]
        previous_translations = new_context[-CONTEXT_WINDOW:]

    if log_cb:
        if deferred_to_review_count:
            log_cb(
                "AI translation complete: "
                f"{khmer_count} segment(s) translated, "
                f"{deferred_to_review_count} deferred to AI review"
            )
        else:
            log_cb(f"AI translation complete: {translated_count} segments translated")

    return segments


BULK_AI_RECOVERY_THRESHOLD = 5


def recover_missing_khmer_with_ai(
    segments: list[Segment],
    source_language: str,
    progress_cb: ProgressCallback | None,
    log_cb: LogCallback | None,
    cancel_event: Event,
    content_style: str = "casual_vlog",
    khmer_style: str = "simple",
    glossary_path: Path | None = None,
) -> list[Segment]:
    """Translate and review segments that are missing Khmer dubbing text."""
    if not segments:
        return segments

    targets = [segment for segment in segments if segment.enabled and segment.text.strip()]
    if not targets:
        return segments

    if log_cb:
        log_cb(f"  AI recovery: translating {len(targets)} segment(s) to Khmer")

    translate_segments_ai(
        targets,
        source_language,
        progress_cb,
        log_cb,
        cancel_event,
        content_style=content_style,
        khmer_style=khmer_style,
        glossary_path=glossary_path,
        allow_review_recovery=False,
    )

    still_missing = [segment for segment in targets if not segment.tts_text.strip()]
    review_targets = still_missing or targets
    if log_cb:
        log_cb(f"  AI recovery: reviewing {len(review_targets)} segment(s)")

    from modules.transcript_review import review_segments

    review_segments(
        review_targets,
        khmer_style,
        "ai",
        glossary_path,
        None,
        None,
        progress_cb,
        log_cb,
        cancel_event,
        content_style=content_style,
        source_language=source_language,
    )

    for segment in targets:
        if segment.tts_text.strip():
            note = "recovered Khmer via AI translation/review"
            if note not in (segment.review_notes or ""):
                segment.review_notes = f"{segment.review_notes}; {note}".strip("; ").strip()

    return segments
