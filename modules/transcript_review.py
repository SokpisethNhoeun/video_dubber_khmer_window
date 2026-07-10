from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Callable

from config.env import resolve_review_api_credentials
from core.context import CancellationError, Segment
from modules.glossary_builder import extract_glossary_terms


ProgressCallback = Callable[[int], None]
LogCallback = Callable[[str], None]


STYLE_LABELS = {
    "natural": "Natural",
    "simple": "Simple/Easy",
    "formal": "Formal",
    "short_dub": "Short Dub",
}
REVIEW_MAX_RETRIES = 2
DEFAULT_REVIEW_MAX_TOKENS = 12000

# Content-style biasing for the AI reviewer. Chosen once per project in the
# wizard; shapes the *voice* of the Khmer rewrite independently of the
# register-focused `khmer_style` above.
CONTENT_STYLE_INSTRUCTIONS = {
    "casual_vlog": (
        "Casual YouTube/TikTok vlog. Write in first person, informal spoken "
        "Khmer. Contractions and everyday phrasing are welcome. Avoid stiff "
        "written-Khmer constructions."
    ),
    "educational": (
        "Educational / explainer content. Address the viewer directly, keep "
        "sentences clear and self-contained, and preserve technical terms "
        "consistently. Prefer everyday Khmer over borrowed jargon where the "
        "meaning is unchanged."
    ),
    "reaction": (
        "Reaction / commentary. Preserve emotional beats — surprise, "
        "excitement, laughter cues — and let exclamations land. Keep it "
        "punchy; short sentences over long ones."
    ),
    "movie_dialogue": (
        "Movie / drama dialogue. Preserve character voice, slang, insults, "
        "profanity, threats, and emotional intensity faithfully in natural "
        "spoken Khmer. Do not make dialogue polite unless the original is polite."
    ),
    "generic": (
        "Match the original tone as closely as possible without adding a "
        "specific voice preset."
    ),
}
SRT_SKIP_MARKERS = {"[skip]", "[mute]", "skip", "mute"}
LATIN_ALLOWED_TERMS = {
    "ai",
    "api",
    "app",
    "cpu",
    "gpu",
    "hd",
    "ios",
    "iphone",
    "mac",
    "pc",
    "pro",
    "tv",
    "usb",
    "wifi",
}

SPOKEN_KHMER_REPLACEMENTS = {
    "លោកអ្នក": "អ្នក",
    "ពួកយើង": "យើង",
    "ធ្វើការសម្រេចចិត្ត": "សម្រេចចិត្ត",
    "ធ្វើការពិនិត្យមើល": "ពិនិត្យមើល",
    "នៅក្នុងពេលនេះ": "ឥឡូវនេះ",
    "នៅពេលដែល": "ពេល",
    "ដោយសារតែ": "ព្រោះ",
    "យ៉ាងណាក៏ដោយ": "តែ",
    "បន្ទាប់ពីនោះ": "បន្ទាប់មក",
}

LOW_VALUE_PHRASES = [
    "ជាក់ស្តែង",
    "យ៉ាងពិតប្រាកដ",
    "ដែលបានរៀបរាប់ខាងលើ",
    "នៅក្នុងករណីនេះ",
    "គួរឱ្យកត់សម្គាល់ថា",
]


@dataclass(frozen=True)
class StoryContext:
    summary: str
    terms: list[str]
    speaker_notes: list[str]


def _segment_payload(segment: Segment) -> dict:
    return {
        "index": segment.index,
        "start": segment.start,
        "end": segment.end,
        "speaker": segment.speaker_label or segment.speaker_id or "",
        "source_text": segment.text,
        "raw_khmer_text": segment.raw_khmer_text or segment.translated_text,
        "improved_khmer_text": segment.improved_khmer_text or segment.translated_text,
        "user_edited_text": segment.user_edited_text,
        "enabled": segment.enabled,
        "review_notes": segment.review_notes,
    }


def _normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    text = text.replace(" ,", ",").replace(" .", ".").replace(" ?", "?").replace(" !", "!")
    return text


def _review_max_tokens() -> int:
    raw_value = os.getenv("TRANSCRIPT_REVIEW_MAX_TOKENS", str(DEFAULT_REVIEW_MAX_TOKENS))
    try:
        return max(1024, int(raw_value))
    except ValueError:
        return DEFAULT_REVIEW_MAX_TOKENS


def _target_non_space_chars(duration: float) -> int:
    return max(6, int(max(0.1, duration) * 22))


def _hard_non_space_chars(duration: float) -> int:
    return max(8, int(max(0.1, duration) * 26))


def _text_non_space_chars(text: str) -> int:
    return len(re.sub(r"\s+", "", text or ""))


def _latin_words(text: str) -> list[str]:
    words = re.findall(r"\b[A-Za-z][A-Za-z0-9_-]*\b", text or "")
    return [word for word in words if word.casefold() not in LATIN_ALLOWED_TERMS]


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


def _apply_glossary(text: str, glossary: dict[str, str]) -> str:
    improved = text
    for source, target in glossary.items():
        improved = re.sub(re.escape(source), target, improved, flags=re.IGNORECASE)
    return improved


def _style_cleanup(text: str, style: str, duration: float) -> str:
    cleaned = _normalize_text(text)
    if not cleaned:
        return cleaned

    for source, target in SPOKEN_KHMER_REPLACEMENTS.items():
        cleaned = cleaned.replace(source, target)

    if style == "formal":
        cleaned = cleaned.replace("អូន", "អ្នក").replace("បង", "លោកអ្នក")
    elif style == "simple":
        cleaned = cleaned.replace("លោកអ្នក", "អ្នក").replace("ពួកយើង", "យើង")
    elif style == "short_dub":
        words = cleaned.split()
        max_words = max(4, int(duration * 3.2))
        if len(words) > max_words:
            cleaned = " ".join(words[:max_words]).rstrip(",។.!?") + "..."

    return _fit_text_to_segment(cleaned, duration)


def _fit_text_to_segment(text: str, duration: float) -> str:
    cleaned = _normalize_text(text)
    hard_limit = _hard_non_space_chars(duration)
    if _text_non_space_chars(cleaned) <= hard_limit:
        return cleaned

    for phrase in LOW_VALUE_PHRASES:
        shortened = _normalize_text(cleaned.replace(phrase, ""))
        if shortened != cleaned:
            cleaned = shortened
        if _text_non_space_chars(cleaned) <= hard_limit:
            return cleaned

    separators = ["។", "?", "!", "៕", ","]
    best = ""
    for sep in separators:
        parts = [part.strip() for part in cleaned.split(sep) if part.strip()]
        candidate = ""
        for part in parts:
            next_candidate = f"{candidate}{sep} {part}" if candidate else part
            if _text_non_space_chars(next_candidate) > hard_limit:
                break
            candidate = next_candidate
        if candidate and _text_non_space_chars(candidate) <= hard_limit:
            best = candidate
            break
    if best:
        return best.rstrip(",។.!?") + "..."

    words = cleaned.split()
    if len(words) > 1:
        selected: list[str] = []
        for word in words:
            candidate = " ".join([*selected, word])
            if _text_non_space_chars(candidate) > hard_limit:
                break
            selected.append(word)
        if selected:
            return " ".join(selected).rstrip(",។.!?") + "..."

    return cleaned


def build_story_context(segments: list[Segment]) -> StoryContext:
    speakers: dict[str, int] = {}
    source_text: list[str] = []
    for segment in segments:
        if segment.speaker_label or segment.speaker_id:
            speaker = segment.speaker_label or segment.speaker_id or ""
            speakers[speaker] = speakers.get(speaker, 0) + 1
        source_text.append(segment.text)

    # Rich glossary extraction: catches Chinese personal names and recurring
    # domain terms that the previous ASCII-only regex missed. This is what
    # keeps a speaker's name spelled the same way across the whole dub.
    glossary_terms = extract_glossary_terms(source_text, max_terms=25)
    terms = [t.source for t in glossary_terms]

    joined = " ".join(source_text)
    summary = _normalize_text(joined[:700])
    if len(joined) > 700:
        summary += "..."
    speaker_notes = [f"{speaker}: {count} segment(s)" for speaker, count in sorted(speakers.items())]
    return StoryContext(summary=summary, terms=terms, speaker_notes=speaker_notes)


def load_review_json(path: Path, segments: list[Segment]) -> list[Segment]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_segments = payload.get("segments", payload if isinstance(payload, list) else [])
    by_index = {
        int(item.get("index")): item
        for item in raw_segments
        if isinstance(item, dict) and str(item.get("index", "")).isdigit()
    }
    for segment in segments:
        item = by_index.get(segment.index)
        if not item:
            continue
        segment.raw_khmer_text = str(item.get("raw_khmer_text") or segment.raw_khmer_text or segment.translated_text)
        segment.improved_khmer_text = str(
            item.get("improved_khmer_text") or item.get("translated_text") or segment.improved_khmer_text
        )
        segment.user_edited_text = str(item.get("user_edited_text") or "")
        segment.enabled = bool(item.get("enabled", True))
        segment.review_notes = str(item.get("review_notes") or "")
    return segments


def parse_srt_text(text: str) -> list[str]:
    entries: list[str] = []
    blocks = re.split(r"\n\s*\n", text.replace("\r\n", "\n").replace("\r", "\n").strip())
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        if lines[0].isdigit():
            lines = lines[1:]
        if lines and "-->" in lines[0]:
            lines = lines[1:]
        if not lines:
            entries.append("")
            continue
        entries.append(_normalize_text(" ".join(lines)))
    return entries


def load_review_srt(path: Path, segments: list[Segment]) -> list[Segment]:
    entries = parse_srt_text(path.read_text(encoding="utf-8"))
    for segment, text in zip(segments, entries):
        clean_text = _normalize_text(text)
        if clean_text.lower() in SRT_SKIP_MARKERS:
            segment.enabled = False
            segment.user_edited_text = ""
            segment.review_notes = "disabled by SRT"
            continue
        if not clean_text:
            continue
        segment.raw_khmer_text = segment.raw_khmer_text or segment.translated_text
        segment.improved_khmer_text = clean_text
        segment.user_edited_text = clean_text
        segment.enabled = True
        segment.review_notes = "loaded from SRT"
    return segments


def save_review_json(path: Path, segments: list[Segment], context: StoryContext, style: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "style": style,
        "story": {
            "summary": context.summary,
            "terms": context.terms,
            "speaker_notes": context.speaker_notes,
        },
        "segments": [_segment_payload(segment) for segment in segments],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _chat_payloads_from_raw(raw: str) -> list[dict]:
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("empty response body")

    if raw.startswith("data:"):
        payloads = []
        for line in raw.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            data = line.removeprefix("data:").strip()
            if not data or data == "[DONE]":
                continue
            payloads.append(json.loads(data))
        if payloads:
            return payloads
        raise ValueError("stream response did not include JSON data")

    payload = json.loads(raw)
    if isinstance(payload, dict):
        return [payload]
    raise ValueError("response JSON was not an object")


def _content_parts(content: object) -> list[str]:
    if isinstance(content, str):
        return [content]
    if not isinstance(content, list):
        return []
    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if isinstance(text, str):
            parts.append(text)
            continue
        nested_text = item.get("content")
        if isinstance(nested_text, str):
            parts.append(nested_text)
    return parts


def _extract_chat_message(raw: str) -> str:
    try:
        payloads = _chat_payloads_from_raw(raw)
    except json.JSONDecodeError:
        raw = (raw or "").strip()
        if raw:
            return raw
        raise

    parts: list[str] = []
    for payload in payloads:
        if isinstance(payload.get("segments"), list):
            return json.dumps(payload, ensure_ascii=False)

        choices = payload.get("choices")
        if not isinstance(choices, list):
            continue
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message")
            if isinstance(message, dict):
                parts.extend(_content_parts(message.get("content")))

            delta = choice.get("delta")
            if isinstance(delta, dict):
                parts.extend(_content_parts(delta.get("content")))

            text = choice.get("text")
            if isinstance(text, str):
                parts.append(text)

    message = "".join(parts).strip()
    if message:
        return message
    raise ValueError("response did not include assistant text")


def _strip_json_fence(text: str) -> str:
    text = text.strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text


def _parse_review_payload(raw: str) -> dict:
    message = _strip_json_fence(_extract_chat_message(raw))
    decoder = json.JSONDecoder()
    for index, character in enumerate(message):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(message[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("segments"), list):
            return payload
    raise ValueError("AI response did not contain a JSON object with a segments list")


def _validate_review_payload(payload: dict, segments: list[Segment]) -> tuple[dict[int, str], list[str]]:
    expected = [segment.index for segment in segments]
    expected_set = set(expected)
    seen: set[int] = set()
    by_index: dict[int, str] = {}
    issues: list[str] = []

    raw_segments = payload.get("segments")
    if not isinstance(raw_segments, list):
        return {}, ["response does not include a segments list"]

    for position, item in enumerate(raw_segments, start=1):
        if not isinstance(item, dict):
            issues.append(f"segment payload #{position} is not an object")
            continue
        raw_index = item.get("index")
        if not str(raw_index).isdigit():
            issues.append(f"segment payload #{position} has an invalid index")
            continue
        index = int(raw_index)
        if index not in expected_set:
            issues.append(f"segment index {index} is not part of the input script")
            continue
        if index in seen:
            issues.append(f"segment index {index} is duplicated")
            continue
        seen.add(index)
        text = str(item.get("text") or "").strip()
        if not text:
            issues.append(f"segment index {index} has empty text")
            continue
        by_index[index] = text

    missing = [index for index in expected if index not in by_index]
    if missing:
        issues.append(f"missing segment indexes: {missing[:8]}")
    return by_index, issues


def _detect_review_quality_issues(segments: list[Segment], reviewed_texts: dict[int, str]) -> list[str]:
    issues: list[str] = []
    normalized_seen: dict[str, int] = {}
    for segment in segments:
        text = _normalize_text(reviewed_texts.get(segment.index, ""))
        if not text:
            issues.append(f"segment {segment.index}: empty reviewed text")
            continue

        compact_len = len(re.sub(r"\s+", "", text))
        hard_limit = _hard_non_space_chars(segment.duration)
        if compact_len > hard_limit:
            issues.append(
                f"segment {segment.index}: likely too long for {segment.duration:.2f}s "
                f"({compact_len} non-space characters, hard limit {hard_limit})"
            )

        if len(text) < 2 and len(segment.text.strip()) > 5:
            issues.append(f"segment {segment.index}: reviewed text is suspiciously short")

        latin_words = _latin_words(text)
        if latin_words:
            issues.append(
                f"segment {segment.index}: contains Latin word(s) that should usually "
                f"be transliterated to Khmer: {latin_words[:4]}"
            )

        key = text.casefold()
        previous = normalized_seen.get(key)
        if previous is not None and text:
            issues.append(f"segment {segment.index}: duplicates segment {previous}")
        else:
            normalized_seen[key] = segment.index
    return issues


def _chat_completion_raw(
    payload: dict,
    api_key: str,
    base_url: str,
    timeout: float,
    cancel_event: Event,
) -> str:
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
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8")


def _review_segment_payload(segment: Segment, reviewed_text: str | None = None) -> dict:
    payload = {
        "index": segment.index,
        "duration": round(segment.duration, 3),
        "target_non_space_chars": _target_non_space_chars(segment.duration),
        "hard_non_space_chars": _hard_non_space_chars(segment.duration),
        "speaker": segment.speaker_label or segment.speaker_id or "",
        "source": segment.text,
        "raw_khmer": segment.raw_khmer_text or segment.translated_text,
        "review_notes": segment.review_notes,
    }
    if reviewed_text is not None:
        payload["first_pass_khmer"] = reviewed_text
    return payload


def _payload_with_validation_retry(payload: dict, issues: list[str]) -> dict:
    retry_payload = dict(payload)
    retry_payload["messages"] = [
        *payload.get("messages", []),
        {
            "role": "user",
            "content": json.dumps(
                {
                    "previous_response_rejected": True,
                    "validation_issues": issues[:12],
                    "required_fix": (
                        "Return one complete JSON object with every original segment index "
                        "exactly once. Preserve order. Do not omit the last segments. "
                        "Each text must be natural Khmer and must fit the target/hard "
                        "non-space character limits already provided for each segment."
                    ),
                },
                ensure_ascii=False,
            ),
        },
    ]
    return retry_payload


def _request_review_texts(
    payload: dict,
    segments: list[Segment],
    api_key: str,
    base_url: str,
    timeout: float,
    cancel_event: Event,
    log_cb: LogCallback | None,
    label: str,
) -> dict[int, str] | None:
    request_payload = payload
    for attempt in range(REVIEW_MAX_RETRIES):
        try:
            raw = _chat_completion_raw(request_payload, api_key, base_url, timeout, cancel_event)
        except Exception as e:
            if log_cb:
                error_msg = e.read().decode("utf-8") if hasattr(e, "read") else str(e)
                log_cb(f"{label} request failed: {error_msg}")
            return None

        if cancel_event.is_set():
            raise CancellationError("Processing cancelled by user")

        try:
            reviewed = _parse_review_payload(raw)
            by_index, response_issues = _validate_review_payload(reviewed, segments)
        except Exception as e:
            response_issues = [f"parse error: {e}"]
            by_index = {}
            if log_cb:
                log_cb(f"Failed to parse {label} response: {e}\nRaw output: {raw[:200]}...")

        if not response_issues:
            return by_index

        if attempt < REVIEW_MAX_RETRIES - 1:
            if log_cb:
                log_cb(f"{label} failed validation; retrying: {'; '.join(response_issues[:4])}")
            request_payload = _payload_with_validation_retry(payload, response_issues)
            continue

        if log_cb:
            log_cb(f"{label} failed validation: {'; '.join(response_issues[:4])}")
    return None


def _second_pass_review_with_chat_api(
    segments: list[Segment],
    context: StoryContext,
    style: str,
    reviewed_texts: dict[int, str],
    local_issues: list[str],
    source_language: str | None,
    cancel_event: Event,
    api_key: str,
    model: str,
    base_url: str,
    timeout: float,
    log_cb: LogCallback | None,
    content_style: str,
) -> dict[int, str] | None:
    content_instruction = CONTENT_STYLE_INSTRUCTIONS.get(
        content_style, CONTENT_STYLE_INSTRUCTIONS["casual_vlog"]
    )
    source_label = "Chinese" if source_language == "zh" else source_language or "unknown"
    payload = {
        "model": model,
        "temperature": 0.1,
        "max_tokens": _review_max_tokens(),
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": f"""
You are a Khmer dubbing quality reviewer.

This is a second pass after another AI already rewrote the script.
Review the ENTIRE script before changing any segment.

Highest priority:
1. Fit each line inside its segment duration
2. Preserve meaning
3. Natural spoken Khmer
4. Consistent names and terms
5. Style

This is dubbing audio, not subtitles. The listener will hear it in real time.
Write for easy listening: one clear idea per segment, short spoken sentences,
common Khmer words, and direct subject-verb-object order. Avoid literary Khmer
and avoid dictionary-style translation.

If a segment is weak, rewrite it shorter and more natural.
If a segment is already good, keep the same text.
If first_pass_khmer or raw_khmer is empty, recover it by translating the
source text into natural Khmer.

Check for:
- missing or duplicated indexes
- empty output
- missing meaning
- duplicated meaning
- unnatural Khmer wording
- inconsistent names, places, products, or technical terms
- text too long for the segment duration
- personal names left in Latin letters instead of Khmer script
- target_non_space_chars is the comfortable timing target
- hard_non_space_chars is the absolute maximum; rewrite shorter if over it

Content integrity:
- This can be movie/dialogue content. If the source or first-pass Khmer contains
  profanity, insults, crude slang, sexual references, violence, or other mature
  language, preserve that meaning faithfully in natural Khmer.
- Do not censor, sanitize, moralize, soften, replace with [censored], or refuse
  because of bad words. Keep the same intensity, but do not add new insults,
  slurs, or stronger meaning that was not present in the original.

Content voice:
{content_instruction}

Return ONLY valid JSON.
Do NOT include markdown.
Do NOT explain your reasoning.

Schema:
{{
  "segments":[
    {{
      "index":0,
      "text":"...",
      "notes":""
    }}
  ]
}}

Rules:
- preserve every input index exactly once
- preserve segment order
- do not output any extra fields
- keep every line suitable for Khmer TTS dubbing
- keep every line easy for a Khmer listener to understand on the first listen
- use simple spoken words instead of formal written words unless the content requires formality
- transliterate personal names into Khmer script, for example John -> ចន
- keep Latin letters only for real brands, acronyms, or technical labels
""",
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "style": STYLE_LABELS.get(style, style),
                        "content_style": content_style,
                        "source_language": source_label,
                        "story_summary": context.summary,
                        "terms": context.terms,
                        "speaker_notes": context.speaker_notes,
                        "local_quality_flags": local_issues,
                        "full_source_script": "\n".join(
                            f"{segment.index}: {segment.text}" for segment in segments
                        ),
                        "first_pass_khmer_script": "\n".join(
                            f"{segment.index}: {reviewed_texts.get(segment.index, '')}"
                            for segment in segments
                        ),
                        "segments": [
                            _review_segment_payload(
                                segment, reviewed_texts.get(segment.index, "")
                            )
                            for segment in segments
                        ],
                    },
                    ensure_ascii=False,
                ),
            },
        ],
    }
    revised_texts = _request_review_texts(
        payload,
        segments,
        api_key,
        base_url,
        timeout,
        cancel_event,
        log_cb,
        "AI second-pass review",
    )
    if revised_texts is None:
        return None

    quality_issues = _detect_review_quality_issues(segments, revised_texts)
    if len(quality_issues) > len(local_issues):
        if log_cb:
            log_cb(
                "AI second-pass review made timing/quality warnings worse "
                f"({len(local_issues)} -> {len(quality_issues)}); keeping first pass"
            )
        return None
    if quality_issues and log_cb:
        log_cb(f"AI second-pass kept {len(quality_issues)} timing/quality warning(s)")
    return revised_texts


def _review_with_chat_api(
    segments: list[Segment],
    context: StoryContext,
    style: str,
    source_language: str | None,
    cancel_event: Event,
    log_cb: LogCallback | None = None,
    content_style: str = "casual_vlog",
) -> dict[int, str] | None:
    api_key, base_url, model = resolve_review_api_credentials()
    if not api_key:
        return None
    timeout = float(os.getenv("TRANSCRIPT_REVIEW_TIMEOUT", "120"))
    content_instruction = CONTENT_STYLE_INSTRUCTIONS.get(
        content_style, CONTENT_STYLE_INSTRUCTIONS["casual_vlog"]
    )
    source_label = "Chinese" if source_language == "zh" else source_language or "unknown"
    payload = {
        "model": model,
        "temperature": 0.2,
        "max_tokens": _review_max_tokens(),
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": f"""
You are an expert Khmer dubbing editor.

Your task is NOT to translate.
Your task is to improve an already translated Khmer dubbing script.
Exception: if raw_khmer is empty, or review_notes says AI translation missing,
translate that segment from source into natural Khmer before improving it.

Read the ENTIRE script before editing any segment.
This is dubbing audio, not subtitles. The listener will hear it in real time.

==================================================
PRIMARY GOAL
==================================================

Produce Khmer narration that:

• sounds like a native speaker
• is easy to understand by listening only
• uses simple everyday Khmer whenever possible
• keeps one clear idea per segment
• flows naturally from one segment to the next
• fits inside each segment duration
• can be spoken by TTS without being cut off or rushed

==================================================
STRICT PRIORITY ORDER
==================================================

1. TIMING (highest priority)
2. Preserve meaning
3. Natural spoken Khmer
4. Full-script consistency
5. Style

Never sacrifice timing.

==================================================
TIMING RULES
==================================================

Every segment includes:

duration (seconds)
target_non_space_chars
hard_non_space_chars

Your rewritten Khmer MUST comfortably fit inside that duration.
Use target_non_space_chars as the normal limit.
Never exceed hard_non_space_chars unless the source is only a name, number, or fixed term.

Assume narration speed:

• normal: 4–5 Khmer syllables/sec
• maximum: 1.6× playback
• effective limit: about 6–8 syllables/sec

If a segment is too long:

Rewrite it SHORTER.

Do NOT:

- keep unnecessary modifiers
- repeat information
- use formal wording
- create text that would be cut off during TTS
- preserve literal wording when natural compression is better

A complete short sentence is ALWAYS better than a long sentence.

==================================================
MEANING
==================================================

Do not invent information.

Do not remove important facts.

Compress wording instead of removing meaning.

This can be movie/dialogue content. If the source or first-pass Khmer contains
profanity, insults, crude slang, sexual references, violence, or other mature
language, preserve that meaning faithfully in natural Khmer.

Do not censor, sanitize, moralize, soften, replace with [censored], or refuse
because of bad words. Keep the same intensity, but do not add new insults,
slurs, or stronger meaning that was not present in the original.

==================================================
NATURAL KHMER
==================================================

Use conversational spoken Khmer.
Prefer direct subject-verb-object order and short sentences.
Use common Khmer words over formal/literary words.
Transliterate personal names into Khmer script, for example John -> ចន.
Keep Latin letters only for real brands, acronyms, or technical labels.

Avoid:

- literal translation
- awkward English sentence order
- overly formal writing
- robotic wording

Write as if a Khmer YouTube narrator is speaking.

==================================================
SCRIPT FLOW
==================================================

Consider the whole script.

Each segment should connect naturally.

If needed, add a very short connector such as:

- បន្ទាប់មក
- ខណៈនោះ
- នៅពេលនោះ
- ដូច្នេះ

only when it improves listening.

==================================================
REFERENCES
==================================================

Avoid unclear pronouns.

When a new topic starts, use the person's name or a clear noun once before using pronouns.

==================================================
CONSISTENCY
==================================================

Use identical Khmer spelling for:

- names
- places
- products
- technical terms

throughout the script.

==================================================
TECHNICAL TERMS
==================================================

The first time an uncommon foreign term appears,
briefly explain it naturally.

Later occurrences may simply use the term.

==================================================
STYLE
==================================================

Content voice:
{content_instruction}

==================================================
OUTPUT
==================================================

Return ONLY valid JSON.

Do NOT include markdown.

Do NOT include explanations.

Schema:

{{
  "segments":[
    {{
      "index":0,
      "text":"...",
      "notes":""
    }}
  ]
}}

Rules:

- preserve every index
- preserve segment order
- one output segment for every input segment
- do not output any extra fields
""",
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "style": STYLE_LABELS.get(style, style),
                        "content_style": content_style,
                        "story_summary": context.summary,
                        "terms": context.terms,
                        "speaker_notes": context.speaker_notes,
                        "full_source_script": "\n".join(
                            f"{segment.index}: {segment.text}"
                            for segment in segments
                        ),
                        "full_raw_khmer_script": "\n".join(
                            f"{segment.index}: {segment.raw_khmer_text or segment.translated_text}"
                            for segment in segments
                        ),
                        "segments": [
                            _review_segment_payload(segment)
                            for segment in segments
                        ],
                    },
                    ensure_ascii=False,
                ),
            },
        ],
    }
    by_index = _request_review_texts(
        payload,
        segments,
        api_key,
        base_url,
        timeout,
        cancel_event,
        log_cb,
        "AI review",
    )
    if by_index is None:
        return None

    local_issues = _detect_review_quality_issues(segments, by_index)
    if log_cb:
        if local_issues:
            log_cb(f"AI review first pass returned {len(local_issues)} timing/quality warning(s); running second pass")
        else:
            log_cb("AI review first pass complete; running second-pass quality check")

    revised = _second_pass_review_with_chat_api(
        segments,
        context,
        style,
        by_index,
        local_issues,
        source_language,
        cancel_event,
        api_key,
        model,
        base_url,
        timeout,
        log_cb,
        content_style,
    )
    if revised is not None:
        if log_cb:
            log_cb("AI second-pass review complete")
        return revised
    if log_cb:
        log_cb("Using first-pass AI transcript review result")
    return by_index


def review_segments(
    segments: list[Segment],
    style: str,
    mode: str,
    glossary_path: Path | None,
    review_json_path: Path | None,
    save_json_path: Path | None,
    progress_cb: ProgressCallback | None,
    log_cb: LogCallback | None,
    cancel_event: Event,
    content_style: str = "casual_vlog",
    source_language: str | None = None,
) -> list[Segment]:
    if not segments:
        raise ValueError("No segments available for transcript review")

    if mode == "auto":
        mode = "ai"
    elif mode in {"manual", "skip"}:
        mode = "off"

    if mode not in {"off", "local", "ai"}:
        raise ValueError(f"Unsupported transcript review mode: {mode}")

    for segment in segments:
        segment.raw_khmer_text = segment.raw_khmer_text or segment.translated_text
        segment.improved_khmer_text = segment.improved_khmer_text or segment.translated_text

    context = build_story_context(segments)
    if review_json_path and review_json_path.exists():
        if log_cb:
            log_cb(f"Loading transcript review edits: {review_json_path}")
        if review_json_path.suffix.lower() == ".srt":
            segments = load_review_srt(review_json_path, segments)
        else:
            segments = load_review_json(review_json_path, segments)
        if progress_cb:
            progress_cb(100)
        return segments

    if mode == "off":
        if progress_cb:
            progress_cb(100)
        return segments

    glossary = _load_glossary(glossary_path)
    ai_texts = None
    if mode == "ai":
        if log_cb:
            style_label = CONTENT_STYLE_INSTRUCTIONS.get(content_style, "").split(".")[0]
            log_cb(
                f"Reviewing full transcript with AI reviewer "
                f"(content style: {content_style} — {style_label})"
            )
            if context.terms:
                # Surface the auto-extracted glossary so the user can spot bad
                # extractions in the log without opening the JSON.
                log_cb(f"  Auto-glossary ({len(context.terms)}): {', '.join(context.terms[:10])}"
                       + (" ..." if len(context.terms) > 10 else ""))
        ai_texts = _review_with_chat_api(
            segments,
            context,
            style,
            source_language,
            cancel_event,
            log_cb,
            content_style=content_style,
        )
        if ai_texts is None and log_cb:
            log_cb("AI transcript review unavailable; using local Khmer cleanup")

    for position, segment in enumerate(segments, start=1):
        if cancel_event.is_set():
            raise CancellationError("Processing cancelled by user")
        base_text = ai_texts.get(segment.index, "") if ai_texts is not None else segment.raw_khmer_text or segment.translated_text
        improved = _style_cleanup(_apply_glossary(base_text, glossary), style, segment.duration)
        segment.improved_khmer_text = improved or segment.raw_khmer_text or segment.translated_text
        if _text_non_space_chars(base_text) > _hard_non_space_chars(segment.duration):
            note = "shortened for dubbing timing"
            if ai_texts is None:
                note = "local cleanup; " + note
            segment.review_notes = note
            if log_cb:
                log_cb(
                    f"  Segment {segment.index + 1}: shortened Khmer line for "
                    f"{segment.duration:.2f}s dubbing slot"
                )
        if ai_texts is None:
            segment.review_notes = segment.review_notes or "local cleanup"
        if progress_cb:
            progress_cb(int((position / len(segments)) * 100))

    if save_json_path is not None:
        save_review_json(save_json_path, segments, context, style)
        if log_cb:
            log_cb(f"Saved transcript review JSON: {save_json_path}")
    return segments
