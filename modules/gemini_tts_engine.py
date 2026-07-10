from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
import wave
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Callable

from core.context import CancellationError, Segment


ProgressCallback = Callable[[int], None]
LogCallback = Callable[[str], None]

DEFAULT_GEMINI_TTS_MODEL = "gemini-3.1-flash-tts-preview"
MAX_REQUESTS_PER_VIDEO = 15
DEFAULT_MAX_REQUESTS = MAX_REQUESTS_PER_VIDEO
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_RATE_LIMIT_RETRIES = 20
DEFAULT_MAX_CHUNK_DURATION_SEC = 30.0
DEFAULT_MAX_SEGMENTS_PER_CHUNK = 8
GEMINI_INTERACTIONS_API_REVISION = "2026-05-20"
GEMINI_MAX_CONFIGURED_SPEAKERS = 2
GEMINI_TTS_PROMPT_VERSION = 3
# Adult-sounding presets first (avoid Leda/Youthful, Puck/Upbeat, Zephyr/Bright).
MATURE_PRESET_VOICES = [
    "Gacrux",
    "Charon",
    "Kore",
    "Orus",
    "Algenib",
    "Schedar",
    "Rasalgethi",
    "Sadaltager",
    "Alnilam",
    "Sulafat",
    "Despina",
    "Callirrhoe",
    "Algieba",
    "Erinome",
    "Umbriel",
    "Achird",
]
PRESET_VOICES = [
    *MATURE_PRESET_VOICES,
    "Puck",
    "Fenrir",
    "Aoede",
    "Leda",
    "Zephyr",
    "Autonoe",
    "Enceladus",
    "Iapetus",
    "Laomedeia",
    "Achernar",
    "Pulcherrima",
    "Zubenelgenubi",
    "Vindemiatrix",
    "Sadachbia",
]


class GeminiTTSError(RuntimeError):
    pass


class GeminiTTSRateLimitError(GeminiTTSError):
    def __init__(self, message: str, retry_after_sec: float = 60.0) -> None:
        super().__init__(message)
        self.retry_after_sec = max(1.0, float(retry_after_sec))


class GeminiTTSNoAudioError(GeminiTTSError):
    pass


@dataclass(frozen=True)
class GeminiTTSChunk:
    index: int
    segments: list[Segment]

    @property
    def group_id(self) -> str:
        return f"gemini_chunk_{self.index:03d}"

    @property
    def start(self) -> float:
        return self.segments[0].start

    @property
    def end(self) -> float:
        return self.segments[-1].end

    @property
    def duration(self) -> float:
        return max(0.01, self.end - self.start)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return max(0.0, float(raw))
    except ValueError:
        return default


def _resolve_max_requests(max_requests: int | None) -> int:
    requested = max_requests or _env_int("GEMINI_TTS_MAX_REQUESTS", DEFAULT_MAX_REQUESTS)
    return max(1, min(int(requested), MAX_REQUESTS_PER_VIDEO))


def _max_chunk_duration_sec() -> float:
    return max(5.0, _env_float("GEMINI_TTS_MAX_CHUNK_DURATION_SEC", DEFAULT_MAX_CHUNK_DURATION_SEC))


def _max_segments_per_chunk() -> int:
    return max(1, _env_int("GEMINI_TTS_MAX_SEGMENTS_PER_CHUNK", DEFAULT_MAX_SEGMENTS_PER_CHUNK))


def _chunk_duration(segments: list[Segment]) -> float:
    if not segments:
        return 0.0
    return max(0.01, segments[-1].end - segments[0].start)


def _chunk_within_limits(segments: list[Segment]) -> bool:
    return (
        len(segments) <= _max_segments_per_chunk()
        and _chunk_duration(segments) <= _max_chunk_duration_sec()
    )


def _split_segment_groups(groups: list[list[Segment]]) -> list[list[Segment]]:
    max_segments = _max_segments_per_chunk()
    max_duration = _max_chunk_duration_sec()
    split: list[list[Segment]] = []
    for group in groups:
        if not group:
            continue
        current: list[Segment] = []
        current_duration = 0.0
        for segment in group:
            segment_duration = max(0.01, segment.end - segment.start)
            would_exceed_segments = len(current) >= max_segments
            would_exceed_duration = current and (current_duration + segment_duration) > max_duration
            if current and (would_exceed_segments or would_exceed_duration):
                split.append(current)
                current = [segment]
                current_duration = segment_duration
            else:
                current.append(segment)
                current_duration = _chunk_duration(current)
        if current:
            split.append(current)
    return split


def resolve_gemini_api_keys(explicit_key: str | None = None) -> list[str]:
    """Return ordered, de-duplicated Gemini API keys for TTS requests."""
    keys: list[str] = []
    if explicit_key and explicit_key.strip():
        keys.append(explicit_key.strip())

    for env_name in ("GEMINI_API_KEYS", "GEMINI_API_KEY", "GEMINI_API_KEY_FALLBACK"):
        raw = os.getenv(env_name, "").strip()
        if not raw:
            continue
        if env_name == "GEMINI_API_KEYS":
            keys.extend(part.strip() for part in raw.split(",") if part.strip())
        else:
            keys.append(raw)

    seen: set[str] = set()
    ordered: list[str] = []
    for key in keys:
        if key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    return ordered


def _parse_retry_delay(detail: str, headers: object | None = None) -> float:
    if headers is not None:
        retry_after = getattr(headers, "get", lambda _key, _default=None: None)("Retry-After")
        if retry_after:
            try:
                return max(1.0, float(retry_after))
            except ValueError:
                pass

    match = re.search(r"retry in ([\d.]+)s", detail, flags=re.IGNORECASE)
    if match:
        return max(1.0, float(match.group(1)))

    match = re.search(r"retry after ([\d.]+)", detail, flags=re.IGNORECASE)
    if match:
        return max(1.0, float(match.group(1)))

    return 60.0


def _sleep_with_cancel(seconds: float, cancel_event: Event) -> None:
    remaining = max(0.0, float(seconds))
    while remaining > 0:
        if cancel_event.is_set():
            raise CancellationError("Processing cancelled by user")
        step = min(remaining, 0.5)
        time.sleep(step)
        remaining -= step


def _has_khmer(text: str) -> bool:
    return any("\u1780" <= ch <= "\u17ff" for ch in text)


def _speaker_key(segment: Segment) -> str:
    return segment.speaker_id or segment.speaker_label or "default"


def gemini_voices_from_mappings(mappings: dict[str, dict[str, str]] | None) -> dict[str, str]:
    """Load persisted Gemini preset voices for each speaker_id."""
    saved: dict[str, str] = {}
    if not mappings:
        return saved
    preset_set = set(PRESET_VOICES)
    for speaker_id, entry in mappings.items():
        if not isinstance(entry, dict):
            continue
        voice = str(entry.get("gemini_voice") or "").strip()
        if voice in preset_set:
            saved[speaker_id] = voice
    return saved


def _speaker_first_appearance(segments: list[Segment]) -> list[str]:
    """Order speakers by their first line in the video for stable voice assignment."""
    first_seen: dict[str, float] = {}
    for segment in segments:
        speaker = _speaker_key(segment)
        if speaker not in first_seen:
            first_seen[speaker] = segment.start
    return sorted(first_seen.keys(), key=lambda speaker: (first_seen[speaker], speaker))


def _apply_gemini_voices_to_mappings(
    mappings: dict[str, dict[str, str]] | None,
    voices: dict[str, str],
) -> None:
    if not mappings:
        return
    for speaker_id, voice in voices.items():
        entry = mappings.setdefault(speaker_id, {})
        entry["gemini_voice"] = voice
        entry.setdefault("label", speaker_id.replace("_", " ").title())


def _speaker_count(segments: list[Segment]) -> int:
    return len({_speaker_key(segment) for segment in segments})


def group_segments_for_gemini(segments: list[Segment], max_requests: int = DEFAULT_MAX_REQUESTS) -> list[GeminiTTSChunk]:
    active = [segment for segment in segments if segment.enabled and segment.tts_text.strip()]
    if not active:
        raise GeminiTTSError("No enabled Khmer text segments available for Gemini TTS")
    max_requests = max(1, int(max_requests))

    grouped: list[list[Segment]] = []
    current: list[Segment] = []
    current_speakers: set[str] = set()
    for segment in active:
        speaker = _speaker_key(segment)
        would_add_third_speaker = (
            current
            and speaker not in current_speakers
            and len(current_speakers) >= GEMINI_MAX_CONFIGURED_SPEAKERS
        )
        if would_add_third_speaker:
            grouped.append(current)
            current = [segment]
            current_speakers = {speaker}
        else:
            current.append(segment)
            current_speakers.add(speaker)
    if current:
        grouped.append(current)

    target_size = max(1, min(_max_segments_per_chunk(), (len(active) + max_requests - 1) // max_requests))
    sized_groups: list[list[Segment]] = []
    for group in grouped:
        for start in range(0, len(group), target_size):
            sized_groups.append(group[start:start + target_size])
    grouped = _split_segment_groups(sized_groups)

    while len(grouped) > max_requests:
        mergeable = [
            index
            for index in range(len(grouped) - 1)
            if _chunk_within_limits(grouped[index] + grouped[index + 1])
        ]
        if not mergeable:
            break
        merge_index = min(
            mergeable,
            key=lambda index: (
                _speaker_count(grouped[index] + grouped[index + 1]),
                len(grouped[index]) + len(grouped[index + 1]),
            ),
        )
        grouped[merge_index] = grouped[merge_index] + grouped[merge_index + 1]
        del grouped[merge_index + 1]

    return [GeminiTTSChunk(index=index, segments=chunk) for index, chunk in enumerate(grouped)]


def speaker_voice_map(
    segments: list[Segment],
    saved_voices: dict[str, str] | None = None,
) -> dict[str, str]:
    """Assign one Gemini preset voice per speaker, stable across reruns."""
    saved = dict(saved_voices or {})
    speakers = _speaker_first_appearance(segments)
    assigned: dict[str, str] = {}
    used_voices: set[str] = set()

    for speaker in speakers:
        voice = saved.get(speaker)
        if voice in PRESET_VOICES:
            assigned[speaker] = voice
            used_voices.add(voice)

    pool_index = 0
    voice_pool = MATURE_PRESET_VOICES
    for speaker in speakers:
        if speaker in assigned:
            continue
        while pool_index < len(voice_pool):
            candidate = voice_pool[pool_index % len(voice_pool)]
            pool_index += 1
            if candidate not in used_voices:
                assigned[speaker] = candidate
                used_voices.add(candidate)
                break
        else:
            assigned[speaker] = voice_pool[len(assigned) % len(voice_pool)]

    return assigned


def _emotion_note(segment: Segment, emotion_analyses: dict[int, object] | None) -> str:
    analysis = emotion_analyses.get(segment.index) if emotion_analyses else None
    if analysis is not None and not bool(getattr(analysis, "is_neutral_fallback", False)):
        label = str(getattr(analysis, "label", "") or "emotional").strip()
        instruction = str(getattr(analysis, "instruct_text", "") or "").strip()
        confidence = getattr(analysis, "confidence", None)
        if isinstance(confidence, (int, float)):
            return f"{label} delivery, confidence {confidence:.2f}. {instruction}".strip()
        return f"{label} delivery. {instruction}".strip()

    text = segment.tts_text.strip()
    source_text = segment.text.strip()
    if "?" in text or "?" in source_text:
        return "questioning delivery with natural rising Khmer intonation"
    if "!" in text or "!" in source_text:
        return "strong emotional emphasis with controlled energy"
    if "..." in text or "…" in text:
        return "hesitant delivery with a short natural pause"
    return "infer emotion from the Khmer words and scene context; keep it natural"


def _emotion_tag(segment: Segment, emotion_analyses: dict[int, object] | None) -> str:
    analysis = emotion_analyses.get(segment.index) if emotion_analyses else None
    label = ""
    instruction = ""
    if analysis is not None and not bool(getattr(analysis, "is_neutral_fallback", False)):
        label = str(getattr(analysis, "label", "") or "").lower()
        instruction = str(getattr(analysis, "instruct_text", "") or "").lower()

    text = f"{label} {instruction} {segment.tts_text} {segment.text}".lower()
    if any(word in text for word in ("angry", "anger", "furious", "frustrated", "threat", "ខឹង")):
        return "angry, sharp consonants, controlled intensity, not shouting unless the line has an exclamation"
    if any(word in text for word in ("sad", "cry", "grief", "hurt", "regret", "សោក", "យំ")):
        return "sad, soft, restrained, slightly trembling, slower pace"
    if any(word in text for word in ("fear", "scared", "panic", "terrified", "afraid", "ភ័យ")):
        return "panicked, trembling, tense, slightly faster pace"
    if any(word in text for word in ("happy", "joy", "excited", "laugh", "smile", "សប្បាយ")):
        return "excited, warm, lively, natural smile in the voice"
    if any(word in text for word in ("surprise", "shocked", "amazed", "wow", "ភ្ញាក់")):
        return "amazed, quick reaction, lifted intonation"
    if any(word in text for word in ("whisper", "secret", "quiet", "ស្ងាត់")):
        return "whispers, intimate, careful, close-mic feeling"
    if any(word in text for word in ("tired", "weak", "exhausted", "sick", "ហត់")):
        return "tired, low energy, breathy, slower pace"
    if "?" in segment.tts_text or "?" in segment.text:
        return "curious, conversational, natural rising Khmer question intonation"
    if "!" in segment.tts_text or "!" in segment.text:
        return "energetic, urgent, clear emphasis"
    if "..." in segment.tts_text or "…" in segment.tts_text:
        return "hesitant, thoughtful, short pause before continuing"
    return "native Cambodian Khmer, mature adult 25-40, grounded conversational tone, emotionally natural, medium pace"


def build_gemini_tts_prompt(
    chunk: GeminiTTSChunk,
    voices: dict[str, str],
    emotion_analyses: dict[int, object] | None = None,
) -> str:
    lines = []
    delivery_notes = []
    speaker_rows: list[str] = []
    seen_speakers: set[str] = set()
    for segment in chunk.segments:
        speaker = _speaker_key(segment)
        label = segment.speaker_label or speaker.replace("_", " ").title()
        voice = voices.get(speaker, PRESET_VOICES[0])
        if speaker not in seen_speakers:
            seen_speakers.add(speaker)
            speaker_rows.append(
                f"- {label}: Gemini voice {voice}; mature adult (25-40 years old), grounded chest voice, "
                "not teen-like or child-like; keep a distinct texture, energy, and rhythm from other speakers."
            )
        timing = f"[{segment.start:.2f}-{segment.end:.2f}]"
        delivery_notes.append(f"- {timing} {label}: {_emotion_note(segment, emotion_analyses)}")
        lines.append(f"{timing} {label}: [{_emotion_tag(segment, emotion_analyses)}] {segment.tts_text.strip()}")

    return (
        "Synthesize speech only. Do not say stage directions, timestamps, speaker labels, voice names, "
        "emotion tags, explanations, markdown, or any text that is not Khmer dialogue.\n"
        "Do not translate, summarize, rewrite, add extra words, add Chinese, add English, or add sound effects. "
        "Speak only the Khmer dialogue exactly as written after each emotion tag.\n\n"
        "Audio profile:\n"
        "- Language: Khmer only, Cambodian Khmer pronunciation.\n"
        "- Performance type: professional film/video dubbing, close to a native Cambodian adult actor replacing the original voice.\n"
        "- Age target: every speaker must sound like a mature adult (roughly 25-45 years old), never child-like, teen-like, or cartoonish.\n"
        "- Quality target: natural human conversation with emotion, not narration, announcement, reading voice, or robotic TTS.\n\n"
        "Character voices:\n"
        "- These are preset Gemini voices, not cloned voices from the original video.\n"
        "- Give each speaker a clearly different mature adult voice color, energy level, and speaking rhythm.\n"
        "- Keep the same speaker consistent across all lines in this chunk.\n"
        "- Use the speaker label only to choose the voice. Never read the speaker label aloud.\n"
        "- Avoid bright, bubbly, or youthful delivery unless a line's emotion tag explicitly calls for it.\n\n"
        "Scene and director notes:\n"
        "- Treat this as a dubbed movie scene, not an audiobook. The voice should feel connected to the on-screen moment.\n"
        "- Match emotion line by line using the bracketed audio tags. The tags are performance controls only; never speak them.\n"
        "- Use Cambodian Khmer sentence stress, final-particle softness, natural vowel length, and realistic question rises.\n"
        "- Calm lines should breathe and stay relaxed. Urgent lines should become tighter and faster. Sad lines should soften. "
        "Angry lines should become sharper but controlled. Fearful lines should sound tense. Questions should rise naturally.\n"
        "- Keep speech clear and easy to understand for Khmer listeners; avoid foreign accent, monotone delivery, or unnatural syllable timing.\n"
        "- Add only natural micro-pauses and breath timing from the performance; do not add laughs, gasps, filler words, or extra dialogue unless written.\n\n"
        f"Timing target: generate one continuous chunk that naturally fits about {chunk.duration:.2f} seconds. "
        "Follow the line order and approximate each timestamp window, but never read timestamps aloud. "
        "Leave natural silence for gaps between lines where the timing suggests a pause. "
        "If the text is too long for a window, speak a little faster while keeping Khmer understandable.\n\n"
        "Speaker voice map:\n"
        + "\n".join(speaker_rows)
        + "\n\nLine delivery notes, for direction only, do not speak these notes:\n"
        + "\n".join(delivery_notes)
        + "\n\nTranscript to perform:\n"
        + "\n".join(lines)
    )


def _speech_config_for_chunk(chunk: GeminiTTSChunk, voices: dict[str, str]) -> list[dict[str, str]]:
    speakers: list[tuple[str, str]] = []
    for segment in chunk.segments:
        speaker = _speaker_key(segment)
        label = segment.speaker_label or speaker.replace("_", " ").title()
        voice = voices.get(speaker, PRESET_VOICES[0])
        if (label, voice) not in speakers:
            speakers.append((label, voice))

    if len(speakers) == 1:
        return [{"voice": speakers[0][1]}]

    # Gemini multi-speaker TTS is limited, so configure the first two
    # speakers and keep the full speaker map in the prompt.
    return [
        {"speaker": label, "voice": voice}
        for label, voice in speakers[:2]
    ]


def build_gemini_payload(
    chunk: GeminiTTSChunk,
    voices: dict[str, str],
    emotion_analyses: dict[int, object] | None = None,
    model: str = DEFAULT_GEMINI_TTS_MODEL,
) -> dict:
    return {
        "model": model,
        "input": build_gemini_tts_prompt(chunk, voices, emotion_analyses),
        "response_format": {"type": "audio"},
        "generation_config": {
            "speech_config": _speech_config_for_chunk(chunk, voices),
        },
    }


def _emotion_cache_material(segment: Segment, emotion_analyses: dict[int, object] | None) -> dict[str, object]:
    analysis = emotion_analyses.get(segment.index) if emotion_analyses else None
    if analysis is None:
        return {}
    return {
        "label": str(getattr(analysis, "label", "") or ""),
        "instruction": str(getattr(analysis, "instruct_text", "") or ""),
        "neutral": bool(getattr(analysis, "is_neutral_fallback", False)),
    }


def _chunk_cache_name(
    chunk: GeminiTTSChunk,
    model: str,
    voices: dict[str, str],
    emotion_analyses: dict[int, object] | None = None,
) -> str:
    material = {
        "prompt_version": GEMINI_TTS_PROMPT_VERSION,
        "model": model,
        "start": round(chunk.start, 3),
        "end": round(chunk.end, 3),
        "voices": voices,
        "segments": [
            {
                "index": segment.index,
                "start": round(segment.start, 3),
                "end": round(segment.end, 3),
                "speaker": _speaker_key(segment),
                "text": segment.tts_text.strip(),
                "emotion": _emotion_cache_material(segment, emotion_analyses),
            }
            for segment in chunk.segments
        ],
    }
    digest = hashlib.sha256(json.dumps(material, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    return f"{chunk.group_id}_{digest[:16]}.wav"


def _sample_rate_from_mime(mime_type: str) -> int:
    match = re.search(r"rate=(\d+)", mime_type or "")
    if match:
        return int(match.group(1))
    return DEFAULT_SAMPLE_RATE


def _is_audio_content_item(item: dict) -> bool:
    item_type = str(item.get("type") or "").strip().lower()
    mime_type = str(item.get("mime_type") or item.get("mimeType") or "").strip().lower()
    if item_type == "audio":
        return True
    if mime_type.startswith("audio/"):
        return True
    if item.get("data") or item.get("uri"):
        return mime_type.startswith("audio/") or item_type == "audio"
    return False


def _fetch_audio_uri(uri: str, api_key: str, timeout: int = 120) -> bytes:
    request = urllib.request.Request(
        uri,
        headers={
            "x-goog-api-key": api_key,
            "Api-Revision": GEMINI_INTERACTIONS_API_REVISION,
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        message = detail or str(exc)
        if exc.code == 429:
            raise GeminiTTSRateLimitError(
                f"Gemini TTS audio URI HTTP 429: {message[:500]}",
                retry_after_sec=_parse_retry_delay(detail, exc.headers),
            ) from exc
        raise GeminiTTSError(f"Gemini TTS audio URI HTTP {exc.code}: {message[:500]}") from exc


def _audio_content_bytes(item: dict, api_key: str | None = None) -> tuple[bytes, int]:
    mime_type = str(item.get("mime_type") or item.get("mimeType") or "")
    sample_rate = _sample_rate_from_mime(mime_type)
    if item.get("sample_rate") or item.get("sampleRate"):
        try:
            sample_rate = int(item.get("sample_rate") or item.get("sampleRate"))
        except (TypeError, ValueError):
            pass

    data = item.get("data")
    if data:
        return base64.b64decode(data), sample_rate

    uri = str(item.get("uri") or "").strip()
    if uri:
        if not api_key:
            raise GeminiTTSError("Gemini TTS audio URI requires an API key to download")
        return _fetch_audio_uri(uri, api_key), sample_rate

    return b"", sample_rate


def _response_audio_block(response: dict) -> dict:
    output_audio = response.get("output_audio") or response.get("outputAudio") or {}
    if isinstance(output_audio, dict) and output_audio.get("data"):
        return output_audio
    interaction = response.get("interaction") or {}
    if isinstance(interaction, dict):
        output_audio = interaction.get("output_audio") or interaction.get("outputAudio") or {}
        if isinstance(output_audio, dict) and output_audio.get("data"):
            return output_audio
    return {}


def _audio_blocks_from_steps(response: dict, api_key: str | None = None) -> list[tuple[bytes, int]]:
    blocks: list[tuple[bytes, int]] = []
    for step in response.get("steps") or []:
        if not isinstance(step, dict):
            continue
        step_type = str(step.get("type") or "").strip()
        if step_type and step_type != "model_output":
            continue
        for item in step.get("content") or []:
            if not isinstance(item, dict) or not _is_audio_content_item(item):
                continue
            pcm_bytes, sample_rate = _audio_content_bytes(item, api_key)
            if pcm_bytes:
                blocks.append((pcm_bytes, sample_rate))
    return blocks


def _gemini_response_diagnostic(response: dict) -> str:
    details: list[str] = []
    status = str(response.get("status") or "").strip()
    if status and status not in {"completed", "COMPLETED"}:
        details.append(f"status={status}")

    error = response.get("error") or {}
    if isinstance(error, dict):
        message = str(error.get("message") or "").strip()
        status = str(error.get("status") or "").strip()
        if message or status:
            details.append(f"error={status or 'unknown'} {message}".strip())

    output_text = str(response.get("output_text") or response.get("outputText") or "").strip()
    if output_text:
        details.append(f"returned text instead of audio: {output_text[:180]}")

    for step in response.get("steps") or []:
        if not isinstance(step, dict):
            continue
        for item in step.get("content") or []:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type") or "").strip().lower()
            text = str(item.get("text") or "").strip()
            if text and item_type in {"", "text"}:
                details.append(f"returned text instead of audio: {text[:180]}")
                break
        if details and details[-1].startswith("returned text"):
            break

    prompt_feedback = response.get("promptFeedback") or response.get("prompt_feedback") or {}
    if isinstance(prompt_feedback, dict):
        block_reason = str(prompt_feedback.get("blockReason") or prompt_feedback.get("block_reason") or "").strip()
        if block_reason:
            details.append(f"blocked={block_reason}")

    for index, candidate in enumerate(response.get("candidates") or []):
        finish_reason = str(candidate.get("finishReason") or candidate.get("finish_reason") or "").strip()
        if finish_reason:
            details.append(f"candidate {index + 1} finishReason={finish_reason}")
        content = candidate.get("content") or {}
        text_parts = [
            str(part.get("text") or "").strip()
            for part in content.get("parts") or []
            if str(part.get("text") or "").strip()
        ]
        if text_parts:
            details.append(f"candidate {index + 1} text={text_parts[0][:180]}")

    return "; ".join(details)


def _extract_audio_data(response: dict, api_key: str | None = None) -> tuple[bytes, int]:
    output_audio = _response_audio_block(response)
    if output_audio:
        pcm_bytes, sample_rate = _audio_content_bytes(output_audio, api_key)
        if pcm_bytes:
            return pcm_bytes, sample_rate

    step_blocks = _audio_blocks_from_steps(response, api_key)
    if step_blocks:
        sample_rate = step_blocks[0][1]
        return b"".join(block for block, _rate in step_blocks), sample_rate

    candidates = response.get("candidates") or []
    for candidate in candidates:
        content = candidate.get("content") or {}
        for part in content.get("parts") or []:
            inline_data = part.get("inlineData") or part.get("inline_data") or {}
            data = inline_data.get("data")
            if data:
                mime_type = str(inline_data.get("mimeType") or inline_data.get("mime_type") or "")
                return base64.b64decode(data), _sample_rate_from_mime(mime_type)
    diagnostic = _gemini_response_diagnostic(response)
    message = "Gemini TTS response did not contain audio data"
    if diagnostic:
        message = f"{message}: {diagnostic}"
    raise GeminiTTSNoAudioError(message)


def write_pcm_wav(path: Path, pcm_bytes: bytes, sample_rate: int = DEFAULT_SAMPLE_RATE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm_bytes)
    return path


def call_gemini_tts(payload: dict, api_key: str, model: str, timeout: int = 120) -> dict:
    url = "https://generativelanguage.googleapis.com/v1beta/interactions"
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
            "Api-Revision": GEMINI_INTERACTIONS_API_REVISION,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        message = detail or str(exc)
        if exc.code == 429:
            raise GeminiTTSRateLimitError(
                f"Gemini TTS HTTP 429: {message[:500]}",
                retry_after_sec=_parse_retry_delay(detail, exc.headers),
            ) from exc
        raise GeminiTTSError(f"Gemini TTS HTTP {exc.code}: {message[:500]}") from exc


def _call_with_retries(
    payload: dict,
    api_keys: list[str],
    model: str,
    cancel_event: Event,
    transport: Callable[[dict, str, str], dict] | None,
    log_cb: LogCallback | None = None,
    *,
    start_key_index: int = 0,
) -> tuple[dict, str]:
    if not api_keys:
        raise GeminiTTSError("Gemini TTS requires at least one API key.")

    last_error: Exception | None = None
    rate_limit_cycles = 0
    max_rate_limit_cycles = _env_int("GEMINI_TTS_RATE_LIMIT_RETRIES", DEFAULT_RATE_LIMIT_RETRIES)
    transient_attempts = 0
    max_transient_attempts = 3
    key_index = start_key_index % len(api_keys)
    keys_tried_this_cycle = 0

    while True:
        if cancel_event.is_set():
            raise CancellationError("Processing cancelled by user")

        api_key = api_keys[key_index]
        try:
            if transport is not None:
                response = transport(payload, api_key, model)
            else:
                response = call_gemini_tts(payload, api_key, model)
            return response, api_key
        except GeminiTTSRateLimitError as exc:
            last_error = exc
            keys_tried_this_cycle += 1
            if keys_tried_this_cycle < len(api_keys):
                key_index = (key_index + 1) % len(api_keys)
                if log_cb:
                    log_cb(
                        "  Gemini TTS rate limited; switching to fallback API key "
                        f"({key_index + 1}/{len(api_keys)})"
                    )
                continue

            rate_limit_cycles += 1
            if rate_limit_cycles > max_rate_limit_cycles:
                break
            wait_sec = exc.retry_after_sec
            if log_cb:
                log_cb(
                    "  Gemini TTS rate limited on all API keys; "
                    f"waiting {wait_sec:.0f}s before retry "
                    f"({rate_limit_cycles}/{max_rate_limit_cycles})"
                )
            _sleep_with_cancel(wait_sec, cancel_event)
            key_index = start_key_index % len(api_keys)
            keys_tried_this_cycle = 0
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, GeminiTTSError) as exc:
            if isinstance(exc, GeminiTTSNoAudioError):
                raise
            last_error = exc
            transient_attempts += 1
            if transient_attempts >= max_transient_attempts:
                break
            wait_sec = 1.5 * transient_attempts
            if log_cb:
                log_cb(
                    "  Gemini TTS request failed; retrying in "
                    f"{wait_sec:.1f}s ({transient_attempts}/{max_transient_attempts})"
                )
            _sleep_with_cancel(wait_sec, cancel_event)

    raise GeminiTTSError(f"Gemini TTS request failed: {last_error}") from last_error


def _request_chunk_audio(
    payload: dict,
    api_keys: list[str],
    model: str,
    cancel_event: Event,
    transport: Callable[[dict, str, str], dict] | None,
    log_cb: LogCallback | None = None,
    *,
    start_key_index: int = 0,
) -> tuple[bytes, int]:
    no_audio_attempts = 0
    max_no_audio_attempts = max(2, len(api_keys))
    key_index = start_key_index % len(api_keys)

    while True:
        if cancel_event.is_set():
            raise CancellationError("Processing cancelled by user")

        try:
            response, used_key = _call_with_retries(
                payload,
                api_keys,
                model,
                cancel_event,
                transport,
                log_cb=log_cb,
                start_key_index=key_index,
            )
            return _extract_audio_data(response, used_key)
        except GeminiTTSNoAudioError as exc:
            last_error = exc
            no_audio_attempts += 1
            if no_audio_attempts < len(api_keys):
                key_index = (key_index + 1) % len(api_keys)
                if log_cb:
                    log_cb(
                        "  Gemini TTS response had no audio; switching to fallback API key "
                        f"({key_index + 1}/{len(api_keys)})"
                    )
                continue
            if no_audio_attempts >= max_no_audio_attempts:
                break
            wait_sec = 2.0 * no_audio_attempts
            if log_cb:
                log_cb(
                    "  Gemini TTS response had no audio; retrying in "
                    f"{wait_sec:.1f}s ({no_audio_attempts}/{max_no_audio_attempts})"
                )
            _sleep_with_cancel(wait_sec, cancel_event)
            key_index = start_key_index % len(api_keys)

    raise GeminiTTSNoAudioError(str(last_error)) from last_error


def _synthesize_chunk_audio(
    chunk: GeminiTTSChunk,
    voices: dict[str, str],
    api_keys: list[str],
    model: str,
    cancel_event: Event,
    transport: Callable[[dict, str, str], dict] | None,
    emotion_analyses: dict[int, object] | None,
    log_cb: LogCallback | None,
    *,
    start_key_index: int = 0,
) -> tuple[bytes, int]:
    payload = build_gemini_payload(chunk, voices, emotion_analyses, model)
    try:
        return _request_chunk_audio(
            payload,
            api_keys,
            model,
            cancel_event,
            transport,
            log_cb=log_cb,
            start_key_index=start_key_index,
        )
    except GeminiTTSNoAudioError as exc:
        if len(chunk.segments) <= 1:
            raise GeminiTTSError(str(exc)) from exc

        midpoint = len(chunk.segments) // 2
        left = GeminiTTSChunk(index=chunk.index, segments=chunk.segments[:midpoint])
        right = GeminiTTSChunk(index=chunk.index, segments=chunk.segments[midpoint:])
        if log_cb:
            log_cb(
                "  Gemini TTS could not synthesize "
                f"{len(chunk.segments)} segment(s) / {chunk.duration:.1f}s in one request; "
                f"splitting into {len(left.segments)} + {len(right.segments)} segment(s)"
            )

        left_audio, sample_rate = _synthesize_chunk_audio(
            left,
            voices,
            api_keys,
            model,
            cancel_event,
            transport,
            emotion_analyses,
            log_cb,
            start_key_index=start_key_index,
        )
        right_audio, right_rate = _synthesize_chunk_audio(
            right,
            voices,
            api_keys,
            model,
            cancel_event,
            transport,
            emotion_analyses,
            log_cb,
            start_key_index=(start_key_index + 1) % len(api_keys),
        )
        if right_rate != sample_rate:
            raise GeminiTTSError(
                f"Gemini TTS split chunk sample-rate mismatch: {sample_rate} vs {right_rate}"
            )
        return left_audio + right_audio, sample_rate


def synthesize_gemini_tts(
    segments: list[Segment],
    work_dir: Path,
    progress_cb: ProgressCallback | None,
    log_cb: LogCallback | None,
    cancel_event: Event,
    *,
    persistent_cache_dir: Path | None = None,
    max_requests: int | None = None,
    api_key: str | None = None,
    model: str | None = None,
    transport: Callable[[dict, str, str], dict] | None = None,
    emotion_analyses: dict[int, object] | None = None,
    speaker_voice_mappings: dict[str, dict[str, str]] | None = None,
) -> list[Segment]:
    api_keys = resolve_gemini_api_keys(api_key)
    if not api_keys:
        raise GeminiTTSError(
            "Gemini TTS requires GEMINI_API_KEY (or GEMINI_API_KEY_FALLBACK / GEMINI_API_KEYS) to be set."
        )

    model = (model or os.getenv("GEMINI_TTS_MODEL", DEFAULT_GEMINI_TTS_MODEL)).strip()
    max_requests = _resolve_max_requests(max_requests)

    invalid = [
        segment.index + 1
        for segment in segments
        if segment.enabled and segment.tts_text.strip() and not _has_khmer(segment.tts_text)
    ]
    if invalid:
        raise GeminiTTSError(
            "Gemini TTS refused non-Khmer text in segment(s): "
            + ", ".join(str(index) for index in invalid[:20])
        )

    chunks = group_segments_for_gemini(segments, max_requests)
    chunk_segments = [segment for chunk in chunks for segment in chunk.segments]
    saved_voices = gemini_voices_from_mappings(speaker_voice_mappings)
    voices = speaker_voice_map(chunk_segments, saved_voices=saved_voices)
    _apply_gemini_voices_to_mappings(speaker_voice_mappings, voices)
    cache_dir = (persistent_cache_dir / "gemini_tts") if persistent_cache_dir else (work_dir / "gemini_tts")
    cache_dir.mkdir(parents=True, exist_ok=True)

    if log_cb:
        key_note = f"{len(api_keys)} API key(s)" if len(api_keys) > 1 else "1 API key"
        log_cb(
            f"Generating expressive Khmer audio with Gemini TTS "
            f"({model}, {len(chunks)} request(s), max {max_requests} per video, {key_note})"
        )
        log_cb(
            "  Chunk limits: "
            f"max {_max_segments_per_chunk()} segment(s), "
            f"max {_max_chunk_duration_sec():.0f}s per request"
        )
        real_speakers = [speaker for speaker in voices if speaker != "default"]
        if real_speakers:
            log_cb(f"  One Gemini preset voice per speaker ({len(real_speakers)} speaker(s)):")
            for speaker_id in _speaker_first_appearance(chunk_segments):
                if speaker_id in voices:
                    log_cb(f"    {speaker_id} → {voices[speaker_id]}")
        else:
            log_cb(
                "  No speaker IDs on segments — all lines share one Gemini voice. "
                "Use Voice Mode: Auto per-speaker voices to split speakers first."
            )

    for chunk in chunks:
        if cancel_event.is_set():
            raise CancellationError("Processing cancelled by user")
        output_path = cache_dir / _chunk_cache_name(chunk, model, voices, emotion_analyses)
        if not output_path.exists():
            if log_cb:
                log_cb(
                    f"  Gemini TTS chunk {chunk.index + 1}/{len(chunks)}: "
                    f"segments {chunk.segments[0].index + 1}-{chunk.segments[-1].index + 1}, "
                    f"{chunk.duration:.2f}s target"
                )
            audio_bytes, sample_rate = _synthesize_chunk_audio(
                chunk,
                voices,
                api_keys,
                model,
                cancel_event,
                transport,
                emotion_analyses,
                log_cb,
                start_key_index=chunk.index % len(api_keys),
            )
            write_pcm_wav(output_path, audio_bytes, sample_rate)
        elif log_cb:
            log_cb(f"  Using cached Gemini TTS chunk {chunk.index + 1}/{len(chunks)}")

        for segment in chunk.segments:
            segment.tts_path = output_path
            segment.tts_group_id = chunk.group_id

        if progress_cb:
            progress_cb(int(((chunk.index + 1) / len(chunks)) * 100))

    return segments
