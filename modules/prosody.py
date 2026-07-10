from __future__ import annotations

import re
from dataclasses import dataclass
from statistics import median


# Source-side character counting. Chinese caption text is measured in
# characters, not words — one hanzi is roughly one syllable, so chars/second
# is a decent proxy for speaking rate.
_WHITESPACE_RE = re.compile(r"\s+")

# Clamp per-speaker rate offsets so a single very short outlier segment can't
# push a speaker to +40% (which would make the Khmer TTS sound comical).
MIN_RATE_OFFSET_PCT = -12
MAX_RATE_OFFSET_PCT = 12

# Emphasis carryover thresholds. Only apply when the punctuation is emphatic
# — a comma with a question mark buried in Cantonese-style captions doesn't
# count.
EXCLAMATION_RE = re.compile(r"[！!]")
QUESTION_RE = re.compile(r"[？?]")

EXCLAMATION_RATE_DELTA = 5   # slightly faster on ! for energy
EXCLAMATION_PITCH_DELTA = 10  # +10Hz on ! for brightness
QUESTION_PITCH_DELTA = 8      # +8Hz on ? for the rising-tone feel


@dataclass(frozen=True)
class SpeakerRateProfile:
    """Per-speaker speech-rate offset relative to the crowd median.

    A positive ``rate_offset_pct`` means "this speaker talks faster than the
    average person in this video; nudge the Khmer TTS to match."
    """
    speaker_id: str
    chars_per_second: float
    rate_offset_pct: int


def _clean_source_text(text: str) -> str:
    return _WHITESPACE_RE.sub("", text or "")


def _segment_chars_per_second(segment) -> float | None:
    duration = getattr(segment, "duration", None)
    if duration is None:
        duration = max(0.001, float(segment.end - segment.start))
    if duration <= 0.05:
        return None
    text = _clean_source_text(getattr(segment, "text", "") or "")
    if not text:
        return None
    return len(text) / float(duration)


def compute_speaker_rate_profiles(segments) -> dict[str, SpeakerRateProfile]:
    """Measure each speaker's chars-per-second on the *source* audio and
    produce a per-speaker rate offset relative to the median speaker.

    We deliberately pool across all segments per speaker rather than treating
    each segment independently — one very short segment (a stray "yes.") has
    unstable timing, and we want the offset that reflects the whole video.
    """
    if not segments:
        return {}

    per_speaker: dict[str, list[float]] = {}
    for segment in segments:
        speaker_id = getattr(segment, "speaker_id", None)
        if not speaker_id:
            continue
        cps = _segment_chars_per_second(segment)
        if cps is None:
            continue
        per_speaker.setdefault(speaker_id, []).append(cps)

    if not per_speaker:
        return {}

    # Median-of-medians so one chatty speaker with many segments doesn't
    # dominate the crowd baseline.
    speaker_medians = {sid: median(vals) for sid, vals in per_speaker.items() if vals}
    crowd_median = median(speaker_medians.values())
    if crowd_median <= 0:
        return {}

    profiles: dict[str, SpeakerRateProfile] = {}
    for speaker_id, cps in speaker_medians.items():
        # Rate offset is roughly the % faster/slower this speaker is vs the
        # crowd. Scaled by 0.6 because a 20% source rate difference tends to
        # sound like only ~12% in the dub (Khmer is more syllable-dense than
        # Mandarin per unit meaning).
        raw_offset = int(round((cps / crowd_median - 1.0) * 60.0))
        offset = max(MIN_RATE_OFFSET_PCT, min(MAX_RATE_OFFSET_PCT, raw_offset))
        profiles[speaker_id] = SpeakerRateProfile(
            speaker_id=speaker_id,
            chars_per_second=cps,
            rate_offset_pct=offset,
        )
    return profiles


def emphasis_from_source(source_text: str) -> tuple[int, int]:
    """Look at the source (Chinese) text for emotional cues and return
    (rate_offset_pct, pitch_offset_hz) to nudge the TTS."""
    if not source_text:
        return 0, 0

    has_exclaim = bool(EXCLAMATION_RE.search(source_text))
    has_question = bool(QUESTION_RE.search(source_text))
    # Exclamation dominates when both are present ("really?!") — the more
    # emphatic reading is nearly always the correct one for a dub.
    if has_exclaim:
        return EXCLAMATION_RATE_DELTA, EXCLAMATION_PITCH_DELTA
    if has_question:
        return 0, QUESTION_PITCH_DELTA
    return 0, 0


def per_segment_prosody(
    segment,
    base_rate_pct: int,
    base_pitch_hz: int,
    profiles: dict[str, SpeakerRateProfile] | None,
    emotion_analyses: dict[int, object] | None = None,
) -> tuple[int, int]:
    """Combine base rate/pitch with per-speaker profile and per-segment
    emphasis carryover. Returns (rate_pct, pitch_hz)."""
    rate = int(base_rate_pct)
    pitch = int(base_pitch_hz)

    if profiles:
        speaker_id = getattr(segment, "speaker_id", None)
        profile = profiles.get(speaker_id) if speaker_id else None
        if profile is not None:
            rate += profile.rate_offset_pct

    source_text = getattr(segment, "text", "") or ""
    rate_delta, pitch_delta = emphasis_from_source(source_text)
    rate += rate_delta
    pitch += pitch_delta

    if emotion_analyses:
        analysis = emotion_analyses.get(getattr(segment, "index", -1))
        if analysis is not None and not getattr(analysis, "is_neutral_fallback", True):
            rate += int(getattr(analysis, "pacing_offset_pct", 0))
            pitch += int(getattr(analysis, "pitch_offset_hz", 0))

    # Hard clamp: edge-tts accepts a wide range but extreme values start
    # producing artifacts.
    rate = max(-50, min(50, rate))
    pitch = max(-50, min(50, pitch))
    return rate, pitch
