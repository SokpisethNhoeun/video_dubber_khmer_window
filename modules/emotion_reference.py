from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Callable

from core.context import CancellationError, Segment
from modules.audio_utils import ffprobe_duration, trim_audio_segment
from modules.diarizer import SpeakerTurn


LogCallback = Callable[[str], None]

MIN_USABLE_SECONDS = 1.0
DEFAULT_MIN_CLIP_SECONDS = 1.5
DEFAULT_MAX_CLIP_SECONDS = 8.0
DEFAULT_CONTEXT_PADDING_SECONDS = 0.35
MIN_SNR_DB = 6.0


@dataclass
class EmotionClip:
    segment_index: int
    clip_path: Path
    duration: float
    snr_db: float
    usable: bool
    fallback_reason: str


def _find_covering_turn(
    segment: Segment,
    turns: list[SpeakerTurn],
) -> SpeakerTurn | None:
    best: SpeakerTurn | None = None
    best_overlap = 0.0
    for turn in turns:
        if segment.speaker_id and turn.speaker_id != segment.speaker_id:
            continue
        overlap = min(segment.end, turn.end) - max(segment.start, turn.start)
        if overlap > best_overlap:
            best_overlap = overlap
            best = turn
    return best


def _quick_snr(clip_path: Path) -> float:
    try:
        import numpy as np
        from modules.reference_quality import _load_mono_16k, _voiced_ratio_and_snr
        samples, sr = _load_mono_16k(clip_path)
        _, snr_db = _voiced_ratio_and_snr(samples, sr)
        return snr_db
    except Exception:
        return 99.0


def extract_emotion_clips(
    source_wav: Path,
    segments: list[Segment],
    turns: list[SpeakerTurn],
    output_dir: Path,
    cancel_event: Event,
    log_cb: LogCallback | None = None,
    min_clip_seconds: float = DEFAULT_MIN_CLIP_SECONDS,
    max_clip_seconds: float = DEFAULT_MAX_CLIP_SECONDS,
    context_padding_seconds: float = DEFAULT_CONTEXT_PADDING_SECONDS,
) -> dict[int, EmotionClip]:
    output_dir.mkdir(parents=True, exist_ok=True)
    active_segments = [s for s in segments if s.enabled]
    clips: dict[int, EmotionClip] = {}
    usable_count = 0

    for segment in active_segments:
        if cancel_event.is_set():
            raise CancellationError("Processing cancelled by user")

        covering_turn = _find_covering_turn(segment, turns) if turns else None
        if covering_turn is not None:
            clip_start = max(covering_turn.start, segment.start - context_padding_seconds)
            clip_end = min(covering_turn.end, segment.end + context_padding_seconds)
        else:
            clip_start = max(0.0, segment.start - context_padding_seconds)
            clip_end = segment.end + context_padding_seconds
        clip_duration = clip_end - clip_start

        if clip_duration < min_clip_seconds and covering_turn is not None:
            clip_start = covering_turn.start
            clip_end = covering_turn.end
            clip_duration = clip_end - clip_start

        if clip_duration > max_clip_seconds:
            center = (segment.start + segment.end) / 2.0
            clip_start = max(clip_start, center - max_clip_seconds / 2.0)
            clip_end = clip_start + max_clip_seconds
            clip_duration = max_clip_seconds

        clip_start = max(0.0, clip_start)
        clip_duration = clip_end - clip_start

        if clip_duration < MIN_USABLE_SECONDS:
            clips[segment.index] = EmotionClip(
                segment_index=segment.index,
                clip_path=Path(""),
                duration=clip_duration,
                snr_db=0.0,
                usable=False,
                fallback_reason=f"too short ({clip_duration:.2f}s < {MIN_USABLE_SECONDS}s)",
            )
            continue

        clip_path = output_dir / f"{segment.index:05d}_emotion.wav"
        try:
            trim_audio_segment(source_wav, clip_path, clip_start, clip_duration, cancel_event)
        except Exception as exc:
            clips[segment.index] = EmotionClip(
                segment_index=segment.index,
                clip_path=Path(""),
                duration=0.0,
                snr_db=0.0,
                usable=False,
                fallback_reason=f"extraction failed: {exc}",
            )
            continue

        actual_duration = ffprobe_duration(clip_path)
        snr_db = _quick_snr(clip_path)

        if actual_duration < MIN_USABLE_SECONDS:
            reason = f"too short after extraction ({actual_duration:.2f}s)"
            clips[segment.index] = EmotionClip(
                segment_index=segment.index,
                clip_path=clip_path,
                duration=actual_duration,
                snr_db=snr_db,
                usable=False,
                fallback_reason=reason,
            )
        elif snr_db < MIN_SNR_DB:
            reason = f"low SNR ({snr_db:.1f} dB < {MIN_SNR_DB} dB)"
            clips[segment.index] = EmotionClip(
                segment_index=segment.index,
                clip_path=clip_path,
                duration=actual_duration,
                snr_db=snr_db,
                usable=False,
                fallback_reason=reason,
            )
        else:
            clips[segment.index] = EmotionClip(
                segment_index=segment.index,
                clip_path=clip_path,
                duration=actual_duration,
                snr_db=snr_db,
                usable=True,
                fallback_reason="",
            )
            usable_count += 1

    if log_cb:
        total = len(active_segments)
        log_cb(
            f"  Extracted {usable_count}/{total} emotion clips "
            f"({total - usable_count} will use generic reference)"
        )

    return clips
