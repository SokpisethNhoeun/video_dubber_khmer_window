from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from threading import Event
from typing import Callable

from core.context import CancellationError, Segment
from modules.prosody import SpeakerRateProfile, per_segment_prosody
from modules.ai_translator import BULK_AI_RECOVERY_THRESHOLD


ProgressCallback = Callable[[int], None]
LogCallback = Callable[[str], None]


def _format_rate(rate_percent: int) -> str:
    sign = "+" if rate_percent >= 0 else ""
    return f"{sign}{rate_percent}%"


def _format_pitch(pitch_hz: int) -> str:
    sign = "+" if pitch_hz >= 0 else ""
    return f"{sign}{pitch_hz}Hz"


def _cache_name(text: str, voice: str, rate: str, pitch: str) -> str:
    digest = hashlib.sha256(f"{voice}|{rate}|{pitch}|{text}".encode("utf-8")).hexdigest()
    return f"{digest}.mp3"


async def _synthesize_one(
    segment: Segment,
    voice: str,
    rate: str,
    pitch: str,
    output_path: Path,
    semaphore: asyncio.Semaphore,
    cancel_event: Event | None = None,
) -> None:
    import edge_tts
    import subprocess
    import logging

    async with semaphore:
        for attempt in range(3):
            if cancel_event is not None and cancel_event.is_set():
                raise CancellationError("Processing cancelled by user")
            try:
                communicate = edge_tts.Communicate(segment.tts_text, voice=voice, rate=rate, pitch=pitch)
                await communicate.save(str(output_path))
                return
            except edge_tts.exceptions.NoAudioReceived:
                if attempt == 2:
                    logging.warning(f"edge-tts returned no audio for segment {segment.index}. Creating silent audio.")
                    subprocess.run(
                        [
                            "ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono",
                            "-t", "0.1", "-q:a", "9", "-acodec", "libmp3lame", str(output_path)
                        ],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                    )
                    return
                await asyncio.sleep(1)
            except Exception:
                if attempt == 2:
                    raise
                await asyncio.sleep(1)


def _write_silent_audio(output_path: Path, duration_sec: float) -> None:
    import subprocess

    duration = max(0.05, min(float(duration_sec), 30.0))
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono",
            "-t", f"{duration:.3f}",
            "-q:a", "9", "-acodec", "libmp3lame", str(output_path),
        ],
        check=True,
    )


def repair_segments_for_tts(
    segments: list[Segment],
    source_language: str,
    log_cb: LogCallback | None,
    cancel_event: Event,
    translation_backend: str = "nllb",
    content_style: str = "casual_vlog",
    khmer_style: str = "simple",
    glossary_path: Path | None = None,
) -> list[Segment]:
    """Fill or disable segments that would otherwise block Khmer TTS."""
    missing = [segment for segment in segments if segment.enabled and not segment.tts_text.strip()]
    if not missing:
        return segments

    if log_cb:
        log_cb(
            f"  Repairing {len(missing)} segment(s) with missing Khmer TTS text "
            f"(indices: {', '.join(str(s.index) for s in missing[:12])}"
            f"{'...' if len(missing) > 12 else ''})"
        )

    if source_language == "km":
        for segment in missing:
            if segment.text.strip():
                segment.translated_text = segment.text
                segment.raw_khmer_text = segment.text
                segment.improved_khmer_text = segment.text
        missing = [segment for segment in segments if segment.enabled and not segment.tts_text.strip()]

    use_ai_recovery = (
        source_language != "km"
        and missing
        and (
            translation_backend == "ai"
            or len(missing) >= BULK_AI_RECOVERY_THRESHOLD
        )
    )
    if use_ai_recovery:
        try:
            from modules.ai_translator import recover_missing_khmer_with_ai

            if log_cb:
                log_cb(
                    "  Using AI translation + AI review for missing Khmer script "
                    f"({len(missing)} segment(s))"
                )
            recover_missing_khmer_with_ai(
                missing,
                source_language,
                None,
                log_cb,
                cancel_event,
                content_style=content_style,
                khmer_style=khmer_style,
                glossary_path=glossary_path,
            )
        except Exception as exc:
            if log_cb:
                log_cb(f"  Warning: bulk AI Khmer recovery failed: {exc}")
        missing = [segment for segment in segments if segment.enabled and not segment.tts_text.strip()]

    translator = None
    src_code = None
    allow_google = (
        source_language != "km"
        and missing
        and translation_backend != "ai"
        and len(missing) < BULK_AI_RECOVERY_THRESHOLD
    )
    if allow_google:
        from modules.google_translator import LANGUAGE_MAP, _translate_with_retry
        from deep_translator import GoogleTranslator

        src_code = LANGUAGE_MAP.get(source_language)
        if src_code:
            translator = GoogleTranslator(source=src_code, target="km")

    repaired = 0
    disabled = 0
    for segment in missing:
        if cancel_event.is_set():
            raise CancellationError("Processing cancelled by user")

        khmer = (
            segment.user_edited_text.strip()
            or segment.improved_khmer_text.strip()
            or segment.translated_text.strip()
            or segment.raw_khmer_text.strip()
        )
        if khmer:
            segment.translated_text = segment.translated_text or khmer
            segment.raw_khmer_text = segment.raw_khmer_text or khmer
            segment.improved_khmer_text = segment.improved_khmer_text or khmer
            repaired += 1
            continue

        source = segment.text.strip()
        if not source:
            segment.enabled = False
            note = "disabled: no source or Khmer text for TTS"
            segment.review_notes = f"{segment.review_notes}; {note}".strip("; ").strip()
            disabled += 1
            continue

        if translator is not None and src_code is not None:
            try:
                khmer = _translate_with_retry(translator, source, log_cb).strip()
            except Exception as exc:
                if log_cb:
                    log_cb(f"  Warning: could not recover Khmer for segment {segment.index}: {exc}")
                khmer = ""
            if khmer:
                segment.translated_text = khmer
                segment.raw_khmer_text = khmer
                segment.improved_khmer_text = khmer
                segment.review_notes = (
                    f"{segment.review_notes}; recovered Khmer via Google Translate"
                ).strip("; ").strip()
                repaired += 1
                if log_cb:
                    log_cb(f'  Recovered segment {segment.index}: "{source}" -> "{khmer}" (Google Translate)')
                continue

        segment.enabled = False
        note = "disabled: Khmer translation missing after recovery"
        segment.review_notes = f"{segment.review_notes}; {note}".strip("; ").strip()
        disabled += 1

    if log_cb and (repaired or disabled):
        log_cb(f"  TTS repair complete: {repaired} recovered, {disabled} disabled")
    return segments


async def _synthesize_all(
    segments: list[Segment],
    voice_gender: str,
    voice_female: str,
    voice_male: str,
    segment_genders: dict[int, str] | None,
    base_rate_pct: int,
    base_pitch_hz: int,
    speaker_rate_profiles: dict[str, SpeakerRateProfile] | None,
    emotion_analyses: dict[int, object] | None,
    cache_dir: Path,
    cache_hits: dict[str, int] | None,
    progress_cb: ProgressCallback | None,
    log_cb: LogCallback | None,
    cancel_event: Event,
) -> list[Segment]:
    semaphore = asyncio.Semaphore(4)
    completed = 0

    async def run_segment(segment: Segment) -> None:
        nonlocal completed
        if cancel_event.is_set():
            raise CancellationError("Processing cancelled by user")

        tts_text = segment.tts_text.strip()
        if not tts_text:
            output_path = cache_dir / f"silent_{segment.index:05d}.mp3"
            if not output_path.exists():
                if log_cb:
                    log_cb(
                        f"  Segment {segment.index + 1}/{len(segments)}: "
                        "no Khmer text — using silent placeholder"
                    )
                _write_silent_audio(output_path, segment.duration)
            segment.tts_path = output_path
            segment.tts_group_id = ""
            completed += 1
            if progress_cb:
                progress_cb(int((completed / len(segments)) * 100))
            return

        if voice_gender in {"auto", "per_person", "per_person_auto", "per_speaker_auto"}:
            gender = segment_genders.get(segment.index, "female") if segment_genders else "female"
            voice = voice_female if gender == "female" else voice_male
        else:
            voice = voice_female if voice_gender == "female" else voice_male

        # Per-segment prosody: speaker rate profile + source-side emphasis.
        seg_rate_pct, seg_pitch_hz = per_segment_prosody(
            segment, base_rate_pct, base_pitch_hz, speaker_rate_profiles, emotion_analyses
        )
        rate = _format_rate(seg_rate_pct)
        pitch = _format_pitch(seg_pitch_hz)

        tts_text = segment.tts_text
        output_path = cache_dir / _cache_name(tts_text, voice, rate, pitch)
        if not output_path.exists():
            if log_cb:
                prosody_note = ""
                if seg_rate_pct != base_rate_pct or seg_pitch_hz != base_pitch_hz:
                    prosody_note = f" @ rate {rate}, pitch {pitch}"
                log_cb(
                    f"  Synthesizing TTS segment {segment.index + 1}/{len(segments)} "
                    f"({voice}{prosody_note}): \"{tts_text}\""
                )
            await _synthesize_one(segment, voice, rate, pitch, output_path, semaphore, cancel_event)
        else:
            if cache_hits is not None:
                cache_hits["tts"] = cache_hits.get("tts", 0) + 1
            if log_cb:
                log_cb(f"  Using cached TTS for segment {segment.index + 1}/{len(segments)} ({voice})")
        segment.tts_path = output_path
        segment.tts_group_id = ""
        completed += 1
        if progress_cb:
            progress_cb(int((completed / len(segments)) * 100))

    tasks = [asyncio.create_task(run_segment(segment)) for segment in segments]
    try:
        await asyncio.gather(*tasks)
    except CancellationError:
        for task in tasks:
            if not task.done():
                task.cancel()
        raise
    return segments


def synthesize_tts(
    segments: list[Segment],
    voice_gender: str,
    speech_rate: int,
    pitch_hz: int,
    work_dir: Path,
    progress_cb: ProgressCallback | None,
    log_cb: LogCallback | None,
    cancel_event: Event,
    voice_female: str = "km-KH-SreymomNeural",
    voice_male: str = "km-KH-PisethNeural",
    segment_genders: dict[int, str] | None = None,
    persistent_cache_dir: Path | None = None,
    cache_hits: dict[str, int] | None = None,
    speaker_rate_profiles: dict[str, SpeakerRateProfile] | None = None,
    emotion_analyses: dict[int, object] | None = None,
    source_language: str = "en",
    translation_backend: str = "nllb",
    content_style: str = "casual_vlog",
    khmer_style: str = "simple",
    glossary_path: Path | None = None,
    tts_provider: str = "edge",
    speaker_voice_mappings: dict[str, dict[str, str]] | None = None,
) -> list[Segment]:
    if voice_gender not in ["female", "male", "auto", "per_person", "per_person_auto", "per_speaker_auto"]:
        raise ValueError(f"Unsupported voice gender: {voice_gender}")
    all_segments = repair_segments_for_tts(
        segments,
        source_language,
        log_cb,
        cancel_event,
        translation_backend=translation_backend,
        content_style=content_style,
        khmer_style=khmer_style,
        glossary_path=glossary_path,
    )
    active_segments = [segment for segment in all_segments if segment.enabled]
    if not active_segments:
        raise ValueError(
            "No segments available for TTS: this session has no enabled Khmer transcript rows "
            "after translation recovery. Open the session editor and make sure at least one row "
            "is enabled with Khmer text, or rerun the video from translation/review before resuming audio."
        )

    if tts_provider == "gemini":
        from modules.gemini_tts_engine import synthesize_gemini_tts

        return synthesize_gemini_tts(
            all_segments,
            work_dir,
            progress_cb,
            log_cb,
            cancel_event,
            persistent_cache_dir=persistent_cache_dir,
            emotion_analyses=emotion_analyses,
            speaker_voice_mappings=speaker_voice_mappings,
        )
    if tts_provider != "edge":
        raise ValueError(f"Unsupported TTS provider: {tts_provider}")

    cache_dir = persistent_cache_dir / "tts" if persistent_cache_dir else work_dir / "tts_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    rate = _format_rate(speech_rate)
    pitch = _format_pitch(pitch_hz)

    if log_cb:
        if voice_gender == "auto":
            log_cb(f"Synthesizing Khmer speech in AUTO mode (female: {voice_female}, male: {voice_male})")
        elif voice_gender in {"per_person", "per_person_auto"}:
            log_cb(
                "Synthesizing Khmer speech for per-person dubbing "
                f"(female: {voice_female}, male: {voice_male}; "
                "gender detected for TTS fallback, speakers with references are cloned)"
            )
        elif voice_gender == "per_speaker_auto":
            log_cb(
                "Synthesizing Khmer speech with one consistent voice per detected speaker "
                f"(female: {voice_female}, male: {voice_male} fallback for unassigned lines)"
            )
        else:
            voice_name = voice_female if voice_gender == "female" else voice_male
            log_cb(f"Synthesizing Khmer speech with {voice_name}, rate {rate}, pitch {pitch}")
        if speaker_rate_profiles:
            profile_summary = ", ".join(
                f"{sid} {p.rate_offset_pct:+d}%" for sid, p in speaker_rate_profiles.items()
            )
            log_cb(f"  Per-speaker rate profile: {profile_summary}")

    asyncio.run(
        _synthesize_all(
            active_segments,
            voice_gender,
            voice_female,
            voice_male,
            segment_genders,
            speech_rate,
            pitch_hz,
            speaker_rate_profiles,
            emotion_analyses,
            cache_dir,
            cache_hits,
            progress_cb,
            log_cb,
            cancel_event,
        )
    )
    return all_segments
