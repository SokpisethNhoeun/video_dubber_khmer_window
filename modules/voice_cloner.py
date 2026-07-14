from __future__ import annotations

import os
import re
import shlex
import subprocess
import concurrent.futures
from collections import defaultdict
from pathlib import Path
from threading import Event, Lock
from typing import Callable

from core.context import CancellationError, Segment
from modules.audio_matching import post_clone_match


def _release_gpu() -> None:
    """Free GPU memory between speakers to avoid OOM during voice cloning."""
    try:
        import gc
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


ProgressCallback = Callable[[int], None]
LogCallback = Callable[[str], None]


def _uses_token(template: str, token: str) -> bool:
    return f"{{{token}}}" in template


def _render_command(
    template: str,
    input_path: Path,
    output_path: Path,
    model_path: Path | None,
    index_path: Path | None,
    reference_audio_path: Path | None,
) -> list[str]:
    replacements = {
        "input": str(input_path),
        "output": str(output_path),
        "model": str(model_path) if model_path else "",
        "index": str(index_path) if index_path else "",
        "reference": str(reference_audio_path) if reference_audio_path else "",
    }
    rendered = template
    for token, value in replacements.items():
        rendered = rendered.replace(f"{{{token}}}", value)
    unknown = set()
    for match in re.finditer(r"\{(\w+)\}", rendered):
        unknown.add(match.group(1))
    if unknown:
        raise ValueError(f"Unsupported voice clone command token(s): {', '.join(sorted(unknown))}")
    return shlex.split(rendered)


def _validate_command_setup(command_template: str) -> None:
    try:
        parts = shlex.split(command_template)
    except ValueError as exc:
        raise ValueError(f"Invalid voice clone command: {exc}") from exc
    if "modules.openvoice_voice_clone" not in parts:
        return

    python_path = os.getenv("OPENVOICE_PYTHON", "").strip()
    if not python_path:
        raise ValueError(
            "OpenVoice clone backend selected, but OPENVOICE_PYTHON is not set. "
            "Set it to the python executable in a separate OpenVoice Python 3.10/3.11 environment."
        )
    if not Path(python_path).expanduser().exists():
        raise FileNotFoundError(f"OPENVOICE_PYTHON does not exist: {python_path}")

    checkpoint_dir = os.getenv("OPENVOICE_CHECKPOINT_DIR", "").strip()
    if not checkpoint_dir:
        raise ValueError(
            "OpenVoice clone backend selected, but OPENVOICE_CHECKPOINT_DIR is not set. "
            "Set it to the OpenVoice checkpoint directory."
        )
    if not Path(checkpoint_dir).expanduser().exists():
        raise FileNotFoundError(f"OPENVOICE_CHECKPOINT_DIR does not exist: {checkpoint_dir}")


def _segment_gender(
    segment: Segment,
    voice_gender: str,
    segment_genders: dict[int, str] | None,
) -> str:
    if voice_gender in {"female", "male"}:
        return voice_gender
    if segment_genders:
        return segment_genders.get(segment.index, "female")
    return "female"


def _reference_for_speaker(
    segment: Segment,
    speaker_voice_mappings: dict[str, dict[str, str]] | None,
) -> Path | None:
    if not segment.speaker_id or not speaker_voice_mappings:
        return None
    mapping = speaker_voice_mappings.get(segment.speaker_id, {})
    reference = (
        mapping.get("cleaned_reference_audio_path", "").strip()
        or mapping.get("reference_audio_path", "").strip()
        or mapping.get("original_reference_audio_path", "").strip()
    )
    return Path(reference).expanduser() if reference else None


def _speaker_quality_tier(
    segment: Segment,
    speaker_voice_mappings: dict[str, dict[str, str]] | None,
) -> str | None:
    if not segment.speaker_id or not speaker_voice_mappings:
        return None
    mapping = speaker_voice_mappings.get(segment.speaker_id, {})
    tier = mapping.get("quality_tier", "").strip().lower()
    return tier or None


def _reference_fallback_reason(
    segment: Segment,
    speaker_voice_mappings: dict[str, dict[str, str]] | None,
) -> str:
    if not segment.speaker_id or not speaker_voice_mappings:
        return "no speaker reference mapping"
    mapping = speaker_voice_mappings.get(segment.speaker_id, {})
    return mapping.get("fallback_reason", "").strip() or "no usable reference audio"


def _is_openvoice_template(command_template: str) -> bool:
    try:
        parts = shlex.split(command_template)
    except ValueError:
        return False
    return "modules.openvoice_voice_clone" in parts


def _attach_segment_emotion(
    item: dict,
    segment: Segment,
    emotion_clips: dict | None,
    emotion_analyses: dict | None,
    emotion_mode: str,
) -> None:
    from modules.emotion_detector import apply_emotion_to_clone_item

    clip = emotion_clips.get(segment.index) if emotion_clips else None
    analysis = emotion_analyses.get(segment.index) if emotion_analyses else None
    apply_emotion_to_clone_item(
        item,
        clip,
        emotion_mode,
        analysis,
        source_text=_segment_source_text(segment),
    )


def _post_clone_reference_path(item: dict | None, fallback: Path) -> Path:
    if not item:
        return fallback
    emotion_ref = item.get("emotion_reference_path") or item.get("reference_path")
    if emotion_ref and Path(emotion_ref).exists():
        return Path(emotion_ref)
    return fallback


def _log_emotion_analysis_summary(
    emotion_analyses: dict | None,
    log_cb: LogCallback | None,
) -> None:
    if not log_cb or not emotion_analyses:
        return
    labels: dict[str, int] = {}
    neutral = 0
    for analysis in emotion_analyses.values():
        if getattr(analysis, "is_neutral_fallback", True):
            neutral += 1
        else:
            label = getattr(analysis, "label", "neutral")
            labels[label] = labels.get(label, 0) + 1
    if labels:
        parts = ", ".join(f"{name}={count}" for name, count in sorted(labels.items()))
        log_cb(f"  Detected segment emotions: {parts}")
    if neutral:
        log_cb(f"  {neutral} segment(s) using neutral speaking style (low confidence or unclear audio)")


def _log_clone_result_summary(
    segments: list[Segment],
    enabled: bool,
    log_cb: LogCallback | None,
) -> None:
    if not enabled or log_cb is None:
        return
    active = [segment for segment in segments if segment.enabled]
    if not active:
        return
    cloned = sum(1 for segment in active if segment.cloned_path is not None and segment.cloned_path.exists())
    fallback = len(active) - cloned
    if fallback:
        log_cb(
            f"Voice clone result: {cloned}/{len(active)} segment(s) cloned; "
            f"{fallback} segment(s) use default Khmer TTS fallback"
        )
    else:
        log_cb(f"Voice clone result: all {len(active)} active segment(s) cloned")


def _segment_source_text(segment: Segment) -> str:
    """Source-language transcript that matches per-segment emotion clips."""
    return segment.text.strip()


def _segment_ref_text(segment: Segment) -> str:
    """Transcript for Qwen3 ICL when no dedicated emotion clip text is available."""
    return (
        _segment_source_text(segment)
        or getattr(segment, "user_edited_text", "").strip()
        or segment.translated_text.strip()
    )


def _group_segments_by_gender(
    active_segments: list[Segment],
    gender_reference_paths: dict[str, Path] | None,
    segment_genders: dict[int, str] | None,
    log_cb: LogCallback | None,
) -> dict[str, list[tuple[int, Segment]]]:
    grouped: dict[str, list[tuple[int, Segment]]] = defaultdict(list)
    for position, segment in enumerate(active_segments, start=1):
        gender = _segment_gender(segment, "auto", segment_genders)
        reference_path = gender_reference_paths.get(gender) if gender_reference_paths else None
        if reference_path is None or not reference_path.exists():
            label = segment.speaker_label or segment.speaker_id or "unknown"
            if log_cb:
                log_cb(
                    f"  Skipping segment {position}: no {gender} voice profile available for {label}"
                )
            continue
        grouped[gender].append((position, segment))
    return grouped


def _clone_batch_qwen3(
    speaker_segments: list[tuple[int, Segment]],
    reference_path: Path,
    clone_dir: Path,
    log_cb: LogCallback | None,
    quality_report,
    cancel_event: Event,
    emotion_clips: dict | None = None,
    emotion_analyses: dict | None = None,
    emotion_mode: str = "auto",
    prompt_cache_key: str | None = None,
) -> None:
    """Clone all segments for one speaker using Qwen3-TTS 1.7B."""
    from modules.qwen3_voice_clone import clone_batch, Qwen3CloneError

    batch_items = []
    segment_map: dict[int, Segment] = {}
    item_map: dict[int, dict] = {}
    speaker_id = speaker_segments[0][1].speaker_id or "__default__"
    for position, segment in speaker_segments:
        if cancel_event.is_set():
            raise CancellationError("Processing cancelled by user")
        output_path = clone_dir / f"{segment.index:05d}_qwen3.wav"
        if output_path.exists():
            segment.cloned_path = output_path
            continue
        item = {
            "segment_index": segment.index,
            "text": segment.tts_text,
            "output_path": str(output_path),
            "speaker_reference_path": str(reference_path),
            "emotion_ref_text": _segment_source_text(segment),
            "ref_text": _segment_ref_text(segment),
            "prompt_cache_key": speaker_id,
        }
        _attach_segment_emotion(item, segment, emotion_clips, emotion_analyses, emotion_mode)
        batch_items.append(item)
        item_map[segment.index] = item
        segment_map[segment.index] = segment

    if not batch_items:
        return

    label = speaker_segments[0][1].speaker_label or speaker_segments[0][1].speaker_id or "unknown"
    if log_cb:
        log_cb(f"  Qwen3-TTS batch-cloning {len(batch_items)} segments for {label}")

    try:
        results = clone_batch(batch_items, reference_path, log_cb=log_cb, cancel_event=cancel_event)
    except Qwen3CloneError as exc:
        msg = f"Qwen3-TTS batch failed for {label}: {str(exc)[:300]}"
        if quality_report is not None:
            quality_report.voice_clone_failures.append({"segment": -1, "message": msg})
        if log_cb:
            log_cb(f"  Warning: {msg}")
        return

    for r in results:
        seg = segment_map.get(r["segment_index"])
        if seg is None:
            continue
        out = clone_dir / f"{seg.index:05d}_qwen3.wav"
        if r.get("ok") and out.exists():
            try:
                post_clone_match(
                    out,
                    _post_clone_reference_path(item_map.get(seg.index), reference_path),
                )
            except Exception as exc:
                if log_cb:
                    log_cb(f"  Warning: post-clone matching failed for segment {seg.index}: {exc}")
            seg.cloned_path = out
        else:
            error = r.get("error", "unknown error")
            if quality_report is not None:
                quality_report.voice_clone_failures.append(
                    {"segment": seg.index + 1, "message": f"Qwen3-TTS failed: {error}"}
                )
            if log_cb:
                log_cb(f"  Warning: Qwen3-TTS clone failed for segment {seg.index}: {error}")


def _clone_batch_xtts(
    speaker_segments: list[tuple[int, Segment]],
    reference_path: Path,
    clone_dir: Path,
    log_cb: LogCallback | None,
    quality_report,
    cancel_event: Event,
    emotion_clips: dict | None = None,
    emotion_analyses: dict | None = None,
    emotion_mode: str = "auto",
) -> None:
    """Clone all segments for one speaker using XTTS-v2 (in-process)."""
    from modules.xtts_voice_clone import clone_batch, XTTSCloneError

    batch_items = []
    segment_map: dict[int, Segment] = {}
    for position, segment in speaker_segments:
        if cancel_event.is_set():
            raise CancellationError("Processing cancelled by user")
        output_path = clone_dir / f"{segment.index:05d}_xtts.wav"
        if output_path.exists():
            segment.cloned_path = output_path
            continue
        item = {
            "segment_index": segment.index,
            "text": segment.tts_text,
            "output_path": str(output_path),
            "speaker_reference_path": str(reference_path),
        }
        _attach_segment_emotion(item, segment, emotion_clips, emotion_analyses, emotion_mode)
        batch_items.append(item)
        segment_map[segment.index] = segment

    if not batch_items:
        return

    label = speaker_segments[0][1].speaker_label or speaker_segments[0][1].speaker_id or "unknown"
    if log_cb:
        log_cb(f"  XTTS batch-cloning {len(batch_items)} segments for {label}")

    try:
        results = clone_batch(batch_items, reference_path, log_cb=log_cb, cancel_event=cancel_event)
    except XTTSCloneError as exc:
        msg = f"XTTS batch failed for {label}: {str(exc)[:300]}"
        if quality_report is not None:
            quality_report.voice_clone_failures.append({"segment": -1, "message": msg})
        if log_cb:
            log_cb(f"  Warning: {msg}")
        return

    for r in results:
        seg = segment_map.get(r["segment_index"])
        if seg is None:
            continue
        out = clone_dir / f"{seg.index:05d}_xtts.wav"
        if r.get("ok") and out.exists():
            try:
                post_clone_match(out, reference_path)
            except Exception as exc:
                if log_cb:
                    log_cb(f"  Warning: post-clone matching failed for segment {seg.index}: {exc}")
            seg.cloned_path = out
        else:
            error = r.get("error", "unknown error")
            if quality_report is not None:
                quality_report.voice_clone_failures.append(
                    {"segment": seg.index + 1, "message": f"XTTS failed: {error}"}
                )
            if log_cb:
                log_cb(f"  Warning: XTTS clone failed for segment {seg.index}: {error}")


def _clone_batch_cosyvoice(
    speaker_segments: list[tuple[int, Segment]],
    reference_path: Path,
    clone_dir: Path,
    log_cb: LogCallback | None,
    quality_report,
    cancel_event: Event,
    emotion_clips: dict | None = None,
    emotion_analyses: dict | None = None,
    emotion_mode: str = "auto",
) -> None:
    """Clone all segments for one speaker using CosyVoice 2 (subprocess)."""
    from modules.cosyvoice_voice_clone import clone_batch, CosyVoiceCloneError

    batch_items = []
    segment_map: dict[int, Segment] = {}
    for position, segment in speaker_segments:
        if cancel_event.is_set():
            raise CancellationError("Processing cancelled by user")
        output_path = clone_dir / f"{segment.index:05d}_cosyvoice.wav"
        if output_path.exists():
            segment.cloned_path = output_path
            continue
        input_path = segment.tts_path
        if input_path is None or not input_path.exists():
            if log_cb:
                log_cb(
                    f"  Warning: Missing Khmer TTS audio for segment {segment.index}; "
                    "CosyVoice clone skipped to avoid non-Khmer text synthesis"
                )
            continue
        item = {
            "segment_index": segment.index,
            "text": segment.tts_text,
            "input_path": str(input_path),
            "output_path": str(output_path),
            "speaker_reference_path": str(reference_path),
        }
        _attach_segment_emotion(item, segment, emotion_clips, emotion_analyses, emotion_mode)
        batch_items.append(item)
        segment_map[segment.index] = segment

    if not batch_items:
        return

    label = speaker_segments[0][1].speaker_label or speaker_segments[0][1].speaker_id or "unknown"
    if log_cb:
        log_cb(f"  CosyVoice batch-cloning {len(batch_items)} segments for {label}")

    try:
        results = clone_batch(batch_items, reference_path, log_cb=log_cb, cancel_event=cancel_event)
    except CosyVoiceCloneError as exc:
        msg = f"CosyVoice batch failed for {label}: {str(exc)[:300]}"
        if quality_report is not None:
            quality_report.voice_clone_failures.append({"segment": -1, "message": msg})
        if log_cb:
            log_cb(f"  Warning: {msg}")
        return

    for r in results:
        seg = segment_map.get(r["segment_index"])
        if seg is None:
            continue
        out = clone_dir / f"{seg.index:05d}_cosyvoice.wav"
        if r.get("ok") and out.exists():
            try:
                post_clone_match(out, reference_path)
            except Exception as exc:
                if log_cb:
                    log_cb(f"  Warning: post-clone matching failed for segment {seg.index}: {exc}")
            seg.cloned_path = out
        else:
            error = r.get("error", "unknown error")
            if quality_report is not None:
                quality_report.voice_clone_failures.append(
                    {"segment": seg.index + 1, "message": f"CosyVoice failed: {error}"}
                )
            if log_cb:
                log_cb(f"  Warning: CosyVoice clone failed for segment {seg.index}: {error}")


def _clone_batch_openvoice(
    speaker_segments: list[tuple[int, Segment]],
    reference_path: Path,
    clone_dir: Path,
    log_cb: LogCallback | None,
    quality_report,
    cancel_event: Event,
) -> None:
    """Clone all segments for one speaker in a single OpenVoice subprocess."""
    from modules.openvoice_voice_clone import clone_batch_openvoice, OpenVoiceCloneError

    batch_items = []
    segment_map: dict[int, Segment] = {}
    for position, segment in speaker_segments:
        if cancel_event.is_set():
            raise CancellationError("Processing cancelled by user")
        input_path = segment.tts_path
        if input_path is None or not input_path.exists():
            if log_cb:
                log_cb(f"  Warning: Missing TTS audio for segment {segment.index}")
            continue
        output_path = clone_dir / f"{segment.index:05d}_rvc.wav"
        if output_path.exists():
            segment.cloned_path = output_path
            continue
        batch_items.append({
            "segment_index": segment.index,
            "input_path": str(input_path),
            "output_path": str(output_path),
        })
        segment_map[segment.index] = segment

    if not batch_items:
        return

    label = speaker_segments[0][1].speaker_label or speaker_segments[0][1].speaker_id or "unknown"
    if log_cb:
        log_cb(f"  Batch-cloning {len(batch_items)} segments for {label} (single subprocess)")

    try:
        results = clone_batch_openvoice(batch_items, reference_path, log_cb=log_cb, cancel_event=cancel_event)
    except OpenVoiceCloneError as exc:
        msg = f"OpenVoice batch failed for {label}: {str(exc)[:300]}"
        if quality_report is not None:
            quality_report.voice_clone_failures.append({"segment": -1, "message": msg})
        if log_cb:
            log_cb(f"  Warning: {msg}")
        return

    for r in results:
        seg = segment_map.get(r["segment_index"])
        if seg is None:
            continue
        out = clone_dir / f"{seg.index:05d}_rvc.wav"
        if r.get("ok") and out.exists():
            try:
                post_clone_match(out, reference_path)
            except Exception as exc:
                if log_cb:
                    log_cb(f"  Warning: post-clone matching failed for segment {seg.index}: {exc}")
            seg.cloned_path = out
        else:
            error = r.get("error", "unknown error")
            if quality_report is not None:
                quality_report.voice_clone_failures.append(
                    {"segment": seg.index + 1, "message": f"OpenVoice failed: {error}"}
                )
            if log_cb:
                log_cb(f"  Warning: clone failed for segment {seg.index}: {error}")


def optional_voice_clone(
    segments: list[Segment],
    enabled: bool,
    model_path: Path | None,
    index_path: Path | None,
    reference_audio_path: Path | None,
    gender_reference_paths: dict[str, Path] | None,
    clone_gender: str,
    voice_gender: str,
    segment_genders: dict[int, str] | None,
    command_template: str,
    work_dir: Path,
    progress_cb: ProgressCallback | None,
    log_cb: LogCallback | None,
    cancel_event: Event,
    speaker_voice_mappings: dict[str, dict[str, str]] | None = None,
    quality_report=None,
    clone_backend: str = "openvoice",
    emotion_aware: bool = False,
    source_wav: Path | None = None,
    diarization_turns: list | None = None,
    emotion_mode: str = "auto",
    emotion_clips: dict | None = None,
    emotion_analyses: dict | None = None,
) -> list[Segment]:
    if not enabled:
        if progress_cb:
            progress_cb(100)
        return segments

    per_person = voice_gender in {"per_person", "per_person_auto"}

    has_gender_references = bool(gender_reference_paths)
    use_internal_backend = clone_backend in ("xtts", "cosyvoice", "qwen3") and (per_person or has_gender_references)
    if use_internal_backend:
        if clone_backend == "cosyvoice":
            clone_dir = work_dir / "cosyvoice"
            backend_label = "CosyVoice 2"
        elif clone_backend == "qwen3":
            clone_dir = work_dir / "qwen3"
            backend_label = "Qwen3-TTS 1.7B"
        else:
            clone_dir = work_dir / "xtts"
            backend_label = "XTTS-v2"
        clone_dir.mkdir(parents=True, exist_ok=True)
        if log_cb:
            mode_label = "per-person" if per_person else "gender-specific"
            log_cb(f"Running {backend_label} voice clone with {mode_label} reference audio")
        active_segments = [segment for segment in segments if segment.enabled]
        if not active_segments:
            if progress_cb:
                progress_cb(100)
            return segments

        if emotion_clips is None and emotion_aware and source_wav is not None and source_wav.exists():
            from modules.emotion_reference import extract_emotion_clips
            if log_cb:
                log_cb("  Extracting per-segment emotion clips from source audio")
            clips_dir = work_dir / "emotion_clips"
            turns = diarization_turns or []
            emotion_clips = extract_emotion_clips(
                source_wav, active_segments, turns, clips_dir, cancel_event, log_cb,
            )

        if emotion_clips and emotion_analyses is None:
            from modules.emotion_detector import analyze_emotion_clips
            if log_cb:
                log_cb("  Analyzing emotional expression per segment")
            emotion_analyses = analyze_emotion_clips(emotion_clips)
            _log_emotion_analysis_summary(emotion_analyses, log_cb)

        if clone_backend == "cosyvoice":
            if per_person:
                _clone_per_person_batch_cosyvoice(
                    active_segments, clone_dir, speaker_voice_mappings,
                    log_cb, quality_report, cancel_event, progress_cb,
                    emotion_clips=emotion_clips,
                    emotion_analyses=emotion_analyses,
                    emotion_mode=emotion_mode,
                )
            else:
                _clone_gender_batch_cosyvoice(
                    active_segments, clone_dir, gender_reference_paths,
                    log_cb, quality_report, cancel_event, progress_cb,
                    emotion_clips=emotion_clips,
                    emotion_analyses=emotion_analyses,
                    emotion_mode=emotion_mode,
                    segment_genders=segment_genders,
                )
        elif clone_backend == "qwen3":
            if per_person:
                _clone_per_person_batch_qwen3(
                    active_segments, clone_dir, speaker_voice_mappings,
                    log_cb, quality_report, cancel_event, progress_cb,
                    emotion_clips=emotion_clips,
                    emotion_analyses=emotion_analyses,
                    emotion_mode=emotion_mode,
                )
            else:
                _clone_gender_batch_qwen3(
                    active_segments, clone_dir, gender_reference_paths,
                    log_cb, quality_report, cancel_event, progress_cb,
                    emotion_clips=emotion_clips,
                    emotion_analyses=emotion_analyses,
                    emotion_mode=emotion_mode,
                    segment_genders=segment_genders,
                )
        else:
            if per_person:
                _clone_per_person_batch_xtts(
                    active_segments, clone_dir, speaker_voice_mappings,
                    log_cb, quality_report, cancel_event, progress_cb,
                    emotion_clips=emotion_clips,
                    emotion_analyses=emotion_analyses,
                    emotion_mode=emotion_mode,
                )
            else:
                _clone_gender_batch_xtts(
                    active_segments, clone_dir, gender_reference_paths,
                    log_cb, quality_report, cancel_event, progress_cb,
                    emotion_clips=emotion_clips,
                    emotion_analyses=emotion_analyses,
                    emotion_mode=emotion_mode,
                    segment_genders=segment_genders,
                )
        _log_clone_result_summary(segments, enabled, log_cb)
        return segments

    if not command_template.strip():
        raise ValueError("Voice cloning is enabled but no command template was provided")
    _validate_command_setup(command_template)
    if clone_gender not in {"all", "female", "male"}:
        raise ValueError(f"Unsupported voice clone gender: {clone_gender}")
    if _uses_token(command_template, "model") and (model_path is None or not model_path.exists()):
        raise FileNotFoundError("RVC command uses {model}, but the .pth model file is missing")
    if _uses_token(command_template, "index") and (index_path is None or not index_path.exists()):
        raise FileNotFoundError("RVC command uses {index}, but the .index file is missing")
    has_gender_references = bool(gender_reference_paths)
    if not per_person and not has_gender_references and _uses_token(command_template, "reference") and (
        reference_audio_path is None or not reference_audio_path.exists()
    ):
        raise FileNotFoundError("RVC command uses {reference}, but the reference audio file is missing")

    clone_dir = work_dir / "rvc"
    clone_dir.mkdir(parents=True, exist_ok=True)

    if log_cb:
        if per_person:
            log_cb("Running external voice clone command with per-person reference audio")
        elif clone_gender == "all":
            log_cb("Running external voice clone command for all synthesized segments")
        else:
            log_cb(f"Running external voice clone command for {clone_gender} segments only")

    active_segments = [segment for segment in segments if segment.enabled]
    if not active_segments:
        if progress_cb:
            progress_cb(100)
        return segments

    use_batch = per_person and _is_openvoice_template(command_template)

    if use_batch:
        _clone_per_person_batch(
            active_segments, clone_dir, speaker_voice_mappings,
            log_cb, quality_report, cancel_event, progress_cb,
        )
    else:
        _clone_per_segment(
            active_segments, clone_dir, per_person, clone_gender, voice_gender,
            segment_genders, command_template, model_path, index_path,
            reference_audio_path, gender_reference_paths, has_gender_references,
            speaker_voice_mappings, log_cb, quality_report, cancel_event, progress_cb,
        )

    _log_clone_result_summary(segments, enabled, log_cb)
    return segments


def _clone_per_person_batch(
    active_segments: list[Segment],
    clone_dir: Path,
    speaker_voice_mappings: dict[str, dict[str, str]] | None,
    log_cb: LogCallback | None,
    quality_report,
    cancel_event: Event,
    progress_cb: ProgressCallback | None,
) -> None:
    """Group segments by speaker, clone each speaker's segments in one batch subprocess."""
    speaker_groups: dict[str, list[tuple[int, Segment]]] = defaultdict(list)
    skipped = 0

    for position, segment in enumerate(active_segments, start=1):
        ref = _reference_for_speaker(segment, speaker_voice_mappings)
        if ref is None or not ref.exists():
            label = segment.speaker_label or segment.speaker_id or "unknown"
            if log_cb:
                log_cb(f"  Warning: no reference audio for {label}; segment {position} will use normal TTS")
            skipped += 1
            continue
        tier = _speaker_quality_tier(segment, speaker_voice_mappings)
        if tier == "bad":
            label = segment.speaker_label or segment.speaker_id or "unknown"
            if log_cb:
                log_cb(f"  Skipping clone for {label} (segment {position}): reference failed quality gate")
            if quality_report is not None:
                quality_report.voice_clone_failures.append(
                    {"segment": segment.index + 1, "message": f"reference quality gate failed for {label}"}
                )
            skipped += 1
            continue
        if tier == "weak" and ref.exists():
            from modules.audio_utils import ffprobe_duration
            ref_duration = ffprobe_duration(ref)
            if ref_duration < 8.0:
                label = segment.speaker_label or segment.speaker_id or "unknown"
                if log_cb:
                    log_cb(f"  Skipping clone for {label} (segment {position}): weak reference too short ({ref_duration:.1f}s < 8s)")
                skipped += 1
                continue
        speaker_id = segment.speaker_id or "__default__"
        speaker_groups[speaker_id].append((position, segment))

    completed_speakers = 0
    total_segments = sum(len(segs) for segs in speaker_groups.values()) + skipped

    for speaker_id, speaker_segs in speaker_groups.items():
        if cancel_event.is_set():
            raise CancellationError("Processing cancelled by user")
        ref = _reference_for_speaker(speaker_segs[0][1], speaker_voice_mappings)
        _clone_batch_openvoice(speaker_segs, ref, clone_dir, log_cb, quality_report, cancel_event)
        completed_speakers += 1
        if progress_cb and total_segments > 0:
            done = skipped + sum(len(speaker_groups[s]) for s in list(speaker_groups)[:completed_speakers])
            progress_cb(int((done / total_segments) * 100) if total_segments > 0 else 100)

    if progress_cb:
        progress_cb(100)


def _clone_per_person_batch_xtts(
    active_segments: list[Segment],
    clone_dir: Path,
    speaker_voice_mappings: dict[str, dict[str, str]] | None,
    log_cb: LogCallback | None,
    quality_report,
    cancel_event: Event,
    progress_cb: ProgressCallback | None,
    emotion_clips: dict | None = None,
    emotion_analyses: dict | None = None,
    emotion_mode: str = "auto",
) -> None:
    """Group segments by speaker, clone each speaker's segments with XTTS-v2."""
    speaker_groups: dict[str, list[tuple[int, Segment]]] = defaultdict(list)
    skipped = 0

    for position, segment in enumerate(active_segments, start=1):
        ref = _reference_for_speaker(segment, speaker_voice_mappings)
        if ref is None or not ref.exists():
            label = segment.speaker_label or segment.speaker_id or "unknown"
            reason = _reference_fallback_reason(segment, speaker_voice_mappings)
            if log_cb:
                log_cb(
                    f"  Using default TTS for {label} (segment {position}): {reason}"
                )
            skipped += 1
            continue
        tier = _speaker_quality_tier(segment, speaker_voice_mappings)
        if tier == "bad":
            label = segment.speaker_label or segment.speaker_id or "unknown"
            if log_cb:
                log_cb(
                    f"  Using default TTS for {label} (segment {position}): reference failed quality gate"
                )
            skipped += 1
            continue
        speaker_id = segment.speaker_id or "__default__"
        speaker_groups[speaker_id].append((position, segment))

    completed_speakers = 0
    total_segments = sum(len(segs) for segs in speaker_groups.values()) + skipped

    for speaker_id, speaker_segs in speaker_groups.items():
        if cancel_event.is_set():
            raise CancellationError("Processing cancelled by user")
        ref = _reference_for_speaker(speaker_segs[0][1], speaker_voice_mappings)
        _clone_batch_xtts(
            speaker_segs, ref, clone_dir, log_cb, quality_report, cancel_event,
            emotion_clips, emotion_analyses, emotion_mode,
        )
        _release_gpu()
        completed_speakers += 1
        if progress_cb:
            done = skipped + sum(len(speaker_groups[s]) for s in list(speaker_groups)[:completed_speakers])
            progress_cb(int((done / total_segments) * 100) if total_segments > 0 else 100)

    if progress_cb:
        progress_cb(100)


def _clone_per_person_batch_cosyvoice(
    active_segments: list[Segment],
    clone_dir: Path,
    speaker_voice_mappings: dict[str, dict[str, str]] | None,
    log_cb: LogCallback | None,
    quality_report,
    cancel_event: Event,
    progress_cb: ProgressCallback | None,
    emotion_clips: dict | None = None,
    emotion_analyses: dict | None = None,
    emotion_mode: str = "auto",
) -> None:
    """Clone all per-speaker CosyVoice segments in one subprocess.

    CosyVoice model startup is expensive. Passing each segment's speaker
    reference in one manifest avoids reloading the model once per speaker.
    """
    from modules.cosyvoice_voice_clone import clone_batch, CosyVoiceCloneError

    batch_items = []
    segment_map: dict[int, Segment] = {}
    reference_map: dict[int, Path] = {}
    fallback_reference: Path | None = None
    speaker_ids: set[str] = set()
    skipped = 0

    for position, segment in enumerate(active_segments, start=1):
        if cancel_event.is_set():
            raise CancellationError("Processing cancelled by user")
        ref = _reference_for_speaker(segment, speaker_voice_mappings)
        if ref is None or not ref.exists():
            label = segment.speaker_label or segment.speaker_id or "unknown"
            reason = _reference_fallback_reason(segment, speaker_voice_mappings)
            if log_cb:
                log_cb(
                    f"  Using default TTS for {label} (segment {position}): {reason}"
                )
            skipped += 1
            continue
        tier = _speaker_quality_tier(segment, speaker_voice_mappings)
        if tier == "bad":
            label = segment.speaker_label or segment.speaker_id or "unknown"
            if log_cb:
                log_cb(
                    f"  Using default TTS for {label} (segment {position}): reference failed quality gate"
                )
            skipped += 1
            continue
        fallback_reference = fallback_reference or ref
        speaker_ids.add(segment.speaker_id or "__default__")

        output_path = clone_dir / f"{segment.index:05d}_cosyvoice.wav"
        if output_path.exists():
            segment.cloned_path = output_path
            skipped += 1
            continue
        input_path = segment.tts_path
        if input_path is None or not input_path.exists():
            if log_cb:
                log_cb(
                    f"  Warning: Missing Khmer TTS audio for segment {segment.index}; "
                    "CosyVoice clone skipped to avoid non-Khmer text synthesis"
                )
            if quality_report is not None:
                quality_report.voice_clone_failures.append(
                    {
                        "segment": segment.index + 1,
                        "message": "missing Khmer TTS source audio; skipped CosyVoice voice conversion",
                    }
                )
            skipped += 1
            continue

        item = {
            "segment_index": segment.index,
            "text": segment.tts_text,
            "input_path": str(input_path),
            "output_path": str(output_path),
            "reference_path": str(ref),
            "speaker_reference_path": str(ref),
        }
        _attach_segment_emotion(item, segment, emotion_clips, emotion_analyses, emotion_mode)
        batch_items.append(item)
        segment_map[segment.index] = segment
        reference_map[segment.index] = ref

    total_segments = len(batch_items) + skipped
    if not batch_items:
        if progress_cb:
            progress_cb(100)
        return

    if fallback_reference is None:
        if progress_cb:
            progress_cb(100)
        return

    if log_cb:
        log_cb(
            "  CosyVoice batch-cloning "
            f"{len(batch_items)} segments across {len(speaker_ids)} speakers (single model load)"
        )

    try:
        results = clone_batch(batch_items, fallback_reference, log_cb=log_cb)
    except CosyVoiceCloneError as exc:
        msg = f"CosyVoice batch failed: {str(exc)[:300]}"
        if quality_report is not None:
            quality_report.voice_clone_failures.append({"segment": -1, "message": msg})
        if log_cb:
            log_cb(f"  Warning: {msg}")
        if progress_cb:
            progress_cb(100)
        return

    completed = skipped
    for r in results:
        seg = segment_map.get(r["segment_index"])
        if seg is None:
            continue
        out = clone_dir / f"{seg.index:05d}_cosyvoice.wav"
        if r.get("ok") and out.exists():
            try:
                post_clone_match(out, reference_map.get(seg.index, fallback_reference))
            except Exception as exc:
                if log_cb:
                    log_cb(f"  Warning: post-clone matching failed for segment {seg.index}: {exc}")
            seg.cloned_path = out
        else:
            error = r.get("error", "unknown error")
            if quality_report is not None:
                quality_report.voice_clone_failures.append(
                    {"segment": seg.index + 1, "message": f"CosyVoice failed: {error}"}
                )
            if log_cb:
                log_cb(f"  Warning: CosyVoice clone failed for segment {seg.index}: {error}")
        completed += 1
        if progress_cb and total_segments > 0:
            progress_cb(int((completed / total_segments) * 100))

    if progress_cb:
        progress_cb(100)


def _clone_per_person_batch_qwen3(
    active_segments: list[Segment],
    clone_dir: Path,
    speaker_voice_mappings: dict[str, dict[str, str]] | None,
    log_cb: LogCallback | None,
    quality_report,
    cancel_event: Event,
    progress_cb: ProgressCallback | None,
    emotion_clips: dict | None = None,
    emotion_analyses: dict | None = None,
    emotion_mode: str = "auto",
) -> None:
    """Clone all per-speaker Qwen3-TTS segments in one subprocess."""
    from modules.qwen3_voice_clone import clone_batch, Qwen3CloneError

    batch_items = []
    segment_map: dict[int, Segment] = {}
    item_map: dict[int, dict] = {}
    reference_map: dict[int, Path] = {}
    fallback_reference: Path | None = None
    speaker_ids: set[str] = set()
    skipped = 0

    for position, segment in enumerate(active_segments, start=1):
        if cancel_event.is_set():
            raise CancellationError("Processing cancelled by user")
        ref = _reference_for_speaker(segment, speaker_voice_mappings)
        if ref is None or not ref.exists():
            label = segment.speaker_label or segment.speaker_id or "unknown"
            reason = _reference_fallback_reason(segment, speaker_voice_mappings)
            if log_cb:
                log_cb(f"  Using default TTS for {label} (segment {position}): {reason}")
            skipped += 1
            continue
        tier = _speaker_quality_tier(segment, speaker_voice_mappings)
        if tier == "bad":
            label = segment.speaker_label or segment.speaker_id or "unknown"
            if log_cb:
                log_cb(
                    f"  Using default TTS for {label} (segment {position}): reference failed quality gate"
                )
            skipped += 1
            continue
        fallback_reference = fallback_reference or ref
        speaker_key = segment.speaker_id or "__default__"
        speaker_ids.add(speaker_key)

        output_path = clone_dir / f"{segment.index:05d}_qwen3.wav"
        if output_path.exists():
            segment.cloned_path = output_path
            skipped += 1
            continue

        item = {
            "segment_index": segment.index,
            "text": segment.tts_text,
            "output_path": str(output_path),
            "speaker_reference_path": str(ref),
            "emotion_ref_text": _segment_source_text(segment),
            "ref_text": _segment_ref_text(segment),
            "prompt_cache_key": speaker_key,
        }
        _attach_segment_emotion(item, segment, emotion_clips, emotion_analyses, emotion_mode)
        batch_items.append(item)
        item_map[segment.index] = item
        segment_map[segment.index] = segment
        reference_map[segment.index] = ref

    total_segments = len(batch_items) + skipped
    if not batch_items or fallback_reference is None:
        if progress_cb:
            progress_cb(100)
        return

    if log_cb:
        log_cb(
            "  Qwen3-TTS batch-cloning "
            f"{len(batch_items)} segments across {len(speaker_ids)} speakers (single model load)"
        )

    def update_batch_progress(done: int, total: int) -> None:
        if progress_cb and total_segments > 0:
            progress_cb(int(((skipped + done) / total_segments) * 100))

    try:
        results = clone_batch(
            batch_items, fallback_reference, log_cb=log_cb, progress_cb=update_batch_progress,
        )
    except Qwen3CloneError as exc:
        msg = f"Qwen3-TTS batch failed: {str(exc)[:300]}"
        if quality_report is not None:
            quality_report.voice_clone_failures.append({"segment": -1, "message": msg})
        if log_cb:
            log_cb(f"  Warning: {msg}")
        if progress_cb:
            progress_cb(100)
        return

    completed = skipped
    for r in results:
        seg = segment_map.get(r["segment_index"])
        if seg is None:
            continue
        out = clone_dir / f"{seg.index:05d}_qwen3.wav"
        if r.get("ok") and out.exists():
            try:
                post_clone_match(
                    out,
                    _post_clone_reference_path(
                        item_map.get(seg.index),
                        reference_map.get(seg.index, fallback_reference),
                    ),
                )
            except Exception as exc:
                if log_cb:
                    log_cb(f"  Warning: post-clone matching failed for segment {seg.index}: {exc}")
            seg.cloned_path = out
        else:
            error = r.get("error", "unknown error")
            if quality_report is not None:
                quality_report.voice_clone_failures.append(
                    {"segment": seg.index + 1, "message": f"Qwen3-TTS failed: {error}"}
                )
            if log_cb:
                log_cb(f"  Warning: Qwen3-TTS clone failed for segment {seg.index}: {error}")
        completed += 1
        if progress_cb and total_segments > 0:
            progress_cb(int((completed / total_segments) * 100))

    if progress_cb:
        progress_cb(100)


def _clone_gender_batch_qwen3(
    active_segments: list[Segment],
    clone_dir: Path,
    gender_reference_paths: dict[str, Path] | None,
    log_cb: LogCallback | None,
    quality_report,
    cancel_event: Event,
    progress_cb: ProgressCallback | None,
    emotion_clips: dict | None = None,
    emotion_analyses: dict | None = None,
    emotion_mode: str = "auto",
    segment_genders: dict[int, str] | None = None,
) -> None:
    """Clone gender-profile Qwen3-TTS segments in one subprocess."""
    from modules.qwen3_voice_clone import clone_batch, Qwen3CloneError

    grouped = _group_segments_by_gender(
        active_segments, gender_reference_paths, segment_genders, log_cb,
    )
    grouped_count = sum(len(segments) for segments in grouped.values())
    skipped = len(active_segments) - grouped_count

    batch_items = []
    segment_map: dict[int, Segment] = {}
    item_map: dict[int, dict] = {}
    reference_map: dict[int, Path] = {}
    fallback_reference: Path | None = None

    for gender, gender_segments in grouped.items():
        reference_path = gender_reference_paths.get(gender) if gender_reference_paths else None
        if reference_path is None or not reference_path.exists():
            skipped += len(gender_segments)
            continue
        fallback_reference = fallback_reference or reference_path
        prompt_cache_key = f"gender::{gender}"

        for _position, segment in gender_segments:
            if cancel_event.is_set():
                raise CancellationError("Processing cancelled by user")
            output_path = clone_dir / f"{segment.index:05d}_qwen3.wav"
            if output_path.exists():
                segment.cloned_path = output_path
                skipped += 1
                continue

            item = {
                "segment_index": segment.index,
                "text": segment.tts_text,
                "output_path": str(output_path),
                "speaker_reference_path": str(reference_path),
                "emotion_ref_text": _segment_source_text(segment),
                "ref_text": _segment_ref_text(segment),
                "prompt_cache_key": prompt_cache_key,
            }
            _attach_segment_emotion(item, segment, emotion_clips, emotion_analyses, emotion_mode)
            batch_items.append(item)
            item_map[segment.index] = item
            segment_map[segment.index] = segment
            reference_map[segment.index] = reference_path

    total_segments = len(batch_items) + skipped
    if not batch_items or fallback_reference is None:
        if progress_cb:
            progress_cb(100)
        return

    if log_cb:
        log_cb(
            "  Qwen3-TTS batch-cloning "
            f"{len(batch_items)} gender-specific segments across {len(grouped)} voice profiles "
            "(single model load)"
        )

    def update_batch_progress(done: int, total: int) -> None:
        if progress_cb and total_segments > 0:
            progress_cb(int(((skipped + done) / total_segments) * 100))

    try:
        results = clone_batch(
            batch_items, fallback_reference, log_cb=log_cb, progress_cb=update_batch_progress,
        )
    except Qwen3CloneError as exc:
        msg = f"Qwen3-TTS batch failed: {str(exc)[:300]}"
        if quality_report is not None:
            quality_report.voice_clone_failures.append({"segment": -1, "message": msg})
        if log_cb:
            log_cb(f"  Warning: {msg}")
        if progress_cb:
            progress_cb(100)
        raise

    completed = skipped
    for r in results:
        seg = segment_map.get(r["segment_index"])
        if seg is None:
            continue
        out = clone_dir / f"{seg.index:05d}_qwen3.wav"
        if r.get("ok") and out.exists():
            try:
                post_clone_match(
                    out,
                    _post_clone_reference_path(
                        item_map.get(seg.index),
                        reference_map.get(seg.index, fallback_reference),
                    ),
                )
            except Exception as exc:
                if log_cb:
                    log_cb(f"  Warning: post-clone matching failed for segment {seg.index}: {exc}")
            seg.cloned_path = out
        else:
            error = r.get("error", "unknown error")
            if quality_report is not None:
                quality_report.voice_clone_failures.append(
                    {"segment": seg.index + 1, "message": f"Qwen3-TTS failed: {error}"}
                )
            if log_cb:
                log_cb(f"  Warning: Qwen3-TTS clone failed for segment {seg.index}: {error}")
        completed += 1
        if progress_cb and total_segments > 0:
            progress_cb(int((completed / total_segments) * 100))

    if progress_cb:
        progress_cb(100)


def _clone_per_segment(
    active_segments: list[Segment],
    clone_dir: Path,
    per_person: bool,
    clone_gender: str,
    voice_gender: str,
    segment_genders: dict[int, str] | None,
    command_template: str,
    model_path: Path | None,
    index_path: Path | None,
    reference_audio_path: Path | None,
    gender_reference_paths: dict[str, Path] | None,
    has_gender_references: bool,
    speaker_voice_mappings: dict[str, dict[str, str]] | None,
    log_cb: LogCallback | None,
    quality_report,
    cancel_event: Event,
    progress_cb: ProgressCallback | None,
) -> None:
    """Original per-segment subprocess path (used for RVC and other non-OpenVoice backends)."""
    completed = 0
    completed_lock = Lock()

    def process_segment(position: int, segment: Segment) -> None:
        nonlocal completed
        if cancel_event.is_set():
            raise CancellationError("Processing cancelled by user")

        current_gender = _segment_gender(segment, voice_gender, segment_genders)
        if not per_person and clone_gender != "all" and current_gender != clone_gender:
            if log_cb:
                log_cb(f"  Skipping segment {position}/{len(active_segments)} ({current_gender})")
            return

        segment_reference_audio = reference_audio_path
        if per_person:
            segment_reference_audio = _reference_for_speaker(segment, speaker_voice_mappings)
            if segment_reference_audio is None or not segment_reference_audio.exists():
                label = segment.speaker_label or segment.speaker_id or "unknown speaker"
                if log_cb:
                    log_cb(
                        f"  Warning: no reference audio for {label}; "
                        f"segment {position}/{len(active_segments)} will use normal TTS"
                    )
                return
            tier = _speaker_quality_tier(segment, speaker_voice_mappings)
            if tier == "bad":
                label = segment.speaker_label or segment.speaker_id or "unknown speaker"
                if log_cb:
                    log_cb(
                        f"  Skipping voice clone for {label} (segment {position}/{len(active_segments)}): "
                        f"reference audio failed quality gate; using base TTS instead"
                    )
                if quality_report is not None:
                    quality_report.voice_clone_failures.append(
                        {
                            "segment": segment.index + 1,
                            "message": f"reference quality gate failed for {label}; skipped clone",
                        }
                    )
                return
        elif has_gender_references:
            segment_reference_audio = gender_reference_paths.get(current_gender) if gender_reference_paths else None
            if segment_reference_audio is None or not segment_reference_audio.exists():
                if log_cb:
                    log_cb(
                        f"  Skipping segment {position}/{len(active_segments)}: no {current_gender} voice profile reference"
                    )
                return

        input_path = segment.tts_path
        if input_path is None or not input_path.exists():
            if log_cb:
                log_cb(f"  Warning: Missing TTS audio for segment {segment.index}")
            return

        output_path = clone_dir / f"{segment.index:05d}_rvc.wav"

        if output_path.exists():
            segment.cloned_path = output_path
            return

        try:
            command = _render_command(
                command_template,
                input_path,
                output_path,
                model_path,
                index_path,
                segment_reference_audio,
            )
        except Exception as e:
            if log_cb:
                log_cb(f"  Warning: {e}")
            return

        if log_cb:
            log_cb(f"  Cloning voice for segment {position}/{len(active_segments)}: \"{segment.tts_text[:40]}...\"")
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            stdout = ""
            stderr = ""
            while True:
                try:
                    stdout, stderr = process.communicate(timeout=0.1)
                    break
                except subprocess.TimeoutExpired:
                    pass
                if cancel_event.is_set():
                    raise CancellationError("Processing cancelled by user")
        except Exception:
            process.terminate()
            try:
                stdout, stderr = process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
            raise
        if process.returncode != 0:
            message = f"RVC command failed for segment {segment.index}"
            details = (stderr or stdout or "").strip()
            if details:
                message = f"{message}: {details[-1200:]}"
            if quality_report is not None:
                quality_report.voice_clone_failures.append({"segment": segment.index + 1, "message": message})
            if log_cb:
                log_cb(f"  Warning: {message}; using normal TTS for this segment")
            return
        if not output_path.exists():
            message = f"RVC command completed but did not create {output_path}"
            if quality_report is not None:
                quality_report.voice_clone_failures.append({"segment": segment.index + 1, "message": message})
            if log_cb:
                log_cb(f"  Warning: {message}; using normal TTS for this segment")
            return

        segment.cloned_path = output_path

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = []
        for position, segment in enumerate(active_segments, start=1):
            futures.append(executor.submit(process_segment, position, segment))

        for future in concurrent.futures.as_completed(futures):
            future.result()
            with completed_lock:
                completed += 1
                current = completed
            if progress_cb:
                progress_cb(int((current / len(active_segments)) * 100))
