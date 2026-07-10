from __future__ import annotations

import shutil
from collections import defaultdict
from pathlib import Path
from threading import Event
from typing import Callable

from core.context import CancellationError
from modules.audio_quality import media_cache_key
from modules.audio_utils import concat_wavs, ffprobe_duration, trim_audio_segment
from modules.diarizer import SpeakerTurn, speaker_ids_from_turns
from modules.reference_quality import ReferenceQuality, assess_reference


LogCallback = Callable[[str], None]

MIN_TURN_SECONDS = 1.5
TURN_PADDING_SECONDS = 0.15
TARGET_REFERENCE_SECONDS = 30.0
MAX_QUALITY_CANDIDATES = 12
MIN_AUTO_REFERENCE_SECONDS = 15.0


def _safe_id(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value).strip("_") or "speaker"


def _has_other_speaker_overlap(turn: SpeakerTurn, turns: list[SpeakerTurn]) -> bool:
    for other in turns:
        if other.speaker_id == turn.speaker_id:
            continue
        overlap = min(turn.end, other.end) - max(turn.start, other.start)
        if overlap > 0.05:
            return True
    return False


def _clean_turns_by_speaker(turns: list[SpeakerTurn]) -> dict[str, list[SpeakerTurn]]:
    grouped: dict[str, list[SpeakerTurn]] = defaultdict(list)
    for turn in turns:
        duration = turn.end - turn.start
        if duration < MIN_TURN_SECONDS:
            continue
        if _has_other_speaker_overlap(turn, turns):
            continue
        grouped[turn.speaker_id].append(turn)

    # For extreme precision, do not fallback to overlapping turns.
    return {speaker_id: sorted(items, key=lambda item: (item.start, item.end)) for speaker_id, items in grouped.items()}


def _select_reference_turns(turns: list[SpeakerTurn], min_seconds: float) -> list[SpeakerTurn]:
    selected: list[SpeakerTurn] = []
    total = 0.0
    target_seconds = max(min_seconds, min(TARGET_REFERENCE_SECONDS, max(min_seconds, 10.0)))
    for turn in turns:
        selected.append(turn)
        total += turn.end - turn.start
        if total >= target_seconds:
            break
    return selected


def _select_reference_turns_longest_first(turns: list[SpeakerTurn], min_seconds: float) -> list[SpeakerTurn]:
    """Alternative selector used to auto-repair a bad reference: prefer the
    longest, presumably cleanest turns first."""
    target_seconds = max(min_seconds, min(TARGET_REFERENCE_SECONDS, max(min_seconds, 10.0)))
    by_length = sorted(turns, key=lambda t: (t.end - t.start), reverse=True)
    selected: list[SpeakerTurn] = []
    total = 0.0
    for turn in by_length:
        selected.append(turn)
        total += turn.end - turn.start
        if total >= target_seconds:
            break
    return sorted(selected, key=lambda t: (t.start, t.end))


def _select_natural_reference_turns(
    source_wav: Path,
    turns: list[SpeakerTurn],
    min_seconds: float,
    clips_dir: Path,
    safe_speaker_id: str,
    cancel_event: Event,
    log_cb: LogCallback | None = None,
) -> list[SpeakerTurn]:
    """Prefer clean, speech-heavy turns for a natural per-speaker clone reference."""
    if not turns:
        return []

    longest_candidates = sorted(turns, key=lambda t: (t.end - t.start), reverse=True)[:MAX_QUALITY_CANDIDATES]
    scored: list[tuple[ReferenceQuality, float, SpeakerTurn]] = []
    for index, turn in enumerate(longest_candidates, start=1):
        if cancel_event.is_set():
            raise CancellationError("Processing cancelled by user")
        duration = turn.end - turn.start
        if duration <= 0:
            continue
        candidate_path = clips_dir / f"{safe_speaker_id}_quality_{index:03d}.wav"
        try:
            start = max(0.0, turn.start + TURN_PADDING_SECONDS)
            end = max(start + 0.01, turn.end - TURN_PADDING_SECONDS)
            trim_audio_segment(source_wav, candidate_path, start, end - start, cancel_event)
            quality = assess_reference(candidate_path)
        except Exception as exc:
            if log_cb:
                log_cb(f"    Could not score turn {turn.start:.2f}s-{turn.end:.2f}s for {safe_speaker_id}: {exc}")
            continue
        scored.append((quality, duration, turn))

    if not scored:
        return _select_reference_turns_longest_first(turns, min_seconds)

    tier_rank = {"good": 2, "weak": 1, "bad": 0}
    scored.sort(key=lambda item: (tier_rank.get(item[0].tier, 0), item[0].score, item[1]), reverse=True)
    usable = [item for item in scored if item[0].tier != "bad"]
    ranked = usable if usable else scored

    selected: list[SpeakerTurn] = []
    total = 0.0
    target_seconds = max(min_seconds, min(TARGET_REFERENCE_SECONDS, max(min_seconds, 10.0)))
    for quality, duration, turn in ranked:
        selected.append(turn)
        total += duration
        if total >= target_seconds:
            break

    if log_cb and selected:
        best_quality = ranked[0][0]
        log_cb(
            f"    Selected natural reference turns for {safe_speaker_id}: "
            f"best {best_quality.tier} score {best_quality.score:.0f}/100"
        )

    return sorted(selected, key=lambda t: (t.start, t.end))


def _build_reference_from_turns(
    source_wav: Path,
    selected_turns: list[SpeakerTurn],
    output_path: Path,
    clips_dir: Path,
    safe_speaker_id: str,
    cancel_event: Event,
) -> list[Path]:
    clip_paths: list[Path] = []
    for index, turn in enumerate(selected_turns, start=1):
        start = max(0.0, turn.start + TURN_PADDING_SECONDS)
        end = max(start + 0.01, turn.end - TURN_PADDING_SECONDS)
        clip_path = clips_dir / f"{safe_speaker_id}_{index:03d}.wav"
        trim_audio_segment(source_wav, clip_path, start, end - start, cancel_event)
        clip_paths.append(clip_path)
    concat_wavs(clip_paths, output_path, clips_dir / f"{safe_speaker_id}_concat.txt", cancel_event)
    return clip_paths


def _total_turn_seconds(turns: list[SpeakerTurn]) -> float:
    return sum(max(0.0, turn.end - turn.start) for turn in turns)


def speaker_has_clone_reference(
    speaker_id: str | None,
    speaker_voice_mappings: dict[str, dict[str, str]] | None,
) -> bool:
    """Return True when a diarized speaker has usable audio for voice cloning."""
    if not speaker_id or not speaker_voice_mappings:
        return False
    mapping = speaker_voice_mappings.get(speaker_id, {})
    if mapping.get("reference_status", "").strip().lower() == "missing":
        return False
    reference = (
        mapping.get("cleaned_reference_audio_path", "").strip()
        or mapping.get("reference_audio_path", "").strip()
        or mapping.get("original_reference_audio_path", "").strip()
    )
    if not reference:
        return False
    if not Path(reference).expanduser().exists():
        return False
    tier = mapping.get("quality_tier", "").strip().lower()
    return tier != "bad"


def _missing_auto_reference_mapping(speaker_id: str, reason: str) -> dict[str, str]:
    return {
        "label": speaker_id.replace("_", " ").title(),
        "reference_audio_path": "",
        "original_reference_audio_path": "",
        "cleaned_reference_audio_path": "",
        "reference_status": "missing",
        "auto_reference": "true",
        "fallback_voice": "default_tts",
        "fallback_reason": reason,
    }


def _copy_cached_reference(cache_path: Path, job_path: Path) -> Path:
    job_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cache_path, job_path)
    return job_path


def build_auto_speaker_references(
    source_wav: Path,
    source_media: Path,
    turns: list[SpeakerTurn],
    work_dir: Path,
    min_reference_seconds: float,
    cancel_event: Event,
    persistent_cache_dir: Path | None = None,
    log_cb: LogCallback | None = None,
) -> dict[str, dict[str, str]]:
    if not turns:
        return {}

    reference_dir = work_dir / "auto_speaker_references"
    reference_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = reference_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    cache_root = None
    if persistent_cache_dir is not None:
        cache_root = persistent_cache_dir / "auto_speaker_references" / media_cache_key(source_media)
        cache_root.mkdir(parents=True, exist_ok=True)

    turns_by_speaker = _clean_turns_by_speaker(turns)
    mappings: dict[str, dict[str, str]] = {}

    for speaker_id in speaker_ids_from_turns(turns):
        if cancel_event.is_set():
            raise CancellationError("Processing cancelled by user")

        safe_speaker_id = _safe_id(speaker_id)
        output_path = reference_dir / f"{safe_speaker_id}.wav"
        cached_path = cache_root / f"{safe_speaker_id}.wav" if cache_root is not None else None
        min_required_seconds = max(min_reference_seconds, MIN_AUTO_REFERENCE_SECONDS)
        if cached_path is not None and cached_path.exists():
            _copy_cached_reference(cached_path, output_path)
            if log_cb:
                log_cb(f"  Using cached auto reference for {speaker_id}: {output_path}")
        else:
            speaker_turns = turns_by_speaker.get(speaker_id, [])
            verified_turns = []
            if speaker_turns:
                longest_turn = max(speaker_turns, key=lambda item: item.end - item.start)
                verified_turns.append(longest_turn)
                try:
                    from modules.speaker_verification import get_segment_embedding, compute_similarity
                    if log_cb:
                        log_cb(f"  Verifying speaker turns for {speaker_id} using SpeechBrain ECAPA-TDNN")
                    anchor_emb = get_segment_embedding(source_wav, longest_turn.start, longest_turn.end - longest_turn.start)
                    for turn in speaker_turns:
                        if turn == longest_turn:
                            continue
                        turn_emb = get_segment_embedding(source_wav, turn.start, turn.end - turn.start)
                        sim = compute_similarity(anchor_emb, turn_emb)
                        if sim >= 0.78:
                            verified_turns.append(turn)
                        else:
                            if log_cb:
                                log_cb(f"    Discarding turn ({turn.start:.2f}s - {turn.end:.2f}s) for {speaker_id} due to low similarity ({sim:.2f} < 0.78)")
                except Exception as exc:
                    if log_cb:
                        log_cb(
                            f"    Speaker verification failed: {exc}. "
                            "Using all clean non-overlapping turns for this diarized speaker."
                        )
                    verified_turns = list(speaker_turns)
            
            verified_turns.sort(key=lambda item: (item.start, item.end))
            verified_seconds = _total_turn_seconds(verified_turns)
            if verified_seconds < min_required_seconds:
                reason = (
                    f"only {verified_seconds:.1f}s clean speech "
                    f"(< {min_required_seconds:.1f}s required)"
                )
                if log_cb:
                    log_cb(
                        f"  Auto reference skipped for {speaker_id}: {reason}; "
                        "using default TTS for this speaker"
                    )
                mappings[speaker_id] = _missing_auto_reference_mapping(speaker_id, reason)
                continue
            selected_turns = _select_natural_reference_turns(
                source_wav,
                verified_turns,
                min_required_seconds,
                clips_dir,
                safe_speaker_id,
                cancel_event,
                log_cb,
            )
            if not selected_turns:
                reason = "no clean diarized speech found"
                if log_cb:
                    log_cb(f"  Warning: {reason} for {speaker_id}; using default TTS for this speaker")
                mappings[speaker_id] = _missing_auto_reference_mapping(speaker_id, reason)
                continue
            selected_seconds = _total_turn_seconds(selected_turns)
            if selected_seconds < min_required_seconds:
                reason = (
                    f"selected clean speech is {selected_seconds:.1f}s "
                    f"(< {min_required_seconds:.1f}s required)"
                )
                if log_cb:
                    log_cb(
                        f"  Auto reference skipped for {speaker_id}: {reason}; "
                        "using default TTS for this speaker"
                    )
                mappings[speaker_id] = _missing_auto_reference_mapping(speaker_id, reason)
                continue

            clip_paths = _build_reference_from_turns(
                source_wav, selected_turns, output_path, clips_dir, safe_speaker_id, cancel_event
            )

            # Quality gate: assess the newly built reference. If it comes out
            # "bad" and we have enough alternate turns, try once more with a
            # longest-first selection strategy before accepting the result.
            initial_quality = assess_reference(output_path)
            if initial_quality.tier == "bad" and len(verified_turns) > len(selected_turns):
                if log_cb:
                    log_cb(
                        f"  Reference for {speaker_id} scored {initial_quality.score:.0f}/100 "
                        f"({initial_quality.tier}); retrying with longest-turn selection"
                    )
                retry_turns = _select_reference_turns_longest_first(verified_turns, min_required_seconds)
                if retry_turns and retry_turns != selected_turns:
                    clip_paths = _build_reference_from_turns(
                        source_wav, retry_turns, output_path, clips_dir, safe_speaker_id, cancel_event
                    )

            built_duration = ffprobe_duration(output_path)
            if built_duration < min_required_seconds:
                reason = f"merged reference is {built_duration:.1f}s (< {min_required_seconds:.1f}s required)"
                if log_cb:
                    log_cb(
                        f"  Auto reference skipped for {speaker_id}: {reason}; "
                        "using default TTS for this speaker"
                    )
                mappings[speaker_id] = _missing_auto_reference_mapping(speaker_id, reason)
                continue

            if cached_path is not None:
                shutil.copy2(output_path, cached_path)

            if log_cb:
                log_cb(
                    f"  Built auto reference for {speaker_id}: {built_duration:.1f}s "
                    f"from {len(clip_paths)} clean turn(s)"
                )

        if output_path.exists():
            cached_or_built_duration = ffprobe_duration(output_path)
            if cached_or_built_duration < min_required_seconds:
                reason = (
                    f"merged reference is {cached_or_built_duration:.1f}s "
                    f"(< {min_required_seconds:.1f}s required)"
                )
                if log_cb:
                    log_cb(
                        f"  Auto reference skipped for {speaker_id}: {reason}; "
                        "using default TTS for this speaker"
                    )
                mappings[speaker_id] = _missing_auto_reference_mapping(speaker_id, reason)
                continue

        # Cross-video global speaker verification & matching
        display_label = speaker_id.replace("_", " ").title()
        if persistent_cache_dir is not None and output_path.exists():
            global_registry_dir = persistent_cache_dir / "global_speaker_registry"
            global_registry_dir.mkdir(parents=True, exist_ok=True)
            try:
                from modules.speaker_verification import get_file_embedding, compute_similarity
                import torch
                
                # Compute embedding of the candidate speaker
                candidate_emb = get_file_embedding(output_path)
                
                # Compare against all existing registered speakers
                matched_global_id = None
                best_sim = -1.0
                
                for pt_file in global_registry_dir.glob("*.pt"):
                    try:
                        registered_emb = torch.load(pt_file, weights_only=True)
                        sim = compute_similarity(candidate_emb, registered_emb)
                        if sim > best_sim:
                            best_sim = sim
                            if sim >= 0.88:
                                matched_global_id = pt_file.stem
                    except Exception:
                        continue
                
                if matched_global_id is not None:
                    if log_cb:
                        log_cb(
                            f"  Matched local speaker {speaker_id} to global profile {matched_global_id} "
                            f"(similarity: {best_sim:.2f}); keeping this video's local reference "
                            "to preserve per-speaker voice differences"
                        )
                else:
                    # Register as a new global speaker profile if the match is low (not in gray-zone)
                    if best_sim < 0.75:
                        existing_ids = [
                            int(f.stem.split("_")[-1]) 
                            for f in global_registry_dir.glob("global_speaker_*.pt") 
                            if f.stem.split("_")[-1].isdigit()
                        ]
                        next_num = max(existing_ids) + 1 if existing_ids else 1
                        new_global_id = f"global_speaker_{next_num}"
                        
                        if log_cb:
                            log_cb(f"  Registering local speaker {speaker_id} as new global identity: {new_global_id} (best similarity: {best_sim:.2f})")
                        
                        torch.save(candidate_emb, global_registry_dir / f"{new_global_id}.pt")
                        shutil.copy2(output_path, global_registry_dir / f"{new_global_id}.wav")
                        display_label = new_global_id.replace("_", " ").title()
                    else:
                        if log_cb:
                            log_cb(f"  Local speaker {speaker_id} is in the similarity gray-zone ({best_sim:.2f}). Treating as separate speaker but not enrolling globally to prevent corruption.")
            except Exception as exc:
                if log_cb:
                    log_cb(f"  Warning: global speaker matching failed for {speaker_id}: {exc}")

        final_quality: ReferenceQuality | None = None
        if output_path.exists():
            try:
                final_quality = assess_reference(output_path)
            except Exception as exc:
                if log_cb:
                    log_cb(f"  Warning: quality assessment failed for {speaker_id}: {exc}")

        mapping: dict[str, str] = {
            "label": display_label,
            "reference_audio_path": str(output_path),
            "original_reference_audio_path": str(output_path),
            "cleaned_reference_audio_path": "",
            "reference_status": "auto",
            "auto_reference": "true",
        }
        if final_quality is not None:
            mapping["quality_tier"] = final_quality.tier
            mapping["quality_score"] = f"{final_quality.score:.1f}"
            mapping["quality_reasons"] = "; ".join(final_quality.reasons)
            if log_cb:
                tag = "OK" if final_quality.tier == "good" else final_quality.tier.upper()
                extras = f" — {'; '.join(final_quality.reasons)}" if final_quality.reasons else ""
                log_cb(
                    f"  Quality gate for {speaker_id}: {tag} "
                    f"(score {final_quality.score:.0f}/100, SNR {final_quality.snr_db:.1f} dB, "
                    f"voiced {final_quality.voiced_ratio*100:.0f}%){extras}"
                )
        mappings[speaker_id] = mapping

    return mappings
