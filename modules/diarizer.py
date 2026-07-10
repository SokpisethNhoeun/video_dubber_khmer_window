from __future__ import annotations

import os
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Callable

from core.context import CancellationError, Segment


ProgressCallback = Callable[[int], None]
LogCallback = Callable[[str], None]


SEGMENT_VERIFICATION_MIN_DURATION_SECONDS = 0.8
SEGMENT_VERIFICATION_REJECT_THRESHOLD = 0.25
SEGMENT_VERIFICATION_LOW_CONFIDENCE_THRESHOLD = 0.40


@dataclass(frozen=True)
class SpeakerTurn:
    start: float
    end: float
    speaker_id: str

    def to_dict(self) -> dict[str, float | str]:
        return {"start": self.start, "end": self.end, "speaker_id": self.speaker_id}


class DiarizationUnavailableError(RuntimeError):
    """Raised when optional diarization dependencies or credentials are missing."""


def _patch_huggingface_hub_legacy_auth_keyword() -> None:
    """Allow pyannote 3.x to run with newer huggingface_hub releases."""
    try:
        import huggingface_hub
    except ImportError:
        return

    hf_hub_download = getattr(huggingface_hub, "hf_hub_download", None)
    if hf_hub_download is None or getattr(hf_hub_download, "_video_dubber_auth_patch", False):
        return

    parameters = inspect.signature(hf_hub_download).parameters
    if "use_auth_token" in parameters or "token" not in parameters:
        return

    def hf_hub_download_with_legacy_auth(*args, **kwargs):
        use_auth_token = kwargs.pop("use_auth_token", None)
        if "token" not in kwargs and use_auth_token is not None:
            kwargs["token"] = use_auth_token
        return hf_hub_download(*args, **kwargs)

    hf_hub_download_with_legacy_auth._video_dubber_auth_patch = True
    huggingface_hub.hf_hub_download = hf_hub_download_with_legacy_auth
    for module_name in (
        "pyannote.audio.core.pipeline",
        "pyannote.audio.core.model",
        "pyannote.audio.pipelines.speaker_verification",
    ):
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, "hf_hub_download"):
            module.hf_hub_download = hf_hub_download_with_legacy_auth


def _allowlist_pytorch_checkpoint_globals() -> None:
    """Allow pyannote checkpoints to load under PyTorch 2.6 weights-only defaults."""
    try:
        import torch
        from pyannote.audio.core.task import Problem, Resolution, Specifications
        from torch.torch_version import TorchVersion
    except ImportError:
        return

    add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
    if add_safe_globals is not None:
        add_safe_globals([TorchVersion, Specifications, Resolution, Problem])


def turns_from_dicts(raw_turns: list[dict[str, float | str]]) -> list[SpeakerTurn]:
    turns: list[SpeakerTurn] = []
    for raw_turn in raw_turns:
        turns.append(
            SpeakerTurn(
                start=float(raw_turn["start"]),
                end=float(raw_turn["end"]),
                speaker_id=str(raw_turn["speaker_id"]),
            )
        )
    return turns


def speaker_ids_from_turns(turns: list[SpeakerTurn]) -> list[str]:
    seen: set[str] = set()
    speaker_ids: list[str] = []
    for turn in sorted(turns, key=lambda item: (item.start, item.end)):
        if turn.speaker_id not in seen:
            seen.add(turn.speaker_id)
            speaker_ids.append(turn.speaker_id)
    return speaker_ids


def _is_auto_reference_mapping(mapping: dict[str, str]) -> bool:
    return (
        mapping.get("auto_reference", "").strip().lower() == "true"
        or mapping.get("reference_status", "").strip().lower() == "auto"
    )


def _annotation_from_diarization_output(diarization):
    if hasattr(diarization, "itertracks"):
        return diarization
    for attribute in ("exclusive_speaker_diarization", "speaker_diarization"):
        annotation = getattr(diarization, attribute, None)
        if annotation is not None and hasattr(annotation, "itertracks"):
            return annotation
    raise TypeError(
        "Unsupported pyannote diarization output. Expected an Annotation-like object "
        "or a DiarizeOutput with speaker_diarization."
    )


def detect_speakers(
    audio_wav: Path,
    device: str,
    progress_cb: ProgressCallback | None,
    log_cb: LogCallback | None,
    cancel_event: Event,
) -> list[SpeakerTurn]:
    if not audio_wav.exists():
        raise FileNotFoundError(f"Audio file does not exist: {audio_wav}")

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        available = ", ".join(
            name for name in ("HF_TOKEN", "HUGGINGFACE_TOKEN") if name in os.environ
        )
        detail = f" Visible token variables: {available}." if available else " No token variables are visible to this app process."
        raise DiarizationUnavailableError(
            "Speaker diarization requires HF_TOKEN or HUGGINGFACE_TOKEN in the environment."
            f"{detail} If you use a launcher, put HF_TOKEN in the project .env file and restart the app."
        )

    try:
        _patch_huggingface_hub_legacy_auth_keyword()
        _allowlist_pytorch_checkpoint_globals()
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise DiarizationUnavailableError(
            "Speaker diarization requires optional package pyannote.audio. Install it to use per-person voices."
        ) from exc

    if cancel_event.is_set():
        raise CancellationError("Processing cancelled by user")

    if log_cb:
        log_cb("Loading pyannote speaker diarization pipeline")
    if progress_cb:
        progress_cb(5)

    pretrained_parameters = inspect.signature(Pipeline.from_pretrained).parameters
    auth_kwargs = {"token": token} if "token" in pretrained_parameters else {"use_auth_token": token}
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", **auth_kwargs)
    if pipeline is None:
        raise DiarizationUnavailableError(
            "Could not load pyannote/speaker-diarization-3.1. Check that HF_TOKEN is valid "
            "and that your Hugging Face account accepted the model access conditions."
        )

    if device == "cuda":
        try:
            import torch

            if torch.cuda.is_available():
                pipeline.to(torch.device("cuda"))
        except Exception as exc:
            if log_cb:
                log_cb(f"CUDA diarization setup failed; continuing on default device: {exc}")

    if cancel_event.is_set():
        raise CancellationError("Processing cancelled by user")

    if log_cb:
        log_cb("Running speaker diarization")
    if progress_cb:
        progress_cb(20)

    diarization = pipeline(str(audio_wav))
    if progress_cb:
        progress_cb(80)
    annotation = _annotation_from_diarization_output(diarization)

    speaker_names: dict[str, str] = {}
    turns: list[SpeakerTurn] = []
    for turn, _, raw_speaker in annotation.itertracks(yield_label=True):
        if cancel_event.is_set():
            raise CancellationError("Processing cancelled by user")
        if raw_speaker not in speaker_names:
            speaker_names[raw_speaker] = f"speaker_{len(speaker_names) + 1}"
        turns.append(
            SpeakerTurn(
                start=max(0.0, float(turn.start)),
                end=max(float(turn.start), float(turn.end)),
                speaker_id=speaker_names[raw_speaker],
            )
        )

    turns.sort(key=lambda item: (item.start, item.end))
    if not turns:
        raise ValueError("Speaker diarization did not detect any speakers.")

    if log_cb:
        speakers = ", ".join(speaker_ids_from_turns(turns))
        log_cb(f"Detected {len(speaker_names)} speaker(s): {speakers}")
    if progress_cb:
        progress_cb(100)
    return turns


def merge_similar_speakers(
    turns: list[SpeakerTurn],
    audio_wav: Path,
    similarity_threshold: float = 0.70,
    log_cb: LogCallback | None = None,
) -> list[SpeakerTurn]:
    """Merge speakers whose ECAPA embeddings are too similar (over-segmentation fix)."""
    speaker_ids = speaker_ids_from_turns(turns)
    if len(speaker_ids) <= 1:
        return turns

    try:
        from modules.speaker_verification import get_segment_embedding, compute_similarity
    except ImportError:
        return turns

    speaker_turns_map: dict[str, list[SpeakerTurn]] = {}
    for turn in turns:
        speaker_turns_map.setdefault(turn.speaker_id, []).append(turn)

    speaker_embeddings: dict[str, object] = {}
    for sid, sid_turns in speaker_turns_map.items():
        best = max(sid_turns, key=lambda t: t.end - t.start)
        duration = best.end - best.start
        if duration < 0.5:
            continue
        try:
            emb = get_segment_embedding(audio_wav, best.start, duration)
            speaker_embeddings[sid] = emb
        except Exception:
            continue

    merge_map: dict[str, str] = {}
    merged_ids = list(speaker_embeddings.keys())
    for i, sid_a in enumerate(merged_ids):
        if sid_a in merge_map:
            continue
        for sid_b in merged_ids[i + 1:]:
            if sid_b in merge_map:
                continue
            sim = compute_similarity(speaker_embeddings[sid_a], speaker_embeddings[sid_b])
            if sim >= similarity_threshold:
                merge_map[sid_b] = sid_a
                if log_cb:
                    log_cb(
                        f"  Merging {sid_b} into {sid_a} (similarity {sim:.2f})"
                    )

    if not merge_map:
        return turns

    merged_turns = []
    for turn in turns:
        new_id = merge_map.get(turn.speaker_id, turn.speaker_id)
        merged_turns.append(SpeakerTurn(start=turn.start, end=turn.end, speaker_id=new_id))
    return merged_turns


def assign_speakers_to_segments(
    segments: list[Segment],
    turns: list[SpeakerTurn],
    speaker_mappings: dict[str, dict[str, str]] | None,
    log_cb: LogCallback | None,
    audio_wav: Path | None = None,
) -> list[Segment]:
    if not turns:
        return segments

    from collections import defaultdict
    mappings = speaker_mappings or {}
    for segment in segments:
        best_speaker: str | None = None
        
        # Calculate overlaps for all speakers on this segment
        speaker_overlaps = defaultdict(float)
        for turn in turns:
            overlap = max(0.0, min(segment.end, turn.end) - max(segment.start, turn.start))
            if overlap > 0.0:
                speaker_overlaps[turn.speaker_id] += overlap

        if speaker_overlaps:
            sorted_speakers = sorted(speaker_overlaps.items(), key=lambda x: x[1], reverse=True)
            top_speaker, top_overlap = sorted_speakers[0]
            # Best speaker must cover at least 40% of the segment.
            if top_overlap >= 0.40 * segment.duration:
                # Double talk: if another speaker overlaps by more than 25% of segment, reject.
                if len(sorted_speakers) > 1 and sorted_speakers[1][1] > 0.25 * segment.duration:
                    if log_cb:
                        log_cb(f"  Segment {segment.index + 1} has overlapping speakers (double talk detected). Discarding segment speaker assignment.")
                else:
                    best_speaker = top_speaker

        # Speaker verification is a guard for manual reference mappings. Auto
        # references are derived from these same diarized turns, so using them as
        # a hard veto here only duplicates pyannote and can reject short valid
        # segments due to noisy ECAPA cosine scores.
        if best_speaker is not None:
            mapping = mappings.get(best_speaker, {})
            ref_path = (
                mapping.get("cleaned_reference_audio_path", "").strip()
                or mapping.get("reference_audio_path", "").strip()
                or mapping.get("original_reference_audio_path", "").strip()
            )
            if (
                ref_path
                and not _is_auto_reference_mapping(mapping)
                and segment.duration >= SEGMENT_VERIFICATION_MIN_DURATION_SECONDS
                and audio_wav is not None
                and audio_wav.exists()
            ):
                try:
                    from modules.speaker_verification import get_segment_embedding, get_file_embedding, compute_similarity
                    ref_path_obj = Path(ref_path).expanduser()
                    if ref_path_obj.exists():
                        seg_emb = get_segment_embedding(audio_wav, segment.start, segment.duration)
                        ref_emb = get_file_embedding(ref_path_obj)
                        sim = compute_similarity(seg_emb, ref_emb)
                        if sim < SEGMENT_VERIFICATION_REJECT_THRESHOLD:
                            if log_cb:
                                log_cb(
                                    f"  Segment {segment.index + 1} verification failed for {best_speaker}: "
                                    f"similarity {sim:.2f} < threshold "
                                    f"{SEGMENT_VERIFICATION_REJECT_THRESHOLD:.2f}. Rejecting speaker assignment."
                                )
                            best_speaker = None
                        elif sim < SEGMENT_VERIFICATION_LOW_CONFIDENCE_THRESHOLD:
                            if log_cb:
                                log_cb(
                                    f"  Segment {segment.index + 1} low-confidence verification for {best_speaker} "
                                    f"(similarity: {sim:.2f}); keeping diarization assignment."
                                )
                        else:
                            if log_cb:
                                log_cb(
                                    f"  Segment {segment.index + 1} verified for {best_speaker} (similarity: {sim:.2f})"
                                )
                except Exception as exc:
                    if log_cb:
                        log_cb(f"  Warning: speaker verification failed for segment {segment.index + 1}: {exc}")

        if best_speaker is not None:
            segment.speaker_id = best_speaker
            segment.speaker_label = mappings.get(best_speaker, {}).get("label") or best_speaker.replace("_", " ").title()
            if log_cb:
                log_cb(
                    f"  Segment {segment.index + 1}/{len(segments)} assigned to "
                    f"{segment.speaker_label} ({segment.speaker_id})"
                )
        else:
            segment.speaker_id = None
            segment.speaker_label = None
            if log_cb:
                log_cb(
                    f"  Segment {segment.index + 1}/{len(segments)} unassigned (fallback to default TTS)"
                )
    return segments
