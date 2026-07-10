from __future__ import annotations

import tempfile
from pathlib import Path
from threading import Event

from core.context import Segment


def preview_segment(
    segment: Segment,
    voice_female: str,
    voice_male: str,
    speech_rate: int,
    pitch_hz: int,
    voice_gender: str,
    segment_genders: dict[int, str] | None,
    clone_backend: str = "openvoice",
    speaker_voice_mappings: dict[str, dict[str, str]] | None = None,
    device: str = "cpu",
    cancel_event: Event | None = None,
) -> Path | None:
    """Run TTS (and optionally clone) for a single segment, return the audio path."""
    if cancel_event is None:
        cancel_event = Event()

    if not segment.tts_text.strip():
        return None

    work_dir = Path(tempfile.mkdtemp(prefix="preview_"))
    cache_dir = work_dir / "tts_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    from modules.tts_engine import synthesize_tts

    segments = synthesize_tts(
        [segment],
        voice_gender,
        speech_rate,
        pitch_hz,
        work_dir,
        progress_cb=None,
        log_cb=None,
        cancel_event=cancel_event,
        voice_female=voice_female,
        voice_male=voice_male,
        segment_genders=segment_genders,
    )

    seg = segments[0]
    if seg.tts_path is None or not seg.tts_path.exists():
        return None

    ref_path = _get_speaker_reference(segment, speaker_voice_mappings)
    if ref_path is None:
        return seg.tts_path

    if clone_backend == "xtts":
        try:
            from modules.xtts_voice_clone import clone_batch
            clone_dir = work_dir / "clone"
            clone_dir.mkdir(parents=True, exist_ok=True)
            out = clone_dir / f"preview_{segment.index}.wav"
            results = clone_batch(
                [{"segment_index": segment.index, "text": segment.tts_text, "output_path": str(out)}],
                ref_path,
                device=device,
            )
            if results and results[0].get("ok") and out.exists():
                return out
        except Exception:
            pass
    elif clone_backend == "openvoice":
        try:
            from modules.openvoice_voice_clone import clone_with_openvoice
            clone_dir = work_dir / "clone"
            clone_dir.mkdir(parents=True, exist_ok=True)
            out = clone_dir / f"preview_{segment.index}.wav"
            clone_with_openvoice(seg.tts_path, out, ref_path, device=device)
            if out.exists():
                return out
        except Exception:
            pass
    elif clone_backend == "cosyvoice":
        try:
            from modules.cosyvoice_voice_clone import clone_batch
            clone_dir = work_dir / "clone"
            clone_dir.mkdir(parents=True, exist_ok=True)
            out = clone_dir / f"preview_{segment.index}.wav"
            results = clone_batch(
                [{
                    "segment_index": segment.index,
                    "text": segment.tts_text,
                    "input_path": str(seg.tts_path),
                    "output_path": str(out),
                }],
                ref_path,
                device=device,
            )
            if results and results[0].get("ok") and out.exists():
                return out
        except Exception:
            pass
    elif clone_backend == "qwen3":
        try:
            from modules.qwen3_voice_clone import clone_batch
            clone_dir = work_dir / "clone"
            clone_dir.mkdir(parents=True, exist_ok=True)
            out = clone_dir / f"preview_{segment.index}.wav"
            ref_text = segment.text.strip() or segment.translated_text.strip()
            results = clone_batch(
                [{
                    "segment_index": segment.index,
                    "text": segment.tts_text,
                    "output_path": str(out),
                    "speaker_reference_path": str(ref_path),
                    "ref_text": ref_text,
                }],
                ref_path,
                device=device,
            )
            if results and results[0].get("ok") and out.exists():
                return out
        except Exception:
            pass

    return seg.tts_path


def _get_speaker_reference(
    segment: Segment,
    speaker_voice_mappings: dict[str, dict[str, str]] | None,
) -> Path | None:
    if not segment.speaker_id or not speaker_voice_mappings:
        return None
    mapping = speaker_voice_mappings.get(segment.speaker_id, {})
    ref = (
        mapping.get("cleaned_reference_audio_path", "").strip()
        or mapping.get("reference_audio_path", "").strip()
    )
    if ref and Path(ref).exists():
        return Path(ref)
    return None
