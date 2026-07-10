from __future__ import annotations

import gc
from pathlib import Path
from threading import Event
from typing import Callable

from config.paths import is_whisper_model_downloaded, whisper_cache_dir
from core.context import CancellationError, Segment


ProgressCallback = Callable[[int], None]
LogCallback = Callable[[str], None]


def _load_whisper_model(model_name: str, device: str, compute_type: str):
    from faster_whisper import WhisperModel

    return WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
        download_root=str(whisper_cache_dir()),
        local_files_only=is_whisper_model_downloaded(model_name),
    )


def _free_gpu_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return


def _is_oom(error: BaseException) -> bool:
    text = str(error).lower()
    return "out of memory" in text or "cuda" in text and "memory" in text


def transcribe_audio(
    audio_wav: Path,
    language_code: str,
    model_name: str,
    device: str,
    total_duration: float,
    progress_cb: ProgressCallback | None,
    log_cb: LogCallback | None,
    cancel_event: Event,
) -> list[Segment]:
    if not audio_wav.exists():
        raise FileNotFoundError(f"Audio file does not exist: {audio_wav}")

    selected_device = "cuda" if device == "cuda" else "cpu"
    compute_type = "float16" if selected_device == "cuda" else "int8"
    active_model = model_name
    model = None

    try:
        try:
            if log_cb:
                log_cb(f"Loading faster-whisper {active_model} on {selected_device} ({compute_type})")
            model = _load_whisper_model(active_model, selected_device, compute_type)
        except RuntimeError as error:
            if selected_device == "cuda" and _is_oom(error):
                _free_gpu_memory()
                active_model = "small"
                if log_cb:
                    log_cb("CUDA OOM while loading Whisper model; falling back to small")
                try:
                    model = _load_whisper_model(active_model, "cuda", "float16")
                except RuntimeError as small_error:
                    if not _is_oom(small_error):
                        raise
                    if log_cb:
                        log_cb("CUDA OOM while loading small Whisper model; falling back to CPU int8")
                    _free_gpu_memory()
                    selected_device = "cpu"
                    compute_type = "int8"
                    model = _load_whisper_model(active_model, "cpu", "int8")
            elif selected_device == "cuda":
                if log_cb:
                    log_cb(f"CUDA Whisper load failed; falling back to CPU int8: {error}")
                _free_gpu_memory()
                selected_device = "cpu"
                compute_type = "int8"
                model = _load_whisper_model(active_model, "cpu", "int8")
            else:
                raise

        try:
            raw_segments, _ = model.transcribe(
                str(audio_wav),
                language=language_code,
                vad_filter=True,
                beam_size=5,
            )
        except RuntimeError as error:
            if selected_device == "cuda" and _is_oom(error):
                if log_cb:
                    log_cb("CUDA OOM during transcription; retrying on CPU int8")
                del model
                _free_gpu_memory()
                model = _load_whisper_model(active_model, "cpu", "int8")
                raw_segments, _ = model.transcribe(
                    str(audio_wav),
                    language=language_code,
                    vad_filter=True,
                    beam_size=5,
                )
            else:
                raise

        segments: list[Segment] = []
        for index, item in enumerate(raw_segments):
            if cancel_event.is_set():
                raise CancellationError("Processing cancelled by user")
            text = item.text.strip()
            if not text:
                continue
            segment = Segment(index=len(segments), start=float(item.start), end=float(item.end), text=text)
            segments.append(segment)
            if log_cb:
                log_cb(f"  [{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
            if progress_cb and total_duration > 0:
                progress_cb(min(99, int((float(item.end) / total_duration) * 100)))

        if not segments:
            raise ValueError("Transcription produced no text segments")
        if progress_cb:
            progress_cb(100)
        return segments
    finally:
        if model is not None:
            del model
        _free_gpu_memory()
