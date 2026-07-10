from __future__ import annotations

import gc
from threading import Event
from typing import Callable

from config.models import LANGUAGES, NLLB_MODEL_ID, TARGET_LANGUAGE
from config.paths import nllb_cache_dir
from core.context import CancellationError, Segment


ProgressCallback = Callable[[int], None]
LogCallback = Callable[[str], None]


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


def translate_segments(
    segments: list[Segment],
    source_language: str,
    device: str,
    progress_cb: ProgressCallback | None,
    log_cb: LogCallback | None,
    cancel_event: Event,
    batch_size: int = 8,
) -> list[Segment]:
    if source_language not in LANGUAGES:
        raise ValueError(f"Unsupported source language: {source_language}")
    if not segments:
        raise ValueError("No segments available for translation")

    if source_language == "km":
        for index, segment in enumerate(segments, start=1):
            segment.translated_text = segment.text
            segment.raw_khmer_text = segment.text
            segment.improved_khmer_text = segment.text
            if progress_cb:
                progress_cb(int((index / len(segments)) * 100))
        if log_cb:
            log_cb("Source language is Khmer; translation stage copied original text")
        return segments

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    src_lang = LANGUAGES[source_language].nllb_code
    target_lang = TARGET_LANGUAGE.nllb_code
    selected_device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
    torch_device = torch.device(selected_device)
    tokenizer = None
    model = None

    try:
        if log_cb:
            log_cb(f"Loading NLLB model on {selected_device}")
        tokenizer = AutoTokenizer.from_pretrained(
            NLLB_MODEL_ID, src_lang=src_lang, cache_dir=str(nllb_cache_dir())
        )
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                NLLB_MODEL_ID,
                cache_dir=str(nllb_cache_dir()),
                dtype=torch.float16 if selected_device == "cuda" else torch.float32,
            )
            model.to(torch_device)
        except RuntimeError as error:
            if selected_device == "cuda" and _is_oom(error):
                if log_cb:
                    log_cb("CUDA OOM while loading NLLB; falling back to CPU")
                if model is not None:
                    del model
                    model = None
                _free_gpu_memory()
                selected_device = "cpu"
                torch_device = torch.device("cpu")
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    NLLB_MODEL_ID, cache_dir=str(nllb_cache_dir())
                )
                model.to(torch_device)
            else:
                raise
        model.eval()

        forced_bos_token_id = tokenizer.convert_tokens_to_ids(target_lang)
        translated_count = 0

        for start in range(0, len(segments), batch_size):
            if cancel_event.is_set():
                raise CancellationError("Processing cancelled by user")

            batch = segments[start : start + batch_size]
            texts = [segment.text for segment in batch]

            try:
                encoded = tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                ).to(torch_device)
                with torch.inference_mode():
                    generated = model.generate(
                        **encoded,
                        forced_bos_token_id=forced_bos_token_id,
                        max_length=512,
                        num_beams=4,
                        repetition_penalty=1.5,
                        no_repeat_ngram_size=4,
                    )
            except RuntimeError as error:
                if selected_device == "cuda" and _is_oom(error):
                    if log_cb:
                        log_cb("CUDA OOM during translation; retrying this stage on CPU")
                    del model
                    _free_gpu_memory()
                    selected_device = "cpu"
                    torch_device = torch.device("cpu")
                    model = AutoModelForSeq2SeqLM.from_pretrained(
                        NLLB_MODEL_ID, cache_dir=str(nllb_cache_dir())
                    )
                    model.to(torch_device)
                    model.eval()
                    encoded = tokenizer(
                        texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512,
                    ).to(torch_device)
                    with torch.inference_mode():
                        generated = model.generate(
                            **encoded,
                            forced_bos_token_id=forced_bos_token_id,
                            max_length=512,
                            num_beams=4,
                            repetition_penalty=1.5,
                            no_repeat_ngram_size=4,
                        )
                else:
                    raise

            outputs = tokenizer.batch_decode(generated, skip_special_tokens=True)
            for segment, output in zip(batch, outputs, strict=True):
                segment.translated_text = output.strip()
                segment.raw_khmer_text = segment.translated_text
                segment.improved_khmer_text = segment.translated_text
                if log_cb:
                    log_cb(f"  Translated: \"{segment.text}\" -> \"{segment.translated_text}\"")

            translated_count += len(batch)
            if progress_cb:
                progress_cb(int((translated_count / len(segments)) * 100))

        return segments
    finally:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        _free_gpu_memory()
