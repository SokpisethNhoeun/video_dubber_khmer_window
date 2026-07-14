from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path


QWEN3_BATCH_SCRIPT = r"""
from __future__ import annotations

import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1]).expanduser()
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

reference_path = Path(manifest["reference_path"]).expanduser()
segments = manifest["segments"]
model_name = manifest.get("model_name", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
language = manifest.get("language", "Auto")
device_arg = manifest.get("device", "auto")
dtype_name = str(manifest.get("dtype", "float16")).strip().lower()
low_vram = bool(manifest.get("low_vram", False))
default_max_tokens = int(manifest.get("default_max_new_tokens", 1024 if low_vram else 2048))

if not reference_path.exists():
    raise FileNotFoundError(f"Reference audio does not exist: {reference_path}")

import torch
import soundfile as sf
from transformers.utils import logging as transformers_logging
from qwen_tts import Qwen3TTSModel

transformers_logging.set_verbosity_error()

if device_arg == "auto":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
elif device_arg == "cuda" and not torch.cuda.is_available():
    device = "cpu"
else:
    device = device_arg if ":" in str(device_arg) else f"{device_arg}:0"

dtype_map = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}
dtype = dtype_map.get(dtype_name, torch.float16) if device != "cpu" else torch.float32

if device != "cpu" and torch.cuda.is_available():
    torch.cuda.empty_cache()

attn_candidates = ("sdpa", None) if low_vram else ("flash_attention_2", "sdpa", None)
model = None
for attn in attn_candidates:
    try:
        kwargs = dict(device_map=device, dtype=dtype)
        if attn:
            kwargs["attn_implementation"] = attn
        model = Qwen3TTSModel.from_pretrained(model_name, **kwargs)
        print(f"Qwen3-TTS loaded from {model_name} on {device} (dtype={dtype_name}, attn={attn or 'default'}, low_vram={low_vram})")
        # Ensure a pad token is set to avoid Transformers info/warnings when generating
        try:
            cfg = getattr(model, "config", None)
            generation_cfg = getattr(model, "generation_config", None)
            eos_token_id = (
                getattr(cfg, "eos_token_id", None)
                if cfg is not None
                else getattr(generation_cfg, "eos_token_id", None)
            )
            if cfg is not None and getattr(cfg, "pad_token_id", None) is None:
                setattr(cfg, "pad_token_id", eos_token_id)
            if generation_cfg is not None and getattr(generation_cfg, "pad_token_id", None) is None:
                setattr(generation_cfg, "pad_token_id", eos_token_id)
        except Exception:
            # Best-effort only; do not fail model load if this cannot be set
            pass
        break
    except Exception as exc:
        print(f"Qwen3-TTS load attempt failed (attn={attn}): {exc}")
        model = None
if model is None:
    raise RuntimeError(f"Could not load Qwen3-TTS model: {model_name}")

prompt_cache = {}
results = []
completed_segments = 0
for seg in segments:
    text = seg.get("text", "").strip()
    output_path = Path(seg["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not text:
        results.append({"segment_index": seg["segment_index"], "ok": False, "error": "Empty text"})
        completed_segments += 1
        print(
            f"QWEN3_PROGRESS {completed_segments}/{len(segments)} "
            f"segment={seg['segment_index']} ok=0",
            flush=True,
        )
        continue

    try:
        emotion_ref = (
            seg.get("emotion_reference_path", "").strip()
            or seg.get("reference_path", "").strip()
        )
        speaker_ref = seg.get("speaker_reference_path", "").strip() or str(reference_path)
        emotion_ref_text = (
            seg.get("emotion_ref_text", "").strip()
            or seg.get("ref_text", "").strip()
        )
        cache_key = seg.get("prompt_cache_key", "").strip()
        instruct_text = seg.get("instruct_text", "").strip()
        emotion_label = seg.get("emotion_label", "").strip()

        gen_kwargs = dict(
            text=text,
            language=language,
            max_new_tokens=int(seg.get("max_new_tokens", default_max_tokens)),
            temperature=float(seg.get("temperature", 0.9)),
            top_k=int(seg.get("top_k", 50)),
            top_p=float(seg.get("top_p", 1.0)),
            repetition_penalty=float(seg.get("repetition_penalty", 1.05)),
        )

        use_emotion_ref = bool(
            emotion_ref
            and Path(emotion_ref).exists()
            and emotion_ref != speaker_ref
        )

        if use_emotion_ref and emotion_ref_text:
            print(
                f"  Segment {seg['segment_index']}: emotion ICL"
                + (f" ({emotion_label})" if emotion_label else "")
            )
            wavs, sr = model.generate_voice_clone(
                ref_audio=emotion_ref,
                ref_text=emotion_ref_text,
                x_vector_only_mode=False,
                **gen_kwargs,
            )
        elif use_emotion_ref:
            print(f"  Segment {seg['segment_index']}: emotion x-vector (no source transcript)")
            wavs, sr = model.generate_voice_clone(
                ref_audio=emotion_ref,
                x_vector_only_mode=True,
                **gen_kwargs,
            )
        elif cache_key and cache_key in prompt_cache:
            wavs, sr = model.generate_voice_clone(
                voice_clone_prompt=prompt_cache[cache_key],
                **gen_kwargs,
            )
        elif speaker_ref and Path(speaker_ref).exists():
            if cache_key and cache_key not in prompt_cache:
                prompt_cache[cache_key] = model.create_voice_clone_prompt(
                    ref_audio=speaker_ref,
                    x_vector_only_mode=True,
                )
            if cache_key and cache_key in prompt_cache:
                wavs, sr = model.generate_voice_clone(
                    voice_clone_prompt=prompt_cache[cache_key],
                    **gen_kwargs,
                )
            else:
                wavs, sr = model.generate_voice_clone(
                    ref_audio=speaker_ref,
                    x_vector_only_mode=True,
                    **gen_kwargs,
                )
        else:
            effective_ref = emotion_ref if emotion_ref and Path(emotion_ref).exists() else speaker_ref
            wavs, sr = model.generate_voice_clone(
                ref_audio=effective_ref,
                ref_text=emotion_ref_text or None,
                x_vector_only_mode=not bool(emotion_ref_text),
                **gen_kwargs,
            )

        if instruct_text:
            print(
                f"  Segment {seg['segment_index']}: emotion style hint "
                f"({instruct_text[:80]}{'...' if len(instruct_text) > 80 else ''})"
            )

        sf.write(str(output_path), wavs[0], sr)
        results.append({"segment_index": seg["segment_index"], "ok": output_path.exists()})
    except Exception as exc:
        results.append({"segment_index": seg["segment_index"], "ok": False, "error": str(exc)[:500]})
    finally:
        completed_segments += 1
        ok = 1 if results and results[-1].get("ok") else 0
        print(
            f"QWEN3_PROGRESS {completed_segments}/{len(segments)} "
            f"segment={seg['segment_index']} ok={ok}",
            flush=True,
        )

results_path = manifest_path.with_suffix(".results.json")
results_path.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
print(f"Qwen3-TTS batch complete: {sum(1 for r in results if r['ok'])}/{len(results)} succeeded")
"""


class Qwen3CloneError(RuntimeError):
    pass


def _is_qwen3_noise_line(line: str) -> bool:
    return "Setting `pad_token_id` to `eos_token_id`" in line


def _qwen3_progress_from_line(line: str) -> tuple[int, int] | None:
    prefix = "QWEN3_PROGRESS "
    if not line.startswith(prefix):
        return None
    progress_part = line[len(prefix):].split(maxsplit=1)[0]
    if "/" not in progress_part:
        return None
    done_text, total_text = progress_part.split("/", 1)
    try:
        done = int(done_text)
        total = int(total_text)
    except ValueError:
        return None
    if total <= 0:
        return None
    return max(0, min(done, total)), total


def _qwen3_python() -> str:
    value = os.getenv("QWEN3_TTS_PYTHON", "").strip()
    if not value:
        raise Qwen3CloneError(
            "QWEN3_TTS_PYTHON is not set. Qwen3-TTS 1.7B requires a separate Python environment "
            "with qwen-tts installed. Set QWEN3_TTS_PYTHON to that environment's python."
        )
    path = Path(value).expanduser()
    if not path.exists():
        raise Qwen3CloneError(f"QWEN3_TTS_PYTHON does not exist: {path}")
    return str(path)


def _qwen3_check_sox_available(python: str) -> None:
    try:
        completed = subprocess.run(
            [python, "-c", "import shutil; print(shutil.which('sox') or '')"],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except Exception as exc:
        raise Qwen3CloneError(
            "Failed to validate QWEN3_TTS_PYTHON environment. Ensure it points to a working Python "
            "environment with qwen-tts installed and SoX available in PATH."
            f"\n\n{exc}"
        )

    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        message = (
            "Failed to validate QWEN3_TTS_PYTHON environment. Ensure the configured Python interpreter "
            "is valid and that SoX is installed in the environment."
        )
        if stderr:
            message += f"\n\n{stderr}"
        raise Qwen3CloneError(message)

    if not completed.stdout.strip():
        raise Qwen3CloneError(
            "Qwen3-TTS requires SoX to be installed and available in the environment used by "
            "QWEN3_TTS_PYTHON. Install SoX and make sure `sox` is discoverable on PATH."
        )


def _qwen3_refine_error_detail(detail: str) -> str:
    lower = detail.lower()
    if "sox: not found" in lower or "sox could not be found" in lower or "could not find sox" in lower:
        return (
            "Qwen3-TTS failed because SoX is missing or not available in the Python environment used by "
            "QWEN3_TTS_PYTHON. Install SoX and ensure `sox` is on PATH.\n\n"
            + detail
        )
    if "couldn't connect to 'https://huggingface.co'" in lower or "we couldn't connect to 'https://huggingface.co'" in lower:
        return (
            "Qwen3-TTS could not download model files from Hugging Face and did not find them in the local cache. "
            "Check your internet connection or configure offline mode / cache the model files.\n\n"
            + detail
        )
    return detail


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name, "").strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on"}


def _qwen3_offline_mode() -> bool:
    return _env_bool("HF_HUB_OFFLINE") or _env_bool("TRANSFORMERS_OFFLINE")


def _qwen3_model_name() -> str:
    configured = os.getenv("QWEN3_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base").strip()
    local_path = Path(configured).expanduser()
    if local_path.exists():
        return str(local_path)
    try:
        from huggingface_hub import snapshot_download
        from config.paths import qwen_cache_dir

        try:
            return snapshot_download(
                configured,
                cache_dir=str(qwen_cache_dir()),
                local_files_only=True,
            )
        except Exception:
            if not _qwen3_offline_mode():
                return configured
            raise
    except Exception as exc:
        raise Qwen3CloneError(
            "Qwen3-TTS is running with Hugging Face offline mode enabled, but the model repo "
            f"'{configured}' was not found in the local cache. Download it once while online or set "
            "QWEN3_TTS_MODEL to the local snapshot directory."
        ) from exc


def _qwen3_runtime_options() -> dict[str, str | bool | int]:
    low_vram = _env_bool("QWEN3_TTS_LOW_VRAM", False)
    model_name = _qwen3_model_name()
    if "1.7B" in model_name and not os.getenv("QWEN3_TTS_DTYPE", "").strip():
        low_vram = True
    dtype = os.getenv("QWEN3_TTS_DTYPE", "float16" if low_vram else "bfloat16").strip().lower()
    return {
        "low_vram": low_vram,
        "dtype": dtype,
        "default_max_new_tokens": 1024 if low_vram else 2048,
    }


def clone_batch(
    segments: list[dict],
    reference_path: Path,
    language: str = "Auto",
    device: str = "auto",
    log_cb=None,
    progress_cb=None,
    cancel_event=None,
) -> list[dict]:
    """Clone multiple segments in one subprocess using Qwen3-TTS 1.7B Base."""
    if not segments:
        return []

    python = _qwen3_python()
    _qwen3_check_sox_available(python)
    model_name = _qwen3_model_name()
    runtime = _qwen3_runtime_options()

    manifest = {
        "reference_path": str(reference_path),
        "model_name": model_name,
        "language": language,
        "device": device,
        "dtype": runtime["dtype"],
        "low_vram": runtime["low_vram"],
        "default_max_new_tokens": runtime["default_max_new_tokens"],
        "segments": segments,
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix="qwen3_batch_", delete=False, encoding="utf-8"
    ) as f:
        json.dump(manifest, f, ensure_ascii=False)
        manifest_path = Path(f.name)

    try:
        timeout_seconds = max(900, len(segments) * 120)
        start_time = time.time()
        process = subprocess.Popen(
            [python, "-c", QWEN3_BATCH_SCRIPT, str(manifest_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        output_lines = []
        if process.stdout:
            import select
            import io
            from core.context import CancellationError
            has_fileno = False
            if hasattr(process.stdout, "fileno"):
                try:
                    process.stdout.fileno()
                    has_fileno = True
                except (io.UnsupportedOperation, AttributeError):
                    pass

            while True:
                if cancel_event and cancel_event.is_set():
                    process.terminate()
                    process.wait(timeout=5)
                    raise CancellationError("Qwen3-TTS batch cloning cancelled by user")
                
                if has_fileno:
                    ready, _, _ = select.select([process.stdout], [], [], 0.1)
                    if ready:
                        line = process.stdout.readline()
                        if not line:
                            break
                        line_str = line.strip()
                        if _is_qwen3_noise_line(line_str):
                            continue
                        progress = _qwen3_progress_from_line(line_str)
                        if progress is not None:
                            if progress_cb:
                                progress_cb(*progress)
                            continue
                        if line_str:
                            output_lines.append(line_str)
                            if log_cb:
                                log_cb(f"      [Qwen3-TTS] {line_str}")
                    else:
                        if process.poll() is not None:
                            break
                else:
                    line = process.stdout.readline()
                    if not line:
                        break
                    line_str = line.strip()
                    if _is_qwen3_noise_line(line_str):
                        continue
                    progress = _qwen3_progress_from_line(line_str)
                    if progress is not None:
                        if progress_cb:
                            progress_cb(*progress)
                        continue
                    if line_str:
                        output_lines.append(line_str)
                        if log_cb:
                            log_cb(f"      [Qwen3-TTS] {line_str}")

        process.wait(timeout=max(1.0, timeout_seconds - int(time.time() - start_time)))

        if process.returncode != 0:
            detail = _qwen3_refine_error_detail("\n".join(output_lines))
            if len(detail) > 1600:
                detail = detail[:1600]
            raise Qwen3CloneError(detail or "Qwen3-TTS batch conversion failed.")

        results_path = manifest_path.with_suffix(".results.json")
        if results_path.exists():
            results = json.loads(results_path.read_text(encoding="utf-8"))
            results_path.unlink(missing_ok=True)
            return results
        return [{"segment_index": s["segment_index"], "ok": False, "error": "No results file"} for s in segments]
    except subprocess.TimeoutExpired:
        if "process" in locals():
            process.kill()
        raise Qwen3CloneError(f"Qwen3-TTS batch timed out after {timeout_seconds // 60} minutes")
    finally:
        manifest_path.unlink(missing_ok=True)
