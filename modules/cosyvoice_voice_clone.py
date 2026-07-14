from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path


COSYVOICE_BATCH_SCRIPT = r"""
from __future__ import annotations

import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1]).expanduser()
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

reference_path = Path(manifest["reference_path"]).expanduser()
segments = manifest["segments"]

if not reference_path.exists():
    raise FileNotFoundError(f"Reference audio does not exist: {reference_path}")

import torch
import torchaudio

from cosyvoice.cli.cosyvoice import CosyVoice2

model_dir = manifest.get("model_dir", "pretrained_models/CosyVoice2-0.5B")
load_jit = bool(manifest.get("load_jit", False))
load_trt = bool(manifest.get("load_trt", False))
load_vllm = bool(manifest.get("load_vllm", False))
fp16 = bool(manifest.get("fp16", True))
cache_references = bool(manifest.get("cache_references", True))
clone_mode = manifest.get("clone_mode", "vc")
cosyvoice = CosyVoice2(
    model_dir,
    load_jit=load_jit,
    load_trt=load_trt,
    load_vllm=load_vllm,
    fp16=fp16,
)
print(
    "CosyVoice2 loaded from "
    f"{model_dir} (mode={clone_mode}, fp16={fp16}, jit={load_jit}, trt={load_trt}, vllm={load_vllm})"
)

results = []
speaker_cache = {}
for seg in segments:
    text = seg.get("text", "").strip()
    input_path_value = seg.get("input_path", "")
    input_path = Path(input_path_value).expanduser() if input_path_value else None
    output_path = Path(seg["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    use_vc = clone_mode == "vc" and input_path is not None and input_path.exists()
    if clone_mode == "vc" and not use_vc:
        results.append({
            "segment_index": seg["segment_index"],
            "ok": False,
            "error": f"Missing Khmer source TTS audio for voice conversion: {input_path_value}",
        })
        continue

    if not use_vc and not text:
        results.append({"segment_index": seg["segment_index"], "ok": False, "error": "Empty text"})
        continue

    try:
        seg_ref = seg.get("reference_path")
        effective_ref = seg_ref if seg_ref and Path(seg_ref).exists() else str(reference_path)

        instruct_text = seg.get("instruct_text", "").strip()
        zero_shot_spk_id = ""
        if cache_references and not use_vc:
            cache_key = f"{Path(effective_ref).resolve()}::{instruct_text}"
            if cache_key not in speaker_cache:
                cached_spk_id = f"batch_spk_{len(speaker_cache)}"
                try:
                    cosyvoice.add_zero_shot_spk(instruct_text, effective_ref, cached_spk_id)
                    speaker_cache[cache_key] = cached_spk_id
                    print(f"Cached CosyVoice reference {cached_spk_id}: {effective_ref}")
                except Exception as cache_exc:
                    print(f"Warning: CosyVoice reference cache disabled for {effective_ref}: {cache_exc}")
                    speaker_cache[cache_key] = ""
            zero_shot_spk_id = speaker_cache[cache_key]

        if use_vc:
            output_iter = cosyvoice.inference_vc(
                str(input_path),
                effective_ref,
                stream=False,
            )
        elif instruct_text:
            output_iter = cosyvoice.inference_instruct2(
                text, instruct_text, effective_ref,
                zero_shot_spk_id=zero_shot_spk_id,
                stream=False,
            )
        else:
            output_iter = cosyvoice.inference_cross_lingual(
                text, effective_ref,
                zero_shot_spk_id=zero_shot_spk_id,
                stream=False,
            )

        audio_chunks = []
        for chunk in output_iter:
            audio_chunks.append(chunk["tts_speech"])

        if audio_chunks:
            full_audio = torch.cat(audio_chunks, dim=-1)
            torchaudio.save(str(output_path), full_audio, cosyvoice.sample_rate)
            results.append({"segment_index": seg["segment_index"], "ok": output_path.exists()})
        else:
            results.append({"segment_index": seg["segment_index"], "ok": False, "error": "No audio generated"})
    except Exception as exc:
        results.append({"segment_index": seg["segment_index"], "ok": False, "error": str(exc)[:500]})

results_path = manifest_path.with_suffix(".results.json")
results_path.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
print(f"CosyVoice batch complete: {sum(1 for r in results if r['ok'])}/{len(results)} succeeded")
"""


class CosyVoiceCloneError(RuntimeError):
    pass


def _cosyvoice_python() -> str:
    value = os.getenv("COSYVOICE_PYTHON", "").strip()
    if not value:
        raise CosyVoiceCloneError(
            "COSYVOICE_PYTHON is not set. CosyVoice 2 requires a separate Python environment "
            "with CosyVoice installed. Set COSYVOICE_PYTHON to that environment's python."
        )
    path = Path(value).expanduser()
    if not path.exists():
        raise CosyVoiceCloneError(f"COSYVOICE_PYTHON does not exist: {path}")
    return str(path)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name, "").strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on", "auto"}


def _cosyvoice_model_dir() -> str:
    configured = os.getenv("COSYVOICE_MODEL_DIR", "").strip()
    if configured:
        return str(Path(configured).expanduser())
    try:
        from huggingface_hub import snapshot_download
        from config.paths import cosyvoice_cache_dir

        return snapshot_download(
            "FunAudioLLM/CosyVoice2-0.5B",
            cache_dir=str(cosyvoice_cache_dir()),
            local_files_only=True,
        )
    except Exception:
        return "pretrained_models/CosyVoice2-0.5B"


def clone_batch(
    segments: list[dict],
    reference_path: Path,
    language: str = "km",
    device: str = "auto",
    log_cb=None,
    cancel_event=None,
) -> list[dict]:
    if not segments:
        return []

    python = _cosyvoice_python()
    model_dir = _cosyvoice_model_dir()

    manifest = {
        "reference_path": str(reference_path),
        "language": language,
        "device": device,
        "model_dir": model_dir,
        "fp16": _env_bool("COSYVOICE_FP16", True),
        "load_jit": _env_bool("COSYVOICE_LOAD_JIT", False),
        "load_trt": _env_bool("COSYVOICE_LOAD_TRT", False),
        "load_vllm": _env_bool("COSYVOICE_LOAD_VLLM", False),
        "cache_references": _env_bool("COSYVOICE_CACHE_REFERENCES", True),
        "clone_mode": os.getenv("COSYVOICE_CLONE_MODE", "vc").strip().lower() or "vc",
        "segments": segments,
    }

    import tempfile
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix="cosyvoice_batch_", delete=False, encoding="utf-8",
    ) as f:
        json.dump(manifest, f, ensure_ascii=False)
        manifest_path = Path(f.name)

    try:
        timeout_seconds = max(600, len(segments) * 90)
        start_time = time.time()
        env = os.environ.copy()
        cosyvoice_repo = os.getenv("COSYVOICE_REPO_DIR", "").strip()
        if cosyvoice_repo:
            existing = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = f"{cosyvoice_repo}:{existing}" if existing else cosyvoice_repo
        process = subprocess.Popen(
            [python, "-c", COSYVOICE_BATCH_SCRIPT, str(manifest_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
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
                    raise CancellationError("CosyVoice batch cloning cancelled by user")
                
                if has_fileno:
                    ready, _, _ = select.select([process.stdout], [], [], 0.1)
                    if ready:
                        line = process.stdout.readline()
                        if not line:
                            break
                        line_str = line.strip()
                        if line_str:
                            output_lines.append(line_str)
                            if log_cb:
                                log_cb(f"      [CosyVoice] {line_str}")
                    else:
                        if process.poll() is not None:
                            break
                else:
                    line = process.stdout.readline()
                    if not line:
                        break
                    line_str = line.strip()
                    if line_str:
                        output_lines.append(line_str)
                        if log_cb:
                            log_cb(f"      [CosyVoice] {line_str}")

        process.wait(timeout=max(1.0, timeout_seconds - (time.time() - start_time)))

        if process.returncode != 0:
            detail = "\n".join(output_lines)
            if len(detail) > 1600:
                detail = detail[-1600:]
            raise CosyVoiceCloneError(detail or "CosyVoice batch conversion failed.")

        results_path = manifest_path.with_suffix(".results.json")
        if results_path.exists():
            results = json.loads(results_path.read_text(encoding="utf-8"))
            results_path.unlink(missing_ok=True)
            return results
        return [{"segment_index": s["segment_index"], "ok": False, "error": "No results file"} for s in segments]
    except subprocess.TimeoutExpired:
        if "process" in locals():
            process.kill()
        raise CosyVoiceCloneError(f"CosyVoice batch timed out after {timeout_seconds // 60} minutes")
    finally:
        manifest_path.unlink(missing_ok=True)
