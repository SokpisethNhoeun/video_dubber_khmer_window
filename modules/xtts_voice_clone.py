from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path


XTTS_BATCH_SCRIPT = r"""
from __future__ import annotations

import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1]).expanduser()
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

reference_path = Path(manifest["reference_path"]).expanduser()
language = manifest.get("language", "km")
device_arg = manifest.get("device", "auto")
segments = manifest["segments"]

if not reference_path.exists():
    raise FileNotFoundError(f"Reference audio does not exist: {reference_path}")

import torch
from TTS.api import TTS

if device_arg == "auto":
    device = "cuda" if torch.cuda.is_available() else "cpu"
elif device_arg == "cuda" and not torch.cuda.is_available():
    device = "cpu"
else:
    device = device_arg

lang_map = {"km": "km", "zh": "zh-cn", "en": "en", "khm_Khmr": "km"}
tts_lang = lang_map.get(language, language)

try:
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    print(f"XTTS loaded successfully on device: {device}")
except RuntimeError as exc:
    if "out of memory" in str(exc).lower() and device != "cpu":
        device = "cpu"
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        print(f"XTTS out of memory on GPU, falling back to device: {device}")
    else:
        raise

results = []
for seg in segments:
    text = seg.get("text", "").strip()
    output_path = Path(seg["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not text:
        results.append({"segment_index": seg["segment_index"], "ok": False, "error": "Empty text"})
        continue

    try:
        seg_ref = seg.get("reference_path")
        effective_ref = seg_ref if seg_ref and Path(seg_ref).exists() else str(reference_path)
        tts.tts_to_file(
            text=text,
            speaker_wav=effective_ref,
            language=tts_lang,
            file_path=str(output_path),
        )
        results.append({"segment_index": seg["segment_index"], "ok": output_path.exists()})
    except Exception as exc:
        results.append({"segment_index": seg["segment_index"], "ok": False, "error": str(exc)[:500]})

results_path = manifest_path.with_suffix(".results.json")
results_path.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
print(f"XTTS batch complete: {sum(1 for r in results if r['ok'])}/{len(results)} succeeded")
"""


class XTTSCloneError(RuntimeError):
    pass


def _xtts_python() -> str:
    """Resolve the Python interpreter that has TTS installed.
    Uses OPENVOICE_PYTHON since TTS is installed in the same venv.
    """
    value = os.getenv("OPENVOICE_PYTHON", "").strip()
    if not value:
        raise XTTSCloneError(
            "OPENVOICE_PYTHON is not set. XTTS-v2 runs in the same Python 3.10/3.11 "
            "environment as OpenVoice. Set OPENVOICE_PYTHON to that environment's python."
        )
    path = Path(value).expanduser()
    if not path.exists():
        raise XTTSCloneError(f"OPENVOICE_PYTHON does not exist: {path}")
    return str(path)


def clone_batch(
    segments: list[dict],
    reference_path: Path,
    language: str = "km",
    device: str = "auto",
    log_cb=None,
) -> list[dict]:
    """Clone multiple segments in one subprocess using XTTS-v2.

    segments: list of {"segment_index": int, "text": str, "output_path": str}
    Returns: list of {"segment_index": int, "ok": bool, "error"?: str}
    """
    if not segments:
        return []

    python = _xtts_python()

    manifest = {
        "reference_path": str(reference_path),
        "language": language,
        "device": device,
        "segments": segments,
    }

    import tempfile
    import subprocess
    import time
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix="xtts_batch_", delete=False, encoding="utf-8"
    ) as f:
        json.dump(manifest, f, ensure_ascii=False)
        manifest_path = Path(f.name)

    try:
        timeout_seconds = max(600, len(segments) * 60)
        start_time = time.time()
        env = os.environ.copy()
        env["COQUI_TOS_AGREED"] = "1"  # suppress interactive license prompt
        process = subprocess.Popen(
            [python, "-c", XTTS_BATCH_SCRIPT, str(manifest_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        output_lines = []
        if process.stdout:
            for line in iter(process.stdout.readline, ""):
                line_str = line.strip()
                if line_str:
                    output_lines.append(line_str)
                    if log_cb:
                        log_cb(f"      [XTTS] {line_str}")
        
        process.wait(timeout=timeout_seconds - (time.time() - start_time))
        
        if process.returncode != 0:
            detail = "\n".join(output_lines)
            if len(detail) > 1600:
                detail = detail[-1600:]
            raise XTTSCloneError(detail or "XTTS batch conversion failed.")

        results_path = manifest_path.with_suffix(".results.json")
        if results_path.exists():
            results = json.loads(results_path.read_text(encoding="utf-8"))
            results_path.unlink(missing_ok=True)
            return results
        return [{"segment_index": s["segment_index"], "ok": False, "error": "No results file"} for s in segments]
    except subprocess.TimeoutExpired:
        if 'process' in locals():
            process.kill()
        raise XTTSCloneError(f"XTTS batch timed out after {timeout_seconds // 60} minutes")
    finally:
        manifest_path.unlink(missing_ok=True)
