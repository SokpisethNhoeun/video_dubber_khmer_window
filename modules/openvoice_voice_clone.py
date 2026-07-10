from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path


OPENVOICE_SCRIPT = r"""
from __future__ import annotations

import sys
from pathlib import Path

input_path = Path(sys.argv[1]).expanduser()
output_path = Path(sys.argv[2]).expanduser()
reference_path = Path(sys.argv[3]).expanduser()
checkpoint_dir = Path(sys.argv[4]).expanduser()
device_arg = sys.argv[5]

if not input_path.exists():
    raise FileNotFoundError(f"Input audio does not exist: {input_path}")
if not reference_path.exists():
    raise FileNotFoundError(f"Reference audio does not exist: {reference_path}")

converter_dir = checkpoint_dir
if not (converter_dir / "config.json").exists():
    for candidate in (
        checkpoint_dir / "converter",
        checkpoint_dir / "checkpoints_v2" / "converter",
        checkpoint_dir / "checkpoints" / "converter",
    ):
        if (candidate / "config.json").exists():
            converter_dir = candidate
            break

config_path = converter_dir / "config.json"
checkpoint_path = converter_dir / "checkpoint.pth"
if not config_path.exists() or not checkpoint_path.exists():
    raise FileNotFoundError(
        "OpenVoice converter checkpoint not found. Expected config.json and checkpoint.pth "
        f"under {checkpoint_dir} or a converter subdirectory."
    )

import torch
from openvoice.api import ToneColorConverter

if device_arg == "auto":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
elif device_arg == "cuda" and torch.cuda.is_available():
    device = "cuda:0"
else:
    device = device_arg

converter = ToneColorConverter(str(config_path), device=device, enable_watermark=False)
converter.load_ckpt(str(checkpoint_path))
source_se = converter.extract_se([str(input_path)])
target_se = converter.extract_se([str(reference_path)])
output_path.parent.mkdir(parents=True, exist_ok=True)
converter.convert(
    audio_src_path=str(input_path),
    src_se=source_se,
    tgt_se=target_se,
    output_path=str(output_path),
    message="@video-dubber",
)
"""

OPENVOICE_BATCH_SCRIPT = r"""
from __future__ import annotations

import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1]).expanduser()
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

reference_path = Path(manifest["reference_path"]).expanduser()
checkpoint_dir = Path(manifest["checkpoint_dir"]).expanduser()
device_arg = manifest["device"]
segments = manifest["segments"]

if not reference_path.exists():
    raise FileNotFoundError(f"Reference audio does not exist: {reference_path}")

converter_dir = checkpoint_dir
if not (converter_dir / "config.json").exists():
    for candidate in (
        checkpoint_dir / "converter",
        checkpoint_dir / "checkpoints_v2" / "converter",
        checkpoint_dir / "checkpoints" / "converter",
    ):
        if (candidate / "config.json").exists():
            converter_dir = candidate
            break

config_path = converter_dir / "config.json"
checkpoint_path = converter_dir / "checkpoint.pth"
if not config_path.exists() or not checkpoint_path.exists():
    raise FileNotFoundError(
        "OpenVoice converter checkpoint not found. Expected config.json and checkpoint.pth "
        f"under {checkpoint_dir} or a converter subdirectory."
    )

import torch
from openvoice.api import ToneColorConverter

if device_arg == "auto":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
elif device_arg == "cuda" and torch.cuda.is_available():
    device = "cuda:0"
else:
    device = device_arg

converter = ToneColorConverter(str(config_path), device=device, enable_watermark=False)
converter.load_ckpt(str(checkpoint_path))

# Extract target_se ONCE from the reference — this is the key consistency fix.
target_se = converter.extract_se([str(reference_path)])

results = []
for seg in segments:
    input_path = Path(seg["input_path"]).expanduser()
    output_path = Path(seg["output_path"]).expanduser()
    if not input_path.exists():
        results.append({"segment_index": seg["segment_index"], "ok": False, "error": f"Missing input: {input_path}"})
        continue
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        source_se = converter.extract_se([str(input_path)])
        converter.convert(
            audio_src_path=str(input_path),
            src_se=source_se,
            tgt_se=target_se,
            output_path=str(output_path),
            message="@video-dubber",
        )
        results.append({"segment_index": seg["segment_index"], "ok": True})
    except Exception as exc:
        results.append({"segment_index": seg["segment_index"], "ok": False, "error": str(exc)[:500]})

# Write results back so the caller can check per-segment status.
results_path = manifest_path.with_suffix(".results.json")
results_path.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
print(f"Batch complete: {sum(1 for r in results if r['ok'])}/{len(results)} succeeded")
"""


class OpenVoiceCloneError(RuntimeError):
    pass


def _openvoice_python() -> str:
    value = os.getenv("OPENVOICE_PYTHON", "").strip()
    if not value:
        raise OpenVoiceCloneError(
            "OPENVOICE_PYTHON is not set. Create a separate Python 3.10/3.11 OpenVoice environment "
            "and set OPENVOICE_PYTHON to that environment's python executable."
        )
    path = Path(value).expanduser()
    if not path.exists():
        raise OpenVoiceCloneError(f"OPENVOICE_PYTHON does not exist: {path}")
    return str(path)


def _checkpoint_dir() -> Path:
    value = os.getenv("OPENVOICE_CHECKPOINT_DIR", "").strip()
    if not value:
        raise OpenVoiceCloneError(
            "OPENVOICE_CHECKPOINT_DIR is not set. Set it to your OpenVoice checkpoint directory."
        )
    path = Path(value).expanduser()
    if not path.exists():
        raise OpenVoiceCloneError(f"OPENVOICE_CHECKPOINT_DIR does not exist: {path}")
    return path


def clone_with_openvoice(input_path: Path, output_path: Path, reference_path: Path, device: str = "auto") -> None:
    """Single-segment clone (legacy path, kept for CLI usage)."""
    python = _openvoice_python()
    checkpoint_dir = _checkpoint_dir()
    command = [
        python,
        "-c",
        OPENVOICE_SCRIPT,
        str(input_path),
        str(output_path),
        str(reference_path),
        str(checkpoint_dir),
        device,
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        if len(detail) > 1600:
            detail = detail[-1600:]
        raise OpenVoiceCloneError(detail or "OpenVoice voice conversion failed.")


def clone_batch_openvoice(
    segments: list[dict],
    reference_path: Path,
    device: str = "auto",
) -> list[dict]:
    """Clone multiple segments in one subprocess, sharing the same target_se.

    segments: list of {"segment_index": int, "input_path": str, "output_path": str}
    Returns: list of {"segment_index": int, "ok": bool, "error"?: str}
    """
    if not segments:
        return []

    python = _openvoice_python()
    checkpoint_dir = _checkpoint_dir()

    manifest = {
        "reference_path": str(reference_path),
        "checkpoint_dir": str(checkpoint_dir),
        "device": device,
        "segments": segments,
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix="ov_batch_", delete=False, encoding="utf-8"
    ) as f:
        json.dump(manifest, f, ensure_ascii=False)
        manifest_path = Path(f.name)

    try:
        result = subprocess.run(
            [python, "-c", OPENVOICE_BATCH_SCRIPT, str(manifest_path)],
            capture_output=True, text=True, check=False,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            if len(detail) > 1600:
                detail = detail[-1600:]
            raise OpenVoiceCloneError(detail or "OpenVoice batch conversion failed.")

        results_path = manifest_path.with_suffix(".results.json")
        if results_path.exists():
            results = json.loads(results_path.read_text(encoding="utf-8"))
            results_path.unlink(missing_ok=True)
            return results
        return [{"segment_index": s["segment_index"], "ok": False, "error": "No results file"} for s in segments]
    finally:
        manifest_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Local OpenVoice zero-shot voice conversion adapter.")
    parser.add_argument("--input", required=True, type=Path, help="Input synthesized speech audio.")
    parser.add_argument("--output", required=True, type=Path, help="Output WAV path.")
    parser.add_argument("--reference", required=True, type=Path, help="Target speaker reference audio.")
    parser.add_argument("--device", default=os.getenv("OPENVOICE_DEVICE", "auto"), choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    clone_with_openvoice(args.input, args.output, args.reference, args.device)
    print("openvoice voice clone complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
