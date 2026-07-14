from __future__ import annotations

import importlib.metadata as metadata
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from config.env import resolve_review_api_credentials
from core.context import PipelineSettings


@dataclass(frozen=True)
class SetupCheckResult:
    name: str
    status: str
    message: str

    @property
    def is_error(self) -> bool:
        return self.status == "ERROR"


def _result(name: str, status: str, message: str) -> SetupCheckResult:
    return SetupCheckResult(name=name, status=status, message=message)


def _package_version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _run_version_command(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=8,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    first_line = (completed.stdout or completed.stderr).splitlines()
    return first_line[0].strip() if first_line else None


def _check_ffmpeg() -> list[SetupCheckResult]:
    results: list[SetupCheckResult] = []
    for binary in ("ffmpeg", "ffprobe"):
        path = shutil.which(binary)
        if not path:
            results.append(_result(binary, "ERROR", f"{binary} is not installed or is not in PATH."))
            continue
        version = _run_version_command([binary, "-version"])
        if version:
            results.append(_result(binary, "OK", f"{version} ({path})"))
        else:
            results.append(_result(binary, "WARN", f"{binary} exists at {path}, but version check failed."))
    return results


def _check_python_packages(settings: PipelineSettings) -> list[SetupCheckResult]:
    results: list[SetupCheckResult] = []
    results.append(_result("Python", "OK", sys.version.split()[0]))

    try:
        import torch

        cuda_available = torch.cuda.is_available()
        if settings.device == "cuda" and not cuda_available:
            results.append(
                _result(
                    "PyTorch",
                    "ERROR",
                    f"torch {torch.__version__} is installed, but CUDA is not available. Select CPU or install a CUDA build.",
                )
            )
        else:
            device_text = "CUDA available" if cuda_available else "CPU only"
            results.append(_result("PyTorch", "OK", f"torch {torch.__version__}; {device_text}"))
    except Exception as exc:
        results.append(_result("PyTorch", "ERROR", f"Could not import torch: {exc}"))

    try:
        import torchaudio

        results.append(_result("torchaudio", "OK", torchaudio.__version__))
    except Exception as exc:
        results.append(_result("torchaudio", "ERROR", f"Could not import torchaudio: {exc}"))

    edge_tts_version = _package_version("edge-tts")
    if edge_tts_version:
        results.append(_result("edge-tts", "OK", edge_tts_version))
    else:
        results.append(_result("edge-tts", "ERROR", "edge-tts is not installed."))

    yt_dlp_version = _package_version("yt-dlp")
    if yt_dlp_version:
        results.append(_result("yt-dlp", "OK", yt_dlp_version))
    else:
        results.append(_result("yt-dlp", "WARN", "yt-dlp is not installed. URL video import will be unavailable."))

    if settings.preserve_bgm:
        demucs_version = _package_version("demucs")
        if demucs_version:
            results.append(_result("Demucs", "OK", demucs_version))
        else:
            results.append(
                _result(
                    "Demucs",
                    "ERROR",
                    "Background music preservation is enabled, but demucs is not installed.",
                )
            )

    return results


def _check_torchcodec() -> SetupCheckResult:
    version = _package_version("torchcodec")
    if not version:
        return _result("TorchCodec", "OK", "Not installed. This app does not require TorchCodec.")

    torch_version = _package_version("torch") or ""
    if torch_version.startswith("2.6") and not version.startswith("0.2"):
        return _result(
            "TorchCodec",
            "ERROR",
            f"torchcodec {version} is installed with torch {torch_version}. Remove it or install torchcodec 0.2.x.",
        )

    try:
        import torchcodec  # noqa: F401
    except Exception as exc:
        return _result(
            "TorchCodec",
            "ERROR",
            f"torchcodec {version} is installed but cannot load: {exc}. Remove it unless you explicitly need it.",
        )
    return _result("TorchCodec", "OK", version)


def _is_per_person_mode(settings: PipelineSettings) -> bool:
    return settings.voice_gender in {"per_person", "per_person_auto"}


def _command_setup_error(command_template: str, project_root: Path) -> str | None:
    try:
        parts = shlex.split(command_template)
    except ValueError as exc:
        return f"Invalid clone command: {exc}"

    if "modules.openvoice_voice_clone" in parts:
        python_path = os.getenv("OPENVOICE_PYTHON", "").strip()
        if not python_path:
            return "OpenVoice command selected, but OPENVOICE_PYTHON is not set."
        if not Path(python_path).expanduser().exists():
            return f"OPENVOICE_PYTHON does not exist: {python_path}"
        checkpoint_dir = os.getenv("OPENVOICE_CHECKPOINT_DIR", "").strip()
        if not checkpoint_dir:
            return "OpenVoice command selected, but OPENVOICE_CHECKPOINT_DIR is not set."
        if not Path(checkpoint_dir).expanduser().exists():
            return f"OPENVOICE_CHECKPOINT_DIR does not exist: {checkpoint_dir}"

    if "modules.elevenlabs_voice_clone" in parts and not os.getenv("ELEVENLABS_API_KEY", "").strip():
        return "ElevenLabs command selected, but ELEVENLABS_API_KEY is not set."

    if parts and Path(parts[0]).name == "python" and len(parts) > 1 and parts[1].endswith("infer.py"):
        infer_path = Path(parts[1])
        if not infer_path.is_absolute():
            infer_path = project_root / infer_path
        if not infer_path.exists():
            return "The command uses infer.py, but this project does not include infer.py."

    return None


def _check_cosyvoice_cuda(python_path: str) -> SetupCheckResult:
    script = """
import json
import torch

payload = {
    "torch": getattr(torch, "__version__", "unknown"),
    "cuda_compiled": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count(),
}
if payload["cuda_available"] and payload["device_count"] > 0:
    payload["device_name"] = torch.cuda.get_device_name(0)
print(json.dumps(payload))
"""
    try:
        completed = subprocess.run(
            [python_path, "-c", script],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=20,
        )
    except subprocess.TimeoutExpired:
        return _result(
            "CosyVoice CUDA",
            "WARN",
            "Timed out while checking CUDA in the CosyVoice Python environment. "
            "If cloning is slow, verify the GPU driver and PyTorch CUDA setup.",
        )
    except Exception as exc:
        return _result("CosyVoice CUDA", "WARN", f"Could not check CUDA in CosyVoice env: {exc}")

    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        return _result(
            "CosyVoice CUDA",
            "WARN",
            f"Could not check CUDA in CosyVoice env: {_shorten_message(detail)}",
        )

    try:
        payload = json.loads(completed.stdout.splitlines()[-1])
    except Exception as exc:
        return _result("CosyVoice CUDA", "WARN", f"Could not parse CosyVoice CUDA check: {exc}")

    torch_version = payload.get("torch", "unknown")
    cuda_compiled = payload.get("cuda_compiled") or "CPU build"
    if payload.get("cuda_available") and payload.get("device_count", 0) > 0:
        device_name = payload.get("device_name", "CUDA device")
        return _result(
            "CosyVoice CUDA",
            "OK",
            f"torch {torch_version}; CUDA {cuda_compiled}; {device_name}",
        )

    return _result(
        "CosyVoice CUDA",
        "WARN",
        f"torch {torch_version}; CUDA {cuda_compiled}, but no CUDA device is available. "
        "CosyVoice will run on CPU and can be extremely slow.",
    )


def _check_clone_command(settings: PipelineSettings, project_root: Path) -> list[SetupCheckResult]:
    results: list[SetupCheckResult] = []
    per_person = _is_per_person_mode(settings)
    needs_clone_command = (
        settings.rvc_enabled
        or per_person
        or settings.voice_female_reference_path is not None
        or settings.voice_male_reference_path is not None
    )
    if not needs_clone_command:
        return results

    # XTTS-v2 backend handles cloning internally — no external command template needed
    if settings.clone_backend == "xtts" and per_person:
        python_path = os.getenv("OPENVOICE_PYTHON", "").strip()
        if not python_path:
            results.append(_result("Clone command", "ERROR",
                                   "XTTS-v2 requires OPENVOICE_PYTHON to be set (same Python 3.10/3.11 env)."))
        elif not Path(python_path).expanduser().exists():
            results.append(_result("Clone command", "ERROR",
                                   f"OPENVOICE_PYTHON path does not exist: {python_path}"))
        else:
            results.append(_result("Clone command", "OK", "XTTS-v2 backend is configured."))
        return results

    if settings.clone_backend == "qwen3" and (
        per_person
        or settings.voice_female_reference_path is not None
        or settings.voice_male_reference_path is not None
    ):
        python_path = os.getenv("QWEN3_TTS_PYTHON", "").strip()
        if not python_path:
            results.append(_result("Clone command", "ERROR",
                                   "Qwen3-TTS 1.7B requires QWEN3_TTS_PYTHON to be set "
                                   "(Python environment with qwen-tts installed)."))
        elif not Path(python_path).expanduser().exists():
            results.append(_result("Clone command", "ERROR",
                                   f"QWEN3_TTS_PYTHON path does not exist: {python_path}"))
        else:
            model_name = os.getenv("QWEN3_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base").strip()
            results.append(_result("Clone command", "OK", f"Qwen3-TTS backend is configured ({model_name})."))
        return results

    if settings.clone_backend == "cosyvoice" and per_person:
        python_path = os.getenv("COSYVOICE_PYTHON", "").strip()
        if not python_path:
            results.append(_result("Clone command", "ERROR",
                                   "CosyVoice 2 requires COSYVOICE_PYTHON to be set "
                                   "(Python environment with CosyVoice installed)."))
        elif not Path(python_path).expanduser().exists():
            results.append(_result("Clone command", "ERROR",
                                   f"COSYVOICE_PYTHON path does not exist: {python_path}"))
        else:
            results.append(_result("Clone command", "OK", "CosyVoice 2 backend is configured."))
            results.append(_check_cosyvoice_cuda(str(Path(python_path).expanduser())))
        return results

    command_template = settings.rvc_command_template.strip()
    if not command_template:
        results.append(_result("Clone command", "ERROR", "Voice cloning needs a command template."))
        return results

    setup_error = _command_setup_error(command_template, project_root)
    if setup_error:
        results.append(_result("Clone command", "ERROR", setup_error))
    else:
        results.append(_result("Clone command", "OK", "Command template is usable."))

    if "{reference}" not in command_template and (
        settings.rvc_enabled or _is_per_person_mode(settings) or settings.voice_female_reference_path or settings.voice_male_reference_path
    ):
        results.append(_result("Clone reference", "ERROR", "Clone command must contain {reference}."))

    if "{model}" in command_template and (settings.rvc_model_path is None or not settings.rvc_model_path.exists()):
        results.append(_result("Clone model", "ERROR", "Command uses {model}; select an existing .pth file."))
    if "{index}" in command_template and (settings.rvc_index_path is None or not settings.rvc_index_path.exists()):
        results.append(_result("Clone index", "ERROR", "Command uses {index}; select an existing .index file."))
    if settings.rvc_enabled and "{reference}" in command_template and settings.rvc_reference_audio_path is not None:
        if not settings.rvc_reference_audio_path.exists():
            results.append(_result("Clone reference", "ERROR", f"Missing reference audio: {settings.rvc_reference_audio_path}"))

    return results


def _read_error_body(error: urllib.error.HTTPError) -> str:
    try:
        return error.read().decode("utf-8", errors="replace").strip()
    except Exception:
        return str(error)


def _shorten_message(message: str, limit: int = 500) -> str:
    cleaned = " ".join((message or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def _extract_chat_response(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("empty response body")

    payloads: list[dict] = []
    if raw.startswith("data:"):
        for line in raw.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            data = line.removeprefix("data:").strip()
            if not data or data == "[DONE]":
                continue
            payloads.append(json.loads(data))
    else:
        try:
            payloads.append(json.loads(raw))
        except json.JSONDecodeError:
            return raw

    parts: list[str] = []
    for payload in payloads:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        choice = choices[0]
        if not isinstance(choice, dict):
            continue

        message = choice.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and isinstance(item.get("text"), str):
                        parts.append(item["text"])

        delta = choice.get("delta")
        if isinstance(delta, dict) and isinstance(delta.get("content"), str):
            parts.append(delta["content"])

        text = choice.get("text")
        if isinstance(text, str):
            parts.append(text)

    joined = "".join(parts).strip()
    if joined:
        return joined

    raise ValueError("response did not include assistant text")


def _provider_error_hint(error_body: str) -> str:
    if "No credentials for provider:" not in error_body:
        return ""
    provider = error_body.split("No credentials for provider:", 1)[1].split('"', 1)[0].strip()
    if provider:
        return (
            f" Configure credentials for provider '{provider}' in your local AI router, "
            "or select a model/provider route that already has credentials."
        )
    return " Configure the upstream provider credentials in your local AI router."


def _check_ai_hello(settings: PipelineSettings, api_key: str, base_url: str, model: str) -> SetupCheckResult:
    timeout = float(os.getenv("TRANSCRIPT_REVIEW_HEALTH_TIMEOUT", "20"))
    host = urllib.parse.urlparse(base_url).netloc or base_url
    provider_hint = f" Check that TRANSCRIPT_REVIEW_API_BASE_URL points to your AI provider and that TRANSCRIPT_REVIEW_API_KEY belongs to {host}."

    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 32,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": "Reply with one short greeting. No markdown.",
            },
            {"role": "user", "content": "hello"},
        ],
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        status = "ERROR" if settings.transcript_review_mode == "ai" else "WARN"
        error_body = _read_error_body(exc)
        return _result(
            "Transcript AI hello",
            status,
            f"HTTP {exc.code}: {_shorten_message(error_body)}{_provider_error_hint(error_body)}{provider_hint}",
        )
    except Exception as exc:
        status = "ERROR" if settings.transcript_review_mode == "ai" else "WARN"
        return _result("Transcript AI hello", status, f"Request failed: {_shorten_message(str(exc))}{provider_hint}")

    try:
        reply = _extract_chat_response(raw)
    except Exception as exc:
        status = "ERROR" if settings.transcript_review_mode == "ai" else "WARN"
        return _result("Transcript AI hello", status, f"Model responded, but parsing failed: {exc}")

    mode_note = "" if settings.transcript_review_mode == "ai" else " Review mode is not AI, so this is only a connectivity check."
    return _result(
        "Transcript AI hello",
        "OK",
        f"Model {model} replied: \"{_shorten_message(reply, 160)}\".{mode_note}",
    )


def _check_transcript_review(settings: PipelineSettings) -> list[SetupCheckResult]:
    results: list[SetupCheckResult] = []
    api_key, base_url, model = resolve_review_api_credentials()
    if api_key:
        results.append(_check_ai_hello(settings, api_key, base_url, model))
    elif settings.transcript_review_mode == "ai":
        results.append(
            _result(
                "Transcript AI review",
                "WARN",
                "No TRANSCRIPT_REVIEW_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY found; "
                "local Khmer cleanup will be used.",
            )
        )
    return results


def _check_selected_files(settings: PipelineSettings) -> list[SetupCheckResult]:
    results: list[SetupCheckResult] = []
    if settings.output_dir.exists():
        results.append(_result("Output folder", "OK", str(settings.output_dir)))
    else:
        results.append(_result("Output folder", "ERROR", f"Missing output folder: {settings.output_dir}"))

    videos = settings.input_videos if settings.input_videos else [settings.input_video]
    missing = [video for video in videos if not video.exists()]
    if missing:
        results.append(_result("Input videos", "WARN", f"{len(missing)} selected video(s) do not exist."))
    elif videos:
        results.append(_result("Input videos", "OK", f"{len(videos)} selected video(s)."))
    else:
        results.append(_result("Input videos", "WARN", "No input video selected."))
    return results


def _check_gpu_cuda() -> list[SetupCheckResult]:
    results: list[SetupCheckResult] = []
    try:
        import torch
        if not torch.cuda.is_available():
            results.append(_result("CUDA GPU", "WARN", "CUDA is not available in PyTorch. CPU will be used for execution, which is much slower."))
            return results
        
        device_count = torch.cuda.device_count()
        gpu_names = []
        gpu_vrams = []
        for i in range(device_count):
            name = torch.cuda.get_device_name(i)
            free_bytes, total_bytes = torch.cuda.mem_get_info(i)
            free_gb = free_bytes / (1024 ** 3)
            total_gb = total_bytes / (1024 ** 3)
            gpu_names.append(name)
            gpu_vrams.append(f"{free_gb:.1f} GB / {total_gb:.1f} GB free")
            
        details = ", ".join(f"[{i}] {name} ({vram})" for i, (name, vram) in enumerate(zip(gpu_names, gpu_vrams)))
        results.append(_result("CUDA GPU", "OK", f"{device_count} device(s) found: {details}"))
    except Exception as exc:
        results.append(_result("CUDA GPU", "WARN", f"Could not perform GPU diagnostics: {exc}"))
    return results


def run_setup_checks(settings: PipelineSettings, project_root: Path) -> list[SetupCheckResult]:
    results: list[SetupCheckResult] = []
    results.extend(_check_selected_files(settings))
    results.extend(_check_gpu_cuda())
    results.extend(_check_ffmpeg())
    results.extend(_check_python_packages(settings))
    results.append(_check_torchcodec())
    results.extend(_check_clone_command(settings, project_root))
    results.extend(_check_transcript_review(settings))

    if not importlib.util.find_spec("soundfile"):
        results.append(_result("soundfile", "ERROR", "soundfile is not installed."))
    else:
        results.append(_result("soundfile", "OK", _package_version("soundfile") or "installed"))
    return results
