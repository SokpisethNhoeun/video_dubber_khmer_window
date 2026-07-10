from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


API_BASE_URL = "https://api.elevenlabs.io/v1"
DEFAULT_STS_MODEL = "eleven_multilingual_sts_v2"
DEFAULT_OUTPUT_FORMAT = "mp3_44100_128"


class ElevenLabsCloneError(RuntimeError):
    pass


def _api_key() -> str:
    key = os.getenv("ELEVENLABS_API_KEY", "").strip()
    if not key:
        raise ElevenLabsCloneError("Set ELEVENLABS_API_KEY before using the ElevenLabs production clone backend.")
    return key


def _ensure_ffmpeg() -> None:
    if not shutil.which("ffmpeg"):
        raise ElevenLabsCloneError("ffmpeg is required to write the final WAV output.")


def _cache_path(cache_dir: Path | None) -> Path:
    base_dir = cache_dir or Path("cache")
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / "elevenlabs_voice_cache.json"


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_cache(cache_file: Path) -> dict[str, dict[str, str]]:
    if not cache_file.exists():
        return {}
    try:
        data = json.loads(cache_file.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _save_cache(cache_file: Path, data: dict[str, dict[str, str]]) -> None:
    cache_file.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _guess_content_type(path: Path) -> str:
    return mimetypes.guess_type(str(path))[0] or "application/octet-stream"


def _multipart_form(fields: dict[str, str], files: list[tuple[str, Path]]) -> tuple[bytes, str]:
    boundary = f"----video-dubber-{uuid.uuid4().hex}"
    body = bytearray()
    for name, value in fields.items():
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
        body.extend(str(value).encode("utf-8"))
        body.extend(b"\r\n")

    for field_name, path in files:
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(
            (
                f'Content-Disposition: form-data; name="{field_name}"; '
                f'filename="{path.name}"\r\n'
            ).encode("utf-8")
        )
        body.extend(f"Content-Type: {_guess_content_type(path)}\r\n\r\n".encode("utf-8"))
        body.extend(path.read_bytes())
        body.extend(b"\r\n")

    body.extend(f"--{boundary}--\r\n".encode("utf-8"))
    return bytes(body), f"multipart/form-data; boundary={boundary}"


def _request_json(
    method: str,
    path: str,
    api_key: str,
    body: bytes | None = None,
    content_type: str | None = None,
    query: dict[str, str] | None = None,
) -> dict:
    response = _request_bytes(method, path, api_key, body, content_type, query)
    try:
        data = json.loads(response.decode("utf-8"))
    except Exception as exc:
        raise ElevenLabsCloneError("ElevenLabs returned a non-JSON response.") from exc
    if not isinstance(data, dict):
        raise ElevenLabsCloneError("ElevenLabs returned an unexpected response.")
    return data


def _request_bytes(
    method: str,
    path: str,
    api_key: str,
    body: bytes | None = None,
    content_type: str | None = None,
    query: dict[str, str] | None = None,
) -> bytes:
    url = f"{API_BASE_URL}{path}"
    if query:
        url = f"{url}?{urlencode(query)}"
    headers = {"xi-api-key": api_key}
    if content_type:
        headers["Content-Type"] = content_type
    request = Request(url, data=body, headers=headers, method=method)
    try:
        with urlopen(request, timeout=120) as response:
            return response.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        raise ElevenLabsCloneError(f"ElevenLabs API error {exc.code}: {detail}") from exc
    except URLError as exc:
        raise ElevenLabsCloneError(f"Could not reach ElevenLabs API: {exc.reason}") from exc


def _create_instant_voice(
    api_key: str,
    reference_audio: Path,
    name: str,
    remove_background_noise: bool,
) -> str:
    body, content_type = _multipart_form(
        {
            "name": name,
            "remove_background_noise": "true" if remove_background_noise else "false",
            "description": "Created by Khmer Video Dubber production clone backend",
        },
        [("files[]", reference_audio)],
    )
    data = _request_json("POST", "/voices/add", api_key, body, content_type)
    voice_id = str(data.get("voice_id", "")).strip()
    if not voice_id:
        raise ElevenLabsCloneError("ElevenLabs did not return a voice_id.")
    return voice_id


def _voice_id_for_reference(
    reference_audio: Path,
    voice_name: str,
    cache_file: Path,
    api_key: str,
    remove_background_noise: bool,
    force_recreate: bool,
) -> str:
    reference_hash = _file_hash(reference_audio)
    cache = _load_cache(cache_file)
    cache_key = f"sha256:{reference_hash}"
    if not force_recreate and cache_key in cache and cache[cache_key].get("voice_id"):
        return str(cache[cache_key]["voice_id"])

    voice_id = _create_instant_voice(api_key, reference_audio, voice_name, remove_background_noise)
    cache[cache_key] = {
        "voice_id": voice_id,
        "name": voice_name,
        "reference_audio": str(reference_audio),
        "sha256": reference_hash,
    }
    _save_cache(cache_file, cache)
    return voice_id


def _convert_to_wav(source_audio: Path, output_path: Path) -> None:
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source_audio),
            "-ac",
            "1",
            "-ar",
            "44100",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise ElevenLabsCloneError(result.stderr.strip() or "ffmpeg failed while converting ElevenLabs audio.")


def convert_speech_to_cloned_voice(
    input_audio: Path,
    output_audio: Path,
    reference_audio: Path,
    voice_name: str | None = None,
    model_id: str = DEFAULT_STS_MODEL,
    output_format: str = DEFAULT_OUTPUT_FORMAT,
    cache_dir: Path | None = None,
    remove_background_noise: bool = True,
    force_recreate_voice: bool = False,
) -> str:
    _ensure_ffmpeg()
    api_key = _api_key()
    input_audio = input_audio.expanduser()
    output_audio = output_audio.expanduser()
    reference_audio = reference_audio.expanduser()
    if not input_audio.exists():
        raise FileNotFoundError(f"Input audio does not exist: {input_audio}")
    if not reference_audio.exists():
        raise FileNotFoundError(f"Reference audio does not exist: {reference_audio}")
    output_audio.parent.mkdir(parents=True, exist_ok=True)

    name = voice_name or reference_audio.parent.name or reference_audio.stem
    cache_file = _cache_path(cache_dir)
    voice_id = _voice_id_for_reference(
        reference_audio,
        name,
        cache_file,
        api_key,
        remove_background_noise,
        force_recreate_voice,
    )

    body, content_type = _multipart_form(
        {
            "model_id": model_id,
        },
        [("audio", input_audio)],
    )
    audio_bytes = _request_bytes(
        "POST",
        f"/speech-to-speech/{voice_id}",
        api_key,
        body,
        content_type,
        {"output_format": output_format},
    )

    with tempfile.TemporaryDirectory(prefix="elevenlabs_clone_") as temp_name:
        temp_audio = Path(temp_name) / "speech.mp3"
        temp_audio.write_bytes(audio_bytes)
        _convert_to_wav(temp_audio, output_audio)
    return voice_id


def main() -> int:
    parser = argparse.ArgumentParser(description="Production ElevenLabs instant voice clone and voice changer backend.")
    parser.add_argument("--input", required=True, type=Path, help="Input synthesized speech audio.")
    parser.add_argument("--output", required=True, type=Path, help="Output WAV path.")
    parser.add_argument("--reference", required=True, type=Path, help="Reference speaker MP3/WAV.")
    parser.add_argument("--voice-name", default=None, help="Name used when creating the ElevenLabs cloned voice.")
    parser.add_argument("--model-id", default=DEFAULT_STS_MODEL)
    parser.add_argument("--output-format", default=DEFAULT_OUTPUT_FORMAT)
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--no-background-noise-removal", action="store_true")
    parser.add_argument("--force-recreate-voice", action="store_true")
    args = parser.parse_args()

    try:
        voice_id = convert_speech_to_cloned_voice(
            args.input,
            args.output,
            args.reference,
            args.voice_name,
            args.model_id,
            args.output_format,
            args.cache_dir,
            not args.no_background_noise_removal,
            args.force_recreate_voice,
        )
    except Exception as exc:
        print(f"ElevenLabs production clone failed: {exc}", file=sys.stderr)
        return 2
    print(f"ElevenLabs production clone complete: voice_id={voice_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
