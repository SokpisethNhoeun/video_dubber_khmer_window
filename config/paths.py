from __future__ import annotations

from pathlib import Path

from platformdirs import user_data_dir

from config.models import FASTER_WHISPER_MODEL_PREFIX

APP_NAME = "khmer-video-dubber"


def app_data_dir() -> Path:
    return Path(user_data_dir(APP_NAME, appauthor=False))


def whisper_cache_dir() -> Path:
    path = app_data_dir() / "models" / "whisper"
    path.mkdir(parents=True, exist_ok=True)
    return path


def nllb_cache_dir() -> Path:
    path = app_data_dir() / "models" / "nllb"
    path.mkdir(parents=True, exist_ok=True)
    return path


def qwen_cache_dir() -> Path:
    path = app_data_dir() / "models" / "qwen3"
    path.mkdir(parents=True, exist_ok=True)
    return path


def cosyvoice_cache_dir() -> Path:
    path = app_data_dir() / "models" / "cosyvoice"
    path.mkdir(parents=True, exist_ok=True)
    return path


def repository_snapshot_exists(repo_id: str, cache_dir: Path) -> bool:
    repo_dir = cache_dir / f"models--{repo_id.replace('/', '--')}"
    ref = repo_dir / "refs" / "main"
    try:
        commit = ref.read_text(encoding="utf-8").strip()
    except OSError:
        return False
    snapshot = repo_dir / "snapshots" / commit
    return snapshot.is_dir() and any(path.is_file() for path in snapshot.rglob("*"))


def repository_snapshot_exists(repo_id: str, cache_dir: Path) -> bool:
    repo_dir = cache_dir / f"models--{repo_id.replace('/', '--')}"
    ref = repo_dir / "refs" / "main"
    try:
        commit = ref.read_text(encoding="utf-8").strip()
    except OSError:
        return False
    snapshot = repo_dir / "snapshots" / commit
    return snapshot.is_dir() and any(path.is_file() for path in snapshot.rglob("*"))


def is_whisper_model_downloaded(model_name: str) -> bool:
    from huggingface_hub import try_to_load_from_cache

    repo_id = f"{FASTER_WHISPER_MODEL_PREFIX}-{model_name}"
    result = try_to_load_from_cache(repo_id, filename="model.bin", cache_dir=str(whisper_cache_dir()))
    return isinstance(result, str)
