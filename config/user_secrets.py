from __future__ import annotations

import json
import os
from pathlib import Path

from platformdirs import user_config_dir


def secrets_path() -> Path:
    return Path(user_config_dir("khmer-video-dubber", appauthor=False)) / "secrets.json"


def load_user_secrets() -> dict[str, str]:
    try:
        data = json.loads(secrets_path().read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return {str(key): str(value) for key, value in data.items() if value}


def save_user_secret(name: str, value: str) -> None:
    path = secrets_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = load_user_secrets()
    if value.strip():
        data[name] = value.strip()
    else:
        data.pop(name, None)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass


def apply_user_secrets() -> None:
    for name, value in load_user_secrets().items():
        os.environ[name] = value
