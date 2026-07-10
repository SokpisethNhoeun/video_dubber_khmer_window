from __future__ import annotations

import os
from pathlib import Path

from config.release import apply_public_release_defaults
from config.runtime import is_frozen


def load_project_env(project_root: Path) -> None:
    apply_public_release_defaults()
    env_file = project_root / ".env"
    # Development may use .env. Packaged customers must never receive it.
    if is_frozen() or not env_file.exists():
        from config.user_secrets import apply_user_secrets
        apply_user_secrets()
        return

    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]

        if key and key not in os.environ:
            os.environ[key] = value

    from config.user_secrets import apply_user_secrets
    apply_user_secrets()


def env_path(name: str) -> Path | None:
    value = os.getenv(name, "").strip()
    if not value:
        return None
    return Path(value).expanduser()


def resolve_review_api_credentials() -> tuple[str | None, str, str]:
    """Resolve (api_key, base_url, model) for the OpenAI-compatible client used
    by transcript review and AI translation.

    Falls back to the customer's GEMINI_API_KEY when no dedicated
    TRANSCRIPT_REVIEW_API_KEY/OPENAI_API_KEY is set, since every activated
    customer already has a working Gemini key (checked at startup) — this
    makes AI review work out of the box without a separate provider setup.
    """
    api_key = os.getenv("TRANSCRIPT_REVIEW_API_KEY") or os.getenv("OPENAI_API_KEY")
    if api_key:
        base_url = os.getenv(
            "TRANSCRIPT_REVIEW_API_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta/openai",
        )
        model = os.getenv("TRANSCRIPT_REVIEW_MODEL", "gemini-3.1-flash-lite")
        return api_key, base_url.rstrip("/"), model

    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        base_url = os.getenv(
            "TRANSCRIPT_REVIEW_API_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta/openai",
        )
        model = os.getenv("TRANSCRIPT_REVIEW_MODEL", "gemini-3.1-flash-lite")
        return gemini_key, base_url.rstrip("/"), model

    return None, "", ""
