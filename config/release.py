from __future__ import annotations

import os
import json
import sys
from pathlib import Path


# Public desktop defaults only. The build workflow overrides LICENSE_SERVER_URL.
# Never add API keys, mail credentials, payment secrets, or database URLs here.
PUBLIC_RELEASE_DEFAULTS = {
    "LICENSE_SERVER_URL": "",
    "TRANSCRIPT_REVIEW_API_BASE_URL": "https://generativelanguage.googleapis.com/v1beta/openai",
    "TRANSCRIPT_REVIEW_MODEL": "gemini-3.1-flash-lite",
}


def apply_public_release_defaults() -> None:
    defaults = dict(PUBLIC_RELEASE_DEFAULTS)
    bundle = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))
    try:
        packaged = json.loads((bundle / "release.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        packaged = {}
    for name in defaults:
        value = packaged.get(name)
        if isinstance(value, str):
            defaults[name] = value.strip()
    for name, value in defaults.items():
        if value:
            os.environ.setdefault(name, value)
