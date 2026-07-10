from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request


def validate_gemini_api_key(api_key: str, timeout: float = 15.0) -> tuple[bool, str]:
    key = api_key.strip()
    if not key:
        return False, "Enter a Gemini API key first."
    url = "https://generativelanguage.googleapis.com/v1beta/models?" + urllib.parse.urlencode({"key": key})
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code in {400, 401, 403}:
            return False, "Gemini rejected this API key. Check the key and its API permissions."
        return False, f"Gemini validation failed (HTTP {exc.code})."
    except (OSError, ValueError) as exc:
        return False, f"Could not contact Gemini: {exc}"
    if not payload.get("models"):
        return False, "The key connected, but Gemini returned no available models."
    return True, "Gemini API key is valid and can access Gemini models."
