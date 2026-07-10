from __future__ import annotations

import hashlib
import json
from pathlib import Path


def cache_key(*parts: str) -> str:
    combined = "|".join(str(p) for p in parts)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def stage_cache_path(cache_dir: Path, stage: str, key: str) -> Path:
    stage_dir = cache_dir / "stage_cache" / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    return stage_dir / f"{key}.json"


def load_cached(cache_dir: Path, stage: str, key: str) -> dict | None:
    path = stage_cache_path(cache_dir, stage, key)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_cached(cache_dir: Path, stage: str, key: str, data: dict) -> None:
    path = stage_cache_path(cache_dir, stage, key)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def invalidate_stage(cache_dir: Path, stage: str) -> int:
    stage_dir = cache_dir / "stage_cache" / stage
    if not stage_dir.exists():
        return 0
    count = 0
    for f in stage_dir.glob("*.json"):
        f.unlink(missing_ok=True)
        count += 1
    return count


def invalidate_downstream(cache_dir: Path, from_stage: str) -> int:
    stage_order = ["transcribe", "translate", "review", "tts", "clone", "align", "mix"]
    try:
        start_idx = stage_order.index(from_stage)
    except ValueError:
        return 0
    count = 0
    for stage in stage_order[start_idx:]:
        count += invalidate_stage(cache_dir, stage)
    return count
