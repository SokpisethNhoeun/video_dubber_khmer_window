from __future__ import annotations

import tempfile
from pathlib import Path

from core.stage_cache import (
    cache_key,
    invalidate_downstream,
    invalidate_stage,
    load_cached,
    save_cached,
)


def test_cache_key_deterministic():
    k1 = cache_key("a", "b", "c")
    k2 = cache_key("a", "b", "c")
    assert k1 == k2
    assert len(k1) == 64


def test_cache_key_varies():
    assert cache_key("a") != cache_key("b")


def test_save_and_load():
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = Path(tmp)
        save_cached(cache_dir, "tts", "abc123", {"segments": 5})
        result = load_cached(cache_dir, "tts", "abc123")
        assert result == {"segments": 5}


def test_load_missing_returns_none():
    with tempfile.TemporaryDirectory() as tmp:
        result = load_cached(Path(tmp), "tts", "nonexistent")
        assert result is None


def test_invalidate_stage():
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = Path(tmp)
        save_cached(cache_dir, "tts", "a", {"x": 1})
        save_cached(cache_dir, "tts", "b", {"x": 2})
        count = invalidate_stage(cache_dir, "tts")
        assert count == 2
        assert load_cached(cache_dir, "tts", "a") is None


def test_invalidate_downstream():
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = Path(tmp)
        save_cached(cache_dir, "translate", "k1", {"v": 1})
        save_cached(cache_dir, "tts", "k2", {"v": 2})
        save_cached(cache_dir, "clone", "k3", {"v": 3})
        count = invalidate_downstream(cache_dir, "tts")
        assert count >= 2
        assert load_cached(cache_dir, "translate", "k1") is not None
        assert load_cached(cache_dir, "tts", "k2") is None
        assert load_cached(cache_dir, "clone", "k3") is None
