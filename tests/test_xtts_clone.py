from __future__ import annotations

from pathlib import Path

import pytest

from modules.xtts_voice_clone import XTTSCloneError, _xtts_python


def test_xtts_clone_error_is_runtime_error():
    assert issubclass(XTTSCloneError, RuntimeError)


def test_xtts_python_missing_env(monkeypatch):
    monkeypatch.delenv("OPENVOICE_PYTHON", raising=False)
    with pytest.raises(XTTSCloneError, match="OPENVOICE_PYTHON is not set"):
        _xtts_python()


def test_xtts_python_nonexistent_path(monkeypatch):
    monkeypatch.setenv("OPENVOICE_PYTHON", "/nonexistent/python")
    with pytest.raises(XTTSCloneError, match="does not exist"):
        _xtts_python()


def test_clone_batch_empty():
    from modules.xtts_voice_clone import clone_batch
    # Should not spawn subprocess for empty list
    result = clone_batch([], Path("/fake/ref.wav"))
    assert result == []
