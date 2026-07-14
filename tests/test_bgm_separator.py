from __future__ import annotations

import builtins
from threading import Event

import pytest

from modules import bgm_separator


class FakeProcess:
    def __init__(self) -> None:
        self.stderr = iter(["Demucs warning\n"])
        self.returncode = None
        self._polls = 0

    def poll(self):
        self._polls += 1
        if self._polls > 1:
            self.returncode = 0
        return self.returncode

    def terminate(self):
        self.returncode = -1

    def kill(self):
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


def test_frozen_missing_demucs_tells_user_to_reinstall(monkeypatch) -> None:
    real_import = builtins.__import__

    def missing_demucs(name, *args, **kwargs):
        if name == "demucs":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_demucs)
    monkeypatch.setattr(bgm_separator.sys, "frozen", True, raising=False)

    with pytest.raises(RuntimeError, match="Reinstall the latest Khmer Video Dubber"):
        bgm_separator.install_demucs_if_needed()


def test_demucs_stderr_reader_is_portable_to_windows(monkeypatch) -> None:
    process = FakeProcess()
    monkeypatch.setattr(bgm_separator.subprocess, "Popen", lambda *_args, **_kwargs: process)

    bgm_separator._run_demucs(["app.exe", "-m", "demucs.separate"], Event())

    assert process.returncode == 0
