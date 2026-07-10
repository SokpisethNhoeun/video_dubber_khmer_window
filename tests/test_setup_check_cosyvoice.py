from __future__ import annotations

import json
from types import SimpleNamespace

from core.setup_check import _check_cosyvoice_cuda


def test_cosyvoice_cuda_check_warns_when_cuda_unavailable(monkeypatch):
    def fake_run(*_args, **_kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({
                "torch": "2.3.1+cu121",
                "cuda_compiled": "12.1",
                "cuda_available": False,
                "device_count": 0,
            }),
            stderr="",
        )

    monkeypatch.setattr("core.setup_check.subprocess.run", fake_run)

    result = _check_cosyvoice_cuda("/fake/python")

    assert result.status == "WARN"
    assert "CosyVoice will run on CPU" in result.message


def test_cosyvoice_cuda_check_ok_when_cuda_available(monkeypatch):
    def fake_run(*_args, **_kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({
                "torch": "2.3.1+cu121",
                "cuda_compiled": "12.1",
                "cuda_available": True,
                "device_count": 1,
                "device_name": "NVIDIA RTX",
            }),
            stderr="",
        )

    monkeypatch.setattr("core.setup_check.subprocess.run", fake_run)

    result = _check_cosyvoice_cuda("/fake/python")

    assert result.status == "OK"
    assert "NVIDIA RTX" in result.message
