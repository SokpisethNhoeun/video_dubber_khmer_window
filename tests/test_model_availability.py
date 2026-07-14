from __future__ import annotations

from config import paths


def test_clone_backends_require_runtime_and_checkpoint(monkeypatch, tmp_path) -> None:
    runtime = tmp_path / "python.exe"
    runtime.write_bytes(b"runtime")
    monkeypatch.setenv("QWEN3_TTS_PYTHON", str(runtime))
    monkeypatch.delenv("COSYVOICE_PYTHON", raising=False)
    monkeypatch.delenv("OPENVOICE_PYTHON", raising=False)
    monkeypatch.setattr(
        paths,
        "repository_snapshot_exists",
        lambda repo_id, _cache: repo_id.startswith("Qwen/"),
    )

    assert paths.installed_clone_backends() == [
        ("Qwen3-TTS 1.7B (best clone + emotion)", "qwen3")
    ]


def test_clone_backends_are_empty_without_runtimes(monkeypatch) -> None:
    monkeypatch.delenv("QWEN3_TTS_PYTHON", raising=False)
    monkeypatch.delenv("COSYVOICE_PYTHON", raising=False)
    monkeypatch.delenv("OPENVOICE_PYTHON", raising=False)

    assert paths.installed_clone_backends() == []
