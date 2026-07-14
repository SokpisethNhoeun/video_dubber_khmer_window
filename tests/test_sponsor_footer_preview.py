from __future__ import annotations

from pathlib import Path
import pytest

from modules.footer_overlay import FooterOverlayConfig
from modules.overlay_preview import render_footer_preview


def test_render_footer_preview_when_disabled(monkeypatch, tmp_path: Path) -> None:
    # Setup mock subprocess run to simulate successful ffmpeg run
    captured: dict[str, list[str]] = {}

    class DummyCompletedProcess:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(command, **kwargs):
        captured["command"] = command
        # Write dummy bytes to target file to mock ffmpeg output creation
        Path(command[-1]).write_bytes(b"dummy png data")
        return DummyCompletedProcess()

    monkeypatch.setattr("modules.overlay_preview.subprocess.run", fake_run)
    monkeypatch.setattr("modules.overlay_preview.ensure_ffmpeg", lambda: None)

    config = FooterOverlayConfig(
        enabled=False,
        style="fixed",
        text="Sample Banner",
        texts=[],
        position="bottom",
        bg_color="black",
        text_color="white",
        opacity=0.8,
        scroll_speed=150,
        rotation_interval=5.0,
        font_size=24,
    )

    output_path = tmp_path / "test_preview.png"
    result_path = render_footer_preview(config, output_path)

    assert result_path == output_path
    assert output_path.exists()
    assert output_path.read_bytes() == b"dummy png data"

    # Verify command didn't use the ASS filter
    command = captured.get("command", [])
    assert not any("ass=" in arg for arg in command)
