from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from modules.video_overlay import burn_subtitles_and_overlay


def test_overlay_text_and_image_use_independent_positions(monkeypatch, tmp_path: Path) -> None:
    input_video = tmp_path / "input.mp4"
    input_video.write_bytes(b"video")
    image = tmp_path / "logo.png"
    image.write_bytes(b"image")
    output_video = tmp_path / "output.mp4"
    captured: dict[str, object] = {}

    monkeypatch.setattr("modules.video_overlay._ensure_ffmpeg", lambda: None)
    monkeypatch.setattr("modules.video_overlay._get_video_resolution", lambda _path: (1280, 720))

    def fake_build_overlay_ass(text, output_path, position, opacity, **kwargs):
        captured["text"] = text
        captured["text_position"] = position
        captured["opacity"] = opacity
        Path(output_path).write_text("ass", encoding="utf-8")
        return Path(output_path)

    def fake_run(command, **_kwargs):
        captured["command"] = command
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr("modules.video_overlay._build_overlay_ass", fake_build_overlay_ass)
    monkeypatch.setattr("modules.video_overlay.subprocess.run", fake_run)

    burn_subtitles_and_overlay(
        input_video=input_video,
        output_video=output_video,
        overlay_text="hello",
        overlay_image_path=image,
        overlay_position="bottom_right",
        overlay_text_position="top_left",
        overlay_image_position="center",
        overlay_opacity=0.5,
        work_dir=tmp_path,
    )

    assert captured["text_position"] == "top_left"
    command = captured["command"]
    assert isinstance(command, list)
    filter_complex = command[command.index("-filter_complex") + 1]
    assert "overlay=(W-w)/2:(H-h)/2" in filter_complex


def test_overlay_legacy_position_still_controls_both_overlays(monkeypatch, tmp_path: Path) -> None:
    input_video = tmp_path / "input.mp4"
    input_video.write_bytes(b"video")
    image = tmp_path / "logo.png"
    image.write_bytes(b"image")
    output_video = tmp_path / "output.mp4"
    captured: dict[str, object] = {}

    monkeypatch.setattr("modules.video_overlay._ensure_ffmpeg", lambda: None)
    monkeypatch.setattr("modules.video_overlay._get_video_resolution", lambda _path: (1280, 720))

    def fake_build_overlay_ass(text, output_path, position, opacity, **kwargs):
        captured["text_position"] = position
        Path(output_path).write_text("ass", encoding="utf-8")
        return Path(output_path)

    def fake_run(command, **_kwargs):
        captured["command"] = command
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr("modules.video_overlay._build_overlay_ass", fake_build_overlay_ass)
    monkeypatch.setattr("modules.video_overlay.subprocess.run", fake_run)

    burn_subtitles_and_overlay(
        input_video=input_video,
        output_video=output_video,
        overlay_text="hello",
        overlay_image_path=image,
        overlay_position="top_right",
        overlay_opacity=0.5,
        work_dir=tmp_path,
    )

    assert captured["text_position"] == "top_right"
    command = captured["command"]
    assert isinstance(command, list)
    filter_complex = command[command.index("-filter_complex") + 1]
    assert "overlay=W-w-10:10" in filter_complex
