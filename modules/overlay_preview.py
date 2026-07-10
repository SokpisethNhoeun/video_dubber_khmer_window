from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from modules.audio_utils import ensure_ffmpeg
from modules.sponsor_card import SponsorCardConfig, _auto_font_size, _color_to_ass, _ASS_FONT_NAME
from modules.footer_overlay import FooterOverlayConfig, build_footer_ass


def _build_card_ass(config: SponsorCardConfig, width: int, height: int) -> str:
    text = config.text if config.text.strip() else "Sponsor"
    font_size = config.font_size if config.font_size > 0 else _auto_font_size(text, width)
    primary = _color_to_ass(config.text_color)

    return (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {width}\n"
        f"PlayResY: {height}\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Card,{_ASS_FONT_NAME},{font_size},{primary},{primary},&H00000000,&H00000000,"
        f"1,0,0,0,100,100,0,0,1,2,1,5,20,20,20,1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        f"Dialogue: 0,0:00:00.00,0:00:01.00,Card,,0,0,0,,{text}\n"
    )


def render_card_preview(
    config: SponsorCardConfig,
    output_png: Path,
    resolution: tuple[int, int] = (1920, 1080),
) -> Path:
    """Render a single PNG frame preview of a sponsor card."""
    ensure_ffmpeg()
    output_png.parent.mkdir(parents=True, exist_ok=True)

    width, height = resolution

    if config.card_type == "image" and config.image_path and Path(config.image_path).exists():
        command = [
            "ffmpeg", "-y",
            "-i", str(config.image_path),
            "-vf", (
                f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,setsar=1"
            ),
            "-frames:v", "1",
            str(output_png),
        ]
    else:
        ass_content = _build_card_ass(config, width, height)
        ass_file = output_png.with_suffix(".ass")
        ass_file.write_text(ass_content, encoding="utf-8")
        try:
            command = [
                "ffmpeg", "-y",
                "-f", "lavfi",
                "-i", f"color=c={config.bg_color}:s={width}x{height}:d=0.1",
                "-vf", f"ass={ass_file}",
                "-frames:v", "1",
                str(output_png),
            ]
            result = subprocess.run(command, capture_output=True, text=True, timeout=15)
            if result.returncode != 0:
                raise RuntimeError(f"Preview render failed: {result.stderr[:300]}")
            return output_png
        finally:
            ass_file.unlink(missing_ok=True)

    result = subprocess.run(command, capture_output=True, text=True, timeout=15)
    if result.returncode != 0:
        raise RuntimeError(f"Preview render failed: {result.stderr[:300]}")

    return output_png


def render_footer_preview(
    config: FooterOverlayConfig,
    output_png: Path,
    video_path: Path | None = None,
    resolution: tuple[int, int] = (1920, 1080),
    timestamp: float = 2.0,
) -> Path:
    """Render a single PNG frame preview of a footer overlay."""
    ensure_ffmpeg()
    output_png.parent.mkdir(parents=True, exist_ok=True)

    width, height = resolution
    ass_content = build_footer_ass(config, width, height, 10.0)
    if not ass_content:
        raise ValueError("No footer content to preview")

    ass_file = output_png.with_suffix(".ass")
    ass_file.write_text(ass_content, encoding="utf-8")
    try:
        if video_path and video_path.exists():
            command = [
                "ffmpeg", "-y",
                "-ss", str(timestamp),
                "-i", str(video_path),
                "-vf", f"ass={ass_file}",
                "-frames:v", "1",
                str(output_png),
            ]
        else:
            command = [
                "ffmpeg", "-y",
                "-f", "lavfi",
                "-i", f"color=c=0x1e1e2e:s={width}x{height}:d=0.1",
                "-vf", f"ass={ass_file}",
                "-frames:v", "1",
                str(output_png),
            ]

        result = subprocess.run(command, capture_output=True, text=True, timeout=15)
        if result.returncode != 0:
            raise RuntimeError(f"Preview render failed: {result.stderr[:300]}")
    finally:
        ass_file.unlink(missing_ok=True)

    return output_png
