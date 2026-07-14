from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from threading import Event

from core.context import CancellationError, Segment
from modules.transcript_exports import export_srt

_KHMER_FONT_NAME = "Noto Sans Khmer"

# ASS alignment numpad: 7=TL 8=TC 9=TR / 4=ML 5=MC 6=MR / 1=BL 2=BC 3=BR
_ASS_ALIGNMENT = {
    "top_left": 7,
    "top_right": 9,
    "center": 5,
    "bottom_left": 1,
    "bottom_right": 3,
}

_COLOR_MAP = {
    "white": "&H00FFFFFF",
    "yellow": "&H0000FFFF",
    "green": "&H0000FF00",
    "cyan": "&H00FFFF00",
    "red": "&H000000FF",
    "blue": "&H00FF0000",
}

# Overlay position expressions for image overlay (FFmpeg overlay filter).
_IMAGE_POSITION_MAP = {
    "top_left": "10:10",
    "top_right": "W-w-10:10",
    "bottom_left": "10:H-h-10",
    "bottom_right": "W-w-10:H-h-10",
    "center": "(W-w)/2:(H-h)/2",
}


def _ensure_ffmpeg() -> None:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required for video overlay operations.")


def _build_overlay_ass(
    text: str,
    output_path: Path,
    position: str,
    opacity: float,
    font_size: int = 28,
    video_width: int = 1920,
    video_height: int = 1080,
) -> Path:
    """Generate an ASS subtitle file that renders overlay text with correct
    positioning and mixed Khmer+Latin font fallback via libass."""
    alignment = _ASS_ALIGNMENT.get(position, 3)
    margin = 20

    # ASS opacity is inverted: 00=opaque, FF=transparent.
    alpha = max(0, min(255, int((1.0 - opacity) * 255)))
    alpha_hex = f"{alpha:02X}"
    bg_alpha = max(0, min(255, int((1.0 - opacity * 0.6) * 255)))
    bg_alpha_hex = f"{bg_alpha:02X}"

    ass_content = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {video_width}\n"
        f"PlayResY: {video_height}\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Overlay,{_KHMER_FONT_NAME},{font_size},"
        f"&H{alpha_hex}FFFFFF,&H{alpha_hex}FFFFFF,"
        f"&H{bg_alpha_hex}000000,&H80000000,"
        f"0,0,0,0,100,100,0,0,1,2,1,"
        f"{alignment},{margin},{margin},{margin},1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        f"Dialogue: 0,0:00:00.00,9:59:59.99,Overlay,,0,0,0,,{_ass_escape(text)}\n"
    )

    output_path.write_text(ass_content, encoding="utf-8")
    return output_path


def _ass_escape(text: str) -> str:
    """Escape text for ASS dialogue lines."""
    return text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


def _get_video_resolution(video_path: Path) -> tuple[int, int]:
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "csv=p=0:s=x",
                str(video_path),
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().splitlines()
            if lines:
                parts = lines[0].split("x")
                if len(parts) == 2:
                    return int(parts[0]), int(parts[1])
    except Exception:
        pass
    return 1920, 1080


def burn_subtitles_and_overlay(
    input_video: Path,
    output_video: Path,
    segments: list[Segment] | None = None,
    subtitle_language: str = "khmer",
    subtitle_font_size: int = 24,
    subtitle_font_name: str = "Noto Sans Khmer",
    subtitle_color: str = "white",
    subtitle_bg_opacity: float = 0.0,
    overlay_text: str = "",
    overlay_image_path: Path | None = None,
    overlay_position: str = "bottom_right",
    overlay_text_position: str | None = None,
    overlay_image_position: str | None = None,
    overlay_opacity: float = 0.7,
    work_dir: Path | None = None,
    cancel_event: Event | None = None,
) -> Path:
    _ensure_ffmpeg()
    text_position = overlay_text_position or overlay_position
    image_position = overlay_image_position or overlay_position

    filters: list[str] = []
    temp_files: list[Path] = []
    temp_dir = work_dir or input_video.parent

    if segments:
        temp_srt = temp_dir / f"_burn_{subtitle_language}.srt"
        export_srt(temp_srt, segments, language=subtitle_language)
        temp_files.append(temp_srt)
        srt_path_escaped = str(temp_srt).replace("\\", "/").replace(":", "\\:")
        color_hex = _COLOR_MAP.get(subtitle_color.lower(), "&H00FFFFFF")
        if subtitle_bg_opacity > 0.0:
            bg_alpha = max(0, min(255, int((1.0 - subtitle_bg_opacity) * 255)))
            bg_hex = f"&H{bg_alpha:02X}000000"
            border_style = 3
            outline_style = f"Outline=0,Shadow=0,BackColour={bg_hex}"
        else:
            border_style = 1
            outline_style = "Outline=2,Shadow=1,OutlineColour=&H00000000"
        font_style = f"Fontname={subtitle_font_name},"
        filters.append(
            f"subtitles='{srt_path_escaped}':force_style='{font_style}FontSize={subtitle_font_size},"
            f"PrimaryColour={color_hex},{outline_style},BorderStyle={border_style},"
            f"MarginV=20'"
        )

    if overlay_text.strip():
        vw, vh = _get_video_resolution(input_video)
        overlay_ass = temp_dir / "_overlay_text.ass"
        _build_overlay_ass(
            overlay_text.strip(),
            overlay_ass,
            text_position,
            overlay_opacity,
            font_size=28,
            video_width=vw,
            video_height=vh,
        )
        temp_files.append(overlay_ass)
        ass_path_escaped = str(overlay_ass).replace("\\", "/").replace(":", "\\:")
        filters.append(f"ass='{ass_path_escaped}'")

    has_image_overlay = overlay_image_path and overlay_image_path.exists()

    if not filters and not has_image_overlay:
        return input_video

    command = ["ffmpeg", "-y", "-i", str(input_video)]

    if has_image_overlay:
        command.extend(["-i", str(overlay_image_path)])

    filter_parts = []
    if filters:
        filter_parts.append(",".join(filters))

    if has_image_overlay:
        pos_expr = _IMAGE_POSITION_MAP.get(image_position, "W-w-10:H-h-10")
        if filter_parts:
            overlay_filter = (
                f"[0:v]{filter_parts[0]}[sub];"
                f"[1:v]format=rgba,colorchannelmixer=aa={overlay_opacity}[logo];"
                f"[sub][logo]overlay={pos_expr}"
            )
        else:
            overlay_filter = (
                f"[1:v]format=rgba,colorchannelmixer=aa={overlay_opacity}[logo];"
                f"[0:v][logo]overlay={pos_expr}"
            )
        command.extend(["-filter_complex", overlay_filter])
    else:
        command.extend(["-vf", filter_parts[0]])

    command.extend([
        "-c:a", "copy",
        str(output_video),
    ])

    output_video.parent.mkdir(parents=True, exist_ok=True)

    if cancel_event and cancel_event.is_set():
        raise CancellationError("Processing cancelled by user")

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        detail = (result.stderr or "").strip()[-800:]
        raise RuntimeError(f"Subtitle/overlay burn failed: {detail}")

    for tmp in temp_files:
        if tmp.exists():
            tmp.unlink(missing_ok=True)

    return output_video
