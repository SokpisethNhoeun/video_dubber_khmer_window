from __future__ import annotations

import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event

from core.context import CancellationError
from modules.audio_utils import ensure_ffmpeg


@dataclass
class FooterOverlayConfig:
    enabled: bool = False
    style: str = "fixed"
    text: str = ""
    texts: list[str] = field(default_factory=list)
    position: str = "bottom"
    bg_color: str = "black"
    text_color: str = "white"
    opacity: float = 0.7
    scroll_speed: int = 150
    rotation_interval: float = 5.0
    font_size: int = 0
    height_percent: float = 5.0

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "style": self.style,
            "text": self.text,
            "texts": self.texts,
            "position": self.position,
            "bg_color": self.bg_color,
            "text_color": self.text_color,
            "opacity": self.opacity,
            "scroll_speed": self.scroll_speed,
            "rotation_interval": self.rotation_interval,
            "font_size": self.font_size,
            "height_percent": self.height_percent,
        }

    @classmethod
    def from_dict(cls, data: dict) -> FooterOverlayConfig:
        return cls(
            enabled=data.get("enabled", False),
            style=data.get("style", "fixed"),
            text=data.get("text", ""),
            texts=data.get("texts", []),
            position=data.get("position", "bottom"),
            bg_color=data.get("bg_color", "black"),
            text_color=data.get("text_color", "white"),
            opacity=data.get("opacity", 0.7),
            scroll_speed=data.get("scroll_speed", 150),
            rotation_interval=data.get("rotation_interval", 5.0),
            font_size=data.get("font_size", 0),
            height_percent=data.get("height_percent", 5.0),
        )


_ASS_FONT_NAME = "Noto Sans"


def _run_ffmpeg(command: list[str], cancel_event: Event) -> None:
    process = subprocess.Popen(
        command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
    )
    try:
        while process.poll() is None:
            if cancel_event.is_set():
                process.terminate()
                raise CancellationError("Processing cancelled by user")
            time.sleep(0.2)
    except Exception:
        process.terminate()
        process.wait()
        raise

    if process.returncode != 0:
        stderr = process.stderr.read() if process.stderr else ""
        raise RuntimeError(f"ffmpeg failed: {stderr[:500]}")


def _get_video_info(video_path: Path) -> tuple[int, int, float]:
    """Returns (width, height, duration)."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-show_entries", "format=duration",
                "-of", "csv=p=0:s=,",
                str(video_path),
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            parts = lines[0].split(",") if lines else []
            w = int(parts[0]) if len(parts) >= 1 else 1920
            h = int(parts[1]) if len(parts) >= 2 else 1080
            dur = float(lines[1]) if len(lines) >= 2 else 600.0
            return w, h, dur
    except Exception:
        pass
    return 1920, 1080, 600.0


def _color_to_ass(color_name: str) -> str:
    """Convert color name/hex to ASS &HAABBGGRR format (alpha=00 means opaque)."""
    color_map = {
        "white": "&H00FFFFFF",
        "black": "&H00000000",
        "yellow": "&H0000FFFF",
        "red": "&H000000FF",
        "cyan": "&H00FFFF00",
        "green": "&H0000FF00",
        "blue": "&H00FF0000",
    }
    name = color_name.strip().lower()
    if name in color_map:
        return color_map[name]
    if name.startswith("0x") and len(name) >= 8:
        r = name[2:4]
        g = name[4:6]
        b = name[6:8]
        return f"&H00{b}{g}{r}"
    if name.startswith("#") and len(name) >= 7:
        r = name[1:3]
        g = name[3:5]
        b = name[5:7]
        return f"&H00{b}{g}{r}"
    return "&H00FFFFFF"


def _bg_color_to_ass(color_name: str, opacity: float) -> str:
    """Convert bg color to ASS BackColour with alpha from opacity."""
    if color_name.strip().lower() == "transparent":
        return "&HFF000000"
    alpha = format(int((1.0 - opacity) * 255), "02X")
    base = _color_to_ass(color_name)
    return f"&H{alpha}{base[4:]}"


def _format_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def _build_ass_header(width: int, height: int) -> str:
    return (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {width}\n"
        f"PlayResY: {height}\n"
        "WrapStyle: 0\n\n"
    )


def _build_fixed_ass(
    config: FooterOverlayConfig, width: int, height: int, duration: float,
) -> str:
    font_size = config.font_size if config.font_size > 0 else max(18, int(height * config.height_percent / 100))
    primary = _color_to_ass(config.text_color)
    back = _bg_color_to_ass(config.bg_color, config.opacity)
    alignment = 2 if config.position == "bottom" else 8
    border_style = 3 if config.bg_color.strip().lower() != "transparent" else 1
    outline = font_size // 3 if border_style == 3 else 2
    margin_v = 20

    header = _build_ass_header(width, height)
    styles = (
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Footer,{_ASS_FONT_NAME},{font_size},{primary},{primary},&H00000000,{back},"
        f"1,0,0,0,100,100,0,0,{border_style},{outline},0,{alignment},20,20,{margin_v},1\n\n"
    )
    events = (
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        f"Dialogue: 0,{_format_time(0)},{_format_time(duration)},Footer,,0,0,0,,{config.text}\n"
    )
    return header + styles + events


def _build_marquee_ass(
    config: FooterOverlayConfig, width: int, height: int, duration: float,
) -> str:
    font_size = config.font_size if config.font_size > 0 else max(18, int(height * config.height_percent / 100))
    primary = _color_to_ass(config.text_color)
    back = _bg_color_to_ass(config.bg_color, config.opacity)
    border_style = 3 if config.bg_color.strip().lower() != "transparent" else 1
    outline = font_size // 3 if border_style == 3 else 2

    y_pos = height - font_size - 20 if config.position == "bottom" else 20

    header = _build_ass_header(width, height)
    styles = (
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Marquee,{_ASS_FONT_NAME},{font_size},{primary},{primary},&H00000000,{back},"
        f"1,0,0,0,100,100,0,0,{border_style},{outline},0,7,0,0,0,1\n\n"
    )

    speed = config.scroll_speed
    text_est_width = len(config.text) * font_size * 0.6
    scroll_duration = (width + text_est_width) / max(1, speed)

    events = "[Events]\n" \
             "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"

    t = 0.0
    while t < duration:
        end_t = min(t + scroll_duration, duration)
        x_start = width + 10
        x_end = -int(text_est_width) - 10
        move = f"\\move({x_start},{y_pos},{x_end},{y_pos})"
        events += (
            f"Dialogue: 0,{_format_time(t)},{_format_time(end_t)},Marquee,,0,0,0,,"
            f"{{{move}}}{config.text}\n"
        )
        t += scroll_duration

    return header + styles + events


def _build_circular_ass(
    config: FooterOverlayConfig, width: int, height: int, duration: float,
) -> str:
    texts = config.texts if config.texts else [config.text]
    if not texts:
        return ""

    font_size = config.font_size if config.font_size > 0 else max(18, int(height * config.height_percent / 100))
    primary = _color_to_ass(config.text_color)
    back = _bg_color_to_ass(config.bg_color, config.opacity)
    alignment = 2 if config.position == "bottom" else 8
    border_style = 3 if config.bg_color.strip().lower() != "transparent" else 1
    outline = font_size // 3 if border_style == 3 else 2
    margin_v = 20

    header = _build_ass_header(width, height)
    styles = (
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Rotate,{_ASS_FONT_NAME},{font_size},{primary},{primary},&H00000000,{back},"
        f"1,0,0,0,100,100,0,0,{border_style},{outline},0,{alignment},20,20,{margin_v},1\n\n"
    )

    interval = config.rotation_interval
    events = "[Events]\n" \
             "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"

    t = 0.0
    i = 0
    fade = r"{\fad(300,300)}"
    while t < duration:
        text = texts[i % len(texts)]
        end_t = min(t + interval, duration)
        events += (
            f"Dialogue: 0,{_format_time(t)},{_format_time(end_t)},Rotate,,0,0,0,,"
            f"{fade}{text}\n"
        )
        t += interval
        i += 1

    return header + styles + events


def build_footer_ass(
    config: FooterOverlayConfig,
    width: int,
    height: int,
    duration: float,
) -> str:
    if config.style == "marquee":
        return _build_marquee_ass(config, width, height, duration)
    elif config.style == "circular":
        return _build_circular_ass(config, width, height, duration)
    else:
        return _build_fixed_ass(config, width, height, duration)


def burn_footer(
    input_video: Path,
    output_video: Path,
    config: FooterOverlayConfig,
    cancel_event: Event,
) -> Path:
    """Burn footer/banner overlay onto video using ASS subtitles for proper Khmer+English support."""
    ensure_ffmpeg()

    if not config.enabled:
        return input_video

    if cancel_event.is_set():
        raise CancellationError("Processing cancelled by user")

    width, height, duration = _get_video_info(input_video)
    output_video.parent.mkdir(parents=True, exist_ok=True)

    ass_content = build_footer_ass(config, width, height, duration)
    if not ass_content:
        return input_video

    ass_file = output_video.with_suffix(".ass")
    try:
        ass_file.write_text(ass_content, encoding="utf-8")
        command = [
            "ffmpeg", "-y",
            "-i", str(input_video),
            "-vf", f"ass={ass_file}",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "copy",
            str(output_video),
        ]
        _run_ffmpeg(command, cancel_event)
    finally:
        ass_file.unlink(missing_ok=True)

    return output_video
