from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event

from core.context import CancellationError
from modules.audio_utils import ensure_ffmpeg


@dataclass
class SponsorCardConfig:
    card_type: str = "text"
    position: str = "end"
    text: str = ""
    image_path: str = ""
    duration: float = 3.0
    bg_color: str = "black"
    text_color: str = "white"
    font_size: int = 0

    def to_dict(self) -> dict:
        return {
            "card_type": self.card_type,
            "position": self.position,
            "text": self.text,
            "image_path": self.image_path,
            "duration": self.duration,
            "bg_color": self.bg_color,
            "text_color": self.text_color,
            "font_size": self.font_size,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SponsorCardConfig:
        return cls(
            card_type=data.get("card_type", "text"),
            position=data.get("position", "end"),
            text=data.get("text", ""),
            image_path=data.get("image_path", ""),
            duration=data.get("duration", 3.0),
            bg_color=data.get("bg_color", "black"),
            text_color=data.get("text_color", "white"),
            font_size=data.get("font_size", 0),
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


def _get_video_info(video_path: Path) -> tuple[int, int, float, float]:
    """Returns (width, height, fps, duration)."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height,r_frame_rate",
                "-show_entries", "format=duration",
                "-of", "csv=p=0:s=,",
                str(video_path),
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            stream_parts = lines[0].split(",") if lines else []
            w = int(stream_parts[0]) if len(stream_parts) >= 1 else 1920
            h = int(stream_parts[1]) if len(stream_parts) >= 2 else 1080
            fps_str = stream_parts[2] if len(stream_parts) >= 3 else "30/1"
            fps_parts = fps_str.split("/")
            fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else 30.0
            dur = float(lines[1]) if len(lines) >= 2 else 0.0
            return w, h, fps, dur
    except Exception:
        pass
    return 1920, 1080, 30.0, 0.0


def _auto_font_size(text: str, video_width: int) -> int:
    char_count = max(1, len(text))
    size = min(96, max(24, video_width // char_count))
    return size


def _escape_text(text: str) -> str:
    escaped = text.replace("'", "’")
    escaped = escaped.replace(":", "\\:")
    escaped = escaped.replace("[", "\\[").replace("]", "\\]")
    escaped = escaped.replace("%", "%%")
    return escaped


def generate_card(
    config: SponsorCardConfig,
    resolution: tuple[int, int],
    fps: float,
    output_path: Path,
    cancel_event: Event,
) -> Path:
    """Generate a sponsor card video clip (text or image)."""
    ensure_ffmpeg()
    width, height = resolution
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if config.card_type == "image" and config.image_path and Path(config.image_path).exists():
        _generate_image_card(Path(config.image_path), output_path, width, height, fps, config.duration, cancel_event)
    else:
        font_size = config.font_size if config.font_size > 0 else _auto_font_size(config.text, width)
        _generate_text_card(
            config.text, output_path, width, height, fps,
            config.duration, config.bg_color, config.text_color, font_size, cancel_event,
        )

    return output_path


def _color_to_ass(color_name: str) -> str:
    color_map = {
        "white": "&H00FFFFFF", "black": "&H00000000",
        "yellow": "&H0000FFFF", "red": "&H000000FF",
        "cyan": "&H00FFFF00", "green": "&H0000FF00",
        "blue": "&H00FF0000",
    }
    name = color_name.strip().lower()
    if name in color_map:
        return color_map[name]
    if name.startswith("0x") and len(name) >= 8:
        return f"&H00{name[6:8]}{name[4:6]}{name[2:4]}"
    if name.startswith("#") and len(name) >= 7:
        return f"&H00{name[5:7]}{name[3:5]}{name[1:3]}"
    return "&H00FFFFFF"


def _generate_text_card(
    text: str,
    output: Path,
    width: int,
    height: int,
    fps: float,
    duration: float,
    bg_color: str,
    text_color: str,
    font_size: int,
    cancel_event: Event,
) -> None:
    display_text = text if text.strip() else "Sponsor"
    primary = _color_to_ass(text_color)

    ass_content = (
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
        f"Dialogue: 0,0:00:00.00,0:00:{duration:05.2f},Card,,0,0,0,,{display_text}\n"
    )

    ass_file = output.with_suffix(".ass")
    try:
        ass_file.write_text(ass_content, encoding="utf-8")
        command = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"color=c={bg_color}:s={width}x{height}:r={fps:.2f}:d={duration}",
            "-f", "lavfi",
            "-i", "anullsrc=r=44100:cl=stereo",
            "-vf", f"ass={ass_file}",
            "-t", str(duration),
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            str(output),
        ]
        _run_ffmpeg(command, cancel_event)
    finally:
        ass_file.unlink(missing_ok=True)


def _generate_image_card(
    image_path: Path,
    output: Path,
    width: int,
    height: int,
    fps: float,
    duration: float,
    cancel_event: Event,
) -> None:
    command = [
        "ffmpeg", "-y",
        "-loop", "1",
        "-i", str(image_path),
        "-f", "lavfi",
        "-i", "anullsrc=r=44100:cl=stereo",
        "-vf", (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,setsar=1"
        ),
        "-t", str(duration),
        "-r", f"{fps:.2f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        str(output),
    ]
    _run_ffmpeg(command, cancel_event)


def insert_cards(
    input_video: Path,
    output_video: Path,
    cards: list[SponsorCardConfig],
    work_dir: Path,
    cancel_event: Event,
) -> Path:
    """Insert sponsor cards at front/center/end of the video."""
    ensure_ffmpeg()

    if not cards:
        return input_video

    w, h, fps, total_duration = _get_video_info(input_video)
    resolution = (w, h)

    front_cards: list[Path] = []
    center_cards: list[Path] = []
    end_cards: list[Path] = []

    for i, config in enumerate(cards):
        if cancel_event.is_set():
            raise CancellationError("Processing cancelled by user")

        card_path = work_dir / f"sponsor_card_{i:03d}.mp4"
        generate_card(config, resolution, fps, card_path, cancel_event)

        if config.position == "front":
            front_cards.append(card_path)
        elif config.position == "center":
            center_cards.append(card_path)
        else:
            end_cards.append(card_path)

    current_video = input_video

    if center_cards and total_duration > 0:
        midpoint = total_duration / 2.0
        first_half = work_dir / "sponsor_first_half.mp4"
        second_half = work_dir / "sponsor_second_half.mp4"

        _run_ffmpeg([
            "ffmpeg", "-y", "-i", str(current_video),
            "-t", str(midpoint), "-c", "copy", str(first_half),
        ], cancel_event)

        _run_ffmpeg([
            "ffmpeg", "-y", "-i", str(current_video),
            "-ss", str(midpoint), "-c", "copy", str(second_half),
        ], cancel_event)

        segments = [first_half] + center_cards + [second_half]
        center_output = work_dir / "sponsor_center_merged.mp4"
        _concat_segments(segments, center_output, work_dir, cancel_event)
        current_video = center_output

    if front_cards or end_cards:
        all_segments = front_cards + [current_video] + end_cards
        _concat_segments(all_segments, output_video, work_dir, cancel_event)
    elif current_video != input_video:
        import shutil
        shutil.move(str(current_video), str(output_video))
    else:
        return input_video

    return output_video


def _concat_segments(
    segments: list[Path],
    output: Path,
    work_dir: Path,
    cancel_event: Event,
) -> None:
    concat_file = work_dir / "sponsor_concat.txt"
    try:
        concat_file.write_text(
            "\n".join(f"file '{seg}'" for seg in segments) + "\n",
            encoding="utf-8",
        )
        _run_ffmpeg([
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_file),
            "-c", "copy",
            str(output),
        ], cancel_event)
    finally:
        concat_file.unlink(missing_ok=True)
