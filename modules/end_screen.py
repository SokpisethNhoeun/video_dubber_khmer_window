from __future__ import annotations

import subprocess
import time
from pathlib import Path
from threading import Event

from core.context import CancellationError
from modules.audio_utils import ensure_ffmpeg


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
            parts = result.stdout.strip().split("x")
            if len(parts) == 2:
                return int(parts[0]), int(parts[1])
    except Exception:
        pass
    return 1920, 1080


def _get_video_fps(video_path: Path) -> float:
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split("/")
            if len(parts) == 2:
                return float(parts[0]) / float(parts[1])
    except Exception:
        pass
    return 30.0


def append_end_screen(
    input_video: Path,
    output_video: Path,
    cancel_event: Event,
    text: str = "",
    image_path: Path | None = None,
    duration: float = 3.0,
    bg_color: str = "black",
    font_size: int = 48,
) -> Path:
    """Append an end screen card to the video.

    Either shows centered text on a solid background, or an image scaled to fit.
    Uses a two-step approach: generate card clip, then concat with the main video.
    """
    ensure_ffmpeg()

    if cancel_event.is_set():
        raise CancellationError("Processing cancelled by user")

    width, height = _get_video_resolution(input_video)
    fps = _get_video_fps(input_video)
    output_video.parent.mkdir(parents=True, exist_ok=True)

    card_video = output_video.with_stem(output_video.stem + "_card_tmp")

    try:
        if image_path and image_path.exists():
            _generate_image_card(image_path, card_video, width, height, fps, duration, cancel_event)
        elif text.strip():
            _generate_text_card(text.strip(), card_video, width, height, fps, duration, bg_color, font_size, cancel_event)
        else:
            return input_video

        _concat_videos(input_video, card_video, output_video, cancel_event)
    finally:
        if card_video.exists():
            card_video.unlink(missing_ok=True)

    return output_video


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


_ASS_FONT_NAME = "Noto Sans"


def _generate_text_card(
    text: str,
    output: Path,
    width: int,
    height: int,
    fps: float,
    duration: float,
    bg_color: str,
    font_size: int,
    cancel_event: Event,
) -> None:
    font_color = "&H00FFFFFF" if bg_color == "black" else "&H00000000"

    ass_content = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {width}\n"
        f"PlayResY: {height}\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Card,{_ASS_FONT_NAME},{font_size},{font_color},{font_color},&H00000000,&H00000000,"
        f"1,0,0,0,100,100,0,0,1,2,1,5,20,20,20,1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        f"Dialogue: 0,0:00:00.00,0:00:{duration:05.2f},Card,,0,0,0,,{text}\n"
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
        "-i", f"anullsrc=r=44100:cl=stereo",
        "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
               f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,setsar=1",
        "-t", str(duration),
        "-r", f"{fps:.2f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        str(output),
    ]
    _run_ffmpeg(command, cancel_event)


def _concat_videos(
    main_video: Path,
    card_video: Path,
    output: Path,
    cancel_event: Event,
) -> None:
    concat_file = output.with_suffix(".txt")
    try:
        concat_file.write_text(
            f"file '{main_video}'\nfile '{card_video}'\n",
            encoding="utf-8",
        )
        command = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_file),
            "-c", "copy",
            str(output),
        ]
        _run_ffmpeg(command, cancel_event)
    finally:
        concat_file.unlink(missing_ok=True)
