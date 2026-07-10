from __future__ import annotations

import shutil
from pathlib import Path

from core.context import Segment


def _stamp(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    hours = total_ms // 3_600_000
    total_ms %= 3_600_000
    minutes = total_ms // 60_000
    total_ms %= 60_000
    secs = total_ms // 1000
    millis = total_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _line_stamp(segment: Segment) -> str:
    return f"[{segment.start:08.2f} -> {segment.end:08.2f}]"


def _write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def export_transcript_text(path: Path, segments: list[Segment], field: str) -> None:
    lines: list[str] = []
    for segment in segments:
        if field == "source":
            text = segment.text
        elif field == "raw_khmer":
            text = segment.raw_khmer_text or segment.translated_text
        elif field == "improved_khmer":
            text = segment.tts_text
        else:
            raise ValueError(f"Unsupported transcript export field: {field}")
        speaker = segment.speaker_label or segment.speaker_id or ""
        prefix = f"{_line_stamp(segment)}"
        if speaker:
            prefix += f" {speaker}:"
        lines.append(f"{prefix} {text}".rstrip())
    _write_lines(path, lines)


def _srt_text_for_language(segment: Segment, language: str) -> str:
    if language == "chinese":
        return segment.text
    if language == "english":
        return segment.translated_text or segment.text
    return segment.tts_text


def export_srt(path: Path, segments: list[Segment], language: str = "khmer") -> None:
    lines: list[str] = []
    subtitle_index = 1
    for segment in segments:
        if not segment.enabled:
            continue
        text = _srt_text_for_language(segment, language)
        if not text:
            continue
        lines.extend(
            [
                str(subtitle_index),
                f"{_stamp(segment.start)} --> {_stamp(segment.end)}",
                text,
                "",
            ]
        )
        subtitle_index += 1
    _write_lines(path, lines)


def copy_export(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def export_pipeline_outputs(
    output_dir: Path,
    stem: str,
    segments: list[Segment],
    final_audio: Path,
    quality_report_dir: Path,
    export_dubbed_audio: bool,
    export_original_transcript: bool,
    export_raw_khmer: bool,
    export_improved_khmer: bool,
    export_subtitles: bool,
    export_quality_report: bool,
) -> list[Path]:
    written: list[Path] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    if export_dubbed_audio and final_audio.exists():
        path = output_dir / f"{stem}_khmer_dubbed.wav"
        copy_export(final_audio, path)
        written.append(path)
    if export_original_transcript:
        path = output_dir / f"{stem}_original_transcript.txt"
        export_transcript_text(path, segments, "source")
        written.append(path)
    if export_raw_khmer:
        path = output_dir / f"{stem}_raw_khmer.txt"
        export_transcript_text(path, segments, "raw_khmer")
        written.append(path)
    if export_improved_khmer:
        path = output_dir / f"{stem}_improved_khmer.txt"
        export_transcript_text(path, segments, "improved_khmer")
        written.append(path)
    if export_subtitles:
        path = output_dir / f"{stem}_khmer_subtitles.srt"
        export_srt(path, segments)
        written.append(path)
    if export_quality_report:
        for name in ("quality_report.json", "quality_report.txt"):
            source = quality_report_dir / name
            if source.exists():
                path = output_dir / f"{stem}_{name}"
                copy_export(source, path)
                written.append(path)
    return written
