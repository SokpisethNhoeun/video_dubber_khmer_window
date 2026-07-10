from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class QualityReport:
    segment_count: int = 0
    speaker_count: int = 0
    missing_references: list[str] = field(default_factory=list)
    bad_references: list[dict[str, str]] = field(default_factory=list)
    long_segments: list[dict[str, float | int | str]] = field(default_factory=list)
    timing_segments: list[dict[str, float | int | str | bool]] = field(default_factory=list)
    voice_clone_failures: list[dict[str, str | int]] = field(default_factory=list)
    cache_hits: dict[str, int] = field(default_factory=lambda: {"tts": 0, "references": 0, "diarization": 0})
    final_output_path: str = ""
    speaker_quality: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "segment_count": self.segment_count,
            "speaker_count": self.speaker_count,
            "missing_references": self.missing_references,
            "bad_or_short_references": self.bad_references,
            "long_running_segments_after_speed_cap": self.long_segments,
            "timing_segments": self.timing_segments,
            "voice_clone_failures": self.voice_clone_failures,
            "cache_hits": self.cache_hits,
            "final_output_path": self.final_output_path,
            "speaker_quality": self.speaker_quality,
        }

    def write(self, work_dir: Path) -> None:
        payload = self.to_dict()
        (work_dir / "quality_report.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        weak_or_bad = sum(1 for item in self.speaker_quality if item.get("tier") in {"weak", "bad"})
        lines = [
            "Khmer Video Dubber Quality Report",
            "",
            f"Segments: {self.segment_count}",
            f"Speakers: {self.speaker_count}",
            f"Missing references: {len(self.missing_references)}",
            f"Bad/short/noisy references: {len(self.bad_references)}",
            f"Speakers with weak/bad reference quality: {weak_or_bad}",
            f"Long segments after speed cap: {len(self.long_segments)}",
            f"Voice clone failures: {len(self.voice_clone_failures)}",
            (
                "Cache hits: "
                f"TTS {self.cache_hits.get('tts', 0)}, "
                f"references {self.cache_hits.get('references', 0)}, "
                f"diarization {self.cache_hits.get('diarization', 0)}"
            ),
            f"Final output: {self.final_output_path or 'not completed'}",
        ]
        if self.missing_references:
            lines.extend(["", "Missing References:"])
            lines.extend(f"- {item}" for item in self.missing_references)
        if self.bad_references:
            lines.extend(["", "Reference Warnings:"])
            lines.extend(
                f"- {item.get('speaker', 'reference')}: {item.get('status', '')} ({item.get('path', '')})"
                for item in self.bad_references
            )
        if self.long_segments:
            lines.extend(["", "Long Segments:"])
            lines.extend(
                f"- Segment {item.get('index')}: {item.get('adjusted_duration')}s audio after speed-up for "
                f"{item.get('target_duration')}s slot at {item.get('speed')}x"
                f"{'; trimmed ' + str(item.get('trim_duration')) + 's' if item.get('trim_required') else ''}"
                for item in self.long_segments
            )
        if self.voice_clone_failures:
            lines.extend(["", "Voice Clone Failures:"])
            lines.extend(
                f"- Segment {item.get('segment')}: {item.get('message')}"
                for item in self.voice_clone_failures
            )
        if self.speaker_quality:
            lines.extend(["", "Speaker Reference Quality:"])
            for item in self.speaker_quality:
                reasons = item.get("reasons") or []
                reasons_text = f" — {'; '.join(reasons)}" if reasons else ""
                lines.append(
                    f"- {item.get('speaker', '?')} [{item.get('tier', '?').upper()}]: "
                    f"score {item.get('score', 0):.0f}/100, SNR {item.get('snr_db', 0):.1f} dB, "
                    f"voiced {float(item.get('voiced_ratio', 0.0)) * 100:.0f}%{reasons_text}"
                )
        (work_dir / "quality_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def summary(self) -> str:
        return (
            "Quality summary: "
            f"{self.segment_count} segments, {self.speaker_count} speaker(s), "
            f"{len(self.missing_references)} missing reference(s), "
            f"{len(self.bad_references)} reference warning(s), "
            f"{len(self.long_segments)} long segment warning(s), "
            f"{len(self.voice_clone_failures)} voice clone failure(s)."
        )
