from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from typing import Callable

from core.quality_report import QualityReport


ProgressCallback = Callable[[str, int], None]
LogCallback = Callable[[str], None]


class CancellationError(RuntimeError):
    """Raised when the pipeline process is cancelled by the user."""
    pass


@dataclass
class Segment:
    index: int
    start: float
    end: float
    text: str
    translated_text: str = ""
    raw_khmer_text: str = ""
    improved_khmer_text: str = ""
    user_edited_text: str = ""
    enabled: bool = True
    review_notes: str = ""
    tts_path: Path | None = None
    cloned_path: Path | None = None
    tts_group_id: str = ""
    speaker_id: str | None = None
    speaker_label: str | None = None

    @property
    def duration(self) -> float:
        if self.end < self.start:
            raise ValueError(f"Malformed segment {self.index}: end ({self.end}) is before start ({self.start})")
        return max(0.01, self.end - self.start)

    @property
    def speech_path(self) -> Path:
        if self.cloned_path is not None:
            return self.cloned_path
        if self.tts_path is None:
            raise ValueError(f"Segment {self.index} has no synthesized speech path")
        return self.tts_path

    @property
    def tts_text(self) -> str:
        return (
            self.user_edited_text.strip()
            or self.improved_khmer_text.strip()
            or self.translated_text.strip()
            or self.raw_khmer_text.strip()
        )


@dataclass
class PipelineSettings:
    input_video: Path
    output_dir: Path
    source_language: str
    voice_gender: str
    voice_female: str
    voice_male: str
    speech_rate: int
    pitch_hz: int
    whisper_model: str
    device: str

    def __post_init__(self) -> None:
        from config.device import resolve_compute_device
        self.device, _ = resolve_compute_device(self.device)

    tts_provider: str = "edge"
    voice_female_reference_path: Path | None = None
    voice_male_reference_path: Path | None = None
    keep_temp: bool = False
    rvc_enabled: bool = False
    rvc_model_path: Path | None = None
    rvc_index_path: Path | None = None
    rvc_reference_audio_path: Path | None = None
    rvc_clone_gender: str = "all"
    rvc_command_template: str = ""
    clone_workflow: str = "auto_per_person"
    input_videos: list[Path] = field(default_factory=list)
    alignment_mode: str = "natural"
    speaker_voice_mappings: dict[str, dict[str, dict[str, str]]] = field(default_factory=dict)
    diarization_turns: dict[str, list[dict[str, float | str]]] = field(default_factory=dict)
    enable_audio_cleanup: bool = True
    enable_final_mastering: bool = True
    enable_persistent_cache: bool = True
    enable_clone_verification: bool = True
    enable_bgm_ducking: bool = True
    duck_depth_db: float = 8.0
    publish_target: str = "youtube"
    custom_lufs: float = -14.0
    enable_per_speaker_prosody: bool = True
    min_reference_seconds: float = 15.0
    auto_speaker_references: bool = False
    preserve_bgm: bool = True
    preset: str = "balanced"
    transcript_review_mode: str = "local"
    khmer_style: str = "simple"
    content_style: str = "casual_vlog"
    glossary_path: Path | None = None
    review_json_path: Path | None = None
    save_review_json: bool = False
    export_dubbed_audio: bool = True
    export_original_transcript: bool = True
    export_raw_khmer: bool = True
    export_improved_khmer: bool = True
    export_subtitles: bool = True
    export_quality_report: bool = True
    voice_volume: float = 1.0
    bgm_volume: float = 0.85
    generate_script_only: bool = False
    burn_subtitles: bool = False
    subtitle_language: str = "khmer"
    subtitle_font_size: int = 24
    overlay_text: str = ""
    overlay_image_path: Path | None = None
    overlay_position: str = "bottom_right"
    overlay_text_position: str = "bottom_right"
    overlay_image_position: str = "bottom_right"
    overlay_opacity: float = 0.7
    clone_backend: str = "openvoice"
    emotion_aware_clone: bool = True
    emotion_clone_mode: str = "auto"
    translation_backend: str = "ai"
    ai_skip_review: bool = True
    narration_style: str = "natural"
    end_screen_enabled: bool = False
    end_screen_text: str = ""
    end_screen_image_path: Path | None = None
    end_screen_duration: float = 3.0
    end_screen_bg_color: str = "black"
    sponsor_cards: list[dict] = field(default_factory=list)
    footer_overlay_enabled: bool = False
    footer_overlay_config: dict = field(default_factory=dict)


@dataclass
class PipelineContext:
    settings: PipelineSettings
    work_dir: Path
    cancel_event: Event = field(default_factory=Event)
    progress: ProgressCallback | None = None
    log: LogCallback | None = None
    quality_report: QualityReport = field(default_factory=QualityReport)

    def emit_log(self, message: str) -> None:
        if self.log:
            self.log(message)

    def emit_progress(self, stage: str, value: int) -> None:
        if self.progress:
            self.progress(stage, max(0, min(100, int(value))))

    def check_cancelled(self) -> None:
        if self.cancel_event.is_set():
            raise CancellationError("Processing cancelled by user")
