from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from core.context import PipelineSettings, Segment
from modules.audio_utils import remove_tree

SCHEMA_VERSION = 1
SESSION_FILENAME = "session.json"

STATUS_RUNNING = "running"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"
STATUS_COMPLETED = "completed"

_SETTINGS_PATH_FIELDS = {
    "input_video",
    "output_dir",
    "voice_female_reference_path",
    "voice_male_reference_path",
    "rvc_model_path",
    "rvc_index_path",
    "rvc_reference_audio_path",
    "glossary_path",
    "review_json_path",
    "overlay_image_path",
}
_SETTINGS_PATH_LIST_FIELDS = {"input_videos"}

_SEGMENT_PATH_FIELDS = {"tts_path", "cloned_path"}


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _encode_path(path: Path | None, work_dir: Path) -> str | None:
    if path is None:
        return None
    path = Path(path)
    try:
        return str(path.relative_to(work_dir))
    except ValueError:
        return str(path)


def _decode_path(value: str | None, work_dir: Path) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = work_dir / path
    return path


def settings_to_dict(settings: PipelineSettings) -> dict:
    data = asdict(settings)
    for key in _SETTINGS_PATH_FIELDS:
        if data.get(key) is not None:
            data[key] = str(data[key])
    for key in _SETTINGS_PATH_LIST_FIELDS:
        data[key] = [str(p) for p in data.get(key) or []]
    return data


def settings_from_dict(data: dict) -> PipelineSettings:
    kwargs = dict(data)
    legacy_overlay_position = kwargs.get("overlay_position", "bottom_right")
    kwargs.setdefault("overlay_text_position", legacy_overlay_position)
    kwargs.setdefault("overlay_image_position", legacy_overlay_position)
    for key in _SETTINGS_PATH_FIELDS:
        if kwargs.get(key) is not None:
            kwargs[key] = Path(kwargs[key])
    for key in _SETTINGS_PATH_LIST_FIELDS:
        kwargs[key] = [Path(p) for p in kwargs.get(key) or []]
    valid = {f for f in PipelineSettings.__dataclass_fields__}
    kwargs = {k: v for k, v in kwargs.items() if k in valid}
    return PipelineSettings(**kwargs)


def segment_to_dict(segment: Segment, work_dir: Path) -> dict:
    data = asdict(segment)
    for key in _SEGMENT_PATH_FIELDS:
        data[key] = _encode_path(getattr(segment, key), work_dir)
    return data


def segment_from_dict(data: dict, work_dir: Path) -> Segment:
    kwargs = dict(data)
    for key in _SEGMENT_PATH_FIELDS:
        kwargs[key] = _decode_path(kwargs.get(key), work_dir)
    valid = {f for f in Segment.__dataclass_fields__}
    kwargs = {k: v for k, v in kwargs.items() if k in valid}
    return Segment(**kwargs)


@dataclass
class DubbingSession:
    work_dir: Path
    settings: PipelineSettings
    session_id: str = ""
    status: str = STATUS_RUNNING
    failed_stage: str = ""
    error: str = ""
    completed_stages: list[str] = field(default_factory=list)
    artifacts: dict[str, Path] = field(default_factory=dict)
    duration: float = 0.0
    segment_genders: dict[int, str] | None = None
    speaker_mappings: dict = field(default_factory=dict)
    segments: list[Segment] = field(default_factory=list)
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    def __post_init__(self) -> None:
        self.work_dir = Path(self.work_dir)
        if not self.session_id:
            self.session_id = self.work_dir.name

    @property
    def path(self) -> Path:
        return self.work_dir / SESSION_FILENAME

    @property
    def video_name(self) -> str:
        return Path(self.settings.input_video).name

    def set_artifact(self, key: str, path: Path | None) -> None:
        if path is None:
            self.artifacts.pop(key, None)
        else:
            self.artifacts[key] = Path(path)

    def get_artifact(self, key: str) -> Path | None:
        return self.artifacts.get(key)

    def is_complete(self, stage: str) -> bool:
        return stage in self.completed_stages

    def mark_stage_complete(self, stage: str) -> None:
        if stage not in self.completed_stages:
            self.completed_stages.append(stage)

    def mark_failed(self, stage: str, error: str) -> None:
        self.status = STATUS_FAILED
        self.failed_stage = stage
        self.error = error

    def mark_cancelled(self, stage: str) -> None:
        self.status = STATUS_CANCELLED
        self.failed_stage = stage

    def mark_completed(self) -> None:
        self.status = STATUS_COMPLETED
        self.failed_stage = ""
        self.error = ""

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status,
            "failed_stage": self.failed_stage,
            "error": self.error,
            "completed_stages": list(self.completed_stages),
            "work_dir": str(self.work_dir),
            "settings": settings_to_dict(self.settings),
            "artifacts": {
                key: _encode_path(path, self.work_dir)
                for key, path in self.artifacts.items()
            },
            "duration": self.duration,
            "segment_genders": (
                {str(k): v for k, v in self.segment_genders.items()}
                if self.segment_genders is not None
                else None
            ),
            "speaker_mappings": self.speaker_mappings,
            "segments": [segment_to_dict(seg, self.work_dir) for seg in self.segments],
        }

    def save(self) -> None:
        self.updated_at = _now()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(".json.tmp")
        tmp_path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(tmp_path, self.path)

    @classmethod
    def load(cls, path: Path) -> DubbingSession:
        path = Path(path)
        if path.is_dir():
            path = path / SESSION_FILENAME
        data = json.loads(path.read_text(encoding="utf-8"))
        work_dir = path.parent
        genders = data.get("segment_genders")
        return cls(
            work_dir=work_dir,
            settings=settings_from_dict(data.get("settings", {})),
            session_id=data.get("session_id", work_dir.name),
            status=data.get("status", STATUS_RUNNING),
            failed_stage=data.get("failed_stage", ""),
            error=data.get("error", ""),
            completed_stages=list(data.get("completed_stages", [])),
            artifacts={
                key: _decode_path(value, work_dir)
                for key, value in (data.get("artifacts") or {}).items()
                if value
            },
            duration=float(data.get("duration", 0.0)),
            segment_genders=(
                {int(k): v for k, v in genders.items()} if genders is not None else None
            ),
            speaker_mappings=data.get("speaker_mappings", {}),
            segments=[
                segment_from_dict(seg, work_dir)
                for seg in data.get("segments", [])
            ],
            created_at=data.get("created_at", _now()),
            updated_at=data.get("updated_at", _now()),
        )


@dataclass
class SessionSummary:
    work_dir: Path
    session_id: str
    video_name: str
    status: str
    failed_stage: str
    segment_count: int
    completed_stages: list[str]
    updated_at: str


def list_sessions(temp_dir: Path) -> list[SessionSummary]:
    summaries: list[SessionSummary] = []
    temp_dir = Path(temp_dir)
    if not temp_dir.exists():
        return summaries
    for session_file in sorted(temp_dir.glob("job_*/" + SESSION_FILENAME)):
        try:
            data = json.loads(session_file.read_text(encoding="utf-8"))
            settings = data.get("settings") or {}
            summaries.append(
                SessionSummary(
                    work_dir=session_file.parent,
                    session_id=data.get("session_id", session_file.parent.name),
                    video_name=Path(settings.get("input_video", "?")).name,
                    status=data.get("status", "unknown"),
                    failed_stage=data.get("failed_stage", ""),
                    segment_count=len(data.get("segments") or []),
                    completed_stages=list(data.get("completed_stages", [])),
                    updated_at=data.get("updated_at", ""),
                )
            )
        except Exception:
            continue
    summaries.sort(key=lambda s: s.updated_at, reverse=True)
    return summaries


def delete_session(work_dir: Path) -> None:
    remove_tree(Path(work_dir))


def prune_sessions(temp_dir: Path, keep: int = 5) -> int:
    """Delete the oldest completed sessions beyond ``keep``. Failed/cancelled
    sessions are kept so they stay resumable. Returns count deleted."""
    completed = [s for s in list_sessions(temp_dir) if s.status == STATUS_COMPLETED]
    removed = 0
    for summary in completed[keep:]:
        delete_session(summary.work_dir)
        removed += 1
    return removed
