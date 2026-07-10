from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from core.context import PipelineSettings
from core.session import settings_from_dict, settings_to_dict


DRAFT_QUEUE_FILENAME = "draft_queue.json"

STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_PAUSED = "paused"

ACTIVE_STATUSES = {STATUS_QUEUED, STATUS_RUNNING, STATUS_PAUSED}
RERUNNABLE_STATUSES = {STATUS_QUEUED, STATUS_FAILED, STATUS_PAUSED}


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _path_or_none(value: str | Path | None) -> Path | None:
    if not value:
        return None
    return Path(value)


@dataclass
class DraftJob:
    draft_id: str
    video_path: Path
    output_dir: Path
    settings: PipelineSettings
    source_url: str = ""
    status: str = STATUS_QUEUED
    output_path: Path | None = None
    error: str = ""
    session_path: Path | None = None
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    @property
    def video_name(self) -> str:
        return self.video_path.name

    def to_dict(self) -> dict:
        return {
            "draft_id": self.draft_id,
            "video_path": str(self.video_path),
            "output_dir": str(self.output_dir),
            "settings": settings_to_dict(self.settings),
            "source_url": self.source_url,
            "status": self.status,
            "output_path": str(self.output_path) if self.output_path else "",
            "error": self.error,
            "session_path": str(self.session_path) if self.session_path else "",
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> DraftJob:
        settings = settings_from_dict(data.get("settings", {}))
        video_path = Path(data.get("video_path") or settings.input_video)
        output_dir = Path(data.get("output_dir") or settings.output_dir)
        return cls(
            draft_id=data.get("draft_id") or uuid.uuid4().hex,
            video_path=video_path,
            output_dir=output_dir,
            settings=settings,
            source_url=data.get("source_url", ""),
            status=data.get("status") or STATUS_QUEUED,
            output_path=_path_or_none(data.get("output_path")),
            error=data.get("error", ""),
            session_path=_path_or_none(data.get("session_path")),
            created_at=data.get("created_at", _now()),
            updated_at=data.get("updated_at", _now()),
        )


class DraftQueue:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.jobs: list[DraftJob] = []

    @classmethod
    def for_project(cls, project_root: Path) -> DraftQueue:
        return cls(Path(project_root) / "temp" / DRAFT_QUEUE_FILENAME)

    @classmethod
    def load(cls, path: Path) -> DraftQueue:
        queue = cls(path)
        if not queue.path.exists():
            return queue
        data = json.loads(queue.path.read_text(encoding="utf-8"))
        jobs = data.get("jobs", data if isinstance(data, list) else [])
        queue.jobs = [DraftJob.from_dict(item) for item in jobs]
        return queue

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(".json.tmp")
        tmp_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "updated_at": _now(),
                    "jobs": [job.to_dict() for job in self.jobs],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        os.replace(tmp_path, self.path)

    def add_job(
        self,
        settings: PipelineSettings,
        video_path: Path | None = None,
        *,
        source_url: str = "",
    ) -> DraftJob:
        from dataclasses import replace

        resolved_video = Path(video_path or settings.input_video)
        snapshot = replace(
            settings,
            input_video=resolved_video,
            input_videos=[resolved_video],
        )
        job = DraftJob(
            draft_id=uuid.uuid4().hex,
            video_path=resolved_video,
            output_dir=snapshot.output_dir,
            settings=snapshot,
            source_url=source_url,
        )
        self.jobs.append(job)
        self.save()
        return job

    def add_jobs(
        self,
        settings: PipelineSettings,
        video_paths: list[Path],
        source_urls: dict[Path, str] | None = None,
    ) -> list[DraftJob]:
        source_urls = source_urls or {}
        jobs = [
            self.add_job(
                settings,
                video_path,
                source_url=source_urls.get(Path(video_path), ""),
            )
            for video_path in video_paths
        ]
        return jobs

    def get(self, draft_id: str) -> DraftJob | None:
        for job in self.jobs:
            if job.draft_id == draft_id:
                return job
        return None

    def pending_jobs(self) -> list[DraftJob]:
        return [job for job in self.jobs if job.status == STATUS_QUEUED]

    def next_queued(self) -> DraftJob | None:
        for job in self.jobs:
            if job.status == STATUS_QUEUED:
                return job
        return None

    def next_runnable(self) -> DraftJob | None:
        for job in self.jobs:
            if job.status in RERUNNABLE_STATUSES:
                return job
        return None

    def has_queued(self) -> bool:
        return self.next_queued() is not None

    def has_runnable(self) -> bool:
        return self.next_runnable() is not None

    def mark_queued(self, draft_id: str) -> None:
        self._update(draft_id, status=STATUS_QUEUED, error="")

    def mark_running(self, draft_id: str, session_path: Path) -> None:
        self._update(
            draft_id,
            status=STATUS_RUNNING,
            session_path=session_path,
            output_path=None,
            error="",
        )

    def mark_completed(self, draft_id: str, output_path: Path) -> None:
        self._update(draft_id, status=STATUS_COMPLETED, output_path=output_path, error="")

    def mark_failed(self, draft_id: str, error: str) -> None:
        self._update(draft_id, status=STATUS_FAILED, error=error)

    def mark_paused(self, draft_id: str) -> None:
        self._update(draft_id, status=STATUS_PAUSED)

    def remove(self, draft_id: str) -> bool:
        before = len(self.jobs)
        self.jobs = [job for job in self.jobs if job.draft_id != draft_id]
        changed = len(self.jobs) != before
        if changed:
            self.save()
        return changed

    def move(self, draft_id: str, offset: int) -> bool:
        index = next((idx for idx, job in enumerate(self.jobs) if job.draft_id == draft_id), -1)
        if index < 0:
            return False
        new_index = max(0, min(len(self.jobs) - 1, index + offset))
        if new_index == index:
            return False
        job = self.jobs.pop(index)
        self.jobs.insert(new_index, job)
        self.save()
        return True

    def reset_running_to_paused(self) -> None:
        changed = False
        for job in self.jobs:
            if job.status == STATUS_RUNNING:
                job.status = STATUS_PAUSED
                job.updated_at = _now()
                changed = True
        if changed:
            self.save()

    def _update(self, draft_id: str, **changes) -> None:
        job = self.get(draft_id)
        if job is None:
            raise KeyError(f"Unknown draft id: {draft_id}")
        for key, value in changes.items():
            setattr(job, key, value)
        job.updated_at = _now()
        self.save()
