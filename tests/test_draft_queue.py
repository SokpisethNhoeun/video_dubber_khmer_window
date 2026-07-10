from __future__ import annotations

from core.draft_queue import (
    DraftQueue,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_PAUSED,
    STATUS_QUEUED,
    STATUS_RUNNING,
)
from conftest import make_pipeline_settings


def test_draft_queue_saves_and_loads_jobs(tmp_path):
    video_a = tmp_path / "a.mp4"
    video_b = tmp_path / "b.mp4"
    video_a.write_bytes(b"a")
    video_b.write_bytes(b"b")
    settings = make_pipeline_settings(
        tmp_path,
        input_video=video_a,
        input_videos=[video_a, video_b],
    )
    queue = DraftQueue(tmp_path / "draft_queue.json")

    jobs = queue.add_jobs(
        settings,
        [video_a, video_b],
        {video_a: "https://www.rednote.com/explore/abc123"},
    )
    queue.mark_running(jobs[0].draft_id, tmp_path / "job_a" / "session.json")
    queue.mark_completed(jobs[0].draft_id, tmp_path / "out" / "a.mp4")
    queue.mark_failed(jobs[1].draft_id, "failed")

    loaded = DraftQueue.load(queue.path)

    assert [job.video_path for job in loaded.jobs] == [video_a, video_b]
    assert loaded.jobs[0].source_url == "https://www.rednote.com/explore/abc123"
    assert loaded.jobs[1].source_url == ""
    assert loaded.jobs[0].status == STATUS_COMPLETED
    assert loaded.jobs[0].output_path == tmp_path / "out" / "a.mp4"
    assert loaded.jobs[0].session_path == tmp_path / "job_a" / "session.json"
    assert loaded.jobs[1].status == STATUS_FAILED
    assert loaded.jobs[1].error == "failed"
    assert loaded.jobs[1].settings.input_videos == [video_b]


def test_draft_queue_reorders_and_resets_running_jobs(tmp_path):
    video_a = tmp_path / "a.mp4"
    video_b = tmp_path / "b.mp4"
    video_a.write_bytes(b"a")
    video_b.write_bytes(b"b")
    settings = make_pipeline_settings(tmp_path, input_video=video_a)
    queue = DraftQueue(tmp_path / "draft_queue.json")
    first = queue.add_job(settings, video_a)
    second = queue.add_job(settings, video_b)

    assert queue.move(second.draft_id, -1) is True
    assert [job.draft_id for job in queue.jobs] == [second.draft_id, first.draft_id]

    queue.mark_running(second.draft_id, tmp_path / "job_b" / "session.json")
    queue.reset_running_to_paused()
    loaded = DraftQueue.load(queue.path)

    assert loaded.get(second.draft_id).status == STATUS_PAUSED
    assert loaded.get(first.draft_id).status == STATUS_QUEUED
