from __future__ import annotations

import json
from pathlib import Path

from conftest import make_pipeline_settings
from core.context import CancellationError
from core.draft_queue import DraftQueue, STATUS_COMPLETED, STATUS_FAILED, STATUS_PAUSED, STATUS_QUEUED


def test_pipeline_worker_processes_all_selected_videos(tmp_path, monkeypatch):
    from gui import workers

    video_a = tmp_path / "a.mp4"
    video_b = tmp_path / "b.mp4"
    video_a.write_bytes(b"a")
    video_b.write_bytes(b"b")
    settings = make_pipeline_settings(
        tmp_path,
        input_video=video_a,
        input_videos=[video_a, video_b],
    )
    processed: list[tuple[Path, list[Path]]] = []
    session_remainders: list[list[Path]] = []

    class FakePipeline:
        def __init__(self, context, session):
            self.context = context
            self.session = session

        def run(self):
            current = self.context.settings.input_video
            processed.append(
                (current, list(self.context.settings.input_videos))
            )
            session_remainders.append(list(self.session.settings.input_videos))
            output = self.context.settings.output_dir / f"{current.stem}_khmer_dubbed.mp4"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"mp4")
            return output

        def cancel(self):
            self.context.cancel_event.set()

    monkeypatch.setattr(workers, "DubbingPipeline", FakePipeline)
    worker = workers.PipelineWorker(settings, tmp_path)
    finished: list[str] = []
    failed: list[str] = []
    worker.signals.finished.connect(finished.append)
    worker.signals.failed.connect(failed.append)

    worker.run()

    assert failed == []
    assert [item[0] for item in processed] == [video_a, video_b]
    assert [item[1] for item in processed] == [[video_a], [video_b]]
    assert session_remainders == [[video_a, video_b], [video_b]]
    assert finished == [str(settings.output_dir / "b_khmer_dubbed.mp4")]


def test_pipeline_worker_continues_batch_after_one_video_fails(tmp_path, monkeypatch):
    from gui import workers

    video_a = tmp_path / "a.mp4"
    video_b = tmp_path / "b.mp4"
    video_a.write_bytes(b"a")
    video_b.write_bytes(b"b")
    settings = make_pipeline_settings(
        tmp_path,
        input_video=video_a,
        input_videos=[video_a, video_b],
    )
    processed: list[Path] = []

    class FakePipeline:
        def __init__(self, context, session):
            self.context = context
            self.session = session

        def run(self):
            current = self.context.settings.input_video
            processed.append(current)
            if current == video_a:
                raise RuntimeError("first failed")
            output = self.context.settings.output_dir / f"{current.stem}_khmer_dubbed.mp4"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"mp4")
            return output

        def cancel(self):
            self.context.cancel_event.set()

    monkeypatch.setattr(workers, "DubbingPipeline", FakePipeline)
    worker = workers.PipelineWorker(settings, tmp_path)
    logs: list[str] = []
    finished: list[str] = []
    failed: list[str] = []
    worker.signals.log.connect(logs.append)
    worker.signals.finished.connect(finished.append)
    worker.signals.failed.connect(failed.append)

    worker.run()

    assert processed == [video_a, video_b]
    assert failed == []
    assert finished == [str(settings.output_dir / "b_khmer_dubbed.mp4")]
    assert any("Failed processing a.mp4" in message for message in logs)
    assert any("Continuing with next selected video" in message for message in logs)


def test_pipeline_worker_resume_session_continues_remaining_batch(tmp_path, monkeypatch):
    from core.session import DubbingSession
    from gui import workers

    video_a = tmp_path / "a.mp4"
    video_b = tmp_path / "b.mp4"
    video_c = tmp_path / "c.mp4"
    for video in [video_a, video_b, video_c]:
        video.write_bytes(video.stem.encode())
    settings = make_pipeline_settings(
        tmp_path,
        input_video=video_b,
        input_videos=[video_b, video_c],
    )
    session = DubbingSession(work_dir=tmp_path / "temp" / "job_resume", settings=settings)
    processed: list[Path] = []

    class FakePipeline:
        def __init__(self, context, session):
            self.context = context
            self.session = session

        def run(self):
            current = self.context.settings.input_video
            processed.append(current)
            output = self.context.settings.output_dir / f"{current.stem}_khmer_dubbed.mp4"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"mp4")
            return output

        def cancel(self):
            self.context.cancel_event.set()

    monkeypatch.setattr(workers, "DubbingPipeline", FakePipeline)
    worker = workers.PipelineWorker(settings, tmp_path, resume_session=session)
    finished: list[str] = []
    failed: list[str] = []
    logs: list[str] = []
    worker.signals.finished.connect(finished.append)
    worker.signals.failed.connect(failed.append)
    worker.signals.log.connect(logs.append)

    worker.run()

    assert failed == []
    assert processed == [video_b, video_c]
    assert finished == [str(settings.output_dir / "c_khmer_dubbed.mp4")]
    assert any("Continuing resumed batch" in message for message in logs)


def test_pipeline_worker_logs_source_url_for_imported_video(tmp_path, monkeypatch):
    from gui import workers

    cache_dir = tmp_path / "cache" / "imports" / "xiaohongshu" / "abc123"
    cache_dir.mkdir(parents=True)
    video_a = cache_dir / "source.mp4"
    video_a.write_bytes(b"a")
    source_url = "https://www.rednote.com/explore/abc123"
    (cache_dir / "import.json").write_text(
        json.dumps({"source_url": source_url}),
        encoding="utf-8",
    )
    settings = make_pipeline_settings(tmp_path, input_video=video_a, input_videos=[video_a])

    class FakePipeline:
        def __init__(self, context, session):
            self.context = context
            self.session = session

        def run(self):
            output = self.context.settings.output_dir / "source_khmer_dubbed.mp4"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"mp4")
            return output

        def cancel(self):
            self.context.cancel_event.set()

    monkeypatch.setattr(workers, "DubbingPipeline", FakePipeline)
    worker = workers.PipelineWorker(settings, tmp_path)
    logs: list[str] = []
    worker.signals.log.connect(logs.append)

    worker.run()

    assert f"Source: {source_url}" in logs
    assert any("Removed imported source cache" in message for message in logs)
    assert not cache_dir.exists()


def test_pipeline_worker_processes_draft_queue_and_continues_after_failure(tmp_path, monkeypatch):
    from gui import workers

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
    queue.add_jobs(
        settings,
        [video_a, video_b],
        {video_b: "https://www.rednote.com/explore/b"},
    )
    processed: list[Path] = []

    class FakePipeline:
        def __init__(self, context, session):
            self.context = context
            self.session = session

        def run(self):
            current = self.context.settings.input_video
            processed.append(current)
            if current == video_a:
                raise RuntimeError("first failed")
            output = self.context.settings.output_dir / f"{current.stem}_khmer_dubbed.mp4"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"mp4")
            return output

        def cancel(self):
            self.context.cancel_event.set()

    monkeypatch.setattr(workers, "DubbingPipeline", FakePipeline)
    worker = workers.PipelineWorker(settings, tmp_path, draft_queue_path=queue.path)
    finished: list[str] = []
    failed: list[str] = []
    worker.signals.finished.connect(finished.append)
    worker.signals.failed.connect(failed.append)

    worker.run()
    loaded = DraftQueue.load(queue.path)

    assert processed == [video_a, video_b]
    assert failed == []
    assert finished == [str(settings.output_dir / "b_khmer_dubbed.mp4")]
    assert [job.status for job in loaded.jobs] == [STATUS_FAILED, STATUS_COMPLETED]
    assert loaded.jobs[0].error == "first failed"
    assert loaded.jobs[1].session_path is not None
    assert loaded.jobs[1].source_url == "https://www.rednote.com/explore/b"


def test_pipeline_worker_logs_source_url_when_imported_draft_completes(tmp_path, monkeypatch):
    from gui import workers

    video_a = tmp_path / "a.mp4"
    video_a.write_bytes(b"a")
    source_url = "https://www.rednote.com/explore/abc123"
    settings = make_pipeline_settings(tmp_path, input_video=video_a, input_videos=[video_a])
    queue = DraftQueue(tmp_path / "draft_queue.json")
    queue.add_jobs(settings, [video_a], {video_a: source_url})

    class FakePipeline:
        def __init__(self, context, session):
            self.context = context
            self.session = session

        def run(self):
            output = self.context.settings.output_dir / "a_khmer_dubbed.mp4"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"mp4")
            return output

        def cancel(self):
            self.context.cancel_event.set()

    monkeypatch.setattr(workers, "DubbingPipeline", FakePipeline)
    worker = workers.PipelineWorker(settings, tmp_path, draft_queue_path=queue.path)
    logs: list[str] = []
    worker.signals.log.connect(logs.append)

    worker.run()

    assert f"Source: {source_url}" in logs


def test_pipeline_worker_marks_current_draft_paused_on_cancel(tmp_path, monkeypatch):
    from gui import workers

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
    queue.add_jobs(settings, [video_a, video_b])

    class FakePipeline:
        def __init__(self, context, session):
            self.context = context
            self.session = session

        def run(self):
            raise CancellationError("cancelled")

        def cancel(self):
            self.context.cancel_event.set()

    monkeypatch.setattr(workers, "DubbingPipeline", FakePipeline)
    worker = workers.PipelineWorker(settings, tmp_path, draft_queue_path=queue.path)
    failed: list[str] = []
    worker.signals.failed.connect(failed.append)

    worker.run()
    loaded = DraftQueue.load(queue.path)

    assert failed == ["Processing paused — current draft saved"]
    assert loaded.jobs[0].status == STATUS_PAUSED
    assert loaded.jobs[0].session_path is not None
    assert loaded.jobs[1].status == STATUS_QUEUED


def test_pipeline_worker_resumes_paused_draft_then_continues_queue(tmp_path, monkeypatch):
    from core.session import DubbingSession
    from gui import workers

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
    jobs = queue.add_jobs(settings, [video_a, video_b])
    paused_session_settings = make_pipeline_settings(
        tmp_path,
        input_video=video_a,
        input_videos=[video_a],
    )
    paused_session = DubbingSession(
        work_dir=tmp_path / "temp" / "job_paused",
        settings=paused_session_settings,
    )
    paused_session.save()
    queue.mark_running(jobs[0].draft_id, paused_session.path)
    queue.mark_paused(jobs[0].draft_id)
    processed: list[tuple[Path, Path]] = []

    class FakePipeline:
        def __init__(self, context, session):
            self.context = context
            self.session = session

        def run(self):
            current = self.context.settings.input_video
            processed.append((current, self.context.work_dir))
            output = self.context.settings.output_dir / f"{current.stem}_khmer_dubbed.mp4"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"mp4")
            return output

        def cancel(self):
            self.context.cancel_event.set()

    monkeypatch.setattr(workers, "DubbingPipeline", FakePipeline)
    worker = workers.PipelineWorker(settings, tmp_path, draft_queue_path=queue.path)
    finished: list[str] = []
    failed: list[str] = []
    worker.signals.finished.connect(finished.append)
    worker.signals.failed.connect(failed.append)

    worker.run()
    loaded = DraftQueue.load(queue.path)

    assert failed == []
    assert [item[0] for item in processed] == [video_a, video_b]
    assert processed[0][1] == paused_session.work_dir
    assert [job.status for job in loaded.jobs] == [STATUS_COMPLETED, STATUS_COMPLETED]
    assert finished == [str(settings.output_dir / "b_khmer_dubbed.mp4")]


def test_pipeline_worker_picks_up_drafts_added_while_queue_is_running(tmp_path, monkeypatch):
    from gui import workers

    video_a = tmp_path / "a.mp4"
    video_b = tmp_path / "b.mp4"
    video_c = tmp_path / "c.mp4"
    for video in [video_a, video_b, video_c]:
        video.write_bytes(video.stem.encode())
    settings = make_pipeline_settings(
        tmp_path,
        input_video=video_a,
        input_videos=[video_a, video_b],
    )
    queue = DraftQueue(tmp_path / "draft_queue.json")
    queue.add_jobs(settings, [video_a, video_b])
    processed: list[Path] = []

    class FakePipeline:
        def __init__(self, context, session):
            self.context = context
            self.session = session

        def run(self):
            current = self.context.settings.input_video
            processed.append(current)
            if current == video_a:
                DraftQueue.load(queue.path).add_job(self.context.settings, video_c)
            output = self.context.settings.output_dir / f"{current.stem}_khmer_dubbed.mp4"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"mp4")
            return output

        def cancel(self):
            self.context.cancel_event.set()

    monkeypatch.setattr(workers, "DubbingPipeline", FakePipeline)
    worker = workers.PipelineWorker(settings, tmp_path, draft_queue_path=queue.path)

    worker.run()
    loaded = DraftQueue.load(queue.path)

    assert processed == [video_a, video_b, video_c]
    assert [job.status for job in loaded.jobs] == [STATUS_COMPLETED, STATUS_COMPLETED, STATUS_COMPLETED]
