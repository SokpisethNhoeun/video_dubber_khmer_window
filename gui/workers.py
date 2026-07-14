from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from threading import Event

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from config.models import STAGES
from core.context import CancellationError, PipelineContext, PipelineSettings
from core.draft_queue import DraftQueue, STATUS_FAILED, STATUS_PAUSED
from core.logger import PipelineLogger
from core.pipeline import DubbingPipeline, create_work_dir
from core.preview import preview_segment
from core.session import DubbingSession
from core.setup_check import run_setup_checks
from gui.signals import WorkerSignals
from modules.audio_utils import extract_audio
from modules.diarizer import detect_speakers
from modules.video_import import DownloadCancelledError, VideoImportError, VideoImportService
from modules.gemini_key_validator import validate_gemini_api_key
from modules.model_downloader import (
    DownloadCancelled,
    DownloadPaused,
    HuggingFaceModelDownloadManager,
)


def _friendly_processing_error(exc: Exception) -> str:
    message = str(exc)
    if "application control policy has blocked this file" in message.lower():
        return (
            "Windows Application Control blocked a native app component. Install the latest "
            "officially signed Khmer Video Dubber setup package. If it is already current, "
            "ask your Windows administrator to allow the Khmer Video Dubber publisher."
        )
    return message


class GeminiValidationWorker(QObject):
    finished = pyqtSignal(bool, str)

    def __init__(self, key: str) -> None:
        super().__init__()
        self.key = key

    @pyqtSlot()
    def run(self) -> None:
        try:
            valid, message = validate_gemini_api_key(self.key)
            self.finished.emit(valid, message)
        except Exception as exc:
            self.finished.emit(False, f"Gemini validation failed: {exc}")


class ModelDownloadWorker(QObject):
    progress = pyqtSignal(str, int, int, float, object)
    status = pyqtSignal(str)
    finished = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, manager: HuggingFaceModelDownloadManager) -> None:
        super().__init__()
        self.manager = manager

    @pyqtSlot()
    def run(self) -> None:
        try:
            self.status.emit("connecting")
            started = False

            def report_progress(filename, done, total, speed, eta):
                nonlocal started
                if not started:
                    started = True
                    self.status.emit("downloading")
                self.progress.emit(filename, done, total, speed, eta)

            result = self.manager.download(report_progress)
            self.finished.emit(str(result))
        except DownloadPaused:
            self.status.emit("paused")
        except DownloadCancelled:
            self.status.emit("cancelled")
        except Exception as exc:
            self.failed.emit(str(exc))


class SpeakerDetectionSignals(QObject):
    log = pyqtSignal(str)
    progress = pyqtSignal(str, int)
    completed = pyqtSignal(object, object, object)
    failed = pyqtSignal(object, object, str)


class SetupCheckSignals(QObject):
    log = pyqtSignal(str)
    finished = pyqtSignal(bool)


class VideoImportSignals(QObject):
    log = pyqtSignal(str)
    progress = pyqtSignal(str, int)
    finished = pyqtSignal(object, object)
    failed = pyqtSignal(str)


class SetupCheckWorker(QObject):
    def __init__(self, settings: PipelineSettings, project_root: Path) -> None:
        super().__init__()
        self.signals = SetupCheckSignals()
        self.settings = settings
        self.project_root = project_root

    @pyqtSlot()
    def run(self) -> None:
        try:
            results = run_setup_checks(self.settings, self.project_root)
            has_errors = any(result.is_error for result in results)
            self.signals.log.emit("")
            self.signals.log.emit("Setup check results:")
            for result in results:
                self.signals.log.emit(f"  [{result.status}] {result.name}: {result.message}")
            self.signals.finished.emit(not has_errors)
        except Exception as exc:
            self.signals.log.emit(f"Setup check failed unexpectedly: {exc}")
            self.signals.finished.emit(False)


class SpeakerDetectionWorker(QObject):
    def __init__(self, video_path: Path, settings: PipelineSettings, project_root: Path) -> None:
        super().__init__()
        self.signals = SpeakerDetectionSignals()
        self.video_path = video_path
        self.settings = settings
        self.project_root = project_root
        self.work_dir = create_work_dir(project_root / "temp")
        self.cancel_event = Event()

    @pyqtSlot()
    def run(self) -> None:
        audio_wav = self.work_dir / "source_mono_16k.wav"
        try:
            self.signals.log.emit(f"Detecting speakers for {self.video_path.name}")
            extract_audio(
                self.video_path,
                audio_wav,
                lambda value: self.signals.progress.emit("extract_audio", value),
                self.cancel_event,
            )
            turns = detect_speakers(
                audio_wav,
                self.settings.device,
                lambda value: self.signals.progress.emit("speaker_detection", value),
                self.signals.log.emit,
                self.cancel_event,
            )
            self.signals.completed.emit(self.video_path, self.work_dir, turns)
        except Exception as exc:
            self.signals.failed.emit(self.video_path, self.work_dir, str(exc))

    def cancel(self) -> None:
        self.cancel_event.set()


class VideoImportWorker(QObject):
    def __init__(
        self,
        urls: list[str],
        project_root: Path,
        *,
        cookies_file: Path | None = None,
        name_prefix: str | None = None,
    ) -> None:
        super().__init__()
        self.signals = VideoImportSignals()
        self.urls = urls
        self.cookies_file = cookies_file
        self.name_prefix = name_prefix
        self.service = VideoImportService(project_root / "cache" / "imports")
        self.cancel_event = Event()

    @pyqtSlot()
    def run(self) -> None:
        imported: list[Path] = []
        failures: list[tuple[str, str]] = []
        total = len(self.urls)
        for index, url in enumerate(self.urls, start=1):
            if self.cancel_event.is_set():
                self.signals.failed.emit("URL import cancelled by user.")
                return
            self.signals.log.emit(f"Importing URL {index}/{total}: {url}")
            try:
                stem = None
                if self.name_prefix:
                    stem = f"{self.name_prefix}_{index}"
                path = self.service.import_video(
                    url,
                    cookies_file=self.cookies_file,
                    cancel_event=self.cancel_event,
                    log=self.signals.log.emit,
                    progress=lambda current_url, value: self.signals.progress.emit(current_url, value),
                    preferred_stem=stem,
                )
            except DownloadCancelledError:
                self.signals.failed.emit("URL import cancelled by user.")
                return
            except VideoImportError as exc:
                failures.append((url, str(exc)))
                self.signals.log.emit(f"Failed importing {url}: {exc}")
                continue
            imported.append(path)
            self.signals.log.emit(f"Imported video: {path.name}")
            self.signals.progress.emit(url, 100)
        self.signals.finished.emit(imported, failures)

    def cancel(self) -> None:
        self.cancel_event.set()


class PipelineWorker(QObject):
    def __init__(
        self,
        settings: PipelineSettings,
        project_root: Path,
        resume_session: DubbingSession | None = None,
        draft_queue_path: Path | None = None,
    ) -> None:
        super().__init__()
        self.signals = WorkerSignals()
        self.settings = settings
        self.project_root = project_root
        self.resume_session = resume_session
        self.draft_queue_path = draft_queue_path
        self.import_service = VideoImportService(project_root / "cache" / "imports")
        self.pipeline: DubbingPipeline | None = None
        self._stop_after_current = False

    @pyqtSlot()
    def run(self) -> None:
        if self.resume_session is not None:
            self._run_resume()
            return
        if self.draft_queue_path is not None:
            self._run_draft_queue()
            return

        videos = self.settings.input_videos
        if not videos:
            videos = [self.settings.input_video]

        self._run_video_batch(videos)

    def _batch_remainder(self, video_path: Path, videos: list[Path]) -> list[Path]:
        for index, candidate in enumerate(videos):
            if Path(candidate) == Path(video_path):
                return [Path(item) for item in videos[index:]]
        return [Path(video_path)]

    def _remaining_after_session(self, session: DubbingSession) -> list[Path]:
        videos = [Path(item) for item in (session.settings.input_videos or [])]
        if not videos:
            return []
        current = Path(session.settings.input_video)
        for index, candidate in enumerate(videos):
            if candidate == current:
                return videos[index + 1:]
        return videos[1:] if videos and videos[0] == current else []

    def _run_video_batch(
        self,
        videos: list[Path],
        *,
        outputs: list[Path] | None = None,
        failures: list[tuple[Path, str]] | None = None,
        start_index: int = 1,
        total: int | None = None,
    ) -> None:
        outputs = outputs if outputs is not None else []
        failures = failures if failures is not None else []
        total = total or len(videos)

        for idx, video_path in enumerate(videos, start=start_index):
            if self.pipeline and self.pipeline.context.cancel_event.is_set():
                break

            self.signals.log.emit(f"\n==================================================")
            self.signals.log.emit(f"PROCESSING VIDEO {idx}/{total}: {video_path.name}")
            self.signals.log.emit(f"==================================================")

            for stage, _ in STAGES:
                self.signals.progress.emit(stage, 0)

            work_dir = create_work_dir(self.project_root / "temp")
            logger = PipelineLogger(work_dir / "pipeline.log", self.signals.log.emit)

            from dataclasses import replace

            context_settings = replace(
                self.settings,
                input_video=video_path,
                input_videos=[video_path],
            )
            session_settings = replace(
                self.settings,
                input_video=video_path,
                input_videos=self._batch_remainder(video_path, videos),
            )

            context = PipelineContext(
                settings=context_settings,
                work_dir=work_dir,
                progress=self.signals.progress.emit,
                log=logger,
            )
            session = DubbingSession(work_dir=work_dir, settings=session_settings)
            self.pipeline = DubbingPipeline(context, session)
            try:
                output = self.pipeline.run()
                outputs.append(output)
                self.signals.log.emit(f"Output video {idx}/{total}: {output}")
                source_url = VideoImportService.source_url_for_video(video_path)
                if source_url:
                    self.signals.log.emit(f"Source: {source_url}")
                self._cleanup_import_cache(video_path)
            except CancellationError:
                self.signals.failed.emit("Processing paused — cancelled by user")
                return
            except Exception as exc:
                message = _friendly_processing_error(exc)
                failures.append((video_path, message))
                self.signals.log.emit(f"Failed processing {video_path.name}: {message}")
                if idx < total:
                    self.signals.log.emit("Continuing with next selected video...")
                continue

        if outputs:
            if len(outputs) > 1:
                self.signals.log.emit("\nBatch completed. Generated videos:")
                for output in outputs:
                    self.signals.log.emit(f"  {output}")
            if failures:
                self.signals.log.emit("\nBatch completed with failed video(s):")
                for video_path, message in failures:
                    self.signals.log.emit(f"  {video_path.name}: {message}")
            self.signals.finished.emit(str(outputs[-1]))
        else:
            if failures:
                details = "\n".join(
                    f"- {video_path.name}: {message}"
                    for video_path, message in failures
                )
                self.signals.failed.emit(f"No videos were processed successfully.\n{details}")
            else:
                self.signals.failed.emit("No videos were processed.")

    def _run_draft_queue(self) -> None:
        outputs: list[Path] = []
        failures: list[tuple[Path, str]] = []
        processed = 0
        attempted_this_run: set[str] = set()

        while True:
            if self._stop_after_current:
                break
            queue = DraftQueue.load(self.draft_queue_path)
            job = next(
                (
                    item for item in queue.jobs
                    if item.status in {"queued", "failed", "paused"}
                    and item.draft_id not in attempted_this_run
                ),
                None,
            )
            if job is None:
                break

            attempted_this_run.add(job.draft_id)
            processed += 1
            queued_total = len([item for item in queue.jobs if item.status in {"queued", "failed", "paused"}]) + len(outputs)
            self.signals.log.emit(f"\n==================================================")
            self.signals.log.emit(
                f"PROCESSING DRAFT {processed}/{max(processed, queued_total)}: {job.video_name}"
            )
            self.signals.log.emit(f"==================================================")

            for stage, _ in STAGES:
                self.signals.progress.emit(stage, 0)

            session = None
            if job.status in {STATUS_FAILED, STATUS_PAUSED} and job.session_path:
                try:
                    session = DubbingSession.load(job.session_path)
                    session.settings = job.settings
                except Exception as exc:
                    self.signals.log.emit(f"Could not restore saved draft session; starting over: {exc}")
            if session is None:
                work_dir = create_work_dir(self.project_root / "temp")
                session = DubbingSession(work_dir=work_dir, settings=job.settings)
            logger = PipelineLogger(session.work_dir / "pipeline.log", self.signals.log.emit)
            context = PipelineContext(
                settings=session.settings,
                work_dir=session.work_dir,
                progress=self.signals.progress.emit,
                log=logger,
            )
            try:
                session.save()
                queue = DraftQueue.load(self.draft_queue_path)
                queue.mark_running(job.draft_id, session.path)
                self.signals.draft_updated.emit()

                self.pipeline = DubbingPipeline(context, session)
                output = self.pipeline.run()
                outputs.append(output)
                queue = DraftQueue.load(self.draft_queue_path)
                queue.mark_completed(job.draft_id, output)
                self.signals.draft_updated.emit()
                self.signals.log.emit(f"Output draft video: {output}")
                if job.source_url:
                    self.signals.log.emit(f"Source: {job.source_url}")
                self._cleanup_import_cache(session.settings.input_video)
            except CancellationError:
                queue = DraftQueue.load(self.draft_queue_path)
                queue.mark_paused(job.draft_id)
                self.signals.draft_updated.emit()
                self.signals.failed.emit("Processing paused — current draft saved")
                return
            except Exception as exc:
                message = str(exc)
                failures.append((job.video_path, message))
                queue = DraftQueue.load(self.draft_queue_path)
                queue.mark_failed(job.draft_id, message)
                self.signals.draft_updated.emit()
                self.signals.log.emit(f"Failed processing draft {job.video_name}: {message}")
                self.signals.log.emit("Continuing with next queued draft...")
                continue

        if outputs:
            if failures:
                self.signals.log.emit("\nDraft queue completed with failed video(s):")
                for video_path, message in failures:
                    self.signals.log.emit(f"  {video_path.name}: {message}")
            self.signals.finished.emit(str(outputs[-1]))
        elif failures:
            details = "\n".join(
                f"- {video_path.name}: {message}"
                for video_path, message in failures
            )
            self.signals.failed.emit(f"No drafts were processed successfully.\n{details}")
        else:
            self.signals.finished.emit("")

    def _run_resume(self) -> None:
        session = self.resume_session
        self.signals.log.emit(f"\nResuming session: {session.video_name}")
        for stage, _ in STAGES:
            self.signals.progress.emit(stage, 0)

        logger = PipelineLogger(session.work_dir / "pipeline.log", self.signals.log.emit)
        context = PipelineContext(
            settings=session.settings,
            work_dir=session.work_dir,
            progress=self.signals.progress.emit,
            log=logger,
        )
        self.pipeline = DubbingPipeline(context, session)
        try:
            output = self.pipeline.run()
            self._cleanup_import_cache(session.settings.input_video)
            remaining = self._remaining_after_session(session)
            if remaining:
                outputs = [output]
                total = 1 + len(remaining)
                self.signals.log.emit(
                    f"Continuing resumed batch with {len(remaining)} remaining video(s)..."
                )
                self._run_video_batch(
                    remaining,
                    outputs=outputs,
                    failures=[],
                    start_index=2,
                    total=total,
                )
            else:
                self.signals.finished.emit(str(output))
        except Exception as exc:
            self.signals.failed.emit(f"Resume failed: {exc}")

    def cancel(self) -> None:
        if self.pipeline:
            self.pipeline.cancel()

    def pause_after_current(self) -> None:
        self._stop_after_current = True

    def _cleanup_import_cache(self, video_path: Path) -> None:
        cache_dir = Path(video_path).parent
        try:
            removed = self.import_service.cleanup_video_cache(Path(video_path))
        except Exception as exc:
            self.signals.log.emit(f"Warning: could not remove imported source cache: {exc}")
            return
        if removed:
            self.signals.log.emit(f"Removed imported source cache: {cache_dir}")


class RedubWorker(QObject):
    def __init__(
        self,
        session: DubbingSession,
        edits: dict[int, str],
        project_root: Path,
    ) -> None:
        super().__init__()
        self.signals = WorkerSignals()
        self.session = session
        self.edits = edits
        self.project_root = project_root
        self.pipeline: DubbingPipeline | None = None

    @pyqtSlot()
    def run(self) -> None:
        session = self.session
        for idx, new_text in self.edits.items():
            for seg in session.segments:
                if seg.index == idx:
                    seg.user_edited_text = new_text
                    break
        session.save()

        logger = PipelineLogger(session.work_dir / "pipeline.log", self.signals.log.emit)
        context = PipelineContext(
            settings=session.settings,
            work_dir=session.work_dir,
            progress=self.signals.progress.emit,
            log=logger,
        )
        self.pipeline = DubbingPipeline(context, session)
        try:
            output = self.pipeline.redub_segments(session, list(self.edits.keys()))
            self.signals.finished.emit(str(output))
        except Exception as exc:
            self.signals.failed.emit(f"Re-dub failed: {exc}")

    def cancel(self) -> None:
        if self.pipeline:
            self.pipeline.cancel()


class PreviewSegmentWorker(QObject):
    def __init__(
        self,
        session: DubbingSession,
        segment_index: int,
        text: str,
    ) -> None:
        super().__init__()
        self.signals = WorkerSignals()
        self.session = session
        self.segment_index = segment_index
        self.text = text
        self.cancel_event = Event()

    @pyqtSlot()
    def run(self) -> None:
        try:
            segment = next(
                seg for seg in self.session.segments
                if seg.index == self.segment_index
            )
        except StopIteration:
            self.signals.failed.emit("Segment not found.")
            return

        settings = self.session.settings
        preview = replace(
            segment,
            user_edited_text=self.text,
            tts_path=None,
            cloned_path=None,
        )
        voice_mode = settings.voice_gender
        tts_voice_mode = "per_person" if voice_mode in {"per_person", "per_person_auto"} else voice_mode
        speech_rate = settings.speech_rate
        pitch_hz = settings.pitch_hz
        if settings.narration_style == "energetic":
            speech_rate = max(speech_rate, 8)
            pitch_hz = max(pitch_hz, 5)

        try:
            output = preview_segment(
                preview,
                voice_female=settings.voice_female,
                voice_male=settings.voice_male,
                speech_rate=speech_rate,
                pitch_hz=pitch_hz,
                voice_gender=tts_voice_mode,
                segment_genders=self.session.segment_genders,
                clone_backend=settings.clone_backend,
                speaker_voice_mappings=self.session.speaker_mappings or {},
                device=settings.device,
                cancel_event=self.cancel_event,
            )
        except Exception as exc:
            self.signals.failed.emit(f"Preview failed: {exc}")
            return

        if output and output.exists():
            self.signals.finished.emit(str(output))
        else:
            self.signals.failed.emit("Preview did not generate audio.")

    def cancel(self) -> None:
        self.cancel_event.set()


class StartupValidationWorker(QObject):
    finished = pyqtSignal(bool, str)

    def __init__(self) -> None:
        super().__init__()

    @pyqtSlot()
    def run(self) -> None:
        try:
            from licensing.client import LicenseClient
            from modules import gemini_key_validator
            import os

            client = LicenseClient()
            if client.required:
                license_result = client.validate()
                if not license_result.valid:
                    self.finished.emit(False, "No active license is available yet. Verify Gmail, purchase, then activate your key.")
                    return

            gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
            valid_key, key_message = gemini_key_validator.validate_gemini_api_key(gemini_key)
            if not valid_key:
                self.finished.emit(False, key_message)
                return

            self.finished.emit(True, "Subscription and Gemini API key are valid.")
        except Exception as exc:
            self.finished.emit(False, f"Validation failed: {exc}")
