from __future__ import annotations

import gc
import json
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from config.models import LANGUAGES
from core.context import CancellationError, PipelineContext
from core.session import DubbingSession
from modules.audio_quality import (
    master_final_audio,
    media_cache_key,
    prepare_reference_audio,
    resolve_publish_target,
    validate_reference_audio,
)
from modules.audio_utils import align_audio_segments, extract_audio, mux_video, remove_tree, safe_output_path, ffprobe_duration
from modules.video_overlay import burn_subtitles_and_overlay
from modules.diarizer import SpeakerTurn, assign_speakers_to_segments, detect_speakers, merge_similar_speakers, turns_from_dicts
from modules.reference_quality import assess_reference
from modules.prosody import compute_speaker_rate_profiles
from modules.speaker_references import build_auto_speaker_references
from modules.speaker_verification import (
    CLONE_SIMILARITY_ACCEPT,
    CLONE_SIMILARITY_STRONG,
    verify_cloned_segments,
)
from modules.transcriber import transcribe_audio
from modules.transcript_exports import export_pipeline_outputs
from modules.transcript_review import review_segments
from modules.translator import translate_segments
from modules.tts_engine import synthesize_tts
from modules.voice_cloner import optional_voice_clone


@dataclass
class _RunState:
    """Shared mutable state passed between pipeline stage methods."""
    segments: list = field(default_factory=list)
    duration: float = 0.0
    bgm_wav: Path | None = None
    segment_genders: dict | None = None
    voice_mode: str = ""
    per_person_mode: bool = False
    turns: list = field(default_factory=list)
    speaker_mappings: dict = field(default_factory=dict)
    emotion_clips: dict = field(default_factory=dict)
    emotion_analyses: dict = field(default_factory=dict)
    audio_wav: Path = field(default_factory=lambda: Path())
    output_video: Path = field(default_factory=lambda: Path())


class DubbingPipeline:
    def __init__(self, context: PipelineContext, session: DubbingSession | None = None) -> None:
        self.context = context
        self.session = session
        self.current_stage = "prepare"

    def cancel(self) -> None:
        self.context.cancel_event.set()

    def _stage_done(self, stage: str) -> bool:
        return self.session is not None and self.session.is_complete(stage)

    def _checkpoint(
        self,
        stage: str,
        *,
        segments=None,
        duration: float | None = None,
        artifacts: dict[str, Path | None] | None = None,
        speaker_mappings: dict | None = None,
        segment_genders: dict[int, str] | None = None,
    ) -> None:
        if self.session is None:
            return
        if segments is not None:
            self.session.segments = list(segments)
        if duration is not None:
            self.session.duration = duration
        if artifacts:
            for key, path in artifacts.items():
                self.session.set_artifact(key, path)
        if speaker_mappings is not None:
            self.session.speaker_mappings = speaker_mappings
        if segment_genders is not None:
            self.session.segment_genders = segment_genders
        self.session.mark_stage_complete(stage)
        try:
            self.session.save()
        except Exception as exc:
            self.context.emit_log(f"Warning: could not save session: {exc}")

    def _restored(self, stage: str, message: str) -> None:
        self.context.emit_progress(stage, 100)
        self.context.emit_log(f"{message}: restored from saved session")

    def _release_memory(self) -> None:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            return

    def _progress(self, stage: str) -> Callable[[int], None]:
        return lambda value: self.context.emit_progress(stage, value)

    def _prepare(self) -> None:
        settings = self.context.settings
        if not settings.input_video.exists():
            raise FileNotFoundError(f"Input video does not exist: {settings.input_video}")
        if settings.source_language not in LANGUAGES:
            raise ValueError(f"Unsupported language: {settings.source_language}")
        settings.output_dir.mkdir(parents=True, exist_ok=True)
        self.context.work_dir.mkdir(parents=True, exist_ok=True)

    def _video_key(self) -> str:
        try:
            return str(self.context.settings.input_video.resolve())
        except OSError:
            return str(self.context.settings.input_video)

    def _persistent_cache_dir(self) -> Path | None:
        if not self.context.settings.enable_persistent_cache:
            return None
        cache_dir = Path(__file__).resolve().parents[1] / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def _prepare_single_reference(self, reference_path: Path | None, label: str) -> Path | None:
        settings = self.context.settings
        if reference_path is None:
            self.context.quality_report.missing_references.append(label)
            return None
        if settings.enable_audio_cleanup:
            try:
                cleaned_path, validation = prepare_reference_audio(
                    reference_path,
                    self.context.work_dir,
                    settings.min_reference_seconds,
                    self.context.cancel_event,
                    self._persistent_cache_dir(),
                    self.context.quality_report.cache_hits,
                )
            except Exception as exc:
                validation = validate_reference_audio(reference_path, settings.min_reference_seconds, cancel_event=self.context.cancel_event)
                cleaned_path = validation.path if validation.exists and validation.supported else None
                validation.warnings.append(f"cleanup failed; using original audio: {exc}")
        else:
            validation = validate_reference_audio(reference_path, settings.min_reference_seconds, cancel_event=self.context.cancel_event)
            cleaned_path = validation.path if validation.exists and validation.supported else None

        if not validation.exists:
            self.context.quality_report.missing_references.append(label)
        if validation.warnings:
            self.context.quality_report.bad_references.append(
                {
                    "speaker": label,
                    "path": str(validation.path),
                    "status": validation.status,
                }
            )
            self.context.emit_log(f"  Reference warning for {label}: {validation.status}")
        return cleaned_path

    def _prepare_speaker_mappings(self, speaker_mappings: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
        prepared = deepcopy(speaker_mappings)
        self.context.quality_report.speaker_count = len(prepared)
        selected_paths: dict[str, list[str]] = {}
        for speaker_id, mapping in prepared.items():
            label = mapping.get("label") or speaker_id
            original = (
                mapping.get("original_reference_audio_path", "").strip()
                or mapping.get("reference_audio_path", "").strip()
            )
            if original:
                mapping["original_reference_audio_path"] = original
                selected_paths.setdefault(str(Path(original).expanduser()), []).append(label)
            else:
                mapping["reference_status"] = "missing"
                self.context.quality_report.missing_references.append(label)
                continue

            cleaned = self._prepare_single_reference(Path(original).expanduser(), label)
            if cleaned:
                mapping["cleaned_reference_audio_path"] = str(cleaned)
                mapping["reference_audio_path"] = str(cleaned)
                mapping["reference_status"] = "ok"
            else:
                mapping["reference_status"] = "invalid"

            # Compute or refresh quality assessment on the resolved reference so
            # both auto and manual mappings report to the quality gate.
            resolved_ref = mapping.get("reference_audio_path", "").strip()
            if resolved_ref and Path(resolved_ref).exists():
                try:
                    quality = assess_reference(Path(resolved_ref))
                    mapping["quality_tier"] = quality.tier
                    mapping["quality_score"] = f"{quality.score:.1f}"
                    mapping["quality_reasons"] = "; ".join(quality.reasons)
                    self.context.quality_report.speaker_quality.append(
                        {"speaker": label, **quality.to_dict()}
                    )
                    if quality.tier != "good":
                        reasons_text = "; ".join(quality.reasons) or "below quality thresholds"
                        self.context.emit_log(
                            f"  Reference quality for {label}: {quality.tier.upper()} "
                            f"(score {quality.score:.0f}/100) — {reasons_text}"
                        )
                except Exception as exc:
                    self.context.emit_log(f"  Quality assessment failed for {label}: {exc}")

        for path, labels in selected_paths.items():
            if len(labels) > 1:
                self.context.emit_log(
                    f"  Warning: same reference file assigned to multiple speakers: {', '.join(labels)} ({path})"
                )
        return prepared

    def _apply_speaker_merges(self, segments, speaker_mappings: dict[str, dict[str, str]]):
        merge_map: dict[str, str] = {}
        for sid, mapping in speaker_mappings.items():
            target = mapping.get("merge_with", "").strip()
            if target and target in speaker_mappings:
                merge_map[sid] = target

        if not merge_map:
            return segments

        for segment in segments:
            if segment.speaker_id and segment.speaker_id in merge_map:
                old_id = segment.speaker_id
                new_id = merge_map[old_id]
                segment.speaker_id = new_id
                target_mapping = speaker_mappings.get(new_id, {})
                segment.speaker_label = target_mapping.get("label") or new_id.replace("_", " ").title()
                self.context.emit_log(
                    f"  Merged segment {segment.index + 1} from {old_id} -> {new_id}"
                )
        return segments

    def _is_per_person_mode(self, voice_mode: str) -> bool:
        return voice_mode in {"per_person", "per_person_auto"}

    def _uses_speaker_split(self, voice_mode: str) -> bool:
        return voice_mode in {"per_person", "per_person_auto", "per_speaker_auto"}

    def _verify_clone_similarity(
        self,
        segments,
        speaker_mappings: dict[str, dict[str, str]],
    ) -> None:
        """Post-clone quality check: compare cloned output to the speaker
        reference. Wipe cloned_path for speakers whose clones don't sound
        like the reference so those segments fall back to base TTS during
        alignment. Records the score into quality_report.speaker_quality."""
        if not speaker_mappings:
            return

        # Group cloned segments per speaker.
        by_speaker: dict[str, list] = {}
        for segment in segments:
            if getattr(segment, "cloned_path", None) is None:
                continue
            speaker_id = getattr(segment, "speaker_id", None)
            if not speaker_id:
                continue
            by_speaker.setdefault(speaker_id, []).append(segment)

        if not by_speaker:
            return

        self.context.emit_log("Verifying cloned voices against speaker references")

        for speaker_id, cloned_segments in by_speaker.items():
            mapping = speaker_mappings.get(speaker_id) or {}
            reference_path = (
                mapping.get("cleaned_reference_audio_path", "").strip()
                or mapping.get("reference_audio_path", "").strip()
            )
            if not reference_path or not Path(reference_path).exists():
                continue

            label = mapping.get("label") or speaker_id
            cloned_paths = [seg.cloned_path for seg in cloned_segments if seg.cloned_path]
            try:
                similarity, sample_count = verify_cloned_segments(
                    speaker_id, Path(reference_path), cloned_paths
                )
            except Exception as exc:
                self.context.emit_log(
                    f"  Voice similarity check failed for {label}: {exc}"
                )
                continue

            if sample_count == 0:
                continue

            mapping["similarity_score"] = f"{similarity:.3f}"
            mapping["similarity_samples"] = str(sample_count)

            if similarity >= CLONE_SIMILARITY_STRONG:
                verdict = "STRONG"
            elif similarity >= CLONE_SIMILARITY_ACCEPT:
                verdict = "OK"
            else:
                verdict = "LOW"

            self.context.emit_log(
                f"  Voice similarity for {label}: {verdict} "
                f"({similarity:.2f}, {sample_count} sample(s))"
            )

            # Fold the result into the speaker_quality entry so it lands in
            # quality_report.json. Match by label (added during
            # _prepare_speaker_mappings).
            for entry in self.context.quality_report.speaker_quality:
                if entry.get("speaker") == label:
                    entry["clone_similarity"] = round(similarity, 3)
                    entry["clone_similarity_samples"] = sample_count
                    entry["clone_verdict"] = verdict.lower()
                    break

            if similarity < CLONE_SIMILARITY_ACCEPT:
                # Fall back to base TTS for this speaker: the clone does not
                # sound like the reference, so the base voice is preferable.
                dropped = 0
                for seg in cloned_segments:
                    if seg.cloned_path is not None:
                        seg.cloned_path = None
                        dropped += 1
                self.context.emit_log(
                    f"  Dropping {dropped} cloned segment(s) for {label}; "
                    f"falling back to base Khmer TTS"
                )
                self.context.quality_report.voice_clone_failures.append(
                    {
                        "segment": 0,
                        "message": (
                            f"voice similarity {similarity:.2f} below "
                            f"{CLONE_SIMILARITY_ACCEPT:.2f} for {label}; "
                            f"reverted {dropped} segment(s) to base TTS"
                        ),
                    }
                )

    def _load_cached_diarization(self) -> list[dict[str, float | str]] | None:
        cache_dir = self._persistent_cache_dir()
        if cache_dir is None:
            return None
        cache_file = cache_dir / "diarization" / f"{media_cache_key(self.context.settings.input_video)}.json"
        if not cache_file.exists():
            return None
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            if isinstance(data, list):
                self.context.quality_report.cache_hits["diarization"] = (
                    self.context.quality_report.cache_hits.get("diarization", 0) + 1
                )
                return data
        except Exception:
            return None
        return None

    def _store_cached_diarization(self, turns: list[SpeakerTurn]) -> None:
        cache_dir = self._persistent_cache_dir()
        if cache_dir is None:
            return
        cache_file = cache_dir / "diarization" / f"{media_cache_key(self.context.settings.input_video)}.json"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(
            json.dumps([turn.to_dict() for turn in turns], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Pipeline stage methods – each encapsulates one stage of run().
    # ------------------------------------------------------------------

    def _run_extract_audio(self, state: _RunState) -> None:
        settings = self.context.settings
        self.current_stage = "extract_audio"
        if self._stage_done("extract_audio"):
            if not state.audio_wav.exists():
                raise RuntimeError(
                    "Cannot resume: extracted audio is missing from the session folder. "
                    "Start a new run instead."
                )
            self._restored("extract_audio", "Stage 1/8")
        elif settings.preserve_bgm:
            self.context.emit_log("Stage 1/8 (BGM Preservation): separating audio into vocals and background tracks")
            original_full_wav = self.context.work_dir / "original_full.wav"

            # First extract full quality stereo audio
            self.context.emit_log("  Extracting original high-quality audio...")
            import subprocess
            from modules.audio_utils import ensure_ffmpeg
            ensure_ffmpeg()
            cmd = [
                "ffmpeg", "-y",
                "-i", str(settings.input_video),
                "-c:a", "pcm_s16le",
                "-ar", "44100",
                str(original_full_wav)
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # Run Demucs
            from modules.bgm_separator import separate_vocals_demucs
            vocals_wav, bgm_wav = separate_vocals_demucs(
                original_full_wav,
                self.context.work_dir / "demucs",
                self.context.cancel_event,
                device=settings.device,
                log_cb=self.context.emit_log
            )

            # Convert vocals_wav to mono 16kHz audio_wav for transcription
            cmd = [
                "ffmpeg", "-y",
                "-i", str(vocals_wav),
                "-ac", "1",
                "-ar", "16000",
                str(state.audio_wav)
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # Get video duration
            state.duration = ffprobe_duration(settings.input_video)
            state.bgm_wav = bgm_wav
            self.bgm_wav = bgm_wav
        else:
            self.context.emit_log("Stage 1/8: extracting mono 16 kHz audio")
            state.duration = extract_audio(
                settings.input_video,
                state.audio_wav,
                self._progress("extract_audio"),
                self.context.cancel_event,
            )
            state.bgm_wav = None
            self.bgm_wav = None
        if not self._stage_done("extract_audio"):
            self._checkpoint(
                "extract_audio",
                duration=state.duration,
                artifacts={"audio_wav": state.audio_wav, "bgm_wav": state.bgm_wav},
            )
        self.context.check_cancelled()
        self._release_memory()

    def _run_transcription(self, state: _RunState) -> None:
        settings = self.context.settings
        self.current_stage = "transcription"
        if self._stage_done("transcription"):
            self._restored("transcription", "Stage 2/8")
        else:
            self.context.emit_log("Stage 2/8: transcribing with faster-whisper")
            state.segments = transcribe_audio(
                state.audio_wav,
                LANGUAGES[settings.source_language].whisper_code,
                settings.whisper_model,
                settings.device,
                state.duration,
                self._progress("transcription"),
                self.context.emit_log,
                self.context.cancel_event,
            )
            self.context.emit_log(f"Transcription produced {len(state.segments)} segments")
            self._checkpoint("transcription", segments=state.segments)
        self.context.check_cancelled()
        self._release_memory()

    def _run_speaker_detection(self, state: _RunState) -> None:
        settings = self.context.settings
        self.current_stage = "speaker_detection"
        state.voice_mode = settings.voice_gender
        state.per_person_mode = self._is_per_person_mode(state.voice_mode)
        video_key = self._video_key()
        state.turns = []
        state.speaker_mappings = settings.speaker_voice_mappings.get(video_key, {})
        if self._stage_done("speaker_detection"):
            state.speaker_mappings = (self.session.speaker_mappings or {}) if self.session else state.speaker_mappings
            self._restored("speaker_detection", "Stage 3/8")
        elif self._uses_speaker_split(state.voice_mode) or state.voice_mode == "auto":
            gemini_tts = getattr(settings, "tts_provider", "edge") == "gemini"
            if (
                not settings.auto_speaker_references
                and not gemini_tts
                and self._is_per_person_mode(state.voice_mode)
            ):
                self.context.emit_log("Preparing and validating per-speaker reference audio")
                state.speaker_mappings = self._prepare_speaker_mappings(state.speaker_mappings)
            if state.voice_mode == "auto":
                self.context.emit_log("Stage 3/8: detecting speakers for male/female voice grouping")
            else:
                self.context.emit_log("Stage 3/8: assigning segments to detected speakers")
            try:
                raw_turns = settings.diarization_turns.get(video_key)
                if raw_turns:
                    state.turns = turns_from_dicts(raw_turns)
                    self.context.emit_log(f"Using pre-detected speaker turns for {settings.input_video.name}")
                    self.context.emit_progress("speaker_detection", 100)
                else:
                    cached_turns = self._load_cached_diarization()
                    if cached_turns:
                        state.turns = turns_from_dicts(cached_turns)
                        self.context.emit_log(f"Using cached speaker turns for {settings.input_video.name}")
                        self.context.emit_progress("speaker_detection", 100)
                    else:
                        state.turns = detect_speakers(
                            state.audio_wav,
                            settings.device,
                            self._progress("speaker_detection"),
                            self.context.emit_log,
                            self.context.cancel_event,
                        )
                        self._store_cached_diarization(state.turns)
                try:
                    state.turns = merge_similar_speakers(state.turns, state.audio_wav, log_cb=self.context.emit_log)
                except Exception as exc:
                    self.context.emit_log(f"  Speaker merge check skipped: {exc}")
                if settings.auto_speaker_references and not gemini_tts and self._is_per_person_mode(state.voice_mode):
                    self.context.emit_log("Building automatic per-speaker reference audio from source video")
                    auto_mappings = build_auto_speaker_references(
                        state.audio_wav,
                        settings.input_video,
                        state.turns,
                        self.context.work_dir,
                        settings.min_reference_seconds,
                        self.context.cancel_event,
                        self._persistent_cache_dir(),
                        self.context.emit_log,
                    )
                    state.speaker_mappings = self._prepare_speaker_mappings(auto_mappings)
                state.segments = assign_speakers_to_segments(
                    state.segments,
                    state.turns,
                    state.speaker_mappings,
                    self.context.emit_log,
                    state.audio_wav,
                )
                state.segments = self._apply_speaker_merges(state.segments, state.speaker_mappings)
            except Exception as exc:
                if state.voice_mode == "auto":
                    self.context.emit_log(
                        f"Speaker diarization unavailable or failed: {exc}. "
                        "Continuing with per-segment male/female gender detection."
                    )
                elif getattr(settings, "tts_provider", "edge") == "gemini":
                    self.context.emit_log(
                        f"Speaker diarization unavailable or failed: {exc}. "
                        "Continuing with Gemini TTS; all lines will share one preset voice."
                    )
                else:
                    self.context.emit_log(
                        f"Speaker diarization unavailable or failed: {exc}. "
                        "Continuing with auto male/female voice mode."
                    )
                    state.voice_mode = "auto"
                state.per_person_mode = False
                self.context.quality_report.speaker_count = 0
                self.context.quality_report.missing_references.clear()
                self.context.quality_report.bad_references.clear()
                self.context.emit_progress("speaker_detection", 100)
            self.context.check_cancelled()
            self._release_memory()
        else:
            self.context.emit_progress("speaker_detection", 100)
        if not self._stage_done("speaker_detection"):
            self._checkpoint(
                "speaker_detection",
                segments=state.segments,
                speaker_mappings=state.speaker_mappings,
            )

    def _run_translation(self, state: _RunState) -> None:
        settings = self.context.settings
        self.current_stage = "translation"
        if self._stage_done("translation"):
            self._restored("translation", "Stage 4/8")
        elif settings.translation_backend == "google":
            self.context.emit_log("Stage 4/8: translating to Khmer with Google Translate")
            from modules.google_translator import translate_segments_google
            state.segments = translate_segments_google(
                state.segments,
                settings.source_language,
                self._progress("translation"),
                self.context.emit_log,
                self.context.cancel_event,
            )
        elif settings.translation_backend == "ai":
            self.context.emit_log("Stage 4/8: translating to Khmer with AI")
            from modules.ai_translator import translate_segments_ai
            state.segments = translate_segments_ai(
                state.segments,
                settings.source_language,
                self._progress("translation"),
                self.context.emit_log,
                self.context.cancel_event,
                content_style=settings.content_style,
                khmer_style=settings.khmer_style,
                glossary_path=settings.glossary_path,
                allow_review_recovery=settings.transcript_review_mode in {"ai", "auto"},
            )
        else:
            self.context.emit_log("Stage 4/8: translating to Khmer with NLLB")
            state.segments = translate_segments(
                state.segments,
                settings.source_language,
                settings.device,
                self._progress("translation"),
                self.context.emit_log,
                self.context.cancel_event,
            )
        if not self._stage_done("translation"):
            self._checkpoint("translation", segments=state.segments)
        self.context.check_cancelled()
        self._release_memory()

    def _run_transcript_review(self, state: _RunState) -> Path | None:
        """Run transcript review stage. Returns early exit path if script-only
        mode is active, otherwise returns None to continue the pipeline."""
        settings = self.context.settings
        self.current_stage = "transcript_review"
        review_path = settings.output_dir / f"{state.output_video.stem}_transcript_review.json" if settings.save_review_json else None
        needs_ai_recovery = any(
            "AI translation missing" in (segment.review_notes or "")
            for segment in state.segments
        )
        if self._stage_done("transcript_review"):
            self._restored("transcript_review", "Stage 4.25/8")
        elif settings.translation_backend == "ai" and settings.ai_skip_review and not needs_ai_recovery:
            self.context.emit_log("Stage 4.25/8: skipping AI review (AI translation already polished)")
            self.context.emit_progress("transcript_review", 100)
            self._checkpoint("transcript_review", segments=state.segments)
        else:
            if needs_ai_recovery and settings.ai_skip_review:
                self.context.emit_log(
                    "Stage 4.25/8: running AI review to recover missing AI translation segment(s)"
                )
            else:
                self.context.emit_log("Stage 4.25/8: reviewing Khmer transcript")
            state.segments = review_segments(
                state.segments,
                settings.khmer_style,
                settings.transcript_review_mode,
                settings.glossary_path,
                settings.review_json_path,
                review_path,
                self._progress("transcript_review"),
                self.context.emit_log,
                self.context.cancel_event,
                content_style=settings.content_style,
                source_language=settings.source_language,
            )
            self._checkpoint("transcript_review", segments=state.segments)
        self.context.check_cancelled()

        if getattr(settings, 'generate_script_only', False) or getattr(settings, 'transcript_review_mode', '') == 'manual':
            self.context.emit_log("Stage 4.5/8: Script generation complete. Stopping early.")
            self.context.emit_progress("gender_detection", 100)
            self.context.emit_progress("tts", 100)
            self.context.emit_progress("voice_clone", 100)
            self.context.emit_progress("alignment", 100)
            self.context.emit_progress("muxing", 100)

            export_pipeline_outputs(
                settings.output_dir,
                state.output_video.stem,
                state.segments,
                Path("dummy"), # final_audio
                Path("dummy"), # quality_report_dir
                False, # export_dubbed_audio
                settings.export_original_transcript,
                settings.export_raw_khmer,
                settings.export_improved_khmer,
                settings.export_subtitles,
                False, # export_quality_report
            )
            self.context.emit_log(f"Exported transcript script files to: {settings.output_dir}")
            if self.session is not None:
                self.session.mark_completed()
                try:
                    self.session.save()
                except Exception:
                    pass
            return review_path if review_path else (settings.output_dir / f"{state.output_video.stem}_transcript_review.json")
        return None

    def _run_gender_detection(self, state: _RunState) -> None:
        settings = self.context.settings
        self.current_stage = "gender_detection"
        tts_voice_mode = self._tts_voice_mode(state)
        if self._stage_done("gender_detection"):
            self._restored("gender_detection", "Stage 4.5/8")
        elif tts_voice_mode in {"auto", "per_person", "per_speaker_auto"}:
            from modules.gender_classifier import classify_genders, log_gender_emotion_summary

            self.context.emit_log(
                "Stage 4.5/8 (gender_detection): detecting speaker gender and emotion"
            )
            state.segment_genders = classify_genders(
                state.audio_wav,
                state.segments,
                settings.device,
                self._progress("gender_detection"),
                self.context.emit_log,
                self.context.cancel_event,
                state.turns,
            )
            self.context.check_cancelled()
            self._extract_emotion_context(state)
            log_gender_emotion_summary(
                state.segments,
                state.segment_genders or {},
                state.emotion_analyses,
                self.context.emit_log,
            )
            self._release_memory()
        else:
            state.segment_genders = None
            self.context.emit_progress("gender_detection", 100)
        if not self._stage_done("gender_detection"):
            self._checkpoint("gender_detection", segment_genders=state.segment_genders)

    def _tts_voice_mode(self, state: _RunState) -> str:
        return "per_person" if state.per_person_mode else state.voice_mode

    def _needs_emotion_for_tts(self, state: _RunState) -> bool:
        return self._tts_voice_mode(state) in {"auto", "per_person"}

    def _extract_emotion_context(self, state: _RunState, log_prefix: str = "") -> None:
        if state.emotion_clips and state.emotion_analyses:
            return
        if not state.audio_wav.exists():
            return
        active_segments = [segment for segment in state.segments if segment.enabled]
        if not active_segments:
            return

        from modules.emotion_detector import analyze_emotion_clips
        from modules.emotion_reference import extract_emotion_clips

        prefix = f"{log_prefix} " if log_prefix else ""
        self.context.emit_log(f"{prefix}Analyzing emotional delivery per segment from source audio")
        clips_dir = self.context.work_dir / "emotion_clips"
        state.emotion_clips = extract_emotion_clips(
            state.audio_wav,
            active_segments,
            state.turns,
            clips_dir,
            self.context.cancel_event,
            self.context.emit_log,
        )
        state.emotion_analyses = analyze_emotion_clips(state.emotion_clips)
        labels: dict[str, int] = {}
        neutral = 0
        for analysis in state.emotion_analyses.values():
            if analysis.is_neutral_fallback:
                neutral += 1
            else:
                labels[analysis.label] = labels.get(analysis.label, 0) + 1
        if labels:
            summary = ", ".join(f"{name}={count}" for name, count in sorted(labels.items()))
            self.context.emit_log(f"  Segment emotions detected: {summary}")
        if neutral:
            self.context.emit_log(
                f"  {neutral} segment(s) will use neutral speaking style (uncertain emotion)"
            )

    def _prepare_emotion_context(self, state: _RunState) -> None:
        settings = self.context.settings
        need_clone = settings.emotion_aware_clone
        need_tts = self._needs_emotion_for_tts(state)
        if not need_clone and not need_tts:
            state.emotion_clips = {}
            state.emotion_analyses = {}
            return
        if state.emotion_analyses and (not need_clone or state.emotion_clips):
            return
        self._extract_emotion_context(state)

    def _run_tts(self, state: _RunState) -> None:
        settings = self.context.settings
        cache_dir = self._persistent_cache_dir()
        tts_voice_mode = "per_person" if state.per_person_mode else state.voice_mode
        self.current_stage = "tts"
        self.context.quality_report.segment_count = len(state.segments)
        if self._stage_done("tts"):
            self._restored("tts", "Stage 5/8")
        else:
            self.context.emit_log("Stage 5/8: generating Khmer TTS")
            tts_provider = getattr(settings, "tts_provider", "edge")
            if tts_provider == "gemini":
                self.context.emit_log("  TTS provider: Gemini expressive TTS")
            else:
                using_generated_profiles = (
                    settings.voice_female_reference_path is not None
                    or settings.voice_male_reference_path is not None
                )
                if using_generated_profiles:
                    self.context.emit_log(
                        "  TTS provider: Edge Khmer TTS base audio "
                        "(generated voice profile is applied in Stage 6)"
                    )
                else:
                    self.context.emit_log("  TTS provider: Edge Khmer TTS")

            # Per-speaker prosody: derive a rate offset from each speaker's
            # source chars/sec so a naturally-fast talker still sounds fast
            # in the dub. Only enabled when per-person mode is active — for
            # single-voice runs, the base rate is what the user asked for.
            speaker_rate_profiles = None
            if self._uses_speaker_split(state.voice_mode) and settings.enable_per_speaker_prosody:
                speaker_rate_profiles = compute_speaker_rate_profiles(state.segments)

            tts_rate = settings.speech_rate
            tts_pitch = settings.pitch_hz
            if settings.narration_style == "energetic":
                tts_rate = max(tts_rate, 8)
                tts_pitch = max(tts_pitch, 5)
                self.context.emit_log("  Energetic narration: boosting speech rate and pitch")

            self._prepare_emotion_context(state)
            emotion_analyses = state.emotion_analyses or None

            state.segments = synthesize_tts(
                state.segments,
                tts_voice_mode,
                tts_rate,
                tts_pitch,
                self.context.work_dir,
                self._progress("tts"),
                self.context.emit_log,
                self.context.cancel_event,
                voice_female=settings.voice_female,
                voice_male=settings.voice_male,
                segment_genders=state.segment_genders,
                persistent_cache_dir=cache_dir,
                cache_hits=self.context.quality_report.cache_hits,
                speaker_rate_profiles=speaker_rate_profiles,
                emotion_analyses=emotion_analyses,
                source_language=settings.source_language,
                translation_backend=settings.translation_backend,
                content_style=settings.content_style,
                khmer_style=settings.khmer_style,
                glossary_path=settings.glossary_path,
                tts_provider=getattr(settings, "tts_provider", "edge"),
                speaker_voice_mappings=state.speaker_mappings,
            )
            video_key = self._video_key()
            settings.speaker_voice_mappings[video_key] = state.speaker_mappings
            self._checkpoint("tts", segments=state.segments)
        self.context.check_cancelled()
        self._release_memory()

    def _run_voice_clone(self, state: _RunState) -> None:
        settings = self.context.settings
        tts_voice_mode = "per_person" if state.per_person_mode else state.voice_mode
        self.current_stage = "voice_clone"
        if not settings.rvc_enabled and not state.per_person_mode:
            self.context.emit_log("Stage 6/8: finalizing TTS voice (voice cloning removed)")
            self.context.emit_progress("voice_clone", 100)
            self._checkpoint("voice_clone", segments=state.segments)
            self._release_memory()
            return
        if self._stage_done("voice_clone"):
            self._restored("voice_clone", "Stage 6/8")
        else:
            if getattr(settings, "tts_provider", "edge") == "gemini":
                self.context.emit_log("Stage 6/8: skipping voice cloning (Gemini TTS uses preset voices per speaker)")
                self.context.emit_progress("voice_clone", 100)
                self._checkpoint("voice_clone", segments=state.segments)
                self._release_memory()
                return
            self.context.emit_log("Stage 6/8: optional voice cloning")
            rvc_reference_audio_path = settings.rvc_reference_audio_path
            gender_reference_paths: dict[str, Path] = {}
            if (
                settings.rvc_enabled
                and not state.per_person_mode
                and settings.rvc_reference_audio_path is not None
                and "{reference}" in settings.rvc_command_template
            ):
                rvc_reference_audio_path = self._prepare_single_reference(
                    settings.rvc_reference_audio_path,
                    "voice clone reference",
                )
            if not state.per_person_mode:
                for gender, reference_path in {
                    "female": settings.voice_female_reference_path,
                    "male": settings.voice_male_reference_path,
                }.items():
                    if reference_path is not None:
                        prepared_reference = self._prepare_single_reference(reference_path, f"{gender} voice profile")
                        if prepared_reference is not None:
                            gender_reference_paths[gender] = prepared_reference
            self._prepare_emotion_context(state)
            state.segments = optional_voice_clone(
                state.segments,
                settings.rvc_enabled or state.per_person_mode or bool(gender_reference_paths),
                settings.rvc_model_path,
                settings.rvc_index_path,
                rvc_reference_audio_path,
                gender_reference_paths,
                settings.rvc_clone_gender,
                tts_voice_mode,
                state.segment_genders,
                settings.rvc_command_template,
                self.context.work_dir,
                self._progress("voice_clone"),
                self.context.emit_log,
                self.context.cancel_event,
                speaker_voice_mappings=state.speaker_mappings,
                quality_report=self.context.quality_report,
                clone_backend=settings.clone_backend,
                emotion_aware=settings.emotion_aware_clone,
                source_wav=state.audio_wav,
                diarization_turns=state.turns,
                emotion_mode=settings.emotion_clone_mode,
                emotion_clips=state.emotion_clips or None,
                emotion_analyses=state.emotion_analyses or None,
            )
            self.context.check_cancelled()

            if state.per_person_mode and settings.enable_clone_verification:
                try:
                    self._verify_clone_similarity(state.segments, state.speaker_mappings)
                except Exception as exc:
                    self.context.emit_log(
                        f"Voice similarity verification skipped: {exc}"
                    )
            self._checkpoint("voice_clone", segments=state.segments)
        self._release_memory()

    def _run_alignment(self, state: _RunState) -> None:
        settings = self.context.settings
        final_audio = self.context.work_dir / "final_khmer.wav"
        self.current_stage = "alignment"
        if self._stage_done("alignment") and final_audio.exists():
            self._restored("alignment", "Stage 7/8")
        else:
            effective_mode = settings.alignment_mode
            if settings.narration_style == "energetic":
                effective_mode = "energetic"
            self.context.emit_log(f"Stage 7/8: aligning audio to source timeline (mode: {effective_mode})")
            align_audio_segments(
                state.segments,
                final_audio,
                self.context.work_dir,
                state.duration,
                self._progress("alignment"),
                self.context.emit_log,
                self.context.cancel_event,
                mode=effective_mode,
                quality_report=self.context.quality_report,
                shorten_pauses=settings.enable_audio_cleanup,
            )
            self._checkpoint("alignment", artifacts={"final_audio": final_audio})
        self.context.check_cancelled()
        self._release_memory()

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------

    def run(self) -> Path:
        self._prepare()
        settings = self.context.settings
        output_video = safe_output_path(settings.input_video, settings.output_dir)

        state = _RunState(
            audio_wav=self.context.work_dir / "source_mono_16k.wav",
            output_video=output_video,
        )

        self.bgm_wav = None
        resuming = self.session is not None and bool(self.session.completed_stages)
        if resuming:
            self.context.emit_log(
                f"Resuming session {self.session.session_id} "
                f"(completed: {', '.join(self.session.completed_stages)})"
            )
            state.segments = list(self.session.segments)
            state.duration = self.session.duration
            self.bgm_wav = self.session.get_artifact("bgm_wav")
            state.bgm_wav = self.bgm_wav
            if self.session.segment_genders is not None:
                state.segment_genders = dict(self.session.segment_genders)
            self.session.status = "running"

        try:
            self._run_extract_audio(state)
            self._run_transcription(state)
            self._run_speaker_detection(state)
            self._run_translation(state)

            early_exit = self._run_transcript_review(state)
            if early_exit is not None:
                return early_exit

            self._run_gender_detection(state)
            self._run_tts(state)
            self._run_voice_clone(state)
            self._run_alignment(state)

            self.current_stage = "muxing"
            output_video = self.assemble_final_output(
                state.segments, state.duration, self.bgm_wav, state.output_video
            )
            if self.session is not None:
                self.session.set_artifact("output_video", output_video)
                self.session.mark_completed()
                try:
                    self.session.save()
                except Exception as exc:
                    self.context.emit_log(f"Warning: could not save session: {exc}")
            return output_video
        except CancellationError:
            if self.session is not None:
                self.session.mark_cancelled(self.current_stage)
                try:
                    self.session.save()
                except Exception:
                    pass
                self.context.emit_log(
                    "Run cancelled — progress saved. Resume it from the Sessions page."
                )
            raise
        except Exception as exc:
            if self.session is not None:
                self.session.mark_failed(self.current_stage, str(exc))
                try:
                    self.session.save()
                except Exception:
                    pass
                self.context.emit_log(
                    f"Run failed at stage '{self.current_stage}' — progress saved. "
                    "Resume it from the Sessions page."
                )
            raise
        finally:
            self._release_memory()
            if settings.keep_temp or self.session is not None:
                try:
                    self.context.quality_report.write(self.context.work_dir)
                except Exception as exc:
                    self.context.emit_log(f"Warning: could not write quality report: {exc}")
            if not settings.keep_temp and self.session is None:
                remove_tree(self.context.work_dir)

    def assemble_final_output(
        self,
        segments,
        duration: float,
        bgm_wav: Path | None,
        output_video: Path,
    ) -> Path:
        """Final assembly tail: mastering -> BGM mix -> mux -> overlay ->
        quality report -> exports. Requires final_khmer.wav to exist in the
        work dir (produced by alignment). Reused by segment re-dub."""
        settings = self.context.settings
        final_audio = self.context.work_dir / "final_khmer.wav"
        mastered_audio = self.context.work_dir / "final_khmer_mastered.wav"

        target_lufs, target_tp = resolve_publish_target(
            settings.publish_target, settings.custom_lufs
        )

        mux_audio = final_audio
        if settings.enable_final_mastering:
            self.context.emit_log(
                f"Mastering final dubbed audio ({settings.publish_target}: "
                f"{target_lufs:.1f} LUFS, {target_tp:.1f} dBTP)"
            )
            master_final_audio(
                final_audio,
                mastered_audio,
                duration,
                self.context.cancel_event,
                target_lufs=target_lufs,
                true_peak_dbtp=target_tp,
            )
            mux_audio = mastered_audio

        if settings.preserve_bgm and bgm_wav is not None:
            mixed_audio = self.context.work_dir / "final_mixed.wav"
            if settings.enable_bgm_ducking:
                self.context.emit_log(
                    f"Mixing dubbed speech with BGM using sidechain ducking "
                    f"({settings.duck_depth_db:.0f} dB duck)"
                )
                from modules.bgm_separator import mix_audio_ducked, mix_audio_tracks
                try:
                    mix_audio_ducked(
                        mux_audio,
                        bgm_wav,
                        mixed_audio,
                        self.context.cancel_event,
                        vocals_volume=settings.voice_volume,
                        bgm_volume=settings.bgm_volume,
                        duck_depth_db=settings.duck_depth_db,
                    )
                except Exception as exc:
                    self.context.emit_log(
                        f"Sidechain ducking failed ({exc}); falling back to flat mix"
                    )
                    mix_audio_tracks(
                        mux_audio,
                        bgm_wav,
                        mixed_audio,
                        self.context.cancel_event,
                        vocals_volume=settings.voice_volume,
                        bgm_volume=settings.bgm_volume,
                    )
            else:
                self.context.emit_log("Mixing dubbed Khmer speech with original background music track...")
                from modules.bgm_separator import mix_audio_tracks
                mix_audio_tracks(
                    mux_audio,
                    bgm_wav,
                    mixed_audio,
                    self.context.cancel_event,
                    vocals_volume=settings.voice_volume,
                    bgm_volume=settings.bgm_volume,
                )
            mux_audio = mixed_audio

        self.context.emit_log("Stage 8/8: muxing dubbed audio into source video")
        mux_video(
            settings.input_video,
            mux_audio,
            output_video,
            self._progress("muxing"),
            self.context.cancel_event,
        )
        needs_overlay = (
            settings.burn_subtitles
            or settings.overlay_text.strip()
            or (settings.overlay_image_path and settings.overlay_image_path.exists())
        )
        if needs_overlay:
            self.context.emit_log("Burning subtitles / overlay into video")
            overlay_output = output_video.with_stem(output_video.stem + "_overlay")
            burn_subtitles_and_overlay(
                input_video=output_video,
                output_video=overlay_output,
                segments=segments if settings.burn_subtitles else None,
                subtitle_language=settings.subtitle_language,
                subtitle_font_size=settings.subtitle_font_size,
                subtitle_font_name=settings.subtitle_font_name,
                subtitle_color=settings.subtitle_color,
                subtitle_bg_opacity=settings.subtitle_bg_opacity,
                overlay_text=settings.overlay_text,
                overlay_image_path=settings.overlay_image_path,
                overlay_position=settings.overlay_position,
                overlay_text_position=settings.overlay_text_position,
                overlay_image_position=settings.overlay_image_position,
                overlay_opacity=settings.overlay_opacity,
                work_dir=self.context.work_dir,
                cancel_event=self.context.cancel_event,
            )
            if overlay_output.exists():
                overlay_output.replace(output_video)

        if settings.end_screen_enabled and (
            settings.end_screen_text.strip() or
            (settings.end_screen_image_path and settings.end_screen_image_path.exists())
        ):
            self.context.emit_log("Appending end screen card...")
            from modules.end_screen import append_end_screen
            end_output = output_video.with_stem(output_video.stem + "_endcard")
            append_end_screen(
                output_video,
                end_output,
                self.context.cancel_event,
                text=settings.end_screen_text,
                image_path=settings.end_screen_image_path,
                duration=settings.end_screen_duration,
                bg_color=settings.end_screen_bg_color,
            )
            if end_output.exists():
                end_output.replace(output_video)

        # Footer banner overlay
        if settings.footer_overlay_enabled and settings.footer_overlay_config:
            self.context.emit_log("Burning footer banner overlay...")
            from modules.footer_overlay import FooterOverlayConfig, burn_footer
            footer_config = FooterOverlayConfig.from_dict(settings.footer_overlay_config)
            footer_output = output_video.with_stem(output_video.stem + "_footer")
            burn_footer(output_video, footer_output, footer_config, self.context.cancel_event)
            if footer_output.exists():
                footer_output.replace(output_video)

        # Sponsor cards (front/center/end insertion)
        if settings.sponsor_cards:
            self.context.emit_log("Inserting sponsor cards...")
            from modules.sponsor_card import SponsorCardConfig, insert_cards
            card_configs = [SponsorCardConfig.from_dict(c) for c in settings.sponsor_cards]
            cards_output = output_video.with_stem(output_video.stem + "_sponsor")
            insert_cards(
                output_video, cards_output, card_configs,
                self.context.work_dir, self.context.cancel_event,
            )
            if cards_output.exists():
                cards_output.replace(output_video)

        self.context.quality_report.final_output_path = str(output_video)
        if settings.export_quality_report:
            self.context.quality_report.write(self.context.work_dir)
        exported = export_pipeline_outputs(
            settings.output_dir,
            output_video.stem,
            segments,
            mux_audio,
            self.context.work_dir,
            settings.export_dubbed_audio,
            settings.export_original_transcript,
            settings.export_raw_khmer,
            settings.export_improved_khmer,
            settings.export_subtitles,
            settings.export_quality_report,
        )
        for path in exported:
            self.context.emit_log(f"Exported: {path}")
        self.context.emit_log(f"Finished: {output_video}")
        self.context.emit_log(self.context.quality_report.summary())
        return output_video

    def redub_segments(self, session: DubbingSession, segment_indices: list[int]) -> Path:
        """Regenerate only the given segments (TTS + optional clone), then
        re-align the full timeline and re-assemble the final output. Used by
        the segment editor after the user edits Khmer text."""
        settings = self.context.settings
        segments = list(session.segments)
        by_index = {seg.index: seg for seg in segments}
        targets = [by_index[i] for i in segment_indices if i in by_index]
        if not targets:
            raise ValueError("No matching segments to re-dub")

        self.context.emit_log(
            f"Re-dubbing {len(targets)} segment(s): "
            + ", ".join(str(seg.index + 1) for seg in targets)
        )
        for seg in targets:
            seg.tts_path = None
            seg.cloned_path = None

        voice_mode = settings.voice_gender
        per_person_mode = self._is_per_person_mode(voice_mode)
        tts_voice_mode = "per_person" if per_person_mode else voice_mode
        segment_genders = session.segment_genders
        speaker_mappings = session.speaker_mappings or {}

        speaker_rate_profiles = None
        if self._uses_speaker_split(voice_mode) and settings.enable_per_speaker_prosody:
            speaker_rate_profiles = compute_speaker_rate_profiles(segments)

        self.context.emit_log("Re-synthesizing TTS for edited segment(s)")
        targets = synthesize_tts(
            targets,
            tts_voice_mode,
            settings.speech_rate,
            settings.pitch_hz,
            self.context.work_dir,
            self._progress("tts"),
            self.context.emit_log,
            self.context.cancel_event,
            voice_female=settings.voice_female,
            voice_male=settings.voice_male,
            segment_genders=segment_genders,
            persistent_cache_dir=self._persistent_cache_dir(),
            cache_hits=self.context.quality_report.cache_hits,
            speaker_rate_profiles=speaker_rate_profiles,
            source_language=settings.source_language,
            translation_backend=settings.translation_backend,
            content_style=settings.content_style,
            khmer_style=settings.khmer_style,
            glossary_path=settings.glossary_path,
            tts_provider=getattr(settings, "tts_provider", "edge"),
            speaker_voice_mappings=speaker_mappings,
        )
        self.context.check_cancelled()

        if (
            getattr(settings, "tts_provider", "edge") != "gemini"
            and (per_person_mode or settings.rvc_enabled)
        ):
            self.context.emit_log("Re-cloning voices for edited segment(s)")
            targets = optional_voice_clone(
                targets,
                settings.rvc_enabled or per_person_mode,
                settings.rvc_model_path,
                settings.rvc_index_path,
                settings.rvc_reference_audio_path,
                {},
                settings.rvc_clone_gender,
                tts_voice_mode,
                segment_genders,
                settings.rvc_command_template,
                self.context.work_dir,
                self._progress("voice_clone"),
                self.context.emit_log,
                self.context.cancel_event,
                speaker_voice_mappings=speaker_mappings,
                quality_report=self.context.quality_report,
                clone_backend=settings.clone_backend,
            )
            self.context.check_cancelled()

        for seg in targets:
            by_index[seg.index] = seg
        segments = [by_index[seg.index] for seg in segments]

        final_audio = self.context.work_dir / "final_khmer.wav"
        self.context.emit_log("Re-aligning audio to source timeline")
        self.context.quality_report.segment_count = len(segments)
        align_audio_segments(
            segments,
            final_audio,
            self.context.work_dir,
            session.duration,
            self._progress("alignment"),
            self.context.emit_log,
            self.context.cancel_event,
            mode=settings.alignment_mode,
            quality_report=self.context.quality_report,
            shorten_pauses=settings.enable_audio_cleanup,
        )
        self.context.check_cancelled()

        output_video = safe_output_path(settings.input_video, settings.output_dir)
        output_video = self.assemble_final_output(
            segments,
            session.duration,
            session.get_artifact("bgm_wav"),
            output_video,
        )
        session.segments = segments
        session.set_artifact("output_video", output_video)
        try:
            session.save()
        except Exception as exc:
            self.context.emit_log(f"Warning: could not save session: {exc}")
        return output_video


def create_work_dir(base_temp: Path) -> Path:
    return base_temp / f"job_{uuid.uuid4().hex}"
