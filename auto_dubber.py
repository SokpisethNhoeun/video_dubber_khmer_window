from __future__ import annotations

import json
import logging
import os
import re
import shutil
import sys
import time
from pathlib import Path
from threading import Event

# Set up logging to stdout and a file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger("AutoDubber")


VOICE_GENDER_LABELS = {
    "Female TTS (single voice)": "female",
    "Male TTS (single voice)": "male",
    "Auto male/female TTS with emotion": "auto",
    "Auto per-speaker voices (no clone)": "per_speaker_auto",
    "Auto per-person clone": "per_person_auto",
    "Per-person manual clone map": "per_person",
}
CLONE_BACKEND_LABELS = {
    "Qwen3-TTS 1.7B (best clone + emotion)": "qwen3",
    "CosyVoice 2 (emotional voice clone)": "cosyvoice",
    "XTTS-v2 (direct voice clone)": "xtts",
    "OpenVoice (timbre transfer)": "openvoice",
}
CLONE_GENDER_LABELS = {
    "All speakers": "all",
    "Female only": "female",
    "Male only": "male",
}
TRANSLATION_BACKEND_LABELS = {
    "NLLB (offline)": "nllb",
    "Google Translate (online)": "google",
    "AI Translation (online)": "ai",
}
REVIEW_MODE_LABELS = {
    "AI review if configured": "auto",
    "Manual review (pause)": "manual",
    "Skip review": "skip",
}
KHMER_STYLE_LABELS = {
    "Natural": "natural",
    "Simple": "simple",
    "Formal": "formal",
}
CONTENT_STYLE_LABELS = {
    "Casual Vlog": "casual_vlog",
    "Reaction": "reaction",
    "Movie / Drama": "movie_dialogue",
    "Documentary": "documentary",
    "Tutorial": "tutorial",
    "News": "news",
}
PRESET_LABELS = {
    "Best Quality": "best",
    "Balanced (recommended)": "balanced",
    "Fast Draft": "fast",
}
PUBLISH_TARGET_LABELS = {
    "YouTube (-14 LUFS)": "youtube",
    "TikTok/Reels (-12 LUFS)": "tiktok",
    "Podcast (-16 LUFS)": "podcast",
    "Custom": "custom",
}
FEMALE_EDGE_FALLBACK = "km-KH-SreymomNeural"
MALE_EDGE_FALLBACK = "km-KH-PisethNeural"


def _internal_value(value: str | None, label_map: dict[str, str], default: str) -> str:
    if not value:
        return default
    return label_map.get(str(value).strip(), str(value).strip() or default)


def _optional_path(value: str | None) -> Path | None:
    if not value:
        return None
    text = str(value).strip()
    return Path(text).expanduser() if text else None


def _profile_key(value: str) -> str:
    value = re.sub(r"\s*\(generated\)\s*$", "", value.strip(), flags=re.IGNORECASE)
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _selected_voice(
    project_root: Path,
    selected: str | None,
    gender: str,
    edge_fallback: str,
) -> tuple[str, Path | None]:
    """Return Edge fallback voice plus generated-profile reference, if selected."""
    text = str(selected or "").strip()
    if not text:
        return edge_fallback, None

    try:
        from modules.voice_profiles import list_voice_profiles

        selected_key = _profile_key(text)
        for profile in list_voice_profiles(project_root):
            if profile.gender != gender:
                continue
            profile_keys = {
                _profile_key(profile.name),
                _profile_key(profile.slug),
                _profile_key(str(profile.reference_audio_path)),
                _profile_key(f"{profile.name} (generated)"),
            }
            if selected_key in profile_keys:
                return edge_fallback, profile.reference_audio_path
    except Exception as exc:
        logger.warning("Could not resolve saved %s voice profile %r: %s", gender, text, exc)

    return text, None


def _settings_defaults_from_saved_ui(project_root: Path) -> dict:
    settings_path = project_root / "settings.json"
    if not settings_path.exists():
        return {}
    try:
        config = json.loads(settings_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not read saved settings.json: %s", exc)
        return {}

    voice_female, voice_female_ref = _selected_voice(
        project_root, config.get("voice_female"), "female", FEMALE_EDGE_FALLBACK,
    )
    voice_male, voice_male_ref = _selected_voice(
        project_root, config.get("voice_male"), "male", MALE_EDGE_FALLBACK,
    )

    return {
        "voice_gender": _internal_value(config.get("voice_gender"), VOICE_GENDER_LABELS, "auto"),
        "tts_provider": str(config.get("tts_provider") or "edge"),
        "voice_female": voice_female,
        "voice_male": voice_male,
        "voice_female_reference_path": voice_female_ref,
        "voice_male_reference_path": voice_male_ref,
        "speech_rate": int(config.get("speech_rate", 0)),
        "pitch_hz": int(config.get("pitch_hz", 0)),
        "whisper_model": str(config.get("whisper_model") or "medium"),
        "device": str(config.get("device") or "cuda"),
        "keep_temp": bool(config.get("keep_temp", False)),
        "rvc_enabled": bool(config.get("rvc_enabled", False)),
        "rvc_clone_gender": _internal_value(config.get("rvc_clone_gender"), CLONE_GENDER_LABELS, "all"),
        "clone_workflow": str(config.get("clone_workflow") or "auto_per_person"),
        "clone_backend": _internal_value(config.get("clone_backend"), CLONE_BACKEND_LABELS, "openvoice"),
        "emotion_aware_clone": bool(config.get("emotion_aware_clone", True)),
        "emotion_clone_mode": _internal_value(
            config.get("emotion_clone_mode"),
            {"Auto (reference + detected emotion)": "auto",
             "Reference-based (source audio clip only)": "reference",
             "Instruction-based (emotion prompts only)": "instruction"},
            "auto",
        ),
        "enable_audio_cleanup": bool(config.get("audio_cleanup", True)),
        "enable_final_mastering": bool(config.get("final_mastering", True)),
        "enable_persistent_cache": bool(config.get("persistent_cache", True)),
        "enable_clone_verification": bool(config.get("clone_verification", True)),
        "enable_bgm_ducking": bool(config.get("bgm_ducking", True)),
        "duck_depth_db": float(config.get("duck_depth_db", 8.0)),
        "publish_target": _internal_value(config.get("publish_target"), PUBLISH_TARGET_LABELS, "youtube"),
        "custom_lufs": float(config.get("custom_lufs", -14.0)),
        "enable_per_speaker_prosody": bool(config.get("per_speaker_prosody", True)),
        "preserve_bgm": bool(config.get("preserve_bgm", True)),
        "preset": _internal_value(config.get("preset"), PRESET_LABELS, "balanced"),
        "translation_backend": _internal_value(
            config.get("translation_backend"), TRANSLATION_BACKEND_LABELS, "ai",
        ),
        "transcript_review_mode": _internal_value(config.get("review_mode"), REVIEW_MODE_LABELS, "auto"),
        "khmer_style": _internal_value(config.get("khmer_style"), KHMER_STYLE_LABELS, "simple"),
        "content_style": _internal_value(config.get("content_style"), CONTENT_STYLE_LABELS, "casual_vlog"),
        "ai_skip_review": bool(config.get("ai_skip_review", True)),
        "narration_style": str(config.get("narration_style") or "natural"),
        "glossary_path": _optional_path(config.get("glossary_path")),
        "review_json_path": _optional_path(config.get("review_json_path")),
        "save_review_json": bool(config.get("save_review_json", False)),
        "export_dubbed_audio": bool(config.get("export_audio", True)),
        "export_original_transcript": bool(config.get("export_original", True)),
        "export_raw_khmer": bool(config.get("export_raw_khmer", True)),
        "export_improved_khmer": bool(config.get("export_improved_khmer", True)),
        "export_subtitles": bool(config.get("export_srt", True)),
        "export_quality_report": bool(config.get("export_quality", True)),
        "burn_subtitles": bool(config.get("burn_subtitles", False)),
        "subtitle_language": str(config.get("subtitle_language") or "khmer").lower(),
        "subtitle_font_size": int(config.get("subtitle_font_size", 24)),
        "overlay_text": str(config.get("overlay_text") or ""),
        "overlay_image_path": _optional_path(config.get("overlay_image_path")),
        "overlay_text_position": str(config.get("overlay_text_position") or config.get("overlay_position") or "bottom_right"),
        "overlay_image_position": str(config.get("overlay_image_position") or config.get("overlay_position") or "bottom_right"),
        "overlay_opacity": float(config.get("overlay_opacity", 0.7)),
        "end_screen_enabled": bool(config.get("end_screen_enabled", False)),
        "end_screen_text": str(config.get("end_screen_text") or ""),
        "end_screen_image_path": _optional_path(config.get("end_screen_image_path")),
        "end_screen_bg_color": str(config.get("end_screen_bg_color") or "black"),
        "end_screen_duration": float(config.get("end_screen_duration", 3.0)),
    }


def get_stable_video_files(directory: Path) -> list[Path]:
    """Finds video files that are not currently being written to."""
    video_extensions = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
    videos = []
    for file in directory.glob("**/*"):
        if file.is_file() and file.suffix.lower() in video_extensions:
            # Skip videos inside the processed directory
            if "processed" in file.parts:
                continue
            videos.append(file)
            
    # Filter to only return files whose size is stable (fully copied)
    stable_videos = []
    for video in videos:
        try:
            size_before = video.stat().st_size
            time.sleep(1.0)
            size_after = video.stat().st_size
            if size_before == size_after and size_before > 0:
                stable_videos.append(video)
        except OSError:
            # File might be locked or deleted
            continue
    return stable_videos

def main() -> int:
    project_root = Path(__file__).resolve().parent
    
    # Load environment variables
    from config.env import load_project_env
    load_project_env(project_root)
    
    # Define and create watchdog directory structure
    input_dir = project_root / "input_videos"
    output_dir = project_root / "output_videos"
    processed_dir = input_dir / "processed"
    
    # Subfolders to specify source language easily without config
    lang_dirs = {
        "en": input_dir / "en",
        "zh": input_dir / "zh",
        "km": input_dir / "km",
    }
    
    input_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    for d in lang_dirs.values():
        d.mkdir(parents=True, exist_ok=True)
        
    logger.info("==================================================")
    logger.info("  KHMER AUTO-DUBBER BACKGROUND SERVICE STARTED  ")
    logger.info(f"  Watching: {input_dir}")
    logger.info(f"  Outputting: {output_dir}")
    logger.info("  Drop videos in specific subfolders to dub:")
    logger.info("    - input_videos/en/ -> Dub English to Khmer")
    logger.info("    - input_videos/zh/ -> Dub Chinese to Khmer")
    logger.info("    - input_videos/km/ -> Khmer audio (Re-dub/Denoise)")
    logger.info("    - input_videos/   -> Defaults to English source")
    logger.info("==================================================")

    # Initialize modules & models caching early to prevent delayed loading logs
    from core.context import PipelineSettings, PipelineContext
    from core.pipeline import DubbingPipeline, create_work_dir
    from core.logger import PipelineLogger

    saved_defaults = _settings_defaults_from_saved_ui(project_root)
    if saved_defaults:
        female_ref = saved_defaults.get("voice_female_reference_path")
        male_ref = saved_defaults.get("voice_male_reference_path")
        logger.info(
            "Loaded saved app defaults: voice mode=%s, female=%s%s, male=%s%s, clone=%s/%s",
            saved_defaults.get("voice_gender"),
            saved_defaults.get("voice_female"),
            f" + profile {female_ref}" if female_ref else "",
            saved_defaults.get("voice_male"),
            f" + profile {male_ref}" if male_ref else "",
            saved_defaults.get("clone_workflow"),
            saved_defaults.get("clone_backend"),
        )
    
    while True:
        try:
            videos = get_stable_video_files(input_dir)
            if not videos:
                time.sleep(3.0)
                continue
                
            for video_path in videos:
                logger.info(f"Detected new video: {video_path.name}")
                
                # Determine source language based on path
                source_lang = "en"  # default
                for lang, ldir in lang_dirs.items():
                    if ldir in video_path.parents:
                        source_lang = lang
                        break
                        
                logger.info(f"Determined source language: {source_lang.upper()}")
                
                # Create job work directory
                work_dir = create_work_dir(project_root / "temp")
                job_log_path = output_dir / f"{video_path.stem}_dubbing.log"
                job_logger = PipelineLogger(work_dir / "pipeline.log", logger.info)
                
                # Initialize PipelineSettings on auto-pilot
                settings_kwargs = {
                    "input_video": video_path,
                    "output_dir": output_dir,
                    "source_language": source_lang,
                    "voice_gender": "per_person_auto",
                    "voice_female": FEMALE_EDGE_FALLBACK,
                    "voice_male": MALE_EDGE_FALLBACK,
                    "speech_rate": 0,
                    "pitch_hz": 0,
                    "whisper_model": "medium",
                    "device": "cuda",
                    "keep_temp": False,
                    "rvc_enabled": False,
                    "alignment_mode": "natural",
                    "enable_audio_cleanup": True,
                    "enable_final_mastering": True,
                    "enable_persistent_cache": True,
                    "min_reference_seconds": 10.0,
                    "auto_speaker_references": True,
                    "preserve_bgm": True,
                }
                settings_kwargs.update(saved_defaults)
                settings_kwargs.update({
                    "input_video": video_path,
                    "input_videos": [video_path],
                    "output_dir": output_dir,
                    "source_language": source_lang,
                })
                settings_kwargs["auto_speaker_references"] = (
                    settings_kwargs.get("voice_gender") == "per_person_auto"
                )
                settings = PipelineSettings(**settings_kwargs)
                
                context = PipelineContext(
                    settings=settings,
                    work_dir=work_dir,
                    log=job_logger
                )
                
                pipeline = DubbingPipeline(context)
                
                start_time = time.time()
                try:
                    logger.info(f"Starting automatic dubbing pipeline for {video_path.name}...")
                    output_video = pipeline.run()
                    elapsed = time.time() - start_time
                    logger.info(f"SUCCESS: Generated dubbed video: {output_video.name} in {elapsed:.1f}s")
                    
                    # Move logs to output directory for inspection
                    shutil.copy2(work_dir / "pipeline.log", job_log_path)
                except Exception as run_err:
                    logger.error(f"FAILED processing {video_path.name}: {run_err}")
                    # Write failure log
                    with job_log_path.open("w", encoding="utf-8") as lf:
                        lf.write(f"Dubbing failed: {run_err}\n")
                        if (work_dir / "pipeline.log").exists():
                            lf.write((work_dir / "pipeline.log").read_text(encoding="utf-8"))
                            
                # Move input file to processed to avoid infinite loop
                dest_video_path = processed_dir / video_path.name
                # Ensure unique destination filename
                if dest_video_path.exists():
                    dest_video_path = processed_dir / f"{video_path.stem}_{int(time.time())}{video_path.suffix}"
                try:
                    shutil.move(str(video_path), str(dest_video_path))
                    logger.info(f"Moved source video to {dest_video_path.relative_to(project_root)}")
                except Exception as move_err:
                    logger.warning(f"Could not move source video: {move_err}")
                    
        except KeyboardInterrupt:
            logger.info("AutoDubber background service stopped by user.")
            break
        except Exception as loop_err:
            logger.error(f"Error in watchdog poll loop: {loop_err}")
            time.sleep(5.0)
            
    return 0

if __name__ == "__main__":
    sys.exit(main())
