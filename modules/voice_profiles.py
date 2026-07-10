from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Event

from modules.audio_quality import prepare_reference_audio, validate_reference_audio
from modules.audio_utils import remove_tree


@dataclass(frozen=True)
class VoiceProfile:
    name: str
    slug: str
    gender: str
    reference_audio_path: Path
    source_audio_path: Path
    created_at: str
    duration: float
    status: str


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", name.strip().lower()).strip("-")
    return slug or "voice"


def _metadata_path(profile_dir: Path) -> Path:
    return profile_dir / "voice.json"


def voice_profiles_dir(project_root: Path) -> Path:
    return project_root / "voice_profiles"


def list_voice_profiles(project_root: Path) -> list[VoiceProfile]:
    profiles: list[VoiceProfile] = []
    root = voice_profiles_dir(project_root)
    if not root.exists():
        return profiles

    for metadata_file in sorted(root.glob("*/voice.json")):
        try:
            data = json.loads(metadata_file.read_text(encoding="utf-8"))
            profile_dir = metadata_file.parent
            reference_path = profile_dir / data["reference_audio"]
            source_path = profile_dir / data["source_audio"]
            if reference_path.exists():
                profiles.append(
                    VoiceProfile(
                        name=str(data["name"]),
                        slug=profile_dir.name,
                        gender=str(data.get("gender", "custom")),
                        reference_audio_path=reference_path,
                        source_audio_path=source_path,
                        created_at=str(data.get("created_at", "")),
                        duration=float(data.get("duration", 0.0)),
                        status=str(data.get("status", "ok")),
                    )
                )
        except Exception:
            continue
    return sorted(profiles, key=lambda profile: profile.name.lower())


def delete_voice_profile(project_root: Path, reference_audio: Path) -> VoiceProfile:
    target = reference_audio.expanduser()
    for profile in list_voice_profiles(project_root):
        if profile.reference_audio_path == target or profile.reference_audio_path.resolve() == target.resolve():
            remove_tree(profile.reference_audio_path.parent)
            return profile
    raise FileNotFoundError("Saved voice profile was not found.")


def create_voice_profile(
    project_root: Path,
    name: str,
    source_audio: Path,
    gender: str,
    min_seconds: float = 10.0,
) -> VoiceProfile:
    clean_name = name.strip()
    if not clean_name:
        raise ValueError("Enter a voice name.")
    if gender not in {"female", "male"}:
        raise ValueError("Select Male or Female for this voice.")

    source_path = source_audio.expanduser()
    validation = validate_reference_audio(source_path, min_seconds)
    if not validation.exists:
        raise FileNotFoundError("Select an existing MP3 or WAV reference audio file.")
    if not validation.supported:
        raise ValueError("Reference voice audio must be an MP3 or WAV file.")
    if any(warning.startswith("silent") for warning in validation.warnings):
        raise ValueError("Reference voice audio is silent or near-silent.")

    root = voice_profiles_dir(project_root)
    root.mkdir(parents=True, exist_ok=True)
    slug = _slugify(clean_name)
    profile_dir = root / slug
    if profile_dir.exists():
        raise FileExistsError(f"A voice named '{clean_name}' already exists.")

    profile_dir.mkdir(parents=True, exist_ok=False)
    source_copy = profile_dir / f"source{source_path.suffix.lower()}"
    reference_copy = profile_dir / "reference.wav"
    work_dir = profile_dir / "work"

    try:
        shutil.copy2(source_path, source_copy)
        cleaned_path, cleaned_validation = prepare_reference_audio(source_path, work_dir, min_seconds, Event())
        if cleaned_path is None:
            raise ValueError(cleaned_validation.status)
        shutil.copy2(cleaned_path, reference_copy)
        metadata = {
            "name": clean_name,
            "gender": gender,
            "source_audio": source_copy.name,
            "reference_audio": reference_copy.name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "duration": validation.duration,
            "status": validation.status,
        }
        _metadata_path(profile_dir).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    except Exception:
        remove_tree(profile_dir)
        raise
    finally:
        remove_tree(work_dir)

    return VoiceProfile(
        name=clean_name,
        slug=slug,
        gender=gender,
        reference_audio_path=reference_copy,
        source_audio_path=source_copy,
        created_at=str(metadata["created_at"]),
        duration=validation.duration,
        status=validation.status,
    )
