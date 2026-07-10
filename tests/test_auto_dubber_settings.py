import json
from pathlib import Path

from auto_dubber import _settings_defaults_from_saved_ui


def test_auto_dubber_uses_saved_generated_voice_profiles(tmp_path: Path) -> None:
    female_dir = tmp_path / "voice_profiles" / "fvoice1"
    male_dir = tmp_path / "voice_profiles" / "mvoice2"
    female_dir.mkdir(parents=True)
    male_dir.mkdir(parents=True)
    (female_dir / "reference.wav").write_bytes(b"female")
    (female_dir / "source.wav").write_bytes(b"female-src")
    (male_dir / "reference.wav").write_bytes(b"male")
    (male_dir / "source.wav").write_bytes(b"male-src")
    (female_dir / "voice.json").write_text(
        json.dumps(
            {
                "name": "Fvoice1",
                "gender": "female",
                "reference_audio": "reference.wav",
                "source_audio": "source.wav",
                "duration": 12.0,
            }
        ),
        encoding="utf-8",
    )
    (male_dir / "voice.json").write_text(
        json.dumps(
            {
                "name": "Mvoice2",
                "gender": "male",
                "reference_audio": "reference.wav",
                "source_audio": "source.wav",
                "duration": 12.0,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "settings.json").write_text(
        json.dumps(
            {
                "voice_gender": "Auto male/female TTS with emotion",
                "voice_female": "Fvoice1 (generated)",
                "voice_male": "Mvoice2 (generated)",
                "clone_workflow": "gender_profiles",
                "rvc_enabled": True,
                "clone_backend": "Qwen3-TTS 1.7B (best clone + emotion)",
            }
        ),
        encoding="utf-8",
    )

    defaults = _settings_defaults_from_saved_ui(tmp_path)

    assert defaults["voice_gender"] == "auto"
    assert defaults["voice_female"] == "km-KH-SreymomNeural"
    assert defaults["voice_male"] == "km-KH-PisethNeural"
    assert defaults["voice_female_reference_path"] == female_dir / "reference.wav"
    assert defaults["voice_male_reference_path"] == male_dir / "reference.wav"
    assert defaults["clone_workflow"] == "gender_profiles"
    assert defaults["rvc_enabled"] is True
    assert defaults["clone_backend"] == "qwen3"


def test_auto_dubber_uses_saved_edge_voice_names(tmp_path: Path) -> None:
    (tmp_path / "settings.json").write_text(
        json.dumps(
            {
                "voice_gender": "Female TTS (single voice)",
                "voice_female": "en-US-AriaNeural",
                "voice_male": "en-US-GuyNeural",
            }
        ),
        encoding="utf-8",
    )

    defaults = _settings_defaults_from_saved_ui(tmp_path)

    assert defaults["voice_gender"] == "female"
    assert defaults["voice_female"] == "en-US-AriaNeural"
    assert defaults["voice_male"] == "en-US-GuyNeural"
    assert defaults["voice_female_reference_path"] is None
    assert defaults["voice_male_reference_path"] is None
