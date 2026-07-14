from __future__ import annotations

from dataclasses import dataclass


NLLB_MODEL_ID = "facebook/nllb-200-distilled-600M"
FASTER_WHISPER_MODEL_PREFIX = "Systran/faster-whisper"


@dataclass(frozen=True)
class LanguageConfig:
    label: str
    whisper_code: str
    nllb_code: str


LANGUAGES: dict[str, LanguageConfig] = {
    "zh": LanguageConfig("Chinese", "zh", "zho_Hans"),
    "km": LanguageConfig("Khmer", "km", "khm_Khmr"),
    "en": LanguageConfig("English", "en", "eng_Latn"),
}

TARGET_LANGUAGE = LanguageConfig("Khmer", "km", "khm_Khmr")

EDGE_TTS_VOICES = {
    "female": "km-KH-SreymomNeural",
    "male": "km-KH-PisethNeural",
}

WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v3"]
DEFAULT_WHISPER_MODEL = "medium"

STAGES = [
    ("extract_audio", "Extract Audio"),
    ("transcription", "Transcription"),
    ("speaker_detection", "Speaker Detection"),
    ("translation", "Translation"),
    ("transcript_review", "Transcript Review"),
    ("gender_detection", "Gender Detection"),
    ("tts", "TTS"),
    ("voice_clone", "Finalize Voice"),
    ("alignment", "Alignment"),
    ("muxing", "Muxing"),
]
