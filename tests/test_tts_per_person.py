from __future__ import annotations

from pathlib import Path
from threading import Event

from core.context import Segment
from modules.speaker_references import speaker_has_clone_reference
from modules.tts_engine import _synthesize_all


def test_speaker_has_clone_reference_respects_missing_status():
    mappings = {
        "spk_1": {
            "reference_audio_path": "/tmp/ref.wav",
            "reference_status": "missing",
        },
        "spk_2": {
            "reference_audio_path": __file__,
            "reference_status": "auto",
            "quality_tier": "good",
        },
    }
    assert speaker_has_clone_reference("spk_1", mappings) is False
    assert speaker_has_clone_reference("spk_2", mappings) is True


async def _collect_voices(monkeypatch, tmp_path, voice_gender):
    voices: list[str] = []

    async def fake_synthesize_one(segment, voice, rate, pitch, output_path, semaphore, cancel_event=None):
        voices.append(voice)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"mp3")
        segment.tts_path = output_path

    monkeypatch.setattr("modules.tts_engine._synthesize_one", fake_synthesize_one)

    segments = [
        Segment(0, 0.0, 1.0, "hello", translated_text="សួស្តី", speaker_id="spk_clone"),
        Segment(1, 1.0, 2.0, "world", translated_text="ពិភពលោក", speaker_id="spk_fallback"),
    ]
    cache_dir = tmp_path / "tts"

    await _synthesize_all(
        segments,
        voice_gender,
        "female-voice",
        "male-voice",
        segment_genders={0: "female", 1: "male"},
        base_rate_pct=0,
        base_pitch_hz=0,
        speaker_rate_profiles=None,
        emotion_analyses=None,
        cache_dir=cache_dir,
        cache_hits=None,
        progress_cb=None,
        log_cb=None,
        cancel_event=Event(),
    )
    return voices


def test_per_person_mode_uses_gender_voices_for_tts_fallback(monkeypatch, tmp_path):
    import asyncio

    voices = asyncio.run(_collect_voices(monkeypatch, tmp_path, "per_person_auto"))
    assert voices == ["female-voice", "male-voice"]


def test_auto_mode_uses_detected_gender_voices(monkeypatch, tmp_path):
    import asyncio

    voices = asyncio.run(_collect_voices(monkeypatch, tmp_path, "auto"))
    assert voices == ["female-voice", "male-voice"]
