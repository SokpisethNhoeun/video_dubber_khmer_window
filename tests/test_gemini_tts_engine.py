from __future__ import annotations

import base64
import wave
from pathlib import Path
from threading import Event
from types import SimpleNamespace

from core.context import Segment
from modules.gemini_tts_engine import (
    MAX_REQUESTS_PER_VIDEO,
    GeminiTTSNoAudioError,
    GeminiTTSRateLimitError,
    _call_with_retries,
    _extract_audio_data,
    _parse_retry_delay,
    _request_chunk_audio,
    _resolve_max_requests,
    build_gemini_tts_prompt,
    build_gemini_payload,
    gemini_voices_from_mappings,
    group_segments_for_gemini,
    resolve_gemini_api_keys,
    speaker_voice_map,
    synthesize_gemini_tts,
    write_pcm_wav,
)
from modules.tts_engine import synthesize_tts


def _khmer_segment(index: int, speaker_id: str = "speaker_1") -> Segment:
    return Segment(
        index=index,
        start=float(index),
        end=float(index + 1),
        text=f"source {index}",
        translated_text=f"សួស្តី {index}",
        speaker_id=speaker_id,
        speaker_label=speaker_id.replace("_", " ").title(),
    )


def _fake_audio_response() -> dict:
    pcm = b"\x00\x00" * 240
    return {
        "output_audio": {
            "mime_type": "audio/pcm",
            "data": base64.b64encode(pcm).decode("ascii"),
        }
    }


def _fake_mime_only_audio_response() -> dict:
    pcm = b"\x03\x04" * 120
    return {
        "id": "int_mime_only",
        "status": "completed",
        "steps": [
            {
                "content": [
                    {
                        "mime_type": "audio/l16",
                        "data": base64.b64encode(pcm).decode("ascii"),
                    }
                ],
            }
        ],
    }


def _fake_steps_audio_response() -> dict:
    pcm = b"\x01\x02" * 120
    return {
        "id": "int_test",
        "status": "completed",
        "steps": [
            {
                "type": "model_output",
                "content": [
                    {
                        "type": "audio",
                        "mime_type": "audio/l16; rate=24000; channels=1",
                        "data": base64.b64encode(pcm).decode("ascii"),
                    }
                ],
            }
        ],
    }


def test_group_segments_for_gemini_never_exceeds_max_requests() -> None:
    segments = [_khmer_segment(i) for i in range(23)]

    chunks = group_segments_for_gemini(segments, max_requests=15)

    assert len(chunks) <= 15
    assert sum(len(chunk.segments) for chunk in chunks) == 23


def test_resolve_max_requests_caps_at_fifteen_per_video(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_TTS_MAX_REQUESTS", "99")
    assert _resolve_max_requests(None) == MAX_REQUESTS_PER_VIDEO
    assert _resolve_max_requests(20) == MAX_REQUESTS_PER_VIDEO
    assert _resolve_max_requests(8) == 8


def test_parse_retry_delay_reads_google_quota_message() -> None:
    detail = (
        'Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests. '
        "Please retry in 42.5s."
    )
    assert _parse_retry_delay(detail) == 42.5


def test_call_with_retries_waits_and_recovers_from_rate_limit(monkeypatch) -> None:
    attempts = {"count": 0}
    sleeps: list[float] = []

    def fake_transport(payload, api_key, model):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise GeminiTTSRateLimitError("rate limited", retry_after_sec=2.0)
        return _fake_steps_audio_response()

    monkeypatch.setattr("modules.gemini_tts_engine._sleep_with_cancel", lambda seconds, _event: sleeps.append(seconds))

    response, used_key = _call_with_retries(
        {"model": "gemini-test"},
        ["test-key"],
        "gemini-test",
        Event(),
        fake_transport,
        log_cb=None,
    )

    assert attempts["count"] == 2
    assert sleeps == [2.0]
    assert used_key == "test-key"
    assert response["status"] == "completed"


def test_call_with_retries_switches_to_fallback_key_on_rate_limit(monkeypatch) -> None:
    attempts: list[str] = []

    def fake_transport(payload, api_key, model):
        attempts.append(api_key)
        if api_key == "primary":
            raise GeminiTTSRateLimitError("rate limited", retry_after_sec=2.0)
        return _fake_steps_audio_response()

    response, used_key = _call_with_retries(
        {"model": "gemini-test"},
        ["primary", "fallback"],
        "gemini-test",
        Event(),
        fake_transport,
        log_cb=None,
    )

    assert attempts == ["primary", "fallback"]
    assert used_key == "fallback"
    assert response["status"] == "completed"


def test_resolve_gemini_api_keys_supports_fallback_env(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "primary")
    monkeypatch.setenv("GEMINI_API_KEY_FALLBACK", "fallback")
    assert resolve_gemini_api_keys() == ["primary", "fallback"]


def test_extract_audio_data_reads_mime_only_steps_schema() -> None:
    pcm, sample_rate = _extract_audio_data(_fake_mime_only_audio_response())

    assert sample_rate == 24000
    assert len(pcm) == 240


def test_request_chunk_audio_switches_key_when_response_has_no_audio(monkeypatch) -> None:
    attempts: list[str] = []

    def fake_transport(payload, api_key, model):
        attempts.append(api_key)
        if api_key == "primary":
            return {"status": "completed", "steps": []}
        return _fake_steps_audio_response()

    monkeypatch.setattr("modules.gemini_tts_engine._sleep_with_cancel", lambda *_args: None)

    pcm, sample_rate = _request_chunk_audio(
        {"model": "gemini-test"},
        ["primary", "fallback"],
        "gemini-test",
        Event(),
        fake_transport,
        log_cb=None,
    )

    assert attempts == ["primary", "fallback"]
    assert sample_rate == 24000
    assert len(pcm) == 240


def test_group_segments_for_gemini_prefers_two_configured_speakers_per_chunk() -> None:
    segments = [
        _khmer_segment(0, "speaker_a"),
        _khmer_segment(1, "speaker_b"),
        _khmer_segment(2, "speaker_c"),
        _khmer_segment(3, "speaker_c"),
    ]

    chunks = group_segments_for_gemini(segments, max_requests=10)

    assert [segment.speaker_id for chunk in chunks for segment in chunk.segments] == [
        "speaker_a",
        "speaker_b",
        "speaker_c",
        "speaker_c",
    ]
    assert all(len({segment.speaker_id for segment in chunk.segments}) <= 2 for chunk in chunks)


def test_gemini_prompt_includes_speaker_text_timing_and_audio_only_instruction() -> None:
    chunk = group_segments_for_gemini(
        [_khmer_segment(0, "speaker_a"), _khmer_segment(1, "speaker_b")],
        max_requests=1,
    )[0]

    prompt = build_gemini_tts_prompt(chunk, {"speaker_a": "Kore", "speaker_b": "Puck"})

    assert "Synthesize speech only" in prompt
    assert "Speak only the Khmer dialogue exactly as written after each emotion tag" in prompt
    assert "professional film/video dubbing" in prompt
    assert "Audio profile" in prompt
    assert "Scene and director notes" in prompt
    assert "Cambodian Khmer sentence stress" in prompt
    assert "The tags are performance controls only; never speak them" in prompt
    assert "[native Cambodian Khmer, mature adult 25-40, grounded conversational tone, emotionally natural, medium pace]" in prompt
    assert "mature adult (25-40 years old)" in prompt
    assert "Age target" in prompt
    assert "robotic TTS" in prompt
    assert "not cloned voices" in prompt
    assert "Line delivery notes" in prompt
    assert "[0.00-1.00]" in prompt
    assert "Speaker A" in prompt
    assert "Speaker B" in prompt
    assert "- Speaker A: Gemini voice Kore" in prompt
    assert "- Speaker B: Gemini voice Puck" in prompt
    assert "Speaker A (Kore):" not in prompt
    assert "សួស្តី 0" in prompt
    assert "សួស្តី 1" in prompt


def test_gemini_prompt_uses_detected_emotion_notes() -> None:
    chunk = group_segments_for_gemini([_khmer_segment(0, "speaker_a")], max_requests=1)[0]
    emotion = SimpleNamespace(
        label="angry",
        instruct_text="Speak firmly with intensity and controlled anger",
        confidence=0.88,
        is_neutral_fallback=False,
    )

    prompt = build_gemini_tts_prompt(chunk, {"speaker_a": "Kore"}, {0: emotion})

    assert "angry delivery, confidence 0.88" in prompt
    assert "controlled anger" in prompt
    assert "[angry, sharp consonants, controlled intensity" in prompt
    assert "for direction only, do not speak these notes" in prompt


def test_gemini_payload_uses_current_interactions_tts_shape() -> None:
    chunk = group_segments_for_gemini(
        [_khmer_segment(0, "speaker_a"), _khmer_segment(1, "speaker_b")],
        max_requests=1,
    )[0]

    payload = build_gemini_payload(
        chunk,
        {"speaker_a": "Kore", "speaker_b": "Puck"},
        model="gemini-test",
    )

    assert payload["model"] == "gemini-test"
    assert payload["response_format"] == {"type": "audio"}
    assert "contents" not in payload
    assert "generationConfig" not in payload
    assert "សួស្តី 0" in payload["input"]
    assert payload["generation_config"]["speech_config"] == [
        {"speaker": "Speaker A", "voice": "Kore"},
        {"speaker": "Speaker B", "voice": "Puck"},
    ]


def test_write_pcm_wav_creates_readable_wav(tmp_path: Path) -> None:
    output = write_pcm_wav(tmp_path / "gemini.wav", b"\x00\x00" * 120, sample_rate=24000)

    with wave.open(str(output), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 24000
        assert wav.getnframes() == 120


def test_extract_audio_data_reads_interactions_steps_schema() -> None:
    pcm, sample_rate = _extract_audio_data(_fake_steps_audio_response())

    assert sample_rate == 24000
    assert len(pcm) == 240


def test_synthesize_gemini_tts_accepts_interactions_steps_schema(tmp_path: Path) -> None:
    def fake_transport(payload, api_key, model):
        return _fake_steps_audio_response()

    segments = [_khmer_segment(0), _khmer_segment(1)]
    result = synthesize_gemini_tts(
        segments,
        tmp_path,
        None,
        None,
        Event(),
        api_key="test-key",
        model="gemini-test",
        max_requests=1,
        transport=fake_transport,
    )

    assert all(segment.tts_path and segment.tts_path.exists() for segment in result)
    assert {segment.tts_group_id for segment in result} == {"gemini_chunk_000"}


def test_synthesize_gemini_tts_sets_chunk_paths_and_group_ids(tmp_path: Path) -> None:
    calls: list[dict] = []

    def fake_transport(payload, api_key, model):
        calls.append(payload)
        assert api_key == "test-key"
        assert model == "gemini-test"
        return _fake_audio_response()

    segments = [_khmer_segment(i) for i in range(4)]
    result = synthesize_gemini_tts(
        segments,
        tmp_path,
        None,
        None,
        Event(),
        api_key="test-key",
        model="gemini-test",
        max_requests=2,
        transport=fake_transport,
        emotion_analyses={
            0: SimpleNamespace(
                label="sad",
                instruct_text="Speak softly with gentle sadness",
                confidence=0.91,
                is_neutral_fallback=False,
            )
        },
    )

    assert len(calls) == 2
    assert calls[0]["model"] == "gemini-test"
    assert calls[0]["response_format"] == {"type": "audio"}
    assert "sad delivery, confidence 0.91" in calls[0]["input"]
    assert all(segment.tts_path and segment.tts_path.exists() for segment in result)
    assert {segment.tts_group_id for segment in result} == {"gemini_chunk_000", "gemini_chunk_001"}


def test_synthesize_tts_dispatches_to_gemini(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_TTS_MODEL", "gemini-test")
    monkeypatch.setenv("GEMINI_TTS_MAX_REQUESTS", "1")
    monkeypatch.setattr("modules.gemini_tts_engine.call_gemini_tts", lambda *_args, **_kwargs: _fake_audio_response())

    segments = [_khmer_segment(0), _khmer_segment(1)]
    result = synthesize_tts(
        segments,
        "auto",
        0,
        0,
        tmp_path,
        None,
        None,
        Event(),
        tts_provider="gemini",
    )

    assert all(segment.tts_path and segment.tts_path.exists() for segment in result)
    assert {segment.tts_group_id for segment in result} == {"gemini_chunk_000"}


def test_speaker_voice_map_assigns_distinct_voices_per_speaker() -> None:
    segments = [
        _khmer_segment(0, "speaker_b"),
        _khmer_segment(1, "speaker_a"),
        _khmer_segment(2, "speaker_b"),
    ]

    voices = speaker_voice_map(segments)

    assert voices["speaker_b"] != voices["speaker_a"]
    assert len({voices["speaker_a"], voices["speaker_b"]}) == 2


def test_speaker_voice_map_reuses_saved_voices() -> None:
    segments = [_khmer_segment(0, "speaker_a"), _khmer_segment(1, "speaker_b")]

    voices = speaker_voice_map(segments, saved_voices={"speaker_a": "Puck", "speaker_b": "Kore"})

    assert voices == {"speaker_a": "Puck", "speaker_b": "Kore"}


def test_speaker_voice_map_prefers_mature_adult_presets() -> None:
    segments = [_khmer_segment(0, "speaker_a"), _khmer_segment(1, "speaker_b")]

    voices = speaker_voice_map(segments)

    assert voices["speaker_a"] == "Gacrux"
    assert voices["speaker_b"] == "Charon"


def test_gemini_voices_from_mappings_reads_persisted_preset() -> None:
    mappings = {
        "speaker_1": {"gemini_voice": "Kore", "label": "Speaker 1"},
        "speaker_2": {"gemini_voice": "invalid", "label": "Speaker 2"},
    }

    assert gemini_voices_from_mappings(mappings) == {"speaker_1": "Kore"}
