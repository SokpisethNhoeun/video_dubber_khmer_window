from __future__ import annotations

from pathlib import Path
from threading import Event

import pytest

from core.context import Segment
from modules.diarizer import SpeakerTurn
from modules.emotion_reference import EmotionClip, extract_emotion_clips


def _segment(index: int, start: float, end: float, speaker_id: str = "speaker_1") -> Segment:
    return Segment(index=index, start=start, end=end, text=f"seg {index}", speaker_id=speaker_id)


def _turn(start: float, end: float, speaker_id: str = "speaker_1") -> SpeakerTurn:
    return SpeakerTurn(start=start, end=end, speaker_id=speaker_id)


class TestExtractEmotionClips:
    def test_basic_extraction(self, tmp_path, monkeypatch):
        source_wav = tmp_path / "source.wav"
        source_wav.write_bytes(b"fake")
        segments = [_segment(0, 1.0, 3.5), _segment(1, 5.0, 7.0)]
        turns = [_turn(0.5, 4.0), _turn(4.5, 7.5)]
        trim_calls: list[tuple] = []

        def mock_trim(src, out, start, duration, cancel):
            trim_calls.append((start, duration))
            out.write_bytes(b"clip")

        def mock_duration(path):
            return 2.5

        def mock_snr(path):
            return 20.0

        monkeypatch.setattr("modules.emotion_reference.trim_audio_segment", mock_trim)
        monkeypatch.setattr("modules.emotion_reference.ffprobe_duration", mock_duration)
        monkeypatch.setattr("modules.emotion_reference._quick_snr", mock_snr)

        clips = extract_emotion_clips(
            source_wav, segments, turns, tmp_path / "clips", Event(),
        )

        assert len(clips) == 2
        assert clips[0].usable is True
        assert clips[1].usable is True
        assert trim_calls[0][0] == pytest.approx(0.65)
        assert trim_calls[0][1] == pytest.approx(3.2)

    def test_short_segment_extends_to_turn(self, tmp_path, monkeypatch):
        source_wav = tmp_path / "source.wav"
        source_wav.write_bytes(b"fake")
        segments = [_segment(0, 2.0, 2.5)]
        turns = [_turn(1.0, 4.0)]
        trim_calls: list[tuple] = []

        def mock_trim(src, out, start, duration, cancel):
            trim_calls.append((start, duration))
            out.write_bytes(b"clip")

        monkeypatch.setattr("modules.emotion_reference.trim_audio_segment", mock_trim)
        monkeypatch.setattr("modules.emotion_reference.ffprobe_duration", lambda p: 3.0)
        monkeypatch.setattr("modules.emotion_reference._quick_snr", lambda p: 15.0)

        clips = extract_emotion_clips(
            source_wav, segments, turns, tmp_path / "clips", Event(),
        )

        assert clips[0].usable is True
        assert trim_calls[0][0] == pytest.approx(1.0)
        assert trim_calls[0][1] == pytest.approx(3.0)

    def test_clip_too_short_marked_unusable(self, tmp_path, monkeypatch):
        source_wav = tmp_path / "source.wav"
        source_wav.write_bytes(b"fake")
        segments = [_segment(0, 1.0, 1.3)]

        clips = extract_emotion_clips(
            source_wav, segments, [], tmp_path / "clips", Event(),
        )

        assert clips[0].usable is False
        assert "too short" in clips[0].fallback_reason

    def test_clip_low_snr_marked_unusable(self, tmp_path, monkeypatch):
        source_wav = tmp_path / "source.wav"
        source_wav.write_bytes(b"fake")
        segments = [_segment(0, 1.0, 4.0)]

        def mock_trim(src, out, start, duration, cancel):
            out.write_bytes(b"clip")

        monkeypatch.setattr("modules.emotion_reference.trim_audio_segment", mock_trim)
        monkeypatch.setattr("modules.emotion_reference.ffprobe_duration", lambda p: 3.0)
        monkeypatch.setattr("modules.emotion_reference._quick_snr", lambda p: 3.0)

        clips = extract_emotion_clips(
            source_wav, segments, [], tmp_path / "clips", Event(),
        )

        assert clips[0].usable is False
        assert "SNR" in clips[0].fallback_reason

    def test_no_turns_uses_segment_times(self, tmp_path, monkeypatch):
        source_wav = tmp_path / "source.wav"
        source_wav.write_bytes(b"fake")
        segments = [_segment(0, 2.0, 5.0)]
        trim_calls: list[tuple] = []

        def mock_trim(src, out, start, duration, cancel):
            trim_calls.append((start, duration))
            out.write_bytes(b"clip")

        monkeypatch.setattr("modules.emotion_reference.trim_audio_segment", mock_trim)
        monkeypatch.setattr("modules.emotion_reference.ffprobe_duration", lambda p: 3.0)
        monkeypatch.setattr("modules.emotion_reference._quick_snr", lambda p: 20.0)

        clips = extract_emotion_clips(
            source_wav, segments, [], tmp_path / "clips", Event(),
        )

        assert clips[0].usable is True
        assert trim_calls[0][0] == pytest.approx(1.65)
        assert trim_calls[0][1] == pytest.approx(3.7)

    def test_context_padding_stays_inside_speaker_turn(self, tmp_path, monkeypatch):
        source_wav = tmp_path / "source.wav"
        source_wav.write_bytes(b"fake")
        segments = [_segment(0, 2.0, 3.0)]
        turns = [_turn(1.9, 3.1)]
        trim_calls: list[tuple] = []

        def mock_trim(src, out, start, duration, cancel):
            trim_calls.append((start, duration))
            out.write_bytes(b"clip")

        monkeypatch.setattr("modules.emotion_reference.trim_audio_segment", mock_trim)
        monkeypatch.setattr("modules.emotion_reference.ffprobe_duration", lambda p: 1.2)
        monkeypatch.setattr("modules.emotion_reference._quick_snr", lambda p: 15.0)

        clips = extract_emotion_clips(
            source_wav, segments, turns, tmp_path / "clips", Event(),
        )

        assert clips[0].usable is True
        assert trim_calls[0][0] == pytest.approx(1.9)
        assert trim_calls[0][1] == pytest.approx(1.2)

    def test_cancellation(self, tmp_path):
        source_wav = tmp_path / "source.wav"
        source_wav.write_bytes(b"fake")
        segments = [_segment(0, 1.0, 3.0)]
        cancel = Event()
        cancel.set()

        with pytest.raises(Exception, match="[Cc]ancell"):
            extract_emotion_clips(
                source_wav, segments, [], tmp_path / "clips", cancel,
            )

    def test_disabled_segments_skipped(self, tmp_path, monkeypatch):
        source_wav = tmp_path / "source.wav"
        source_wav.write_bytes(b"fake")
        seg = _segment(0, 1.0, 4.0)
        seg.enabled = False
        segments = [seg]

        clips = extract_emotion_clips(
            source_wav, segments, [], tmp_path / "clips", Event(),
        )

        assert len(clips) == 0

    def test_long_clip_clamped(self, tmp_path, monkeypatch):
        source_wav = tmp_path / "source.wav"
        source_wav.write_bytes(b"fake")
        segments = [_segment(0, 0.5, 0.8)]
        turns = [_turn(0.0, 20.0)]
        trim_calls: list[tuple] = []

        def mock_trim(src, out, start, duration, cancel):
            trim_calls.append((round(start, 2), round(duration, 2)))
            out.write_bytes(b"clip")

        monkeypatch.setattr("modules.emotion_reference.trim_audio_segment", mock_trim)
        monkeypatch.setattr("modules.emotion_reference.ffprobe_duration", lambda p: 8.0)
        monkeypatch.setattr("modules.emotion_reference._quick_snr", lambda p: 20.0)

        clips = extract_emotion_clips(
            source_wav, segments, turns, tmp_path / "clips", Event(),
        )

        assert clips[0].usable is True
        assert trim_calls[0][1] <= 8.0
