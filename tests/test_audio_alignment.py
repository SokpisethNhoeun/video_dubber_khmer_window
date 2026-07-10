from __future__ import annotations

import shutil
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest

from core.context import Segment
from modules import audio_utils


def _segment(index: int, start: float, end: float, tmp_path: Path, enabled: bool = True) -> Segment:
    speech_path = tmp_path / f"speech_{index}.wav"
    speech_path.write_bytes(b"fake")
    return Segment(index=index, start=start, end=end, text=f"segment {index}", enabled=enabled, tts_path=speech_path)


def test_align_audio_segments_preserves_original_cursor_and_skips_disabled_segments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    segments = [
        _segment(0, 0.5, 1.5, tmp_path),
        _segment(1, 1.5, 2.5, tmp_path, enabled=False),
        _segment(2, 3.0, 4.0, tmp_path),
    ]
    converted: list[Path] = []
    fit_calls: list[tuple[str, float, float]] = []
    silences: list[tuple[str, float]] = []
    concat_pieces: list[str] = []

    monkeypatch.setattr(audio_utils, "ensure_ffmpeg", lambda: None)
    monkeypatch.setattr(
        audio_utils,
        "convert_to_wav",
        lambda _input, output, _cancel: converted.append(output) or output.write_bytes(b"raw"),
    )
    def mock_ffprobe(path):
        if "aligned" in path.name:
            return 0.6 if "00000" in path.name else 0.6
        return 1.6 if "00000" in path.name else 0.6

    monkeypatch.setattr(audio_utils, "ffprobe_duration", mock_ffprobe)

    def fake_fit(input_wav: Path, output_wav: Path, target_duration: float, max_speed: float, _cancel: Event):
        output_wav.write_bytes(b"aligned")
        fit_calls.append((input_wav.name, target_duration, max_speed))
        trim_required = input_wav.name.startswith("00000")
        return {
            "generated_duration": 1.6 if trim_required else 0.6,
            "target_duration": target_duration,
            "speed_used": max_speed if trim_required else 1.0,
            "trim_required": trim_required,
            "trim_duration": 0.481 if trim_required else 0.0,
            "adjusted_duration": 1.481 if trim_required else 0.6,
        }

    def fake_silence(duration: float, output_wav: Path, _cancel: Event) -> None:
        output_wav.write_bytes(b"silence")
        silences.append((output_wav.name, round(duration, 3)))

    def fake_concat(wav_files: list[Path], output_wav: Path, _concat_file: Path, _cancel: Event) -> None:
        concat_pieces.extend(path.name for path in wav_files)
        output_wav.write_bytes(b"final")

    monkeypatch.setattr(audio_utils, "fit_audio_to_duration", fake_fit)
    monkeypatch.setattr(audio_utils, "make_silence", fake_silence)
    monkeypatch.setattr(audio_utils, "concat_wavs", fake_concat)

    quality_report = SimpleNamespace(long_segments=[], timing_segments=[])
    output_wav = tmp_path / "final.wav"

    audio_utils.align_audio_segments(
        segments,
        output_wav,
        tmp_path,
        total_duration=5.0,
        progress_cb=None,
        log_cb=None,
        cancel_event=Event(),
        mode="natural",
        quality_report=quality_report,
    )

    assert [path.name for path in converted] == ["00000_raw.wav", "00002_raw.wav"]
    # Segment 0: 1.6s audio, 1.0s slot — sped_duration (1.6/1.6=1.0) is NOT > target (1.0), no extension
    assert fit_calls[0] == ("00000_raw.wav", pytest.approx(1.0, abs=0.01), 1.6)
    assert fit_calls[1] == ("00002_raw.wav", 1.0, 1.6)
    # Cursor follows original segment.end, not the probed aligned duration.
    assert silences[0] == ("00000_gap.wav", 0.5)
    assert silences[1][0] == "00002_gap.wav"
    assert silences[1][1] == pytest.approx(1.5, abs=0.02)
    assert len(quality_report.timing_segments) == 2


def test_strict_alignment_allows_more_speed_before_trimming(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    segment = _segment(0, 0.0, 1.0, tmp_path)
    fit_calls: list[float] = []

    monkeypatch.setattr(audio_utils, "ensure_ffmpeg", lambda: None)
    monkeypatch.setattr(
        audio_utils,
        "convert_to_wav",
        lambda _input, output, _cancel: output.write_bytes(b"raw"),
    )
    monkeypatch.setattr(audio_utils, "ffprobe_duration", lambda _path: 1.7)
    monkeypatch.setattr(audio_utils, "make_silence", lambda *_args: None)
    monkeypatch.setattr(
        audio_utils,
        "concat_wavs",
        lambda _files, output, _concat, _cancel: output.write_bytes(b"final"),
    )

    def fake_fit(_input_wav: Path, output_wav: Path, _target_duration: float, max_speed: float, _cancel: Event):
        output_wav.write_bytes(b"aligned")
        fit_calls.append(max_speed)
        return {
            "generated_duration": 1.7,
            "target_duration": 1.0,
            "speed_used": max_speed,
            "trim_required": False,
            "trim_duration": 0.0,
            "adjusted_duration": 1.0,
        }

    monkeypatch.setattr(audio_utils, "fit_audio_to_duration", fake_fit)

    audio_utils.align_audio_segments(
        [segment],
        tmp_path / "final.wav",
        tmp_path,
        total_duration=1.0,
        progress_cb=None,
        log_cb=None,
        cancel_event=Event(),
        mode="strict",
    )

    assert fit_calls == [1.8]


def test_fit_audio_to_duration_normalizes_voice_before_padding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    monkeypatch.setattr(audio_utils, "ffprobe_duration", lambda _path: 0.6)

    def fake_run(command: list[str], _cancel: Event) -> None:
        commands.append(command)
        Path(command[-1]).write_bytes(b"fit")

    monkeypatch.setattr(audio_utils, "_run_checked", fake_run)

    audio_utils.fit_audio_to_duration(
        tmp_path / "input.wav",
        tmp_path / "output.wav",
        1.0,
        1.6,
        Event(),
    )

    filter_arg = commands[0][commands[0].index("-af") + 1]
    assert "dynaudnorm" in filter_arg
    assert "alimiter=limit=0.95" in filter_arg
    assert filter_arg.endswith("apad,atrim=0:1.000")


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="ffmpeg is required")
def test_fit_audio_to_duration_speeds_long_audio_and_pads_short_audio(tmp_path: Path) -> None:
    cancel_event = Event()
    long_input = tmp_path / "long.wav"
    long_output = tmp_path / "long_fit.wav"
    short_input = tmp_path / "short.wav"
    short_output = tmp_path / "short_fit.wav"

    audio_utils.make_silence(1.6, long_input, cancel_event)
    long_result = audio_utils.fit_audio_to_duration(long_input, long_output, 1.0, 1.6, cancel_event)

    # No trim: audio is sped up to 1.6x (1.6/1.6=1.25s), output keeps full content
    assert long_result["speed_used"] == pytest.approx(1.6, abs=0.001)
    output_dur = audio_utils.ffprobe_duration(long_output)
    assert output_dur >= 1.0  # at least target, might be slightly longer (no trim)

    audio_utils.make_silence(0.4, short_input, cancel_event)
    short_result = audio_utils.fit_audio_to_duration(short_input, short_output, 1.2, 1.6, cancel_event)

    assert audio_utils.ffprobe_duration(short_output) == pytest.approx(1.2, abs=0.04)
    assert short_result["speed_used"] == pytest.approx(1.0, abs=0.001)
    assert short_result["trim_required"] is False


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="ffmpeg is required")
def test_align_audio_segments_final_wav_matches_source_duration_with_long_lines(tmp_path: Path) -> None:
    cancel_event = Event()
    first_speech = tmp_path / "first_speech.wav"
    second_speech = tmp_path / "second_speech.wav"
    audio_utils.make_silence(2.4, first_speech, cancel_event)
    audio_utils.make_silence(0.2, second_speech, cancel_event)
    segments = [
        Segment(index=0, start=0.5, end=1.5, text="long", tts_path=first_speech),
        Segment(index=1, start=2.5, end=3.0, text="short", tts_path=second_speech),
    ]
    quality_report = SimpleNamespace(long_segments=[], timing_segments=[])
    output_wav = tmp_path / "final.wav"

    audio_utils.align_audio_segments(
        segments,
        output_wav,
        tmp_path,
        total_duration=4.0,
        progress_cb=None,
        log_cb=None,
        cancel_event=cancel_event,
        mode="natural",
        quality_report=quality_report,
        shorten_pauses=False,
    )

    assert audio_utils.ffprobe_duration(output_wav) == pytest.approx(4.0, abs=0.1)
    # Segment 0: 2.4s audio, 1.0s slot. It must not extend into the following gap.
    assert quality_report.timing_segments[0]["trim_required"] is True
    assert quality_report.timing_segments[1]["speed_adjustment"] == pytest.approx(1.0, abs=0.001)
