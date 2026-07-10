from __future__ import annotations

import sys
import types
from pathlib import Path
from threading import Event

import numpy as np

from conftest import write_wav
from core.context import Segment
from modules.diarizer import SpeakerTurn
from modules.gender_classifier import SpeechBrainGenderUnavailableError, classify_genders


def _install_fake_dependencies(monkeypatch, fake_pipeline):
    transformers = types.SimpleNamespace(pipeline=fake_pipeline)
    torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: False,
            empty_cache=lambda: None,
        )
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(
        "modules.gender_classifier._load_speechbrain_gender_model",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            SpeechBrainGenderUnavailableError("forced fallback")
        ),
    )


def test_short_segments_are_skipped_without_loading_classifier(monkeypatch, tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    write_wav(audio, np.zeros(16000, dtype=np.float32), 16000)
    segments = [
        Segment(index=0, start=0.0, end=0.02, text="tiny", speaker_id="speaker_1"),
        Segment(index=1, start=0.02, end=0.08, text="tiny", speaker_id="speaker_1"),
    ]

    def fail_pipeline(*_args, **_kwargs):
        raise AssertionError("classifier should not load when every segment is too short")

    _install_fake_dependencies(monkeypatch, fail_pipeline)

    results = classify_genders(audio, segments, "cpu", None, None, Event())

    assert results == {0: "female", 1: "female"}


def test_batch_failure_retries_individual_segments_and_skips_tiny_clip(monkeypatch, tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    samples = np.zeros(3 * 16000, dtype=np.float32)
    write_wav(audio, samples, 16000)
    segments = [
        Segment(index=0, start=0.0, end=1.0, text="hello", speaker_id="speaker_1"),
        Segment(index=1, start=1.0, end=1.02, text="tiny", speaker_id="speaker_1"),
        Segment(index=2, start=1.1, end=2.1, text="world", speaker_id="speaker_1"),
    ]

    calls: list[int] = []

    class FakeClassifier:
        def __call__(self, inputs, batch_size=16):
            calls.append(len(inputs))
            if len(inputs) > 1:
                raise RuntimeError("Calculated padded input size per channel: (1). Kernel size: (2).")
            return [[{"label": "male", "score": 0.99}]]

    def fake_pipeline(*_args, **_kwargs):
        return FakeClassifier()

    _install_fake_dependencies(monkeypatch, fake_pipeline)

    logs: list[str] = []
    results = classify_genders(audio, segments, "cpu", None, logs.append, Event())

    assert calls == [2, 1, 1]
    assert results == {0: "male", 1: "male", 2: "male"}
    assert any("batch failed" in line for line in logs)
    assert any("too short" in line for line in logs)


def test_speechbrain_gender_uses_diarized_speaker_turns(monkeypatch, tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    samples = np.concatenate(
        [
            np.full(16000, 0.1, dtype=np.float32),
            np.full(16000, -0.1, dtype=np.float32),
        ]
    )
    write_wav(audio, samples, 16000)
    segments = [
        Segment(index=0, start=0.0, end=0.8, text="a", speaker_id="speaker_1"),
        Segment(index=1, start=1.0, end=1.8, text="b", speaker_id="speaker_2"),
    ]
    turns = [
        SpeakerTurn(start=0.0, end=1.0, speaker_id="speaker_1"),
        SpeakerTurn(start=1.0, end=2.0, speaker_id="speaker_2"),
    ]

    monkeypatch.setattr(
        "modules.gender_classifier._load_speechbrain_gender_model",
        lambda *_args, **_kwargs: object(),
    )

    def fake_predict(_model, clip, _samplerate):
        if float(np.mean(clip)) >= 0:
            return "female", 0.91
        return "male", 0.08

    monkeypatch.setattr("modules.gender_classifier._predict_speechbrain_gender", fake_predict)

    logs: list[str] = []
    results = classify_genders(audio, segments, "cpu", None, logs.append, Event(), turns)

    assert results == {0: "female", 1: "male"}
    assert any("pyannote speaker turns" in line for line in logs)


def test_log_gender_emotion_summary_reports_combined_counts():
    from modules.emotion_detector import EmotionAnalysis
    from modules.gender_classifier import log_gender_emotion_summary

    segments = [
        Segment(index=0, start=0.0, end=1.0, text="a"),
        Segment(index=1, start=1.0, end=2.0, text="b"),
    ]
    logs: list[str] = []
    log_gender_emotion_summary(
        segments,
        {0: "female", 1: "male"},
        {
            0: EmotionAnalysis(
                label="excited",
                instruct_text="",
                confidence=0.9,
                energy=0.8,
                pacing_offset_pct=10,
                pitch_offset_hz=8,
                is_neutral_fallback=False,
            ),
            1: EmotionAnalysis(
                label="neutral",
                instruct_text="",
                confidence=0.2,
                energy=0.1,
                pacing_offset_pct=0,
                pitch_offset_hz=0,
                is_neutral_fallback=True,
            ),
        },
        logs.append,
    )
    assert any("female+excited=1" in line for line in logs)
    assert any("male+neutral=1" in line for line in logs)
