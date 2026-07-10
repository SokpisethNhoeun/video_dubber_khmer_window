from __future__ import annotations

import json
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest

from core.context import Segment
from modules import voice_cloner
from modules.emotion_reference import EmotionClip
from modules.emotion_detector import EmotionAnalysis
from modules.qwen3_voice_clone import (
    Qwen3CloneError,
    _is_qwen3_noise_line,
    _qwen3_progress_from_line,
    _qwen3_python,
    clone_batch,
)


def test_qwen3_clone_error_is_runtime_error():
    assert issubclass(Qwen3CloneError, RuntimeError)


def test_qwen3_pad_token_warning_is_filtered_from_logs():
    assert _is_qwen3_noise_line("Setting `pad_token_id` to `eos_token_id`:2150 for open-end generation.")
    assert not _is_qwen3_noise_line("Qwen3-TTS batch complete: 1/1 succeeded")


def test_qwen3_progress_line_is_parsed():
    assert _qwen3_progress_from_line("QWEN3_PROGRESS 3/96 segment=2 ok=1") == (3, 96)
    assert _qwen3_progress_from_line("QWEN3_PROGRESS 120/96 segment=2 ok=1") == (96, 96)
    assert _qwen3_progress_from_line("Qwen3-TTS batch complete: 1/1 succeeded") is None


def test_qwen3_python_missing_env(monkeypatch):
    monkeypatch.delenv("QWEN3_TTS_PYTHON", raising=False)
    with pytest.raises(Qwen3CloneError, match="QWEN3_TTS_PYTHON"):
        _qwen3_python()


def test_qwen3_python_nonexistent_path(monkeypatch):
    monkeypatch.setenv("QWEN3_TTS_PYTHON", "/tmp/nonexistent_qwen3_python_xyz")
    with pytest.raises(Qwen3CloneError, match="does not exist"):
        _qwen3_python()


def test_clone_batch_manifest_includes_ref_text(monkeypatch, tmp_path):
    py = tmp_path / "python"
    py.write_text("#!/bin/sh\n")
    monkeypatch.setenv("QWEN3_TTS_PYTHON", str(py))
    ref = tmp_path / "speaker.wav"
    ref.write_bytes(b"ref")
    out = tmp_path / "out.wav"
    captured = {}

    class FakeProcess:
        stdout = None
        returncode = 0

        def wait(self, timeout=None):
            return 0

        def kill(self):
            pass

    def fake_run(cmd, **kwargs):
        assert cmd[:2] == [str(py), "-c"]
        return SimpleNamespace(returncode=0, stdout="/usr/bin/sox\n", stderr="")

    def fake_popen(cmd, **kwargs):
        manifest_path = Path(cmd[-1])
        captured["manifest_path"] = manifest_path
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        captured["manifest"] = manifest
        out.write_bytes(b"wav")
        results_path = manifest_path.with_suffix(".results.json")
        results_path.write_text(
            json.dumps([{"segment_index": 0, "ok": True}]),
            encoding="utf-8",
        )
        return FakeProcess()

    monkeypatch.setattr("modules.qwen3_voice_clone.subprocess.run", fake_run)
    monkeypatch.setattr("modules.qwen3_voice_clone.subprocess.Popen", fake_popen)

    results = clone_batch(
        [{
            "segment_index": 0,
            "text": "សួស្តី",
            "output_path": str(out),
            "ref_text": "你好",
            "speaker_reference_path": str(ref),
        }],
        ref,
    )
    assert results[0]["ok"] is True
    assert captured["manifest"]["segments"][0]["ref_text"] == "你好"
    assert "Qwen3-TTS-12Hz-1.7B-Base" in captured["manifest"]["model_name"]


def test_clone_batch_reports_subprocess_progress(monkeypatch, tmp_path):
    py = tmp_path / "python"
    py.write_text("#!/bin/sh\n")
    monkeypatch.setenv("QWEN3_TTS_PYTHON", str(py))
    ref = tmp_path / "speaker.wav"
    ref.write_bytes(b"ref")
    out = tmp_path / "out.wav"
    progress = []

    class FakeStdout:
        def __init__(self):
            self.lines = iter([
                "QWEN3_PROGRESS 1/2 segment=0 ok=1\n",
                "Setting `pad_token_id` to `eos_token_id`:2150 for open-end generation.\n",
                "QWEN3_PROGRESS 2/2 segment=1 ok=1\n",
            ])

        def readline(self):
            return next(self.lines, "")

    class FakeProcess:
        stdout = FakeStdout()
        returncode = 0

        def wait(self, timeout=None):
            return 0

        def kill(self):
            pass

    def fake_run(cmd, **kwargs):
        return SimpleNamespace(returncode=0, stdout="/usr/bin/sox\n", stderr="")

    def fake_popen(cmd, **kwargs):
        manifest_path = Path(cmd[-1])
        results_path = manifest_path.with_suffix(".results.json")
        results_path.write_text(
            json.dumps([{"segment_index": 0, "ok": True}, {"segment_index": 1, "ok": True}]),
            encoding="utf-8",
        )
        out.write_bytes(b"wav")
        return FakeProcess()

    monkeypatch.setattr("modules.qwen3_voice_clone.subprocess.run", fake_run)
    monkeypatch.setattr("modules.qwen3_voice_clone.subprocess.Popen", fake_popen)

    clone_batch(
        [
            {"segment_index": 0, "text": "សួស្តី", "output_path": str(out)},
            {"segment_index": 1, "text": "លា", "output_path": str(out)},
        ],
        ref,
        progress_cb=lambda done, total: progress.append((done, total)),
    )

    assert progress == [(1, 2), (2, 2)]


def test_clone_batch_requires_sox_in_qwen3_environment(monkeypatch, tmp_path):
    py = tmp_path / "python"
    py.write_text("#!/bin/sh\n")
    monkeypatch.setenv("QWEN3_TTS_PYTHON", str(py))

    def fake_run(cmd, **kwargs):
        assert cmd[:2] == [str(py), "-c"]
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("modules.qwen3_voice_clone.subprocess.run", fake_run)

    with pytest.raises(Qwen3CloneError, match="SoX.*required|sox.*available|Qwen3-TTS requires SoX"):
        clone_batch(
            [{
                "segment_index": 0,
                "text": "hello",
                "output_path": str(tmp_path / "out.wav"),
                "speaker_reference_path": str(tmp_path / "speaker.wav"),
            }],
            tmp_path / "speaker.wav",
        )


def test_per_person_qwen3_batches_all_speakers_in_one_model_load(monkeypatch, tmp_path):
    speaker_ref = tmp_path / "speaker.wav"
    tts_path = tmp_path / "tts.mp3"
    speaker_ref.write_bytes(b"speaker")
    tts_path.write_bytes(b"khmer")
    clone_dir = tmp_path / "clones"
    clone_dir.mkdir()
    segment = Segment(
        index=0, start=0.0, end=2.0, text="hello", translated_text="សួស្តី",
        speaker_id="spk_a", tts_path=tts_path,
    )
    mappings = {
        "spk_a": {"cleaned_reference_audio_path": str(speaker_ref), "quality_tier": "good"},
    }
    captured = {}

    def fake_clone_batch(batch_items, reference_path, log_cb=None, progress_cb=None):
        captured["items"] = batch_items
        captured["fallback"] = reference_path
        if progress_cb:
            progress_cb(1, 1)
        out = Path(batch_items[0]["output_path"])
        out.write_bytes(b"wav")
        return [{"segment_index": 0, "ok": True}]

    monkeypatch.setattr("modules.qwen3_voice_clone.clone_batch", fake_clone_batch)

    voice_cloner._clone_per_person_batch_qwen3(
        [segment],
        clone_dir,
        mappings,
        log_cb=None,
        quality_report=SimpleNamespace(voice_clone_failures=[]),
        cancel_event=Event(),
        progress_cb=None,
    )

    assert len(captured["items"]) == 1
    item = captured["items"][0]
    assert item["text"] == "សួស្តី"
    assert item["ref_text"] == "hello"
    assert item["prompt_cache_key"] == "spk_a"
    assert segment.cloned_path is not None


def test_per_person_qwen3_uses_emotion_clip_reference(monkeypatch, tmp_path):
    speaker_ref = tmp_path / "speaker.wav"
    emotion_ref = tmp_path / "emotion.wav"
    speaker_ref.write_bytes(b"speaker")
    emotion_ref.write_bytes(b"emotion")
    clone_dir = tmp_path / "clones"
    clone_dir.mkdir()
    segment = Segment(
        index=0, start=0.0, end=2.0, text="hello", translated_text="សួស្តី",
        speaker_id="spk_a",
    )
    mappings = {
        "spk_a": {"cleaned_reference_audio_path": str(speaker_ref), "quality_tier": "good"},
    }
    captured = {}

    def fake_clone_batch(batch_items, reference_path, log_cb=None, progress_cb=None):
        captured["items"] = batch_items
        if progress_cb:
            progress_cb(1, 1)
        out = Path(batch_items[0]["output_path"])
        out.write_bytes(b"wav")
        return [{"segment_index": 0, "ok": True}]

    monkeypatch.setattr("modules.qwen3_voice_clone.clone_batch", fake_clone_batch)

    voice_cloner._clone_per_person_batch_qwen3(
        [segment],
        clone_dir,
        mappings,
        log_cb=None,
        quality_report=SimpleNamespace(voice_clone_failures=[]),
        cancel_event=Event(),
        progress_cb=None,
        emotion_clips={
            0: EmotionClip(
                segment_index=0,
                clip_path=emotion_ref,
                duration=2.0,
                snr_db=20.0,
                usable=True,
                fallback_reason="",
            )
        },
        emotion_analyses={
            0: EmotionAnalysis(
                label="excited",
                instruct_text="Speak excitedly",
                confidence=0.9,
                energy=0.8,
                pacing_offset_pct=10,
                pitch_offset_hz=8,
                is_neutral_fallback=False,
            )
        },
    )

    item = captured["items"][0]
    assert Path(item["reference_path"]) == emotion_ref
    assert Path(item["speaker_reference_path"]) == speaker_ref
    assert item["emotion_ref_text"] == "hello"
    assert item["temperature"] > 0.7


def test_gender_specific_qwen3_clone_uses_gender_reference(monkeypatch, tmp_path):
    female_ref = tmp_path / "female.wav"
    tts_path = tmp_path / "tts.mp3"
    female_ref.write_bytes(b"female")
    tts_path.write_bytes(b"khmer")
    clone_dir = tmp_path / "clones"
    clone_dir.mkdir()
    segment = Segment(
        index=0,
        start=0.0,
        end=2.0,
        text="hello",
        translated_text="សួស្តី",
        tts_path=tts_path,
    )
    captured = {}

    def fake_clone_batch(batch_items, reference_path, log_cb=None, progress_cb=None):
        captured["items"] = batch_items
        captured["reference_path"] = reference_path
        if progress_cb:
            progress_cb(1, 1)
        out = Path(batch_items[0]["output_path"])
        out.write_bytes(b"wav")
        return [{"segment_index": 0, "ok": True}]

    monkeypatch.setattr("modules.qwen3_voice_clone.clone_batch", fake_clone_batch)

    voice_cloner._clone_gender_batch_qwen3(
        [segment],
        clone_dir,
        {"female": female_ref},
        log_cb=None,
        quality_report=SimpleNamespace(voice_clone_failures=[]),
        cancel_event=Event(),
        progress_cb=None,
        segment_genders={0: "female"},
    )

    assert captured["reference_path"] == female_ref
    assert captured["items"][0]["prompt_cache_key"] == "gender::female"
    assert segment.cloned_path is not None


def test_gender_specific_qwen3_failure_stops_without_fallback(monkeypatch, tmp_path):
    female_ref = tmp_path / "female.wav"
    tts_path = tmp_path / "tts.mp3"
    female_ref.write_bytes(b"female")
    tts_path.write_bytes(b"khmer")
    segment = Segment(
        index=0,
        start=0.0,
        end=2.0,
        text="hello",
        translated_text="សួស្តី",
        tts_path=tts_path,
    )

    def fail_clone_batch(_batch_items, _reference_path, log_cb=None, progress_cb=None):
        raise Qwen3CloneError("could not download model files")

    monkeypatch.delenv("QWEN3_TTS_ALLOW_FALLBACK", raising=False)
    monkeypatch.setattr("modules.qwen3_voice_clone.clone_batch", fail_clone_batch)

    with pytest.raises(Qwen3CloneError, match="could not download model files"):
        voice_cloner._clone_gender_batch_qwen3(
            [segment],
            tmp_path / "clones",
            {"female": female_ref},
            log_cb=None,
            quality_report=SimpleNamespace(voice_clone_failures=[]),
            cancel_event=Event(),
            progress_cb=None,
            segment_genders={0: "female"},
        )

    assert segment.cloned_path is None
