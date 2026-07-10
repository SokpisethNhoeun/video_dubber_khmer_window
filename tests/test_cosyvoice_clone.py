from __future__ import annotations

import json
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from core.context import Segment
from modules.cosyvoice_voice_clone import (
    CosyVoiceCloneError,
    _cosyvoice_python,
    _env_bool,
    clone_batch,
)
from modules.emotion_reference import EmotionClip


class TestCosyVoicePython:
    def test_missing_env_raises(self, monkeypatch):
        monkeypatch.delenv("COSYVOICE_PYTHON", raising=False)
        with pytest.raises(CosyVoiceCloneError, match="COSYVOICE_PYTHON is not set"):
            _cosyvoice_python()

    def test_nonexistent_path_raises(self, monkeypatch):
        monkeypatch.setenv("COSYVOICE_PYTHON", "/tmp/nonexistent_cosyvoice_python_xyz")
        with pytest.raises(CosyVoiceCloneError, match="does not exist"):
            _cosyvoice_python()

    def test_valid_path_returns(self, monkeypatch, tmp_path):
        py = tmp_path / "python3"
        py.write_text("#!/bin/sh")
        py.chmod(0o755)
        monkeypatch.setenv("COSYVOICE_PYTHON", str(py))
        assert _cosyvoice_python() == str(py)


class TestCloneBatch:
    def test_empty_segments_returns_empty(self):
        assert clone_batch([], Path("/fake/ref.wav")) == []

    def test_writes_manifest_with_segments(self, monkeypatch, tmp_path):
        py = tmp_path / "python3"
        py.write_text("#!/bin/sh")
        py.chmod(0o755)
        monkeypatch.setenv("COSYVOICE_PYTHON", str(py))

        captured_manifest = {}

        def mock_popen(cmd, **kwargs):
            manifest_path = Path(cmd[3])
            captured_manifest.update(json.loads(manifest_path.read_text()))
            results_path = manifest_path.with_suffix(".results.json")
            results_path.write_text(json.dumps([
                {"segment_index": 0, "ok": True},
                {"segment_index": 1, "ok": True},
            ]))

            class FakeProc:
                stdout = None
                returncode = 0
                def wait(self, timeout=None): pass
            return FakeProc()

        monkeypatch.setattr("modules.cosyvoice_voice_clone.subprocess.Popen", mock_popen)

        segments = [
            {"segment_index": 0, "text": "Hello", "output_path": str(tmp_path / "out0.wav")},
            {"segment_index": 1, "text": "World", "output_path": str(tmp_path / "out1.wav")},
        ]
        ref = tmp_path / "ref.wav"
        ref.write_bytes(b"fake")

        results = clone_batch(segments, ref)

        assert len(results) == 2
        assert results[0]["ok"] is True
        assert captured_manifest["reference_path"] == str(ref)
        assert captured_manifest["fp16"] is True
        assert captured_manifest["cache_references"] is True
        assert captured_manifest["clone_mode"] == "vc"
        assert len(captured_manifest["segments"]) == 2

    def test_manifest_respects_speed_env_overrides(self, monkeypatch, tmp_path):
        py = tmp_path / "python3"
        py.write_text("#!/bin/sh")
        py.chmod(0o755)
        monkeypatch.setenv("COSYVOICE_PYTHON", str(py))
        monkeypatch.setenv("COSYVOICE_FP16", "0")
        monkeypatch.setenv("COSYVOICE_CACHE_REFERENCES", "0")
        monkeypatch.setenv("COSYVOICE_CLONE_MODE", "tts")

        captured_manifest = {}

        def mock_popen(cmd, **kwargs):
            manifest_path = Path(cmd[3])
            captured_manifest.update(json.loads(manifest_path.read_text()))
            results_path = manifest_path.with_suffix(".results.json")
            results_path.write_text(json.dumps([{"segment_index": 0, "ok": True}]))

            class FakeProc:
                stdout = None
                returncode = 0
                def wait(self, timeout=None): pass
            return FakeProc()

        monkeypatch.setattr("modules.cosyvoice_voice_clone.subprocess.Popen", mock_popen)

        segments = [{"segment_index": 0, "text": "Hello", "output_path": str(tmp_path / "out.wav")}]
        ref = tmp_path / "ref.wav"
        ref.write_bytes(b"fake")

        clone_batch(segments, ref)

        assert captured_manifest["fp16"] is False
        assert captured_manifest["cache_references"] is False
        assert captured_manifest["clone_mode"] == "tts"

    def test_env_bool_parses_acceleration_flags(self, monkeypatch):
        monkeypatch.setenv("COSYVOICE_LOAD_TRT", "yes")
        monkeypatch.setenv("COSYVOICE_LOAD_JIT", "0")

        assert _env_bool("COSYVOICE_LOAD_TRT") is True
        assert _env_bool("COSYVOICE_LOAD_JIT") is False
        assert _env_bool("COSYVOICE_LOAD_VLLM", True) is True

    def test_per_segment_reference_in_manifest(self, monkeypatch, tmp_path):
        py = tmp_path / "python3"
        py.write_text("#!/bin/sh")
        py.chmod(0o755)
        monkeypatch.setenv("COSYVOICE_PYTHON", str(py))

        captured_manifest = {}

        def mock_popen(cmd, **kwargs):
            manifest_path = Path(cmd[3])
            captured_manifest.update(json.loads(manifest_path.read_text()))
            results_path = manifest_path.with_suffix(".results.json")
            results_path.write_text(json.dumps([{"segment_index": 0, "ok": True}]))

            class FakeProc:
                stdout = None
                returncode = 0
                def wait(self, timeout=None): pass
            return FakeProc()

        monkeypatch.setattr("modules.cosyvoice_voice_clone.subprocess.Popen", mock_popen)

        emotion_ref = str(tmp_path / "emotion_clip.wav")
        segments = [
            {
                "segment_index": 0,
                "text": "Hello",
                "output_path": str(tmp_path / "out0.wav"),
                "reference_path": emotion_ref,
            },
        ]
        ref = tmp_path / "ref.wav"
        ref.write_bytes(b"fake")

        clone_batch(segments, ref)

        assert captured_manifest["segments"][0]["reference_path"] == emotion_ref

    def test_instruction_text_in_manifest(self, monkeypatch, tmp_path):
        py = tmp_path / "python3"
        py.write_text("#!/bin/sh")
        py.chmod(0o755)
        monkeypatch.setenv("COSYVOICE_PYTHON", str(py))

        captured_manifest = {}

        def mock_popen(cmd, **kwargs):
            manifest_path = Path(cmd[3])
            captured_manifest.update(json.loads(manifest_path.read_text()))
            results_path = manifest_path.with_suffix(".results.json")
            results_path.write_text(json.dumps([{"segment_index": 0, "ok": True}]))

            class FakeProc:
                stdout = None
                returncode = 0
                def wait(self, timeout=None): pass
            return FakeProc()

        monkeypatch.setattr("modules.cosyvoice_voice_clone.subprocess.Popen", mock_popen)

        segments = [
            {
                "segment_index": 0,
                "text": "Hello",
                "output_path": str(tmp_path / "out0.wav"),
                "instruct_text": "speak excitedly and energetically",
            },
        ]
        ref = tmp_path / "ref.wav"
        ref.write_bytes(b"fake")

        clone_batch(segments, ref)

        assert captured_manifest["segments"][0]["instruct_text"] == "speak excitedly and energetically"

    def test_subprocess_failure_raises(self, monkeypatch, tmp_path):
        py = tmp_path / "python3"
        py.write_text("#!/bin/sh")
        py.chmod(0o755)
        monkeypatch.setenv("COSYVOICE_PYTHON", str(py))

        import io

        def mock_popen(cmd, **kwargs):
            class FakeProc:
                stdout = io.StringIO("Error: model not found\n")
                returncode = 1
                def wait(self, timeout=None): pass
            return FakeProc()

        monkeypatch.setattr("modules.cosyvoice_voice_clone.subprocess.Popen", mock_popen)

        segments = [{"segment_index": 0, "text": "Hello", "output_path": str(tmp_path / "out.wav")}]
        ref = tmp_path / "ref.wav"
        ref.write_bytes(b"fake")

        with pytest.raises(CosyVoiceCloneError):
            clone_batch(segments, ref)


def test_per_person_cosyvoice_batches_all_speakers_in_one_model_load(monkeypatch, tmp_path):
    from modules import voice_cloner

    ref_a = tmp_path / "speaker_a.wav"
    ref_b = tmp_path / "speaker_b.wav"
    ref_a.write_bytes(b"fake")
    ref_b.write_bytes(b"fake")
    clone_dir = tmp_path / "clones"
    clone_dir.mkdir()

    tts_a = tmp_path / "tts_a.mp3"
    tts_b = tmp_path / "tts_b.mp3"
    tts_a.write_bytes(b"khmer tts")
    tts_b.write_bytes(b"khmer tts")
    segments = [
        Segment(
            index=0, start=0.0, end=1.0, text="hello", translated_text="សួស្តី",
            speaker_id="spk_a", tts_path=tts_a,
        ),
        Segment(
            index=1, start=1.0, end=2.0, text="world", translated_text="ពិភពលោក",
            speaker_id="spk_b", tts_path=tts_b,
        ),
    ]
    mappings = {
        "spk_a": {"cleaned_reference_audio_path": str(ref_a), "quality_tier": "good"},
        "spk_b": {"cleaned_reference_audio_path": str(ref_b), "quality_tier": "good"},
    }
    calls = []

    def fake_clone_batch(batch_items, reference_path, log_cb=None):
        calls.append((batch_items, reference_path))
        results = []
        for item in batch_items:
            output_path = Path(item["output_path"])
            output_path.write_bytes(b"wav")
            results.append({"segment_index": item["segment_index"], "ok": True})
        return results

    monkeypatch.setattr("modules.cosyvoice_voice_clone.clone_batch", fake_clone_batch)
    monkeypatch.setattr(voice_cloner, "post_clone_match", lambda *_args, **_kwargs: None)
    progress_updates = []

    voice_cloner._clone_per_person_batch_cosyvoice(
        segments,
        clone_dir,
        mappings,
        log_cb=None,
        quality_report=SimpleNamespace(voice_clone_failures=[]),
        cancel_event=Event(),
        progress_cb=progress_updates.append,
    )

    assert len(calls) == 1
    batch_items, fallback_reference = calls[0]
    assert fallback_reference == ref_a
    assert [Path(item["reference_path"]) for item in batch_items] == [ref_a, ref_b]
    assert [Path(item["input_path"]) for item in batch_items] == [tts_a, tts_b]
    assert all(segment.cloned_path is not None for segment in segments)
    assert progress_updates[-1] == 100


def test_per_person_cosyvoice_uses_emotion_prompt_but_keeps_speaker_reference(monkeypatch, tmp_path):
    from modules import voice_cloner

    speaker_ref = tmp_path / "speaker.wav"
    emotion_ref = tmp_path / "emotion.wav"
    tts_path = tmp_path / "tts.mp3"
    speaker_ref.write_bytes(b"speaker")
    emotion_ref.write_bytes(b"emotion")
    tts_path.write_bytes(b"khmer tts")
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

    def fake_clone_batch(batch_items, reference_path, log_cb=None):
        captured["items"] = batch_items
        captured["fallback"] = reference_path
        out = Path(batch_items[0]["output_path"])
        out.write_bytes(b"wav")
        return [{"segment_index": 0, "ok": True}]

    matched_refs = []
    monkeypatch.setattr("modules.cosyvoice_voice_clone.clone_batch", fake_clone_batch)
    monkeypatch.setattr(voice_cloner, "post_clone_match", lambda _out, ref: matched_refs.append(ref))

    voice_cloner._clone_per_person_batch_cosyvoice(
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
    )

    item = captured["items"][0]
    assert Path(item["reference_path"]) == emotion_ref
    assert Path(item["emotion_reference_path"]) == emotion_ref
    assert Path(item["speaker_reference_path"]) == speaker_ref
    assert matched_refs == [speaker_ref]


def test_per_person_cosyvoice_skips_when_khmer_tts_source_is_missing(monkeypatch, tmp_path):
    from modules import voice_cloner

    ref = tmp_path / "speaker.wav"
    ref.write_bytes(b"fake")
    segment = Segment(
        index=0,
        start=0.0,
        end=1.0,
        text="hello",
        translated_text="សួស្តី",
        speaker_id="spk_a",
        tts_path=tmp_path / "missing.mp3",
    )
    mappings = {
        "spk_a": {"cleaned_reference_audio_path": str(ref), "quality_tier": "good"},
    }
    failures = []

    def fail_clone_batch(*_args, **_kwargs):
        raise AssertionError("CosyVoice should not synthesize text when Khmer TTS audio is missing")

    monkeypatch.setattr("modules.cosyvoice_voice_clone.clone_batch", fail_clone_batch)

    voice_cloner._clone_per_person_batch_cosyvoice(
        [segment],
        tmp_path / "clones",
        mappings,
        log_cb=None,
        quality_report=SimpleNamespace(voice_clone_failures=failures),
        cancel_event=Event(),
        progress_cb=None,
    )

    assert segment.cloned_path is None
    assert failures
    assert "missing Khmer TTS source audio" in failures[0]["message"]
