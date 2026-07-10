from __future__ import annotations

from threading import Event

from modules.diarizer import SpeakerTurn
from modules.reference_quality import ReferenceQuality
from modules.speaker_references import build_auto_speaker_references, _select_natural_reference_turns


def _quality(path, tier: str, score: float) -> ReferenceQuality:
    return ReferenceQuality(
        path=path,
        tier=tier,
        score=score,
        duration_seconds=5.0,
        voiced_ratio=0.7,
        snr_db=24.0,
        peak_dbfs=-4.0,
        clipping_ratio=0.0,
        music_harmonicity=0.1,
        reasons=[],
    )


def test_select_natural_reference_turns_prefers_clean_turns(monkeypatch, tmp_path):
    source_wav = tmp_path / "source.wav"
    source_wav.write_bytes(b"fake")
    turns = [
        SpeakerTurn(start=0.0, end=8.0, speaker_id="speaker_1"),
        SpeakerTurn(start=10.0, end=13.0, speaker_id="speaker_1"),
    ]

    def fake_trim(_src, out, _start, _duration, _cancel):
        out.write_bytes(b"clip")

    def fake_assess(path):
        if "quality_001" in path.name:
            return _quality(path, "bad", 20.0)
        return _quality(path, "good", 95.0)

    monkeypatch.setattr("modules.speaker_references.trim_audio_segment", fake_trim)
    monkeypatch.setattr("modules.speaker_references.assess_reference", fake_assess)

    selected = _select_natural_reference_turns(
        source_wav,
        turns,
        min_seconds=2.0,
        clips_dir=tmp_path,
        safe_speaker_id="speaker_1",
        cancel_event=Event(),
    )

    assert selected == [turns[1]]


def test_build_auto_speaker_references_requires_15_seconds_clean_speech(monkeypatch, tmp_path):
    source_wav = tmp_path / "source.wav"
    source_media = tmp_path / "video.mp4"
    source_wav.write_bytes(b"fake")
    source_media.write_bytes(b"fake")
    turns = [
        SpeakerTurn(start=0.0, end=8.0, speaker_id="speaker_1"),
        SpeakerTurn(start=9.0, end=17.0, speaker_id="speaker_1"),
        SpeakerTurn(start=20.0, end=27.0, speaker_id="speaker_2"),
    ]

    def fake_trim(_src, out, _start, _duration, _cancel):
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"clip")

    def fake_concat(_clips, output_path, _list_path, _cancel):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"merged")

    def fake_duration(path):
        return 16.0 if path.name == "speaker_1.wav" else 7.0

    monkeypatch.setattr("modules.speaker_references.trim_audio_segment", fake_trim)
    monkeypatch.setattr("modules.speaker_references.concat_wavs", fake_concat)
    monkeypatch.setattr("modules.speaker_references.ffprobe_duration", fake_duration)
    monkeypatch.setattr("modules.speaker_references.assess_reference", lambda path: _quality(path, "good", 95.0))
    monkeypatch.setattr("modules.speaker_verification.get_segment_embedding", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("offline")))

    mappings = build_auto_speaker_references(
        source_wav,
        source_media,
        turns,
        tmp_path / "work",
        min_reference_seconds=15.0,
        cancel_event=Event(),
        persistent_cache_dir=None,
    )

    assert mappings["speaker_1"]["reference_status"] == "auto"
    assert mappings["speaker_1"]["reference_audio_path"]
    assert mappings["speaker_2"]["reference_status"] == "missing"
    assert mappings["speaker_2"]["fallback_voice"] == "default_tts"
    assert "7.0s clean speech" in mappings["speaker_2"]["fallback_reason"]
