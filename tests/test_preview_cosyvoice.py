from __future__ import annotations

from pathlib import Path

from core.context import Segment
from core.preview import preview_segment


def test_cosyvoice_preview_uses_khmer_tts_source_audio(monkeypatch, tmp_path):
    tts_path = tmp_path / "khmer_tts.mp3"
    tts_path.write_bytes(b"khmer tts")
    reference_path = tmp_path / "reference.wav"
    reference_path.write_bytes(b"reference")
    captured = {}

    def fake_synthesize_tts(segments, *_args, **_kwargs):
        segments[0].tts_path = tts_path
        return segments

    def fake_clone_batch(batch_items, reference_path_arg, **_kwargs):
        captured["batch_items"] = batch_items
        captured["reference_path"] = reference_path_arg
        out = Path(batch_items[0]["output_path"])
        out.write_bytes(b"cosyvoice")
        return [{"segment_index": batch_items[0]["segment_index"], "ok": True}]

    monkeypatch.setattr("modules.tts_engine.synthesize_tts", fake_synthesize_tts)
    monkeypatch.setattr("modules.cosyvoice_voice_clone.clone_batch", fake_clone_batch)

    result = preview_segment(
        Segment(
            index=3,
            start=0.0,
            end=1.0,
            text="hello",
            translated_text="សួស្តី",
            speaker_id="spk_1",
        ),
        voice_female="km-KH-SreymomNeural",
        voice_male="km-KH-PisethNeural",
        speech_rate=0,
        pitch_hz=0,
        voice_gender="female",
        segment_genders=None,
        clone_backend="cosyvoice",
        speaker_voice_mappings={
            "spk_1": {"cleaned_reference_audio_path": str(reference_path)},
        },
    )

    assert result is not None
    assert result.exists()
    assert captured["reference_path"] == reference_path
    assert captured["batch_items"][0]["input_path"] == str(tts_path)
