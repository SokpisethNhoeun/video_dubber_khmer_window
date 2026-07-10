from __future__ import annotations

from dataclasses import dataclass

import pytest

from modules.prosody import (
    MAX_RATE_OFFSET_PCT,
    MIN_RATE_OFFSET_PCT,
    SpeakerRateProfile,
    compute_speaker_rate_profiles,
    emphasis_from_source,
    per_segment_prosody,
)


@dataclass
class _Seg:
    """Minimal shim so tests don't drag in the full Segment dataclass."""
    index: int
    start: float
    end: float
    text: str
    speaker_id: str | None = None

    @property
    def duration(self) -> float:
        return max(0.001, self.end - self.start)


def test_empty_input_returns_empty_profile():
    assert compute_speaker_rate_profiles([]) == {}


def test_segments_without_speaker_id_are_ignored():
    profiles = compute_speaker_rate_profiles([
        _Seg(0, 0.0, 2.0, "大家好", speaker_id=None),
    ])
    assert profiles == {}


def test_single_speaker_gets_zero_offset():
    # With only one speaker, they *are* the crowd median → offset = 0.
    segments = [
        _Seg(0, 0.0, 2.0, "大家好我今天要给大家介绍", speaker_id="spk_1"),
        _Seg(1, 2.0, 4.0, "这个产品非常好用真的", speaker_id="spk_1"),
    ]
    profiles = compute_speaker_rate_profiles(segments)
    assert "spk_1" in profiles
    assert profiles["spk_1"].rate_offset_pct == 0


def test_fast_speaker_gets_positive_offset():
    # spk_fast crams twice as many characters into the same time.
    segments = [
        _Seg(0, 0.0, 2.0, "你好世界", speaker_id="spk_slow"),          # 4 chars / 2s = 2 cps
        _Seg(1, 2.0, 4.0, "你好呀怎么样今天过得开心吗", speaker_id="spk_fast"),  # 12 chars / 2s = 6 cps
    ]
    profiles = compute_speaker_rate_profiles(segments)
    assert profiles["spk_fast"].rate_offset_pct > 0
    assert profiles["spk_slow"].rate_offset_pct < 0


def test_rate_offset_is_clamped():
    # Two speakers, one absurdly faster. Even with a 10x differential the
    # dub rate shouldn't fly off into unintelligible territory.
    segments = [
        _Seg(0, 0.0, 5.0, "好", speaker_id="spk_slow"),
        _Seg(1, 5.0, 6.0, "你好呀怎么样今天过得开心吗真的很不错哈", speaker_id="spk_fast"),
    ]
    profiles = compute_speaker_rate_profiles(segments)
    assert MIN_RATE_OFFSET_PCT <= profiles["spk_slow"].rate_offset_pct <= MAX_RATE_OFFSET_PCT
    assert MIN_RATE_OFFSET_PCT <= profiles["spk_fast"].rate_offset_pct <= MAX_RATE_OFFSET_PCT


def test_exclamation_boosts_rate_and_pitch():
    rate, pitch = emphasis_from_source("太棒了！")
    assert rate > 0
    assert pitch > 0


def test_question_only_shifts_pitch():
    rate, pitch = emphasis_from_source("你在做什么？")
    assert rate == 0
    assert pitch > 0


def test_exclamation_wins_over_question_when_both_present():
    """"?!" — treat as the emphatic reading."""
    rate_excl, pitch_excl = emphasis_from_source("真的吗！")
    rate_both, pitch_both = emphasis_from_source("真的吗？！")
    assert (rate_both, pitch_both) == (rate_excl, pitch_excl)


def test_per_segment_prosody_combines_speaker_and_emphasis():
    profiles = {
        "spk_fast": SpeakerRateProfile(speaker_id="spk_fast", chars_per_second=6.0, rate_offset_pct=8)
    }
    seg = _Seg(0, 0.0, 1.0, "太棒了！", speaker_id="spk_fast")
    rate, pitch = per_segment_prosody(seg, base_rate_pct=0, base_pitch_hz=0, profiles=profiles)
    # base 0 + speaker +8 + exclamation +5 = 13
    assert rate == 13
    # base 0 + exclamation +10Hz
    assert pitch == 10


def test_per_segment_prosody_without_profiles_still_applies_emphasis():
    seg = _Seg(0, 0.0, 1.0, "你在做什么？", speaker_id="spk_a")
    rate, pitch = per_segment_prosody(seg, base_rate_pct=5, base_pitch_hz=0, profiles=None)
    # No profile → base only; emphasis: question → pitch up
    assert rate == 5
    assert pitch == 8


def test_per_segment_prosody_clamps_extreme_values():
    profiles = {
        "spk_x": SpeakerRateProfile(speaker_id="spk_x", chars_per_second=6.0, rate_offset_pct=12)
    }
    seg = _Seg(0, 0.0, 1.0, "！！！", speaker_id="spk_x")
    rate, _ = per_segment_prosody(seg, base_rate_pct=45, base_pitch_hz=0, profiles=profiles)
    # 45 + 12 + 5 = 62 → clamped to 50 (edge-tts sanity bound)
    assert rate == 50
