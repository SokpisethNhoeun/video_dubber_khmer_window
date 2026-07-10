from __future__ import annotations

import json
from threading import Event

from core.context import Segment
from modules.transcript_exports import export_srt, export_transcript_text
import modules.transcript_review as transcript_review
from modules.transcript_review import (
    CONTENT_STYLE_INSTRUCTIONS,
    _style_cleanup,
    _parse_review_payload,
    _validate_review_payload,
    build_story_context,
    parse_srt_text,
    review_segments,
    save_review_json,
)


def _segments() -> list[Segment]:
    return [
        Segment(index=0, start=0.0, end=1.5, text="Hello John", translated_text="សួស្តី John"),
        Segment(index=1, start=2.0, end=3.0, text="Go now", translated_text="ទៅឥឡូវនេះ"),
    ]


class _FakeResponse:
    def __init__(self, payload: dict | str) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        if isinstance(self.payload, str):
            return self.payload.encode("utf-8")
        return json.dumps(self.payload, ensure_ascii=False).encode("utf-8")


def _chat_response(segments: list[dict]) -> dict:
    return {
        "choices": [
            {
                "message": {
                    "content": json.dumps({"segments": segments}, ensure_ascii=False)
                }
            }
        ]
    }


def _mock_ai_responses(monkeypatch, responses: list[dict | str]) -> list[dict]:
    calls: list[dict] = []

    def fake_urlopen(request, timeout):  # noqa: ARG001
        calls.append(json.loads(request.data.decode("utf-8")))
        if not responses:
            raise AssertionError("unexpected extra AI request")
        return _FakeResponse(responses.pop(0))

    monkeypatch.setenv("TRANSCRIPT_REVIEW_API_KEY", "test-key")
    monkeypatch.setattr(transcript_review.urllib.request, "urlopen", fake_urlopen)
    return calls


def test_local_review_preserves_timing_and_sets_improved_text(tmp_path):
    segments = _segments()
    glossary = tmp_path / "glossary.txt"
    glossary.write_text("John=ចន\n", encoding="utf-8")

    reviewed = review_segments(
        segments,
        "simple",
        "local",
        glossary,
        None,
        tmp_path / "review.json",
        None,
        None,
        Event(),
    )

    assert [(segment.start, segment.end) for segment in reviewed] == [(0.0, 1.5), (2.0, 3.0)]
    assert reviewed[0].raw_khmer_text == "សួស្តី John"
    assert reviewed[0].improved_khmer_text == "សួស្តី ចន"
    assert (tmp_path / "review.json").exists()


def test_local_review_simplifies_and_marks_long_lines_for_dubbing(tmp_path):
    segments = [
        Segment(
            index=0,
            start=0.0,
            end=1.0,
            text="long",
            translated_text="លោកអ្នក នៅក្នុងពេលនេះ គួរឱ្យកត់សម្គាល់ថា ត្រូវធ្វើការសម្រេចចិត្ត យ៉ាងពិតប្រាកដ",
        )
    ]
    logs: list[str] = []

    reviewed = review_segments(
        segments,
        "simple",
        "local",
        None,
        None,
        None,
        None,
        logs.append,
        Event(),
    )

    assert "លោកអ្នក" not in reviewed[0].improved_khmer_text
    assert "ធ្វើការសម្រេចចិត្ត" not in reviewed[0].improved_khmer_text
    assert "shortened for dubbing timing" in reviewed[0].review_notes
    assert any("shortened Khmer line" in message for message in logs)


def test_style_cleanup_prefers_spoken_khmer_replacements():
    cleaned = _style_cleanup("លោកអ្នក នៅក្នុងពេលនេះ ត្រូវធ្វើការពិនិត្យមើល", "simple", 4.0)

    assert cleaned == "អ្នក ឥឡូវនេះ ត្រូវពិនិត្យមើល"


def test_review_json_edits_are_loaded_without_timing_changes(tmp_path):
    segments = _segments()
    context = build_story_context(segments)
    segments[0].improved_khmer_text = "សួស្តី"
    segments[0].user_edited_text = "សួស្តីបង"
    segments[1].enabled = False
    review_path = tmp_path / "review.json"
    save_review_json(review_path, segments, context, "natural")

    fresh = _segments()
    reviewed = review_segments(fresh, "natural", "local", None, review_path, None, None, None, Event())

    assert reviewed[0].tts_text == "សួស្តីបង"
    assert reviewed[1].enabled is False
    assert reviewed[1].start == 2.0
    assert reviewed[1].end == 3.0


def test_srt_edits_are_loaded_and_can_skip_segments(tmp_path):
    srt_path = tmp_path / "review.srt"
    srt_path.write_text(
        "\n".join(
            [
                "1",
                "00:00:00,000 --> 00:00:01,500",
                "សួស្តី ចន",
                "",
                "2",
                "00:00:02,000 --> 00:00:03,000",
                "[skip]",
                "",
            ]
        ),
        encoding="utf-8",
    )

    reviewed = review_segments(_segments(), "natural", "local", None, srt_path, None, None, None, Event())

    assert reviewed[0].tts_text == "សួស្តី ចន"
    assert reviewed[0].review_notes == "loaded from SRT"
    assert reviewed[1].enabled is False
    assert reviewed[1].review_notes == "disabled by SRT"


def test_parse_srt_text_handles_multiline_cues():
    entries = parse_srt_text(
        "1\n"
        "00:00:00,000 --> 00:00:01,500\n"
        "ជួរ ១\n"
        "ជួរ ២\n\n"
        "2\n"
        "00:00:02,000 --> 00:00:03,000\n"
        "ទៅ\n"
    )

    assert entries == ["ជួរ ១ ជួរ ២", "ទៅ"]


def test_exports_use_improved_text_and_skip_disabled_srt(tmp_path):
    segments = _segments()
    segments[0].improved_khmer_text = "សួស្តី"
    segments[1].improved_khmer_text = "ទៅ"
    segments[1].enabled = False

    improved_path = tmp_path / "improved.txt"
    srt_path = tmp_path / "subs.srt"
    export_transcript_text(improved_path, segments, "improved_khmer")
    export_srt(srt_path, segments)

    assert "សួស្តី" in improved_path.read_text(encoding="utf-8")
    srt_text = srt_path.read_text(encoding="utf-8")
    assert "សួស្តី" in srt_text
    assert "ទៅ" not in srt_text


def test_parse_review_payload_from_chat_completion_json():
    raw = (
        '{"choices":[{"message":{"content":"```json\\n'
        '{\\"segments\\":[{\\"index\\":0,\\"text\\":\\"សួស្តីចន\\"}]}'
        '\\n```"}}]}'
    )

    payload = _parse_review_payload(raw)

    assert payload["segments"][0]["text"] == "សួស្តីចន"


def test_parse_review_payload_from_streaming_chunks():
    raw = "\n".join(
        [
            'data: {"choices":[{"delta":{"content":"{\\"segments\\":["}}]}',
            'data: {"choices":[{"delta":{"content":"{\\"index\\":0,\\"text\\":\\"សួស្តី\\"}"}}]}',
            'data: {"choices":[{"delta":{"content":"]}"}}]}',
            "data: [DONE]",
        ]
    )

    payload = _parse_review_payload(raw)

    assert payload["segments"][0]["text"] == "សួស្តី"


def test_parse_review_payload_from_direct_segments_json():
    payload = _parse_review_payload('{"segments":[{"index":1,"text":"ទៅឥឡូវនេះ"}]}')

    assert payload["segments"][0]["index"] == 1


def test_validate_review_payload_reports_missing_and_duplicate_segments():
    reviewed, issues = _validate_review_payload(
        {
            "segments": [
                {"index": 0, "text": "សួស្តី"},
                {"index": 0, "text": "សួស្តីម្តងទៀត"},
            ]
        },
        _segments(),
    )

    assert reviewed == {0: "សួស្តី"}
    assert any("duplicated" in issue for issue in issues)
    assert any("missing" in issue for issue in issues)


def test_ai_review_uses_second_pass_revision(tmp_path, monkeypatch):
    calls = _mock_ai_responses(
        monkeypatch,
        [
            _chat_response(
                [
                    {"index": 0, "text": "សួស្តី John ដ៏វែងណាស់ដែលមិនសមសម្រាប់ពេលខ្លីនេះ"},
                    {"index": 1, "text": "ទៅឥឡូវនេះ"},
                ]
            ),
            _chat_response(
                [
                    {"index": 0, "text": "សួស្តី ចន"},
                    {"index": 1, "text": "ទៅឥឡូវ"},
                ]
            ),
        ],
    )

    reviewed = review_segments(
        _segments(),
        "natural",
        "ai",
        None,
        None,
        tmp_path / "review.json",
        None,
        None,
        Event(),
    )

    assert [segment.improved_khmer_text for segment in reviewed] == ["សួស្តី ចន", "ទៅឥឡូវ"]
    assert len(calls) == 2
    assert "second pass" in calls[1]["messages"][0]["content"].lower()
    assert "meanwhile" not in calls[0]["messages"][0]["content"].lower()
    assert "easy to understand" in calls[0]["messages"][0]["content"]
    assert "Do not censor" in calls[0]["messages"][0]["content"]
    assert "Do not censor" in calls[1]["messages"][0]["content"]
    assert "[censored]" in calls[0]["messages"][0]["content"]


def test_ai_review_retries_when_first_pass_omits_segments(monkeypatch):
    calls = _mock_ai_responses(
        monkeypatch,
        [
            _chat_response(
                [
                    {"index": 0, "text": "សួស្តី ចន"},
                ]
            ),
            _chat_response(
                [
                    {"index": 0, "text": "សួស្តី ចន"},
                    {"index": 1, "text": "ទៅឥឡូវនេះ"},
                ]
            ),
            _chat_response(
                [
                    {"index": 0, "text": "សួស្តី ចន"},
                    {"index": 1, "text": "ទៅឥឡូវ"},
                ]
            ),
        ],
    )

    reviewed = review_segments(
        _segments(),
        "natural",
        "ai",
        None,
        None,
        None,
        None,
        None,
        Event(),
    )

    assert [segment.improved_khmer_text for segment in reviewed] == ["សួស្តី ចន", "ទៅឥឡូវ"]
    assert len(calls) == 3
    retry_message = json.loads(calls[1]["messages"][-1]["content"])
    assert retry_message["previous_response_rejected"] is True
    assert "validation_issues" in retry_message


def test_ai_review_request_includes_token_budget_and_timing_limits(monkeypatch):
    monkeypatch.setenv("TRANSCRIPT_REVIEW_MAX_TOKENS", "16000")
    calls = _mock_ai_responses(
        monkeypatch,
        [
            _chat_response(
                [
                    {"index": 0, "text": "សួស្តី ចន"},
                    {"index": 1, "text": "ទៅឥឡូវនេះ"},
                ]
            ),
            _chat_response(
                [
                    {"index": 0, "text": "សួស្តី ចន"},
                    {"index": 1, "text": "ទៅឥឡូវ"},
                ]
            ),
        ],
    )

    review_segments(
        _segments(),
        "natural",
        "ai",
        None,
        None,
        None,
        None,
        None,
        Event(),
    )

    assert calls[0]["max_tokens"] == 16000
    first_user_payload = json.loads(calls[0]["messages"][1]["content"])
    segment_payload = first_user_payload["segments"][0]
    assert segment_payload["target_non_space_chars"] > 0
    assert segment_payload["hard_non_space_chars"] >= segment_payload["target_non_space_chars"]


def test_ai_review_request_marks_missing_translation_for_recovery(monkeypatch):
    segments = _segments()
    segments[1].translated_text = ""
    segments[1].raw_khmer_text = ""
    segments[1].improved_khmer_text = ""
    segments[1].review_notes = "AI translation missing; recover from source during AI review"
    calls = _mock_ai_responses(
        monkeypatch,
        [
            _chat_response(
                [
                    {"index": 0, "text": "សួស្តី ចន"},
                    {"index": 1, "text": "ទៅឥឡូវ"},
                ]
            ),
            _chat_response(
                [
                    {"index": 0, "text": "សួស្តី ចន"},
                    {"index": 1, "text": "ទៅឥឡូវ"},
                ]
            ),
        ],
    )

    reviewed = review_segments(
        segments,
        "natural",
        "ai",
        None,
        None,
        None,
        None,
        None,
        Event(),
    )

    assert reviewed[1].improved_khmer_text == "ទៅឥឡូវ"
    assert "raw_khmer is empty" in calls[0]["messages"][0]["content"]
    first_user_payload = json.loads(calls[0]["messages"][1]["content"])
    missing_payload = first_user_payload["segments"][1]
    assert missing_payload["raw_khmer"] == ""
    assert "AI translation missing" in missing_payload["review_notes"]


def test_ai_review_keeps_first_pass_when_second_pass_is_invalid(monkeypatch):
    calls = _mock_ai_responses(
        monkeypatch,
        [
            _chat_response(
                [
                    {"index": 0, "text": "សួស្តី ចន"},
                    {"index": 1, "text": "ទៅឥឡូវនេះ"},
                ]
            ),
            _chat_response(
                [
                    {"index": 0, "text": "សួស្តី ចន"},
                    {"index": 0, "text": "ចម្លងខុស"},
                ]
            ),
            _chat_response(
                [
                    {"index": 0, "text": "សួស្តី ចន"},
                    {"index": 0, "text": "ចម្លងខុសទៀត"},
                ]
            ),
        ],
    )

    reviewed = review_segments(
        _segments(),
        "natural",
        "ai",
        None,
        None,
        None,
        None,
        None,
        Event(),
    )

    assert [segment.improved_khmer_text for segment in reviewed] == ["សួស្តី ចន", "ទៅឥឡូវនេះ"]
    assert len(calls) == 3


def test_ai_review_ignores_second_pass_when_it_worsens_quality(monkeypatch):
    calls = _mock_ai_responses(
        monkeypatch,
        [
            _chat_response(
                [
                    {"index": 0, "text": "សួស្តី"},
                    {"index": 1, "text": "ទៅ"},
                ]
            ),
            _chat_response(
                [
                    {"index": 0, "text": "សួស្តី ចន ដ៏វែងណាស់ដែលមិនសមសម្រាប់ពេលខ្លីនេះទាល់តែសោះ"},
                    {"index": 1, "text": "ទៅឥឡូវនេះ"},
                ]
            ),
        ],
    )

    reviewed = review_segments(
        _segments(),
        "natural",
        "ai",
        None,
        None,
        None,
        None,
        None,
        Event(),
    )

    assert [segment.improved_khmer_text for segment in reviewed] == ["សួស្តី", "ទៅ"]
    assert len(calls) == 2


def test_build_story_context_extracts_chinese_names_into_terms():
    """The story context terms feed the AI reviewer's consistency hints. If
    Chinese personal names aren't in that list, the LLM re-spells them
    every segment and the dub is riddled with two Khmer versions of the
    same name."""
    segments = [
        Segment(
            index=0,
            start=0.0,
            end=2.0,
            text="大家好，我是王小明。",
            translated_text="",
        ),
        Segment(
            index=1,
            start=2.0,
            end=4.0,
            text="今天王小明要开箱 iPhone。",
            translated_text="",
        ),
    ]
    context = build_story_context(segments)
    assert "王小明" in context.terms
    assert "iPhone" in context.terms


def test_content_style_instructions_cover_expected_presets():
    # These are the presets exposed to the user in the wizard; adding a new
    # preset without wiring an instruction is a silent quality regression
    # (the LLM falls back to the default and the "Reaction" or "Educational"
    # choice has no effect).
    for preset in ("casual_vlog", "educational", "reaction", "movie_dialogue", "generic"):
        assert preset in CONTENT_STYLE_INSTRUCTIONS
        assert CONTENT_STYLE_INSTRUCTIONS[preset].strip()


def test_review_segments_accepts_content_style_kwarg(tmp_path):
    """Regression: the review_segments signature must accept content_style
    so the pipeline call site (core/pipeline.py) doesn't break."""
    segments = [Segment(index=0, start=0.0, end=1.0, text="hi", translated_text="សួស្តី")]
    reviewed = review_segments(
        segments,
        "simple",
        "local",
        None,
        None,
        None,
        None,
        None,
        Event(),
        content_style="educational",
    )
    assert reviewed[0].improved_khmer_text == "សួស្តី"
