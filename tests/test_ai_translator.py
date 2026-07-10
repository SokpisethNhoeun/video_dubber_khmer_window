from __future__ import annotations

import json
from threading import Event
from unittest.mock import MagicMock, patch

import pytest

from core.context import CancellationError, Segment


def _make_segments(count: int, source_lang: str = "zh") -> list[Segment]:
    segments = []
    for i in range(count):
        seg = Segment(
            index=i,
            start=float(i * 3),
            end=float(i * 3 + 2.5),
            text=f"Source text {i}",
        )
        segments.append(seg)
    return segments


def _mock_api_response(segments: list[Segment]) -> str:
    result = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "segments": [
                                {"index": s.index, "text": f"ខ្មែរ {s.index}"}
                                for s in segments
                            ]
                        },
                        ensure_ascii=False,
                    )
                }
            }
        ]
    }
    return json.dumps(result, ensure_ascii=False)


class TestTranslateSegmentsAi:
    def test_empty_segments_raises(self):
        from modules.ai_translator import translate_segments_ai

        with pytest.raises(ValueError, match="No segments"):
            translate_segments_ai([], "zh", None, None, Event())

    def test_khmer_source_copies_text(self):
        from modules.ai_translator import translate_segments_ai

        segments = _make_segments(3)
        result = translate_segments_ai(segments, "km", None, None, Event())
        for seg in result:
            assert seg.translated_text == seg.text
            assert seg.raw_khmer_text == seg.text
            assert seg.improved_khmer_text == seg.text

    @patch("modules.ai_translator.urllib.request.urlopen")
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_basic_translation(self, mock_urlopen):
        from modules.ai_translator import translate_segments_ai

        segments = _make_segments(5)
        response_body = _mock_api_response(segments)
        mock_response = MagicMock()
        mock_response.read.return_value = response_body.encode("utf-8")
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        result = translate_segments_ai(segments, "zh", None, None, Event())
        assert mock_urlopen.call_count == 1
        for seg in result:
            assert seg.translated_text == f"ខ្មែរ {seg.index}"
            assert seg.raw_khmer_text == f"ខ្មែរ {seg.index}"
            assert seg.improved_khmer_text == f"ខ្មែរ {seg.index}"

    def test_system_prompt_preserves_uncensored_movie_dialogue(self):
        from modules.ai_translator import _build_system_prompt

        prompt = _build_system_prompt("zh", "movie_dialogue", "simple", [], {})

        assert "Movie / drama dialogue" in prompt
        assert "profanity" in prompt
        assert "Do not censor" in prompt
        assert "[censored]" in prompt
        assert "Keep the same intensity" in prompt

    @patch("modules.ai_translator.urllib.request.urlopen")
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_batching(self, mock_urlopen):
        from modules.ai_translator import BATCH_SIZE, translate_segments_ai

        count = BATCH_SIZE * 2 + 10
        segments = _make_segments(count)

        def side_effect(request, timeout=None):
            body = json.loads(request.data.decode("utf-8"))
            user_msg = json.loads(body["messages"][-1]["content"])
            batch_segs = user_msg["segments"]
            result = {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "segments": [
                                        {"index": s["index"], "text": f"បកប្រែ {s['index']}"}
                                        for s in batch_segs
                                    ]
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            }
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps(result).encode("utf-8")
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            return mock_resp

        mock_urlopen.side_effect = side_effect

        result = translate_segments_ai(segments, "zh", None, None, Event())
        assert mock_urlopen.call_count == 3
        for seg in result:
            assert seg.translated_text == f"បកប្រែ {seg.index}"

    @patch("modules.ai_translator.urllib.request.urlopen")
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_context_passing(self, mock_urlopen):
        from modules.ai_translator import BATCH_SIZE, translate_segments_ai

        segments = _make_segments(BATCH_SIZE + 5)
        payloads = []

        def side_effect(request, timeout=None):
            body = json.loads(request.data.decode("utf-8"))
            user_msg = json.loads(body["messages"][-1]["content"])
            payloads.append(user_msg)
            batch_segs = user_msg["segments"]
            result = {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "segments": [
                                        {"index": s["index"], "text": f"ខ្មែរ{s['index']}"}
                                        for s in batch_segs
                                    ]
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            }
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps(result).encode("utf-8")
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            return mock_resp

        mock_urlopen.side_effect = side_effect

        translate_segments_ai(segments, "zh", None, None, Event())

        assert "previous_context" not in payloads[0]
        assert "previous_context" in payloads[1]
        context = payloads[1]["previous_context"]
        assert len(context) <= 10
        assert all("translation" in c for c in context)

    def test_cancellation(self):
        from modules.ai_translator import translate_segments_ai

        segments = _make_segments(5)
        cancel = Event()
        cancel.set()

        with pytest.raises(CancellationError):
            translate_segments_ai(segments, "zh", None, None, cancel)

    @patch("modules.ai_translator.urllib.request.urlopen")
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_progress_callback(self, mock_urlopen):
        from modules.ai_translator import translate_segments_ai

        segments = _make_segments(3)
        response_body = _mock_api_response(segments)
        mock_response = MagicMock()
        mock_response.read.return_value = response_body.encode("utf-8")
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        progress_values = []
        translate_segments_ai(
            segments, "zh", lambda v: progress_values.append(v), None, Event()
        )
        assert len(progress_values) == 3
        assert progress_values[-1] == 100

    def test_no_api_key_raises(self):
        from modules.ai_translator import translate_segments_ai

        segments = _make_segments(3)
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(RuntimeError, match="No API key"):
                translate_segments_ai(segments, "zh", None, None, Event())

    @patch("modules.ai_translator.urllib.request.urlopen")
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_bare_list_response_is_unwrapped(self, mock_urlopen):
        from modules.ai_translator import translate_segments_ai

        segments = _make_segments(3)
        bare_list_content = json.dumps(
            [{"index": s.index, "text": f"ខ្មែរ {s.index}"} for s in segments],
            ensure_ascii=False,
        )
        response_body = json.dumps(
            {"choices": [{"message": {"content": bare_list_content}}]},
            ensure_ascii=False,
        )
        mock_response = MagicMock()
        mock_response.read.return_value = response_body.encode("utf-8")
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        result = translate_segments_ai(segments, "zh", None, None, Event())
        assert mock_urlopen.call_count == 1
        for seg in result:
            assert seg.translated_text == f"ខ្មែរ {seg.index}"

    @patch("modules.ai_translator.urllib.request.urlopen")
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_bare_list_translation_key_is_unwrapped(self, mock_urlopen):
        from modules.ai_translator import translate_segments_ai

        segments = _make_segments(3)
        bare_list_content = json.dumps(
            [{"index": s.index, "source": s.text, "translation": f"ខ្មែរ {s.index}"} for s in segments],
            ensure_ascii=False,
        )
        response_body = json.dumps(
            {"choices": [{"message": {"content": bare_list_content}}]},
            ensure_ascii=False,
        )
        mock_response = MagicMock()
        mock_response.read.return_value = response_body.encode("utf-8")
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        result = translate_segments_ai(segments, "zh", None, None, Event())
        assert mock_urlopen.call_count == 1
        for seg in result:
            assert seg.translated_text == f"ខ្មែរ {seg.index}"

    @patch("modules.ai_translator.urllib.request.urlopen")
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_non_khmer_first_response_gets_stricter_retry(self, mock_urlopen, monkeypatch):
        from modules import ai_translator
        from modules.ai_translator import translate_segments_ai

        monkeypatch.setattr(ai_translator, "RETRY_DELAY", 0)

        segments = _make_segments(2)
        english_content = json.dumps(
            [
                {"index": s.index, "source": s.text, "translation": f"English {s.index}"}
                for s in segments
            ],
            ensure_ascii=False,
        )
        khmer_content = json.dumps(
            [
                {"index": s.index, "source": s.text, "translation": f"ខ្មែរ {s.index}"}
                for s in segments
            ],
            ensure_ascii=False,
        )
        responses = [
            {"choices": [{"message": {"content": english_content}}]},
            {"choices": [{"message": {"content": khmer_content}}]},
        ]

        def side_effect(_request, timeout=None):  # noqa: ARG001
            mock_response = MagicMock()
            mock_response.read.return_value = json.dumps(responses.pop(0)).encode("utf-8")
            mock_response.__enter__ = lambda s: s
            mock_response.__exit__ = MagicMock(return_value=False)
            return mock_response

        mock_urlopen.side_effect = side_effect

        logs: list[str] = []
        result = translate_segments_ai(
            segments, "zh", None, lambda m: logs.append(m), Event()
        )

        assert mock_urlopen.call_count == 2
        retry_request = json.loads(mock_urlopen.call_args_list[1].args[0].data.decode("utf-8"))
        retry_payload = json.loads(retry_request["messages"][-1]["content"])
        assert retry_payload["previous_response_rejected"] is True
        assert "Khmer Unicode" in retry_payload["required_output"]
        for seg in result:
            assert seg.translated_text == f"ខ្មែរ {seg.index}"
        assert any("non-Khmer text" in m for m in logs)

    @patch("modules.ai_translator.urllib.request.urlopen")
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_english_translation_key_is_retried_and_rejected(self, mock_urlopen, monkeypatch):
        from modules import ai_translator
        from modules.ai_translator import MAX_RETRIES, translate_segments_ai

        monkeypatch.setattr(ai_translator, "RETRY_DELAY", 0)

        segments = _make_segments(2)
        bare_list_content = json.dumps(
            [
                {"index": s.index, "source": s.text, "translation": f"English {s.index}"}
                for s in segments
            ],
            ensure_ascii=False,
        )
        response_body = json.dumps(
            {"choices": [{"message": {"content": bare_list_content}}]},
            ensure_ascii=False,
        )
        mock_response = MagicMock()
        mock_response.read.return_value = response_body.encode("utf-8")
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        logs: list[str] = []
        with pytest.raises(RuntimeError, match="Stopping before TTS"):
            translate_segments_ai(
                segments, "zh", None, lambda m: logs.append(m), Event()
            )
        assert mock_urlopen.call_count == MAX_RETRIES
        assert any("non-Khmer text" in m for m in logs)

    @patch("modules.ai_translator.urllib.request.urlopen")
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_unrecognized_shape_raises_before_tts(self, mock_urlopen, monkeypatch):
        from modules import ai_translator
        from modules.ai_translator import translate_segments_ai

        monkeypatch.setattr(ai_translator, "RETRY_DELAY", 0)

        segments = _make_segments(2)
        garbage_content = json.dumps({"result": "nope"}, ensure_ascii=False)
        response_body = json.dumps(
            {"choices": [{"message": {"content": garbage_content}}]},
            ensure_ascii=False,
        )
        mock_response = MagicMock()
        mock_response.read.return_value = response_body.encode("utf-8")
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        logs: list[str] = []
        with pytest.raises(RuntimeError, match="Stopping before TTS"):
            translate_segments_ai(
                segments, "zh", None, lambda m: logs.append(m), Event()
            )
        assert mock_urlopen.call_count == 3
        assert any("unexpected shape" in m for m in logs)

    @patch("modules.ai_translator.urllib.request.urlopen")
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_missing_translation_can_be_deferred_to_ai_review(self, mock_urlopen, monkeypatch):
        from modules import ai_translator
        from modules.ai_translator import translate_segments_ai

        monkeypatch.setattr(ai_translator, "RETRY_DELAY", 0)

        segments = _make_segments(2)
        partial_content = json.dumps(
            {"segments": [{"index": 0, "text": "សួស្តី"}]},
            ensure_ascii=False,
        )
        response_body = json.dumps(
            {"choices": [{"message": {"content": partial_content}}]},
            ensure_ascii=False,
        )
        mock_response = MagicMock()
        mock_response.read.return_value = response_body.encode("utf-8")
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        logs: list[str] = []
        result = translate_segments_ai(
            segments,
            "zh",
            None,
            lambda message: logs.append(message),
            Event(),
            allow_review_recovery=True,
        )

        assert mock_urlopen.call_count == 1
        assert result[0].translated_text == "សួស្តី"
        assert result[1].translated_text == ""
        assert result[1].raw_khmer_text == ""
        assert result[1].improved_khmer_text == ""
        assert "AI translation missing" in result[1].review_notes
        assert any("deferred to AI review" in message for message in logs)

    @patch("modules.ai_translator.urllib.request.urlopen")
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_100_segments_use_three_translation_requests_when_review_recovery_enabled(
        self, mock_urlopen, monkeypatch
    ):
        from modules import ai_translator
        from modules.ai_translator import BATCH_SIZE, translate_segments_ai

        monkeypatch.setattr(ai_translator, "RETRY_DELAY", 0)

        segments = _make_segments(100)

        def side_effect(request, timeout=None):  # noqa: ARG001
            body = json.loads(request.data.decode("utf-8"))
            user_msg = json.loads(body["messages"][-1]["content"])
            batch_segments = user_msg["segments"]
            translated = [
                {"index": item["index"], "text": f"ខ្មែរ {item['index']}"}
                for item in batch_segments
            ]
            if batch_segments[-1]["index"] == 99:
                translated = translated[:-1]
            response_body = json.dumps(
                {"choices": [{"message": {"content": json.dumps({"segments": translated}, ensure_ascii=False)}}]},
                ensure_ascii=False,
            )
            mock_response = MagicMock()
            mock_response.read.return_value = response_body.encode("utf-8")
            mock_response.__enter__ = lambda s: s
            mock_response.__exit__ = MagicMock(return_value=False)
            return mock_response

        mock_urlopen.side_effect = side_effect

        logs: list[str] = []
        result = translate_segments_ai(
            segments,
            "zh",
            None,
            lambda message: logs.append(message),
            Event(),
            allow_review_recovery=True,
        )

        assert BATCH_SIZE == 40
        assert mock_urlopen.call_count == 3
        assert result[98].translated_text == "ខ្មែរ 98"
        assert result[99].translated_text == ""
        assert "AI translation missing" in result[99].review_notes
        assert any("99" in message and "deferred to AI review" in message for message in logs)
