from __future__ import annotations

from modules.glossary_builder import extract_glossary_terms, GlossaryTerm


def _sources(*texts: str) -> list[str]:
    return list(texts)


def test_extracts_chinese_personal_names():
    # 王小明 is a canonical Chinese name (surname 王 + given name 小明).
    terms = extract_glossary_terms(_sources("大家好，我是王小明。", "王小明介绍一下这个产品。"))
    names = [t.source for t in terms if t.kind == "person"]
    assert "王小明" in names


def test_names_ranked_above_bigrams():
    sources = _sources(
        "李四说他喜欢这个游戏。",
        "李四今天玩了很久这个游戏。",
        "李四又玩了这个游戏。",
    )
    terms = extract_glossary_terms(sources)
    kinds_in_order = [t.kind for t in terms]
    # Personal name should surface before recurring bigrams, otherwise the
    # LLM's context window fills with noise before it sees the name.
    if "person" in kinds_in_order and "recurring" in kinds_in_order:
        assert kinds_in_order.index("person") < kinds_in_order.index("recurring")


def test_extracts_latin_brand_and_acronym():
    terms = extract_glossary_terms(
        _sources(
            "我今天开箱 iPhone 15 Pro。",
            "AI 让 YouTube 内容制作更容易。",
        )
    )
    sources = {t.source: t.kind for t in terms}
    assert sources.get("iPhone") == "latin"
    assert sources.get("YouTube") == "latin"
    assert sources.get("AI") == "acronym"


def test_returns_empty_for_empty_input():
    assert extract_glossary_terms([]) == []
    assert extract_glossary_terms(["", "  "]) == []


def test_respects_max_terms_cap():
    sources = _sources(
        "王大是我朋友。李四也是朋友。张三来了。刘明说话。陈红笑了。"
        "杨强跑步。黄安唱歌。赵飞跳舞。吴丽画画。周军写字。"
    )
    terms = extract_glossary_terms(sources, max_terms=5)
    assert len(terms) <= 5


def test_glossary_term_dataclass_is_immutable():
    term = GlossaryTerm(source="王小明", kind="person", count=2)
    # dataclass(frozen=True) — mutation should error, keeping the ranked
    # list stable while it flows through the review pipeline.
    try:
        term.count = 99  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("GlossaryTerm must be frozen")
