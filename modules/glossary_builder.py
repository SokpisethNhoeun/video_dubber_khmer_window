from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass


# Common Chinese surname characters. Used as anchors for name extraction —
# a run of 2-4 CJK characters starting with one of these is very likely a
# personal name, which is exactly what the LLM keeps re-romanizing every
# other segment when we don't pin it down.
CHINESE_SURNAMES = set(
    "王李张刘陈杨黄赵吴周徐孙马朱胡郭何高林罗郑梁谢宋唐许韩冯邓曹彭曾"
    "肖田董袁潘于蒋蔡余杜叶程苏魏吕丁任沈姚卢姜崔钟谭陆汪范金石廖贾夏"
    "韦付方白邹孟熊秦邱江尹薛闫段雷侯龙史陶黎贺顾毛郝龚邵万钱严赖覃洪"
    "武莫孔"
)

# Chinese punctuation used to segment sentences without breaking mid-name.
CHINESE_PUNCT = "，。！？；：、,.!?;:"

# Characters that commonly trail after a name but are not part of the name.
# Includes sentence-final particles ("啊", "呀", "了") and common leading
# characters of the next word (verbs, prepositions, aspect markers) that
# would otherwise get glued onto the name by the greedy 2-4 char window.
TRAILING_PARTICLES = set(
    "啊呀吧呢哦嗯了的呗嘛"          # particles
    "要是在会有说用想去来做给对和跟对于对着还也就都从把被让"  # verbs / prepositions
    "把也又还再又再从向到往来去"    # movement / adverbs
)


@dataclass(frozen=True)
class GlossaryTerm:
    source: str
    kind: str  # "person" | "latin" | "acronym" | "recurring"
    count: int


def _iter_cjk_runs(text: str) -> list[str]:
    """Return contiguous runs of CJK characters (no punctuation, no spaces)."""
    return re.findall(r"[一-鿿]+", text or "")


def _extract_chinese_names(text: str) -> list[str]:
    """Best-effort Chinese personal name extraction using surname anchors.

    We deliberately don't pull in jieba or hanlp as a hard dependency: for
    dubbing scripts what matters is *consistency*, not linguistic precision.
    Even an imperfect list of anchors — "王小明", "李四" — gives the LLM
    something to lock onto so those characters spell the same Khmer name
    every time they appear.
    """
    names: list[str] = []
    for run in _iter_cjk_runs(text):
        i = 0
        while i < len(run):
            if run[i] in CHINESE_SURNAMES:
                # Grab the surname + 1-3 following CJK chars, dropping trailing
                # sentence particles that would otherwise stick to the name.
                end = min(i + 4, len(run))
                candidate = run[i:end]
                while len(candidate) > 2 and candidate[-1] in TRAILING_PARTICLES:
                    candidate = candidate[:-1]
                if 2 <= len(candidate) <= 4:
                    names.append(candidate)
                i += len(candidate)
                continue
            i += 1
    return names


def _extract_latin_terms(text: str) -> list[str]:
    """Capitalized ASCII tokens (brand names, English proper nouns embedded
    in Chinese captions like 'iPhone', 'YouTube', 'AI')."""
    if not text:
        return []
    # Two patterns so we catch both "YouTube" (Uppercase-first) and "iPhone"
    # (lowercase-then-uppercase camelCase brand style). Numbers are allowed
    # in the middle so "iPhone15" style tokens still land.
    upper_first = re.findall(r"\b[A-Z][A-Za-z0-9][A-Za-z0-9_-]{1,}\b", text)
    camel = re.findall(r"\b[a-z][A-Z][A-Za-z0-9_-]{1,}\b", text)
    return upper_first + camel


def _extract_acronyms(text: str) -> list[str]:
    """All-caps acronyms 2-6 chars long, common in tech vlogs."""
    return [
        match.group(0)
        for match in re.finditer(r"\b[A-Z]{2,6}\b", text or "")
        if match.group(0) not in {"I", "A"}
    ]


def _extract_recurring_cjk_bigrams(text: str, min_count: int = 3) -> list[str]:
    """Two-character CJK terms that recur throughout the script are often
    domain terminology the LLM should keep consistent (e.g. game/product
    names). Uses a raw character-bigram count — noisy, but cheap."""
    bigrams: Counter = Counter()
    for run in _iter_cjk_runs(text):
        if len(run) < 2:
            continue
        for i in range(len(run) - 1):
            bigrams[run[i : i + 2]] += 1
    return [term for term, count in bigrams.items() if count >= min_count]


def extract_glossary_terms(
    source_texts: list[str],
    max_terms: int = 20,
) -> list[GlossaryTerm]:
    """Extract candidate glossary terms from a batch of source segments.

    Returns terms ranked by frequency (proper names always ranked above raw
    bigrams even at equal count, since name inconsistency is the most
    obvious dubbing artifact to a viewer).
    """
    if not source_texts:
        return []

    joined = "\n".join(source_texts)

    latin_counts: Counter = Counter(_extract_latin_terms(joined))
    acronym_counts: Counter = Counter(_extract_acronyms(joined))
    name_counts: Counter = Counter(_extract_chinese_names(joined))
    bigram_terms = _extract_recurring_cjk_bigrams(joined, min_count=3)

    # De-dup latin vs acronym (acronyms are a subset of latin extraction).
    for term in acronym_counts:
        latin_counts.pop(term, None)

    # Only surface CJK bigrams that don't overlap with detected names to
    # avoid the LLM getting a duplicated 王小 + 王小明 as two "terms".
    name_chars_seen = "".join(name_counts.keys())
    bigrams_filtered = [b for b in bigram_terms if b not in name_chars_seen]

    entries: list[GlossaryTerm] = []
    for term, count in name_counts.most_common():
        entries.append(GlossaryTerm(source=term, kind="person", count=count))
    for term, count in acronym_counts.most_common():
        entries.append(GlossaryTerm(source=term, kind="acronym", count=count))
    for term, count in latin_counts.most_common():
        entries.append(GlossaryTerm(source=term, kind="latin", count=count))
    for term in bigrams_filtered:
        entries.append(GlossaryTerm(source=term, kind="recurring", count=3))

    # De-duplicate while preserving the ranked order.
    seen: set[str] = set()
    ranked: list[GlossaryTerm] = []
    for entry in entries:
        if entry.source in seen:
            continue
        seen.add(entry.source)
        ranked.append(entry)
        if len(ranked) >= max_terms:
            break
    return ranked
