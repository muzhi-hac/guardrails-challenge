"""BM25 检索。

recall@k 在 Task 8 单独测；这里测的是检索器本身的性质，尤其**确定性** ——
同分结果的顺序如果依赖字典遍历或浮点抖动，recall 测试就会在不同机器上给出不同
数字，而那正是这个模块要产出的核心指标。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from guardrails.retrieval.bm25 import Bm25Retriever
from guardrails.retrieval.chunks import Chunk
from guardrails.retrieval.knowledge_base import KnowledgeBase
from guardrails.locale import get_rules
from guardrails.types import Locale

KB_ROOT = Path(__file__).resolve().parents[1] / "kb"


@pytest.fixture(scope="module")
def kb():
    return KnowledgeBase.load(KB_ROOT)


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(chunk_id=chunk_id, doc_id="d", locale=Locale.DE_DE, title="t",
                 section="s", source_path="p", text=text)


def test_returns_at_most_k(kb):
    assert len(kb.for_locale(Locale.DE_DE).search("Kündigungsfrist", k=3)) <= 3


def test_scores_are_descending(kb):
    results = kb.for_locale(Locale.DE_DE).search("Tarif M Preis", k=5)
    assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)


def test_exact_term_wins():
    """§3.8 的核心主张：精确词必须能把正确文档顶上来。"""
    rules = get_rules(Locale.DE_DE)
    retriever = Bm25Retriever(
        [_chunk("a", "Tarif S kostet 19,99 EUR."),
         _chunk("b", "Tarif M kostet 29,99 EUR.")],
        rules,
    )
    assert retriever.search("Tarif M", k=1)[0].item.chunk_id == "b"


def test_compound_query_matches_component_document():
    """分词闭包的检索侧验收：查复合词，命中只写了成分的文档。"""
    rules = get_rules(Locale.DE_DE)
    retriever = Bm25Retriever(
        [_chunk("a", "Die Frist zur Kündigung beträgt einen Monat."),
         _chunk("b", "Der Tarif enthält 20 GB Datenvolumen.")],
        rules,
    )
    assert retriever.search("Kündigungsfrist", k=1)[0].item.chunk_id == "a"


def test_ascii_query_matches_umlaut_document():
    rules = get_rules(Locale.DE_DE)
    retriever = Bm25Retriever(
        [_chunk("a", "Die Kündigungsfrist beträgt einen Monat."),
         _chunk("b", "Der Tarif enthält 20 GB Datenvolumen.")],
        rules,
    )
    assert retriever.search("Kuendigung", k=1)[0].item.chunk_id == "a"


def test_ties_break_deterministically():
    """同分时按 chunk_id 排 —— 否则 recall 数字在不同机器上不同。"""
    rules = get_rules(Locale.DE_DE)
    chunks = [_chunk("z", "Kündigung"), _chunk("a", "Kündigung")]
    forward = Bm25Retriever(chunks, rules).search("Kündigung", k=2)
    backward = Bm25Retriever(list(reversed(chunks)), rules).search("Kündigung", k=2)
    assert [r.item.chunk_id for r in forward] == ["a", "z"]
    assert [r.item.chunk_id for r in backward] == ["a", "z"]


def test_locales_are_isolated(kb):
    """英语查询不得召回德语文档 —— 两个渠道的索引是分开的。"""
    results = kb.for_locale(Locale.EN_GB).search("cancellation notice period", k=5)
    assert results
    assert all(r.item.locale is Locale.EN_GB for r in results)


def test_unknown_locale_names_what_is_available(kb):
    class Fake:
        value = "fr-FR"
    with pytest.raises(KeyError, match="de-DE"):
        kb.for_locale(Fake())  # type: ignore[arg-type]


def test_empty_query_returns_nothing(kb):
    assert kb.for_locale(Locale.DE_DE).search("", k=5) == ()
