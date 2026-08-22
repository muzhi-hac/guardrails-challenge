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
    """§3.8 的核心主张：精确词必须能把正确文档顶上来。

    语料有五个 chunk 而不是两个：BM25Okapi 的经典 idf 是
    ``ln((N - df + 0.5) / (df + 0.5))``，当一个词恰好出现在语料一半的文档里
    （两篇里的一篇），df=1、N=2 时 idf 精确为 0 —— 分数全灭，测的就不是
    "精确词能不能赢" 了。五篇里出现一篇（df=1, N=5）给出 idf ≈ ln(4.5/1.5)
    ≈ 1.10，是正常语料的量级。三条干扰句不含 "M"。
    """
    rules = get_rules(Locale.DE_DE)
    retriever = Bm25Retriever(
        [_chunk("a", "Tarif S kostet 19,99 EUR."),
         _chunk("b", "Tarif M kostet 29,99 EUR."),
         _chunk("c", "Der Kundenservice ist telefonisch von 8 bis 20 Uhr erreichbar."),
         _chunk("d", "Die Rechnung wird monatlich per E-Mail versendet."),
         _chunk("e", "Bei Fragen zur SIM-Karte hilft das Servicecenter weiter.")],
        rules,
    )
    assert retriever.search("Tarif M", k=1)[0].item.chunk_id == "b"


def test_compound_query_matches_component_document():
    """分词闭包的检索侧验收：查复合词，命中只写了成分的文档。

    语料有五个 chunk 而不是两个：同上，"kündigung"/"frist" 只出现在一篇
    文档里，两篇语料下 df=1, N=2 会把 idf 精确压到 0，掩盖了这个测试真正
    要验的东西（复合词分解后能不能命中）。五篇（df=1, N=5）给出健康的正
    idf。三条干扰句不含 "Kündigung" 或 "Frist"。
    """
    rules = get_rules(Locale.DE_DE)
    retriever = Bm25Retriever(
        [_chunk("a", "Die Frist zur Kündigung beträgt einen Monat."),
         _chunk("b", "Der Tarif enthält 20 GB Datenvolumen."),
         _chunk("c", "Die SIM-Karte wird innerhalb von drei Werktagen versendet."),
         _chunk("d", "Der Kundenservice beantwortet Anfragen per Chat und Telefon."),
         _chunk("e", "Ein Tarifwechsel ist online im Kundenportal möglich.")],
        rules,
    )
    assert retriever.search("Kündigungsfrist", k=1)[0].item.chunk_id == "a"


def test_ascii_query_matches_umlaut_document():
    """语料有五个 chunk 而不是两个，理由同上两个测试：两篇语料下

    "kündigung" 的 df=1, N=2 会把 idf 精确压到 0。五篇（df=1, N=5）给出
    健康的正 idf，测的才是 ascii 归一化命中带 Umlaut 的文档，而不是
    "分数是不是全零"。三条干扰句不含 "Kündigung"。
    """
    rules = get_rules(Locale.DE_DE)
    retriever = Bm25Retriever(
        [_chunk("a", "Die Kündigungsfrist beträgt einen Monat."),
         _chunk("b", "Der Tarif enthält 20 GB Datenvolumen."),
         _chunk("c", "Der Kundenservice ist telefonisch von 8 bis 20 Uhr erreichbar."),
         _chunk("d", "Die Rechnung wird monatlich per E-Mail versendet."),
         _chunk("e", "Bei Fragen zur SIM-Karte hilft das Servicecenter weiter.")],
        rules,
    )
    assert retriever.search("Kuendigung", k=1)[0].item.chunk_id == "a"


def test_isolated_term_gets_positive_idf_at_five_chunks():
    """记录原因：为什么这些测试的语料是五篇而不是两篇。

    BM25Okapi 的 idf 在词恰好出现在语料一半文档时（df=1, N=2）精确为 0 ——
    这是两篇玩具语料的退化情况，不是生产语料的性质（真实语料里 747 个词
    中只有 1.6% 的 idf <= 0，且都不是领域词）。这里直接钉住那条数学关系：
    一个词出现在五篇里的恰好一篇，idf 必须严格为正，分数也必须严格为正。
    如果有人把语料缩回两篇，这个测试和上面三个测试会一起变红，而不是
    默默地全部返回零分。
    """
    rules = get_rules(Locale.DE_DE)
    retriever = Bm25Retriever(
        [_chunk("a", "Der Tarif Roaming kostet 5 EUR pro Tag."),
         _chunk("b", "Die Rechnung wird monatlich per E-Mail versendet."),
         _chunk("c", "Der Kundenservice ist telefonisch von 8 bis 20 Uhr erreichbar."),
         _chunk("d", "Die SIM-Karte wird innerhalb von drei Werktagen versendet."),
         _chunk("e", "Ein Tarifwechsel ist online im Kundenportal möglich.")],
        rules,
    )
    results = retriever.search("Roaming", k=1)
    assert results
    assert results[0].item.chunk_id == "a"
    assert results[0].score > 0.0


def test_ties_break_deterministically():
    """同分时按 chunk_id 排 —— 否则 recall 数字在不同机器上不同。

    这个测试原本也是两篇语料，且两篇都含 "Kündigung" —— 即 df=2, N=2。
    BM25Okapi 在词出现在语料**每一篇**文档里时，idf 是
    ``ln((N-df+0.5)/(df+0.5)) = ln(0.5/2.5) < 0``，两个 tied chunk 的分数
    都变成负数，被检索器的正分过滤器直接滤掉，两边都返回空列表 —— 测的
    就不是"同分平局怎么排"了，而是"根本没有命中"。加入三条不含
    "Kündigung" 的干扰句把 df=2, N=5 撑到 idf ≈ ln(3.5/2.5) > 0，两个 tied
    chunk 恢复为正分而且分数相等，测试才真正验的是平局排序。
    """
    rules = get_rules(Locale.DE_DE)
    distractors = [
        _chunk("c", "Die Rechnung wird monatlich per E-Mail versendet."),
        _chunk("d", "Der Kundenservice ist telefonisch von 8 bis 20 Uhr erreichbar."),
        _chunk("e", "Bei Fragen zur SIM-Karte hilft das Servicecenter weiter."),
    ]
    chunks = [_chunk("z", "Kündigung"), _chunk("a", "Kündigung"), *distractors]
    forward = Bm25Retriever(chunks, rules).search("Kündigung", k=2)
    backward = Bm25Retriever(
        [_chunk("a", "Kündigung"), _chunk("z", "Kündigung"), *distractors], rules
    ).search("Kündigung", k=2)
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


def test_none_locale_raises_keyerror(kb):
    """None 是实际的错误参数 —— 一个从未填充的 profile 字段。"""
    with pytest.raises(KeyError, match="de-DE"):
        kb.for_locale(None)  # type: ignore[arg-type]


def test_unhashable_locale_raises_keyerror(kb):
    """不可哈希的参数也必须产生信息性的 KeyError。"""
    with pytest.raises(KeyError, match="de-DE"):
        kb.for_locale(['not', 'hashable'])  # type: ignore[arg-type]


def test_empty_query_returns_nothing(kb):
    assert kb.for_locale(Locale.DE_DE).search("", k=5) == ()
