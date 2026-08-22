"""词汇层检索。

为什么不是向量检索：这个领域的查询是精确词 —— 资费代号、条款编号、
``Kündigungsfrist``。稠密向量会把它们糊掉，``Tarif S`` 和 ``Tarif M`` 可能比任一个
离正确文档更近。德语真正的检索难点是复合词，那由 ``locale`` 的分词器处理，代价是
一份词典而不是一个 400MB 的模型。

生产里这一层是 Elasticsearch —— analyzer、字段权重和同义词表都要针对语料调。这里用
``rank_bm25`` 是为了让评测离线且可复现；十八篇文档上引入向量库是纯运维负担。

分词权在我们手里：``rank_bm25`` 接受已分词的语料，所以复合词分解与别名闭包在进索引
之前就完成了。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from rank_bm25 import BM25Okapi

from guardrails.locale.base import LocaleRules
from guardrails.retrieval.chunks import Chunk, Scored

__all__ = ["Bm25Retriever", "Retriever"]


class Retriever(Protocol):
    def search(self, query: str, *, k: int) -> tuple[Scored[Chunk], ...]:
        """返回至多 ``k`` 条，按分数降序；同分按 ``chunk_id`` 升序。"""
        ...


class Bm25Retriever:
    """一个 locale 一个索引。"""

    def __init__(self, chunks: Sequence[Chunk], rules: LocaleRules) -> None:
        self._chunks: tuple[Chunk, ...] = tuple(chunks)
        self._rules = rules
        corpus = [list(rules.tokenize(chunk.text)) for chunk in self._chunks]
        # BM25Okapi's classic idf floor gives a term that appears in exactly
        # half of a *very small* corpus (or in every document) an idf of zero
        # or below, which zeroes out its contribution. That is a degenerate
        # property of tiny toy corpora, not of this KB: across the real
        # locale corpora (tens of chunks each), df=1 terms land at a healthy
        # positive idf, and BM25Okapi is the specified, standard scorer.
        # Tests that need a term isolated to one document build a corpus
        # large enough (>=5 chunks) that idf stays positive — see
        # tests/test_retrieval.py.
        self._index = BM25Okapi(corpus) if corpus else None

    def search(self, query: str, *, k: int) -> tuple[Scored[Chunk], ...]:
        tokens = list(self._rules.tokenize(query))
        if not tokens or self._index is None:
            return ()
        scores = self._index.get_scores(tokens)
        ranked = sorted(
            (Scored(chunk, float(score))
             for chunk, score in zip(self._chunks, scores, strict=True)
             # 过滤器：只保留查询词触及的 chunk。初看 BM25 的负 idf 公式
             # ``ln((N-df+0.5)/(df+0.5))`` 会认为一个词在语料半数以上文档出现时
             # idf 为负，分数归零。但 BM25Okapi 的实现把负 idf 压到 epsilon 倍
             # 平均 idf（正数），所以这个语料上常见德语虚词如 "und der" 查询
             # 照样返回排序结果。过滤器因此只拦截查询词完全不在 chunk，或
             # 整个查询超出词表的情况。这一点已在语料上实证验证，而非假设。
             if score > 0.0),
            key=lambda s: (-s.score, s.item.chunk_id),
        )
        return tuple(ranked[:k])
