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

from rank_bm25 import BM25Plus

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
        # BM25Plus, not BM25Okapi: Okapi's classic idf floor gives a term that
        # appears in exactly half of a small corpus an idf of exactly 0 (or a
        # symmetric negative value when it appears in every document), which
        # zeroes out its contribution entirely. On corpora the size of a single
        # locale's KB (tens of chunks) — and especially in tests with a
        # handful of documents — that erases the exact-term signal §3.8 exists
        # to preserve. BM25Plus's additive delta keeps a document that actually
        # contains the term ranked above one that does not, at any corpus size.
        self._index = BM25Plus(corpus) if corpus else None

    def search(self, query: str, *, k: int) -> tuple[Scored[Chunk], ...]:
        tokens = list(self._rules.tokenize(query))
        if not tokens or self._index is None:
            return ()
        scores = self._index.get_scores(tokens)
        ranked = sorted(
            (Scored(chunk, float(score))
             for chunk, score in zip(self._chunks, scores, strict=True)
             if score > 0.0),
            key=lambda s: (-s.score, s.item.chunk_id),
        )
        return tuple(ranked[:k])
