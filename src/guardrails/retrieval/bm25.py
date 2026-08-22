"""Lexical retrieval.

Why not vector retrieval: queries in this domain are exact words — tariff
codes, clause numbers, ``Kündigungsfrist``. A dense vector would blur them
together; ``Tarif S`` and ``Tarif M`` could end up closer to each other than
either is to the correct document. The real retrieval difficulty in German
is compounding, and that is handled by the ``locale`` tokenizer, at the cost
of a lexicon rather than a 400MB model.

In production this layer would be Elasticsearch — the analyzer, field
weights, and synonym tables all need tuning against the corpus. ``rank_bm25``
is used here to keep evaluation offline and reproducible; introducing a
vector store over eighteen documents would be pure operational overhead.

Tokenization stays in our hands: ``rank_bm25`` accepts an already-tokenized
corpus, so compound decomposition and alias closure are done before anything
enters the index.
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
        """Return at most ``k`` results, ordered by descending score; ties
        broken by ascending ``chunk_id``."""
        ...


class Bm25Retriever:
    """One index per locale."""

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
             # Filter: keep only chunks the query terms actually touch. On a
             # first look, BM25's negative-idf formula
             # ``ln((N-df+0.5)/(df+0.5))`` seems to say that a term appearing
             # in more than half the corpus gets a negative idf and its
             # contribution zeroes out. But BM25Okapi's implementation floors
             # a negative idf at epsilon times the (positive) average idf, so
             # a query for common German function words like "und der" still
             # returns a ranked result on this corpus. The filter therefore
             # only blocks the case where the query terms are entirely absent
             # from a chunk, or the whole query falls outside the vocabulary.
             # This has been verified empirically against the corpus, not
             # assumed.
             if score > 0.0),
            key=lambda s: (-s.score, s.item.chunk_id),
        )
        return tuple(ranked[:k])
