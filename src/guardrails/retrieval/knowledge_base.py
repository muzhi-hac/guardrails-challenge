"""The knowledge base, organised by locale.

``k`` is not part of the profile. It genuinely is profile-shaped —
different clients might want different context budgets — but right now
there is only one value anyone needs, and introducing a whole configuration
surface for it would be the same kind of over-engineering as the
"sub-check-level mode configuration" that was rejected in ``locale/``.

**Upgrade trigger**: a second client needs a different context budget, or
evaluation shows ``k`` is the leading cause of grounding-guard false
positives.
"""

from __future__ import annotations

from pathlib import Path

from guardrails.locale import get_rules
from guardrails.retrieval.bm25 import Bm25Retriever, Retriever
from guardrails.retrieval.chunks import Chunk, chunk_document
from guardrails.retrieval.documents import load_documents
from guardrails.types import Locale

__all__ = ["KnowledgeBase"]


class KnowledgeBase:
    def __init__(self, chunks: tuple[Chunk, ...]) -> None:
        self._chunks = chunks
        self._retrievers: dict[Locale, Retriever] = {}
        for locale in {chunk.locale for chunk in chunks}:
            subset = [chunk for chunk in chunks if chunk.locale is locale]
            self._retrievers[locale] = Bm25Retriever(subset, get_rules(locale))

    @classmethod
    def load(cls, root: Path) -> KnowledgeBase:
        chunks: list[Chunk] = []
        for doc in load_documents(root):
            chunks.extend(chunk_document(doc))
        return cls(tuple(chunks))

    def for_locale(self, locale: Locale) -> Retriever:
        try:
            return self._retrievers[locale]
        except (KeyError, TypeError) as exc:
            known = ", ".join(sorted(loc.value for loc in self._retrievers))
            name = getattr(locale, "value", locale)
            raise KeyError(
                f"no knowledge base for {name!r}; have: {known}"
            ) from exc

    @property
    def chunks(self) -> tuple[Chunk, ...]:
        return self._chunks
