"""按 locale 组织的知识库。

``k`` 不是 profile 的一部分。它确实是 profile 形状的东西 —— 不同客户可能想要不同的
上下文预算 —— 但现在只有一个取值需求，为它引入一整层配置面和 ``locale/`` 里被否掉的
「子检查级 mode 配置」是同一类过度设计。

**升级触发条件**：出现第二个客户需要不同的上下文预算，或者评测显示 k 是溯源守卫
假阳性的主因。
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
        except KeyError as exc:
            known = ", ".join(sorted(loc.value for loc in self._retrievers))
            raise KeyError(
                f"no knowledge base for {locale.value!r}; have: {known}"
            ) from exc

    @property
    def chunks(self) -> tuple[Chunk, ...]:
        return self._chunks
