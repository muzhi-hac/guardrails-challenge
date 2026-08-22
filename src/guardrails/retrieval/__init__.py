"""The knowledge base and retrieval."""

from __future__ import annotations

from guardrails.retrieval.bm25 import Bm25Retriever, Retriever
from guardrails.retrieval.chunks import Chunk, Scored, chunk_document, slugify
from guardrails.retrieval.documents import Document, load_documents
from guardrails.retrieval.knowledge_base import KnowledgeBase

__all__ = [
    "Bm25Retriever",
    "Chunk",
    "Document",
    "KnowledgeBase",
    "Retriever",
    "Scored",
    "chunk_document",
    "load_documents",
    "slugify",
]
