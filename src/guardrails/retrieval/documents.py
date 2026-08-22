"""Loading knowledge-base documents.

Markdown plus front matter, one file per document. Markdown rather than
JSON: the grounding guard's evidence has to quote the original text, and a
human-readable corpus lets evaluation results be checked by eye.
Front matter rather than a separate manifest: locale travels with the
document, so adding one more document never requires editing a second file.

A document's unique key is ``(locale, doc_id)``. The German and English
mirrors share the same logical ``doc_id`` — they are two channels for the
same client, not two different pieces of content.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from guardrails.types import Locale

__all__ = ["Document", "load_documents"]

_FRONT_MATTER = "---"


@dataclass(frozen=True, slots=True)
class Document:
    doc_id: str
    locale: Locale
    title: str
    version: str
    source_path: str
    body: str


def _parse(path: Path, root: Path) -> Document:
    """Parse a single document; any failure carries the file path in its
    message.

    ``_parse_body`` already hand-writes two ``ValueError``s with the path
    included (missing front matter, missing field) — those pass straight
    through unchanged. Every other failure mode (an unpack error from
    unclosed front matter, a ``ValueError`` from a misspelled locale, a
    ``yaml.YAMLError`` from malformed YAML itself) gets the path attached
    here in one place, instead of being handled separately at each failure
    site.
    """
    try:
        return _parse_body(path, root)
    except Exception as exc:
        message = str(exc)
        if message.startswith(str(path)):
            raise
        raise ValueError(f"{path}: {message}") from exc


def _parse_body(path: Path, root: Path) -> Document:
    text = path.read_text(encoding="utf-8")
    if not text.startswith(_FRONT_MATTER):
        raise ValueError(f"{path}: missing front matter")
    _, raw_meta, body = text.split(_FRONT_MATTER, 2)
    meta = yaml.safe_load(raw_meta) or {}
    missing = {"doc_id", "title", "locale", "version"} - set(meta)
    if missing:
        raise ValueError(f"{path}: front matter missing {sorted(missing)}")
    return Document(
        doc_id=str(meta["doc_id"]),
        locale=Locale(str(meta["locale"])),
        title=str(meta["title"]),
        version=str(meta["version"]),
        source_path=str(path.relative_to(root.parent)),
        body=body.strip(),
    )


def load_documents(root: Path) -> tuple[Document, ...]:
    """Load every document under ``root``, stably sorted by
    ``(locale, doc_id)``.

    The sort is deliberate: filesystem traversal order differs between
    machines, and chunk construction order flows into the trace. Same
    principle as M5's "verdicts come back in registry order, not completion
    order".
    """
    docs = [_parse(path, root) for path in sorted(root.rglob("*.md"))]
    return tuple(sorted(docs, key=lambda d: (d.locale.value, d.doc_id)))
