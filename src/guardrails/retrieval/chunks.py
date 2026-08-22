"""把文档切成检索单元。

按 Markdown 的 ``##`` 小节切，不按固定 token 数。理由直接来自失效模式：把
``Tarif M`` 和 ``29,99 EUR`` 切进两个块，溯源守卫会把一个**正确**的回答判成无依据。
这是可归因于分块策略的假阳性，和检索 miss 导致的误报是同一类错误的不同来源。

``chunk_id`` 用 slug 而不是内容哈希：trace 要人读。哈希的稳定性优势在这里是假的 ——
改一个错别字会让内容哈希全变，而 slug 只在小节标题真的改了时才变，那时候 id 变化
是正确的。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Generic, NamedTuple, TypeVar

from guardrails.locale.base import count_words
from guardrails.retrieval.documents import Document
from guardrails.types import Locale

__all__ = ["Chunk", "Scored", "chunk_document", "slugify"]

T = TypeVar("T")

_HEADING_RE = re.compile(r"^##\s+(?P<title>.+?)\s*$", re.MULTILINE)
_UMLAUT_SLUG = {"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"}


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    locale: Locale
    title: str
    section: str
    source_path: str
    text: str


class Scored(NamedTuple, Generic[T]):
    item: T
    score: float


def slugify(text: str) -> str:
    """``Tarifübersicht`` -> ``tarifuebersicht``。

    变音符按德语惯例转写而不是丢弃 —— ``tarifbersicht`` 既不可读也不可反查。
    """
    lowered = text.casefold()
    for source, target in _UMLAUT_SLUG.items():
        lowered = lowered.replace(source, target)
    ascii_only = unicodedata.normalize("NFKD", lowered).encode("ascii", "ignore").decode()
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", ascii_only)).strip("-")


def _atoms(body: str) -> list[str]:
    """段落、列表项、表格行 —— 二次切分不得穿过这些单位。

    一个表格行天然是「资费名 + 价格」的最小完整单位。
    """
    atoms: list[str] = []
    for block in re.split(r"\n\s*\n", body.strip()):
        lines = block.splitlines()
        if all(line.lstrip().startswith(("|", "-", "*")) for line in lines if line.strip()):
            atoms.extend(line for line in lines if line.strip())
        else:
            atoms.append(block.strip())
    return [a for a in atoms if a]


def chunk_document(doc: Document, *, max_words: int = 180) -> tuple[Chunk, ...]:
    sections = _sections(doc.body)
    chunks: list[Chunk] = []
    for section_title, section_body in sections:
        heading = f"## {section_title}"
        base_id = f"{doc.locale.value}:{doc.doc_id}#{slugify(section_title)}"
        whole = f"{heading}\n\n{section_body}".strip()
        if count_words(whole) <= max_words:
            chunks.append(_make(doc, base_id, section_title, whole))
            continue
        for index, part in enumerate(_pack(_atoms(section_body), max_words, heading)):
            chunks.append(_make(doc, f"{base_id}#{index}", section_title, part))
    return tuple(chunks)


def _sections(body: str) -> list[tuple[str, str]]:
    matches = list(_HEADING_RE.finditer(body))
    if not matches:
        return [("", body.strip())]
    out: list[tuple[str, str]] = []
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        out.append((match.group("title"), body[match.end():end].strip()))
    return out


def _pack(atoms: list[str], max_words: int, heading: str) -> list[str]:
    """把原子装进不超过 ``max_words`` 的片，每片都带上小节标题前缀。

    标题前缀让片段自身可读，也让 BM25 仍能通过标题词命中这一片。
    """
    parts: list[str] = []
    current: list[str] = []
    budget = max_words - count_words(heading)
    for atom in atoms:
        if current and count_words(" ".join(current)) + count_words(atom) > budget:
            parts.append(f"{heading}\n\n" + "\n\n".join(current))
            current = []
        current.append(atom)
    if current:
        parts.append(f"{heading}\n\n" + "\n\n".join(current))
    return parts


def _make(doc: Document, chunk_id: str, section: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc.doc_id,
        locale=doc.locale,
        title=doc.title,
        section=section,
        source_path=doc.source_path,
        text=text,
    )
