"""Splitting documents into retrieval units.

Splitting follows Markdown ``##`` sections, not a fixed token count. The
reason comes directly from a failure mode: splitting ``Tarif M`` and
``29,99 EUR`` into two different chunks makes the grounding guard call a
**correct** answer unsupported. That is a false positive attributable to the
chunking strategy, and it is the same class of error as a false positive
caused by a retrieval miss — just from a different source.

``chunk_id`` uses a slug rather than a content hash: a trace has to be read
by people. The stability a hash would seem to offer is illusory here —
fixing a typo changes the content hash entirely, whereas a slug only changes
when the section heading itself genuinely changes, and that is exactly when
the id *should* change.
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


class _Atom(NamedTuple):
    """An atomic, indivisible unit, tagged with its kind.

    Once ``_atoms`` has decided that a block is "line-wise" (a table row, a
    list item) or "paragraph-wise" (prose), that kind must survive to
    ``_pack``, or reassembly won't know whether to join with ``"\\n"`` or
    ``"\\n\\n"`` — getting the blank line wrong tears a table apart across
    chunks, see Defect 3.
    """
    text: str
    kind: str  # "line" | "block"


def slugify(text: str) -> str:
    """``Tarifübersicht`` -> ``tarifuebersicht``.

    Umlauts are transliterated per German convention rather than dropped —
    ``tarifbersicht`` is neither readable nor traceable back to the original.
    """
    lowered = text.casefold()
    for source, target in _UMLAUT_SLUG.items():
        lowered = lowered.replace(source, target)
    ascii_only = unicodedata.normalize("NFKD", lowered).encode("ascii", "ignore").decode()
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", ascii_only)).strip("-")


def _atoms(body: str) -> list[_Atom]:
    """Paragraphs, list items, table rows — a further split must never cut
    through one of these units.

    A table row is naturally the minimal complete unit of "tariff name +
    price". Each atom is tagged with its kind (``line`` or ``block``), for
    ``_pack`` to decide which separator to reassemble it with.
    """
    atoms: list[_Atom] = []
    for block in re.split(r"\n\s*\n", body.strip()):
        lines = block.splitlines()
        if all(line.lstrip().startswith(("|", "-", "*")) for line in lines if line.strip()):
            atoms.extend(_Atom(line, "line") for line in lines if line.strip())
        else:
            stripped = block.strip()
            if stripped:
                atoms.append(_Atom(stripped, "block"))
    return atoms


def chunk_document(doc: Document, *, max_words: int = 180) -> tuple[Chunk, ...]:
    sections = _sections(doc.body)
    seen: dict[str, int] = {}
    used: set[str] = set()
    chunks: list[Chunk] = []
    for n, (section_title, section_body) in enumerate(sections):
        # An empty heading (the body before the first heading, see _sections)
        # falls back to the positional abschnitt-{n}; a slug that repeats
        # within the same document (whether or not it came from that
        # fallback) gets -2, -3... appended from its second occurrence
        # onward. The first occurrence keeps no suffix, so existing
        # chunk_ids in the corpus are unaffected.
        base = slugify(section_title) or f"abschnitt-{n}"
        seen[base] = seen.get(base, 0) + 1
        slug = base if seen[base] == 1 else f"{base}-{seen[base]}"
        while slug in used:
            seen[base] += 1
            slug = f"{base}-{seen[base]}"
        used.add(slug)
        base_id = f"{doc.locale.value}:{doc.doc_id}#{slug}"
        heading = f"## {section_title}" if section_title else ""
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
    # The body before the first ``##`` is not a continuation of some
    # document-level heading — it is content in its own right. The corpus
    # currently follows the convention that a body always opens with a
    # heading, but that is a convention, not a contract (see documents.py),
    # and dropping this content is Defect 2. It gets a "section" with an
    # empty title; an empty title falls to abschnitt-0 in chunk_document.
    lead = body[:matches[0].start()].strip()
    if lead:
        out.append(("", lead))
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        out.append((match.group("title"), body[match.end():end].strip()))
    return out


def _join(atoms: list[_Atom]) -> str:
    """Reassemble text by kind: consecutive ``line`` atoms are joined with a
    single newline; everything else (``block`` atoms, and the boundary
    between a block and a line) is separated by a blank line.

    A table row is a ``line`` atom (see ``_atoms``); insert a blank line
    between two table rows and GFM stops rendering them as the same table —
    exactly Defect 3.
    """
    pieces: list[str] = []
    for i, atom in enumerate(atoms):
        if i == 0:
            pieces.append(atom.text)
            continue
        sep = "\n" if atom.kind == "line" and atoms[i - 1].kind == "line" else "\n\n"
        pieces.append(sep)
        pieces.append(atom.text)
    return "".join(pieces)


def _pack(atoms: list[_Atom], max_words: int, heading: str) -> list[str]:
    """Pack atoms into pieces no longer than ``max_words``, each piece
    prefixed with its section heading.

    The heading prefix keeps a piece readable on its own, and still lets
    BM25 match this piece via heading words.
    """
    parts: list[str] = []
    current: list[_Atom] = []
    # The budget can be non-positive when the heading is very long; the spec
    # requires that an atom (a paragraph / table row) is never split, so the
    # first atom is always accepted unconditionally — a single piece is
    # allowed to exceed budget sooner than atomicity is allowed to break.
    budget = max_words - count_words(heading)
    for atom in atoms:
        if current and count_words(_join(current)) + count_words(atom.text) > budget:
            parts.append(f"{heading}\n\n{_join(current)}".strip())
            current = []
        current.append(atom)
    if current:
        parts.append(f"{heading}\n\n{_join(current)}".strip())
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
