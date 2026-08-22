"""Chunking.

The chunking strategy is an independent source of false positives for the grounding
guard: split ``Tarif M`` and ``29,99`` into two different chunks, and the guard will
judge a **correct** answer as unsupported. So the tests here watch whether an entity and
what it describes stay together, not just how many chunks come out.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from guardrails.retrieval.chunks import chunk_document
from guardrails.retrieval.documents import Document, load_documents
from guardrails.types import Locale

BODY = """## Tarifübersicht

| Tarif | Preis pro Monat |
|---|---|
| Tarif M | 29,99 EUR |

## Tarifwechsel

Ein Wechsel ist jederzeit möglich.
"""


@pytest.fixture
def doc():
    return Document(
        doc_id="tarife-mobilfunk",
        locale=Locale.DE_DE,
        title="Mobilfunk-Tarife",
        version="2026-01-01",
        source_path="kb/de/tarife-mobilfunk.md",
        body=BODY.strip(),
    )


def test_splits_on_section_headings(doc):
    chunks = chunk_document(doc)
    assert [c.section for c in chunks] == ["Tarifübersicht", "Tarifwechsel"]


def test_chunk_id_carries_locale_and_is_deterministic(doc):
    first = chunk_document(doc)[0]
    assert first.chunk_id == "de-DE:tarife-mobilfunk#tarifuebersicht"
    assert chunk_document(doc)[0].chunk_id == first.chunk_id


def test_chunk_ids_do_not_collide_across_locales(doc):
    english = Document(
        doc_id=doc.doc_id, locale=Locale.EN_GB, title="Mobile tariffs",
        version=doc.version, source_path="kb/en/tarife-mobilfunk.md",
        body="## Tariff overview\n\n| Tariff | Price |\n|---|---|\n| M | 29.99 EUR |",
    )
    de_ids = {c.chunk_id for c in chunk_document(doc)}
    en_ids = {c.chunk_id for c in chunk_document(english)}
    assert not (de_ids & en_ids)


def test_tariff_name_and_price_stay_in_one_chunk(doc):
    """The single most important assertion in this module."""
    (chunk,) = [c for c in chunk_document(doc) if "Tarif M" in c.text]
    assert "29,99" in chunk.text


def test_chunk_carries_provenance(doc):
    chunk = chunk_document(doc)[0]
    assert chunk.doc_id == "tarife-mobilfunk"
    assert chunk.locale is Locale.DE_DE
    assert chunk.title == "Mobilfunk-Tarife"
    assert chunk.source_path == "kb/de/tarife-mobilfunk.md"


def test_long_section_splits_on_paragraphs_and_keeps_heading():
    paragraphs = "\n\n".join(f"Absatz {i} mit ausreichend vielen Wörtern darin." * 4
                             for i in range(12))
    doc = Document(
        doc_id="lang", locale=Locale.DE_DE, title="Lang", version="2026-01-01",
        source_path="kb/de/lang.md", body=f"## Langer Abschnitt\n\n{paragraphs}",
    )
    chunks = chunk_document(doc, max_words=180)
    assert len(chunks) > 1
    assert all(c.text.startswith("## Langer Abschnitt") for c in chunks)
    assert [c.chunk_id for c in chunks] == [
        f"de-DE:lang#langer-abschnitt#{i}" for i in range(len(chunks))
    ]


def test_table_rows_are_atomic():
    rows = "\n".join(f"| Tarif {i} | {i}9,99 EUR |" for i in range(60))
    doc = Document(
        doc_id="tab", locale=Locale.DE_DE, title="Tab", version="2026-01-01",
        source_path="kb/de/tab.md",
        body=f"## Tabelle\n\n| Tarif | Preis |\n|---|---|\n{rows}",
    )
    chunks = chunk_document(doc, max_words=180)
    assert len(chunks) > 1, "the fixture should force a second split, otherwise this test tests nothing"
    for chunk in chunks:
        lines = chunk.text.splitlines()
        for line in lines:
            if line.startswith("|"):
                assert line.endswith("|"), "a table row was cut in half"
        # Stronger check: no blank line may sit between two consecutive table rows --
        # once one does, GFM stops rendering them as the same table. The weak
        # assertion "line.startswith('|') and line.endswith('|')" alone would still
        # pass even with a blank line inserted between table rows, so it would never
        # catch Defect 3.
        table_line_indexes = [i for i, line in enumerate(lines) if line.startswith("|")]
        for a, b in zip(table_line_indexes, table_line_indexes[1:]):
            if b == a + 1:
                continue
            between = lines[a + 1:b]
            assert any(line.strip() for line in between), (
                "a blank line appeared between two adjacent table rows, tearing the table apart"
            )


def test_duplicate_section_titles_get_distinct_chunk_ids():
    body = "## Hinweise\n\nText A kurz.\n\n## Hinweise\n\nText B kurz."
    doc = Document(
        doc_id="dup", locale=Locale.DE_DE, title="Dup", version="2026-01-01",
        source_path="kb/de/dup.md", body=body,
    )
    ids = [c.chunk_id for c in chunk_document(doc)]
    assert len(ids) == len(set(ids))
    assert ids[0] == "de-DE:dup#hinweise"
    assert ids[1] == "de-DE:dup#hinweise-2"


def test_suffix_collision_does_not_produce_duplicates():
    """When a base slug appears twice and then its legitimate suffixed form appears,
    all three must be distinct: avoid collision where the third heading's natural slug
    matches the second heading's disambiguation suffix."""
    body = "## Hinweise\n\nA kurz.\n\n## Hinweise\n\nB kurz.\n\n## Hinweise 2\n\nC kurz."
    doc = Document(
        doc_id="x", locale=Locale.DE_DE, title="T", version="2026-01-01",
        source_path="p.md", body=body,
    )
    ids = [c.chunk_id for c in chunk_document(doc)]
    assert len(ids) == 3
    assert len(ids) == len(set(ids)), f"Collision detected: {ids}"
    assert ids[0] == "de-DE:x#hinweise"
    assert ids[1] == "de-DE:x#hinweise-2"
    assert ids[2] == "de-DE:x#hinweise-2-2"  # hinweise-2 was taken, so it becomes hinweise-2-2


def test_headings_differing_only_in_punctuation_get_distinct_chunk_ids():
    body = "## Preise & Tarife\n\nText A kurz.\n\n## Preise - Tarife\n\nText B kurz."
    doc = Document(
        doc_id="punct", locale=Locale.DE_DE, title="Punct", version="2026-01-01",
        source_path="kb/de/punct.md", body=body,
    )
    ids = [c.chunk_id for c in chunk_document(doc)]
    assert len(ids) == len(set(ids))


def test_non_latin_heading_gets_nonempty_slug_and_no_trailing_hash():
    body = "## 你好\n\nEin kurzer Text."
    doc = Document(
        doc_id="cjk", locale=Locale.DE_DE, title="CJK", version="2026-01-01",
        source_path="kb/de/cjk.md", body=body,
    )
    (chunk,) = chunk_document(doc)
    assert not chunk.chunk_id.endswith("#")
    slug = chunk.chunk_id.rsplit("#", 1)[1]
    assert slug != ""


def test_text_before_first_heading_is_kept():
    body = "Einleitender Text ohne Ueberschrift, der lang genug ist.\n\n## Abschnitt\n\nKoerper."
    doc = Document(
        doc_id="lead", locale=Locale.DE_DE, title="Lead", version="2026-01-01",
        source_path="kb/de/lead.md", body=body,
    )
    chunks = chunk_document(doc)
    assert any("Einleitender Text" in c.text for c in chunks)
    (lead_chunk,) = [c for c in chunks if "Einleitender Text" in c.text]
    assert not lead_chunk.text.startswith("## ")
    assert lead_chunk.chunk_id == "de-DE:lead#abschnitt-0"


def test_real_corpus_chunk_ids_are_unchanged():
    """Regression anchor: this fix must not change any existing chunk_id in the real
    corpus."""
    docs = load_documents(Path("kb"))
    doc = next(d for d in docs if d.doc_id == "tarife-mobilfunk" and d.locale is Locale.DE_DE)
    ids = [c.chunk_id for c in chunk_document(doc)]
    assert ids == [
        "de-DE:tarife-mobilfunk#tarifuebersicht",
        "de-DE:tarife-mobilfunk#tarifwechsel",
    ]
    tk = next(d for d in docs if d.doc_id == "vertragslaufzeit-kuendigung" and d.locale is Locale.DE_DE)
    tk_ids = [c.chunk_id for c in chunk_document(tk)]
    assert tk_ids == [
        "de-DE:vertragslaufzeit-kuendigung#gesetzliche-vorgaben-zur-kuendigungsfrist",
        "de-DE:vertragslaufzeit-kuendigung#unsere-vertragsbedingungen",
        "de-DE:vertragslaufzeit-kuendigung#kuendigungswege",
    ]
