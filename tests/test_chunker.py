"""分块。

分块策略是溯源守卫假阳性的一个独立来源：把 ``Tarif M`` 和 ``29,99`` 切进两个块，
守卫会把一个**正确**的回答判成无依据。所以这里的测试盯的是「实体和它描述的东西
是否还在一起」，而不只是切了几块。
"""

from __future__ import annotations

import pytest

from guardrails.retrieval.chunks import chunk_document
from guardrails.retrieval.documents import Document
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
    """本模块最重要的一条断言。"""
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
    for chunk in chunk_document(doc, max_words=180):
        for line in chunk.text.splitlines():
            if line.startswith("|"):
                assert line.endswith("|"), "表格行被切断"
