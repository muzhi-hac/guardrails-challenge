"""BM25 retrieval.

recall@k is tested separately in Task 8; what's tested here are properties of the
retriever itself, especially **determinism** -- if the order of tied results depends on
dict iteration order or floating-point jitter, the recall tests would produce different
numbers on different machines, and those numbers are exactly the core metric this
module is meant to produce.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from guardrails.retrieval.bm25 import Bm25Retriever
from guardrails.retrieval.chunks import Chunk
from guardrails.retrieval.knowledge_base import KnowledgeBase
from guardrails.locale import get_rules
from guardrails.types import Locale

KB_ROOT = Path(__file__).resolve().parents[1] / "kb"


@pytest.fixture(scope="module")
def kb():
    return KnowledgeBase.load(KB_ROOT)


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(chunk_id=chunk_id, doc_id="d", locale=Locale.DE_DE, title="t",
                 section="s", source_path="p", text=text)


def test_returns_at_most_k(kb):
    assert len(kb.for_locale(Locale.DE_DE).search("Kündigungsfrist", k=3)) <= 3


def test_scores_are_descending(kb):
    results = kb.for_locale(Locale.DE_DE).search("Tarif M Preis", k=5)
    assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)


def test_exact_term_wins():
    """The core claim of §3.8: an exact term must be able to push the correct
    document to the top.

    The corpus has five chunks, not two: BM25Okapi's classic idf is
    ``ln((N - df + 0.5) / (df + 0.5))``. When a term appears in exactly half the
    corpus's documents (one out of two), df=1, N=2 gives an idf of exactly 0 -- the
    score is wiped out entirely, and the test would no longer be testing "can an
    exact term win". With one occurrence out of five (df=1, N=5), idf ≈
    ln(4.5/1.5) ≈ 1.10, the right order of magnitude for a real corpus. The three
    distractor sentences contain no "M".
    """
    rules = get_rules(Locale.DE_DE)
    retriever = Bm25Retriever(
        [_chunk("a", "Tarif S kostet 19,99 EUR."),
         _chunk("b", "Tarif M kostet 29,99 EUR."),
         _chunk("c", "Der Kundenservice ist telefonisch von 8 bis 20 Uhr erreichbar."),
         _chunk("d", "Die Rechnung wird monatlich per E-Mail versendet."),
         _chunk("e", "Bei Fragen zur SIM-Karte hilft das Servicecenter weiter.")],
        rules,
    )
    assert retriever.search("Tarif M", k=1)[0].item.chunk_id == "b"


def test_compound_query_matches_component_document():
    """The retrieval-side acceptance test for tokenizer closure: querying a
    compound word hits a document that only ever writes the component words.

    The corpus has five chunks, not two: as above, "kündigung"/"frist" appear in
    only one document, and with a two-document corpus df=1, N=2 would push idf to
    exactly 0, masking what this test is actually meant to verify (whether compound
    decomposition makes the hit possible). Five documents (df=1, N=5) gives a
    healthy positive idf. The three distractor sentences contain neither
    "Kündigung" nor "Frist".
    """
    rules = get_rules(Locale.DE_DE)
    retriever = Bm25Retriever(
        [_chunk("a", "Die Frist zur Kündigung beträgt einen Monat."),
         _chunk("b", "Der Tarif enthält 20 GB Datenvolumen."),
         _chunk("c", "Die SIM-Karte wird innerhalb von drei Werktagen versendet."),
         _chunk("d", "Der Kundenservice beantwortet Anfragen per Chat und Telefon."),
         _chunk("e", "Ein Tarifwechsel ist online im Kundenportal möglich.")],
        rules,
    )
    assert retriever.search("Kündigungsfrist", k=1)[0].item.chunk_id == "a"


def test_ascii_query_matches_umlaut_document():
    """The corpus has five chunks, not two, for the same reason as the two tests
    above: with a two-document corpus, "kündigung" at df=1, N=2 would push idf to
    exactly 0. Five documents (df=1, N=5) gives a healthy positive idf, so what's
    actually being tested is whether ASCII normalization hits a document with an
    umlaut, not "did the score come out zero". The three distractor sentences
    contain no "Kündigung".
    """
    rules = get_rules(Locale.DE_DE)
    retriever = Bm25Retriever(
        [_chunk("a", "Die Kündigungsfrist beträgt einen Monat."),
         _chunk("b", "Der Tarif enthält 20 GB Datenvolumen."),
         _chunk("c", "Der Kundenservice ist telefonisch von 8 bis 20 Uhr erreichbar."),
         _chunk("d", "Die Rechnung wird monatlich per E-Mail versendet."),
         _chunk("e", "Bei Fragen zur SIM-Karte hilft das Servicecenter weiter.")],
        rules,
    )
    assert retriever.search("Kuendigung", k=1)[0].item.chunk_id == "a"


def test_isolated_term_gets_positive_idf_at_five_chunks():
    """Records the reason: why the corpus in these tests is five documents, not
    two.

    BM25Okapi's idf is exactly 0 when a term appears in exactly half the corpus's
    documents (df=1, N=2) -- that is a degenerate property of a two-document toy
    corpus, not a property of the production corpus (in the real corpus, only 1.6%
    of 747 terms have idf <= 0, and none of them are domain terms). This pins down
    that mathematical relationship directly: for a term occurring in exactly one of
    five documents, idf must be strictly positive, and so must the score. If someone
    shrinks the corpus back down to two documents, this test and the three above it
    will all go red together, instead of all silently returning a score of zero.
    """
    rules = get_rules(Locale.DE_DE)
    retriever = Bm25Retriever(
        [_chunk("a", "Der Tarif Roaming kostet 5 EUR pro Tag."),
         _chunk("b", "Die Rechnung wird monatlich per E-Mail versendet."),
         _chunk("c", "Der Kundenservice ist telefonisch von 8 bis 20 Uhr erreichbar."),
         _chunk("d", "Die SIM-Karte wird innerhalb von drei Werktagen versendet."),
         _chunk("e", "Ein Tarifwechsel ist online im Kundenportal möglich.")],
        rules,
    )
    results = retriever.search("Roaming", k=1)
    assert results
    assert results[0].item.chunk_id == "a"
    assert results[0].score > 0.0


def test_ties_break_deterministically():
    """Ties are broken by chunk_id -- otherwise the recall numbers would differ
    across machines.

    This test originally used a two-document corpus too, both documents containing
    "Kündigung" -- i.e. df=2, N=2. When a term appears in **every** document in the
    corpus, BM25Okapi's idf is ``ln((N-df+0.5)/(df+0.5)) = ln(0.5/2.5) < 0``, so
    both tied chunks' scores turn negative, get filtered out by the retriever's
    positive-score filter, and both sides return an empty list -- the test would no
    longer be checking "how are ties broken" but "did nothing match at all". Adding
    three distractor sentences without "Kündigung" stretches df=2, N=5 to idf ≈
    ln(3.5/2.5) > 0, restoring both tied chunks to a positive, equal score, so the
    test genuinely exercises tie-breaking.
    """
    rules = get_rules(Locale.DE_DE)
    distractors = [
        _chunk("c", "Die Rechnung wird monatlich per E-Mail versendet."),
        _chunk("d", "Der Kundenservice ist telefonisch von 8 bis 20 Uhr erreichbar."),
        _chunk("e", "Bei Fragen zur SIM-Karte hilft das Servicecenter weiter."),
    ]
    chunks = [_chunk("z", "Kündigung"), _chunk("a", "Kündigung"), *distractors]
    forward = Bm25Retriever(chunks, rules).search("Kündigung", k=2)
    backward = Bm25Retriever(
        [_chunk("a", "Kündigung"), _chunk("z", "Kündigung"), *distractors], rules
    ).search("Kündigung", k=2)
    assert [r.item.chunk_id for r in forward] == ["a", "z"]
    assert [r.item.chunk_id for r in backward] == ["a", "z"]


def test_locales_are_isolated(kb):
    """An English query must not retrieve a German document -- the two channels'
    indexes are kept separate."""
    results = kb.for_locale(Locale.EN_GB).search("cancellation notice period", k=5)
    assert results
    assert all(r.item.locale is Locale.EN_GB for r in results)


def test_unknown_locale_names_what_is_available(kb):
    class Fake:
        value = "fr-FR"
    with pytest.raises(KeyError, match="de-DE"):
        kb.for_locale(Fake())  # type: ignore[arg-type]


def test_none_locale_raises_keyerror(kb):
    """None is a real-world bad argument -- a profile field that was never filled
    in."""
    with pytest.raises(KeyError, match="de-DE"):
        kb.for_locale(None)  # type: ignore[arg-type]


def test_unhashable_locale_raises_keyerror(kb):
    """An unhashable argument must also produce an informative KeyError."""
    with pytest.raises(KeyError, match="de-DE"):
        kb.for_locale(['not', 'hashable'])  # type: ignore[arg-type]


def test_empty_query_returns_nothing(kb):
    assert kb.for_locale(Locale.DE_DE).search("", k=5) == ()
