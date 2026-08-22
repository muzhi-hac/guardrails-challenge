"""Fixed query set for retrieval recall.

Judged by ``doc_id``, not tied to ``chunk_id`` or to full ranking order: which section of
a document gets retrieved is an implementation detail of the chunking strategy, and
pinning it into assertions would turn every chunking adjustment into a false alarm of
retrieval regression.

A query is allowed to have more than one correct document -- ``tarife-mobilfunk`` and
``vertragslaufzeit-kuendigung`` necessarily share duration facts, and forcing a single
correct document would write the content's organization into the assertion.

**Paraphrase-style queries are deliberately included**: ``known_limitation`` classifies
by the lexical relationship between query and document -- whether the query uses the
document's own terminology or not -- rather than by expected difficulty ("will this one
hit"). The classification describes only the input; whether it hits is left for the
measurement to report, so the resulting numbers are a finding, not a conclusion already
baked into the classification.

**Derivational forms are a second, distinct known limitation, with a different boundary
than paraphrase.** The German tokenizer restores inflection through lexicon-checked
stemming (e.g. a noun's case/number suffixes), but does not restore derivational
morphology: tokenizing a verb (``gedrosselt``) never reaches its corresponding noun
(``Drosselung``), because stemming only strips registered noun inflection suffixes and
then checks the result against the lexicon -- ``gedrosselt`` itself is not a lexicon
entry, and there is no mechanism to strip the verbal "ge-" prefix. "Ab wann wird
gedrosselt?" and "Wann beginnt die Drosselung meines Datenvolumens?" are two phrasings of
the same customer intent; the first is a real, natural German question that happens to
sit right on this derivational boundary. Keeping both phrasings in the set is
deliberate: keeping only the one that hits would hide where the boundary is; keeping both
writes the boundary into the numbers.

**Measured result**: of the five limitation cases, three hit within k=5, two of them at
rank 1 (see the per-case numbers printed by ``test_recall_report``). Lexical retrieval is
more robust to colloquial phrasing than the classification alone would suggest, because a
customer's actual question usually still carries at least one domain noun that survives
tokenization -- ``Anschluss`` ("Ich ziehe um, was passiert mit meinem Anschluss?" hits
``kb/de/umzug.md``), ``moving`` ("I am moving house" hits ``kb/en/umzug.md``, titled
"Moving home"). Of the two cases that miss entirely, one is missing every distinguishing
word ("Wie komme ich aus meinem Vertrag heraus?"), and the other falls on the far side of
the derivational boundary (``gedrosselt``) -- these are two different failure causes, and
the two ``known_limitation`` values exist precisely to keep that difference visible in
the numbers.
"""

from __future__ import annotations

from typing import NamedTuple

from guardrails.types import Locale


class RecallCase(NamedTuple):
    query: str
    expected_doc_ids: frozenset[str]
    locale: Locale
    known_limitation: str = ""
    """Classifies by the lexical relationship between query and document, not by
    expected difficulty:

    empty string -- the query uses the document's own terminology: the distinguishing
    domain words appear in both query and document.

    ``paraphrase`` -- the query is the customer's own phrasing, not the document's
    terminology; the distinguishing domain words are missing or weak. This is not the
    same as zero lexical overlap: a customer's question usually still carries at least
    one shared word, and two cases in this category do exactly that (``Anschluss``,
    ``moving``).

    ``derivation`` -- the query uses a morphological variant that the tokenizer cannot
    cross: the tokenizer restores inflection through lexicon-checked stemming, but does
    not restore derivational morphology. ``gedrosselt`` (a tokenized verb) and
    ``Drosselung`` (the noun) therefore land on disjoint tokens."""


DE = Locale.DE_DE
EN = Locale.EN_GB

RECALL_QUERIES: tuple[RecallCase, ...] = (
    # --- German: exact terms ---
    RecallCase("Was kostet Tarif M?", frozenset({"tarife-mobilfunk"}), DE),
    RecallCase("Wie hoch ist die Kündigungsfrist?",
               frozenset({"vertragslaufzeit-kuendigung"}), DE),
    RecallCase("Mindestlaufzeit meines Vertrags",
               frozenset({"vertragslaufzeit-kuendigung", "tarife-mobilfunk"}), DE),
    RecallCase("Roaming außerhalb der EU Kosten", frozenset({"roaming-eu"}), DE),
    RecallCase("Wann kommt meine Rechnung?", frozenset({"rechnung-zahlungsarten"}), DE),
    RecallCase("Entstörfrist bei Störung", frozenset({"stoerung-entstoerfrist"}), DE),
    RecallCase("Rufnummernmitnahme Dauer", frozenset({"rufnummernmitnahme"}), DE),
    RecallCase("Wann beginnt die Drosselung meines Datenvolumens?",
               frozenset({"datenvolumen-drosselung"}), DE),
    RecallCase("Widerrufsfrist 14 Tage", frozenset({"widerrufsrecht"}), DE),
    RecallCase("Wann erreiche ich den Kundenservice?", frozenset({"servicezeiten"}), DE),
    # --- German: ASCII umlaut input (real user behavior) ---
    RecallCase("Kuendigung Frist", frozenset({"vertragslaufzeit-kuendigung"}), DE),
    # --- German: paraphrase-style (known limitation: paraphrase, query uses the customer's own wording rather than document terms) ---
    RecallCase("Wie komme ich aus meinem Vertrag heraus?",
               frozenset({"vertragslaufzeit-kuendigung"}), DE,
               known_limitation="paraphrase"),
    RecallCase("Ich ziehe um, was passiert mit meinem Anschluss?",
               frozenset({"umzug"}), DE, known_limitation="paraphrase"),
    RecallCase("Mein Internet ist seit gestern weg",
               frozenset({"stoerung-entstoerfrist"}), DE,
               known_limitation="paraphrase"),
    # --- German: derivational-form (known limitation: derivation, verb tokenization cannot reach the corresponding noun) ---
    RecallCase("Ab wann wird gedrosselt?",
               frozenset({"datenvolumen-drosselung"}), DE,
               known_limitation="derivation"),
    # --- English ---
    RecallCase("How much does Tariff M cost?", frozenset({"tarife-mobilfunk"}), EN),
    RecallCase("notice period for cancellation",
               frozenset({"vertragslaufzeit-kuendigung"}), EN),
    RecallCase("roaming outside the EU", frozenset({"roaming-eu"}), EN),
    RecallCase("when is my invoice issued", frozenset({"rechnung-zahlungsarten"}), EN),
    RecallCase("how long to fix a fault", frozenset({"stoerung-entstoerfrist"}), EN),
    RecallCase("I am moving house", frozenset({"umzug"}), EN,
               known_limitation="paraphrase"),
)
