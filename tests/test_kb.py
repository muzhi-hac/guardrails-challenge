"""Structural constraints on the knowledge base.

Whether the content itself is correct is a human's job to read; what's tested here are
the properties that, once broken, distort everything downstream silently: unique keys,
DE/EN fact agreement, and entity density -- a corpus with sparse entities gives the
grounding guard nothing to test against.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from guardrails.locale import get_rules
from guardrails.retrieval.documents import load_documents
from guardrails.types import EntityKind, Locale

KB_ROOT = Path(__file__).resolve().parents[1] / "kb"

MIRRORED_DOC_IDS = frozenset({
    "tarife-mobilfunk",
    "vertragslaufzeit-kuendigung",
    "roaming-eu",
    "rechnung-zahlungsarten",
    "stoerung-entstoerfrist",
    "umzug",
})

# The fact table is pinned by doc_id -- the grounding guard checks prices in a reply
# against these literal values, so every number in the corpus is a spec, not a detail
# that's satisfied once you've scraped together ">=3 entities".
#
# Comparing only "the German price set == the English price set" (see
# test_mirrored_prices_match_across_locales) cannot catch "both sides broken together":
# change 19,99 EUR and 19.99 EUR together into 18,99 / 18.99, and DE still equals EN, so
# that test would stay green. Here the expected values read out of the documents are
# written as literals, so a deviation on either side fails, regardless of whether the
# other side drifted in step.
#
# Each value is read out of the corresponding document's body, not invented:
#   tarife-mobilfunk           tariff table, three monthly prices: 19,99 / 29,99 / 49,99 EUR
#   rechnung-zahlungsarten     "Zahlungsverzug und Mahngebühr": first reminder fee 5,00 EUR
#   roaming-eu                 "Roaming außerhalb der EU": 0,49 EUR per MB outside the EU
#   umzug                      "Umzugsservice": one-off moving fee 29,90 EUR
#   datenvolumen-drosselung    "Datenautomatik": extra 1 GB auto-billed at 3,00 EUR
#                              (German only, no English mirror -- not in MIRRORED_DOC_IDS)
PRICE_EXPECTATIONS: dict[str, frozenset[str]] = {
    "tarife-mobilfunk": frozenset({"19.99 EUR", "29.99 EUR", "49.99 EUR"}),
    "rechnung-zahlungsarten": frozenset({"5.00 EUR"}),
    "roaming-eu": frozenset({"0.49 EUR"}),
    "umzug": frozenset({"29.90 EUR"}),
    "datenvolumen-drosselung": frozenset({"3.00 EUR"}),
}


@pytest.fixture(scope="module")
def documents():
    return load_documents(KB_ROOT)


def test_loads_expected_counts(documents):
    by_locale = {loc: [d for d in documents if d.locale is loc] for loc in Locale}
    assert len(by_locale[Locale.DE_DE]) == 12
    assert len(by_locale[Locale.EN_GB]) == 6


def test_unique_key_is_locale_and_doc_id(documents):
    keys = [(d.locale, d.doc_id) for d in documents]
    assert len(keys) == len(set(keys))


def test_english_mirrors_reuse_german_doc_ids(documents):
    """A mirror shares its logical doc_id with the original -- locale, not renaming,
    is what tells them apart."""
    english = {d.doc_id for d in documents if d.locale is Locale.EN_GB}
    german = {d.doc_id for d in documents if d.locale is Locale.DE_DE}
    assert english == MIRRORED_DOC_IDS
    assert english <= german


def test_front_matter_complete(documents):
    for doc in documents:
        assert doc.title.strip(), doc.source_path
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", doc.version), doc.source_path


def test_every_document_has_at_least_three_checkable_entities(documents):
    """Counted via extract_entities, not by manual judgment."""
    for doc in documents:
        rules = get_rules(doc.locale)
        mentions = rules.extract_entities(doc.body, tuple(EntityKind))
        assert len(mentions) >= 3, f"{doc.source_path}: {len(mentions)}"


def test_mirrored_prices_match_across_locales(documents):
    """The price sets for the same doc_id in German and English must match item for
    item, or a cross-locale evaluation ends up measuring content differences instead."""
    def prices(doc):
        rules = get_rules(doc.locale)
        return {
            m.normalized
            for m in rules.extract_entities(doc.body, (EntityKind.PRICE,))
        }

    by_key = {(d.locale, d.doc_id): d for d in documents}
    for doc_id in MIRRORED_DOC_IDS:
        de = prices(by_key[(Locale.DE_DE, doc_id)])
        en = prices(by_key[(Locale.EN_GB, doc_id)])
        assert de == en, doc_id


def test_mirrored_durations_match_across_locales(documents):
    """Same rationale as ``test_mirrored_prices_match_across_locales``, extended
    to DURATION: the 48-hour repair deadline, the 24-month minimum term, the
    14-day withdrawal window and other duration facts must agree exactly
    between the German original and its English mirror, not just superficially
    resemble each other."""
    def durations(doc):
        rules = get_rules(doc.locale)
        return {
            m.normalized
            for m in rules.extract_entities(doc.body, (EntityKind.DURATION,))
        }

    by_key = {(d.locale, d.doc_id): d for d in documents}
    for doc_id in MIRRORED_DOC_IDS:
        de = durations(by_key[(Locale.DE_DE, doc_id)])
        en = durations(by_key[(Locale.EN_GB, doc_id)])
        assert de == en, doc_id


def test_mirrored_numbers_match_across_locales(documents):
    """Same rationale as ``test_mirrored_prices_match_across_locales``, extended
    to NUMBER: the 30 GB fair-use limit and other bare-number facts must agree
    exactly between the German original and its English mirror."""
    def numbers(doc):
        rules = get_rules(doc.locale)
        return {
            m.normalized
            for m in rules.extract_entities(doc.body, (EntityKind.NUMBER,))
        }

    by_key = {(d.locale, d.doc_id): d for d in documents}
    for doc_id in MIRRORED_DOC_IDS:
        de = numbers(by_key[(Locale.DE_DE, doc_id)])
        en = numbers(by_key[(Locale.EN_GB, doc_id)])
        assert de == en, doc_id


def test_price_fact_table_matches_documents(documents):
    """Pins the fact table's exact values into the test, rather than only comparing
    whether the German and English sides agree with each other.

    ``test_mirrored_prices_match_across_locales`` can only catch "DE and EN disagree",
    not "DE and EN were broken together": change 19,99 EUR, along with its English
    mirror, into 18,99 EUR on both sides, and the German set still equals the English
    set -- that test stays green. The fact table in the corpus is the very spec the
    grounding guard compares replies against, so here the expected values read out of
    the documents are written as literal constants (``PRICE_EXPECTATIONS``); a
    deviation from the literal on either side must fail, regardless of whether the
    other side drifted in step.
    """

    def prices(doc):
        rules = get_rules(doc.locale)
        return {
            m.normalized
            for m in rules.extract_entities(doc.body, (EntityKind.PRICE,))
        }

    by_doc_id: dict[str, list] = {}
    for doc in documents:
        by_doc_id.setdefault(doc.doc_id, []).append(doc)

    for doc_id, expected in PRICE_EXPECTATIONS.items():
        # The expected set itself must not be empty -- an empty set would let the
        # equality assertion below pass even when the document has no price entities
        # at all, reproducing the "vacuous comparison" problem described in finding 1.
        assert expected, f"{doc_id}: expected price set must not be empty"
        matching = by_doc_id.get(doc_id, [])
        assert matching, f"{doc_id}: no document with this doc_id was loaded"
        for doc in matching:
            assert prices(doc) == expected, (doc.source_path, doc_id)


def test_no_document_contains_credentials_or_endpoints(documents):
    """The corpus is deliverable content -- no credentials or endpoint names may leak
    into it.

    This test enforces the policy by pattern rather than by naming specific
    forbidden strings. A test that hardcodes the secret becomes the repository's
    only violation of the rule it enforces; the acceptance grep then cannot pass.
    Instead, we detect the *shape* of common violations:

    - URLs: http://, https://, www. — customer-service prose has no legitimate
      reason to reference external hosts.
    - API-key-shaped tokens: vendor tag (2-3 lowercase letters) + separator
      (dash or underscore) + long alphanumeric string. We use [a-z]{2,3} for
      the prefix rather than naming a specific vendor, so this test file itself
      cannot be mistaken for a credential.
    """
    # Patterns that detect the *shape* of violations, not named secrets.
    # URL pattern: http://, https://, or www followed by domain chars.
    url_pattern = re.compile(r'https?://|www\.')
    # API key pattern: vendor tag (2-3 lowercase) + dash/underscore + 20+ alnum.
    # Built from character classes, not hardcoded vendor prefixes.
    api_key_pattern = re.compile(r'[a-z]{2,3}[-_][a-zA-Z0-9_\-]{20,}')

    for doc in documents:
        # Check body, title, and version for URLs and API-key-shaped tokens.
        text_fields = [doc.body, doc.title, doc.version]
        for field_name, text in zip(['body', 'title', 'version'], text_fields):
            assert not url_pattern.search(text), \
                f"{doc.source_path} {field_name}: no URLs allowed"
            assert not api_key_pattern.search(text), \
                f"{doc.source_path} {field_name}: no API-key-shaped tokens allowed"


def test_unterminated_front_matter_names_the_file(tmp_path):
    """Front matter opened but never closed: the error must name the offending file,
    not a bare unpack error."""
    bad = tmp_path / "broken.md"
    bad.write_text(
        "---\ndoc_id: x\ntitle: X\nlocale: de-DE\nversion: 2026-01-01\n"
        "no closing delimiter here\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=re.escape(str(bad))):
        load_documents(tmp_path)


def test_invalid_locale_names_the_file(tmp_path):
    """A misspelled locale (e.g. de_DE instead of de-DE): the error must name the
    offending file."""
    bad = tmp_path / "typo-locale.md"
    bad.write_text(
        "---\ndoc_id: x\ntitle: X\nlocale: de_DE\nversion: 2026-01-01\n---\nBody.\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=re.escape(str(bad))):
        load_documents(tmp_path)


def test_malformed_yaml_names_the_file(tmp_path):
    """Malformed YAML itself (e.g. an indentation/syntax error from a missing value
    after a colon): the error must name the file."""
    bad = tmp_path / "bad-yaml.md"
    bad.write_text(
        "---\ndoc_id: x\ntitle: [unclosed\nlocale: de-DE\nversion: 2026-01-01\n---\nBody.\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=re.escape(str(bad))):
        load_documents(tmp_path)
