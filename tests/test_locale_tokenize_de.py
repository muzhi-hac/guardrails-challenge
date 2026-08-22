"""German tokenization: a six-step pipeline.

The pipeline order is part of the design: surface token -> casefold -> hyphenated
components -> compound-word components -> ae/oe/ue aliasing applied to everything
produced so far -> lexicon-validated stem aliases.

Step 5 is a **closure**: aliasing applies to everything produced so far, not just the
surface word. Skip this step, and the query ``Kuendigung`` fails to hit a document that
only ever writes ``Kündigungsfrist`` -- which is exactly the problem the folding was
supposed to solve.
"""

from __future__ import annotations

import pytest

from guardrails.locale import get_rules
from guardrails.types import Locale


@pytest.fixture(scope="module")
def rules():
    return get_rules(Locale.DE_DE)


def tok(rules, text: str) -> set[str]:
    return set(rules.tokenize(text))


# --- casefold and ß --------------------------------------------------------

def test_eszett_casefolds_to_ss(rules):
    assert "strasse" in tok(rules, "Straße")


def test_eszett_spelling_variants_meet(rules):
    """Straße and Strasse must land on the same token, or the two spellings cannot
    retrieve each other."""
    assert tok(rules, "Straße") & tok(rules, "Strasse")


# --- umlaut aliasing -------------------------------------------------------

def test_umlaut_alias_on_surface_word(rules):
    assert {"kündigung", "kuendigung"} <= tok(rules, "Kündigung")


def test_umlaut_alias_is_additive_not_replacing(rules):
    assert "kündigung" in tok(rules, "Kündigung")


def test_no_bare_vowel_folding(rules):
    """Only ü->ue is done, never ü->u. A wider folding would merge words that
    should not be merged."""
    assert "kundigung" not in tok(rules, "Kündigung")


# --- compound decomposition -------------------------------------------------

def test_compound_yields_parts_and_whole(rules):
    tokens = tok(rules, "Kündigungsfrist")
    assert {"kündigungsfrist", "kündigung", "frist"} <= tokens


def test_compound_without_linking_morpheme(rules):
    assert {"mobilfunkvertrag", "mobilfunk", "vertrag"} <= tok(rules, "Mobilfunkvertrag")


def test_unknown_compound_degrades_to_whole_word(rules):
    """An unlisted compound degrades to whole-word matching -- the failure mode is
    conservative and never produces a false recall."""
    tokens = tok(rules, "Quastenflossergehege")
    assert tokens == {"quastenflossergehege"}


# --- alias closure (the direct acceptance test for step 5) -----------------

def test_alias_closure_covers_compound_parts(rules):
    """Components must produce aliases too, or the folding accomplishes nothing."""
    assert "kuendigung" in tok(rules, "Kündigungsfrist")


def test_ascii_query_retrieves_umlaut_compound(rules):
    """A paired assertion: the query side produces the alias, and the document-side
    component produces the same alias too.

    Asserting either side alone proves nothing about the two spellings retrieving
    each other.
    """
    assert "kuendigung" in tok(rules, "Kuendigung")
    assert "kuendigung" in tok(rules, "Kündigungsfrist")


# --- hyphens -----------------------------------------------------------

def test_hyphenated_parts(rules):
    assert {"eu-roaming", "eu", "roaming"} <= tok(rules, "EU-Roaming")


# --- stemming: validated against the lexicon, not by length -----------------

def test_stem_produced_when_in_lexicon(rules):
    assert "frist" in tok(rules, "Fristen")


def test_stem_produced_for_genitive_in_lexicon(rules):
    assert "preis" in tok(rules, "Preises")


def test_stem_suppressed_when_not_in_lexicon(rules):
    """A length-based rule would fabricate a false stem like 'nutz'; lexicon
    validation does not."""
    tokens = tok(rules, "Nutzer")
    assert "nutzer" in tokens
    assert "nutz" not in tokens


def test_inflection_pair_meets(rules):
    assert tok(rules, "Fristen") & tok(rules, "Frist")


# --- numbers -----------------------------------------------------------------

def test_price_stays_one_token(rules):
    assert "29,99" in tok(rules, "Der Tarif kostet 29,99 EUR pro Monat.")


# --- term-frequency discipline ----------------------------------------------

def test_variants_of_one_word_do_not_inflate_tf(rules):
    """One occurrence of Kündigung expands into several variants, but each variant
    is counted only once."""
    tokens = rules.tokenize("Kündigung")
    assert len(tokens) == len(set(tokens))


def test_genuine_repetition_is_preserved(rules):
    """If it genuinely occurs twice in the document, the count should be two --
    otherwise BM25's TF is no longer interpretable."""
    tokens = rules.tokenize("Kündigung Kündigung")
    assert tokens.count("kündigung") == 2


# --- backtracking: stripping a linking morpheme must be reversible ----------

def test_linking_morpheme_strip_is_backtrackable(rules):
    """The `n` after `ruf` is the first letter of `nummer`, not a linking morpheme.

    Greedily consuming it as a morpheme would leave the whole word undecomposable.
    Stripping must be an attempt that can be backtracked.
    """
    assert {"ruf", "nummer", "mitnahme"} <= tok(rules, "Rufnummernmitnahme")


def test_whole_word_survives_backtracking(rules):
    assert "rufnummernmitnahme" in tok(rules, "Rufnummernmitnahme")


# --- the final component may carry inflection -------------------------------

def test_final_component_may_carry_plural(rules):
    assert {"service", "zeit"} <= tok(rules, "Servicezeiten")


def test_final_component_plural_after_linking_morpheme(rules):
    assert {"zahlung", "art"} <= tok(rules, "Zahlungsarten")


def test_inflected_compound_keeps_the_whole_word(rules):
    assert "servicezeiten" in tok(rules, "Servicezeiten")


# --- lexicon gaps -------------------------------------------------------------

def test_entstoerfrist_decomposes(rules):
    assert {"entstör", "frist"} <= tok(rules, "Entstörfrist")


# --- backtracking must not relax "all or nothing" ---------------------------

def test_backtracking_does_not_admit_partial_splits(rules):
    """A lexicon word plus unrecognizable leftover must still degrade to the whole
    word as a unit.

    Backtracking exists to find a **complete** decomposition, not to accept an
    incomplete one.
    """
    tokens = tok(rules, "Vertragxyzq")
    assert tokens == {"vertragxyzq"}


def test_unknown_compound_still_degrades(rules):
    assert tok(rules, "Quastenflossergehege") == {"quastenflossergehege"}
