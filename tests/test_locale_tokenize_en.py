"""English tokenization: a trivial implementation. There is no compound
decomposition, and that asymmetry is deliberate."""

from __future__ import annotations

import pytest

from guardrails.locale import get_rules
from guardrails.types import Locale


@pytest.fixture(scope="module")
def rules():
    return get_rules(Locale.EN_GB)


def test_lowercases_and_strips_punctuation(rules):
    assert set(rules.tokenize("Cancel the Contract, please.")) == {
        "cancel", "the", "contract", "please",
    }


def test_splits_hyphenated_and_keeps_the_whole(rules):
    tokens = set(rules.tokenize("EU-Roaming"))
    assert {"eu-roaming", "eu", "roaming"} <= tokens


def test_prices_stay_one_token(rules):
    assert "29.99" in rules.tokenize("The tariff costs 29.99 EUR per month.")


def test_no_compound_splitting(rules):
    """English does not do compound decomposition -- asserting this asymmetry
    explicitly guards against someone later 'completing' it."""
    assert "contract" not in rules.tokenize("Contractual")
