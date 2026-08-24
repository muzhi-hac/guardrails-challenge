"""English language rules.

The interesting assertions here are the asymmetries with German: the address
form check is weaker and declares itself so, and `01/02/2026` is day-first.
"""

import pytest

from guardrails.locale import get_rules, supported_locales
from guardrails.types import AddressForm, EntityKind, Locale

RULES = get_rules(Locale.EN_GB)
ALL_KINDS = frozenset(EntityKind)


def kinds_of(text, kinds=ALL_KINDS):
    return [(e.kind, e.raw, e.normalized) for e in RULES.extract_entities(text, kinds)]


class TestRegistry:
    def test_both_locales_are_registered(self):
        assert set(supported_locales()) == {Locale.DE_DE, Locale.EN_GB}

    def test_each_implementation_reports_its_own_locale(self):
        for locale in supported_locales():
            assert get_rules(locale).locale is locale


class TestSegmentation:
    def test_plain_sentences(self):
        text = "Your contract continues. Cancellation is possible. Thanks!"
        assert len(RULES.segment_sentences(text)) == 3

    @pytest.mark.parametrize(
        "text",
        [
            "This applies to new customers, e.g. those joining in March.",
            "Please contact Dr. Weber about the account.",
            "The price is £19.99 per month including VAT.",
        ],
    )
    def test_abbreviations_and_decimals_do_not_split(self, text):
        assert len(RULES.segment_sentences(text)) == 1

    def test_spans_index_the_original_text(self):
        text = "  First one. Second one.  "
        for sentence in RULES.segment_sentences(text):
            lo, hi = sentence.span
            assert text[lo:hi] == sentence.text


class TestAddressForm:
    def test_the_check_declares_itself_weaker_than_german(self):
        """English has no grammatical T-V distinction; this is data, not prose,
        so a guard can weigh the finding rather than assume parity."""
        assert RULES.supports_grammatical_address_form is False
        assert get_rules(Locale.DE_DE).supports_grammatical_address_form is True

    def test_contractions_are_reported_as_contractions(self):
        hits = RULES.check_address_form("We can't do that, but you're covered.", AddressForm.FORMAL)
        assert {h.marker for h in hits} == {"contraction"}

    def test_casual_greeting(self):
        hits = RULES.check_address_form("Hi there. How can I help?", AddressForm.FORMAL)
        assert [h.marker for h in hits] == ["casual_greeting"]

    def test_a_formal_reply_produces_nothing(self):
        text = "Thank you for your message. I will check your account and confirm shortly."
        assert RULES.check_address_form(text, AddressForm.FORMAL) == ()


class TestEntities:
    @pytest.mark.parametrize(
        "text,normalized",
        [
            ("It costs £19.99.", "19.99 GBP"),
            ("The price is 19.99 GBP.", "19.99 GBP"),
            ("A one-off £29 connection fee.", "29.00 GBP"),
            ("In total £1,234.50.", "1234.50 GBP"),
        ],
    )
    def test_prices(self, text, normalized):
        prices = [e for e in RULES.extract_entities(text, ALL_KINDS) if e.kind is EntityKind.PRICE]
        assert [p.normalized for p in prices] == [normalized]

    def test_slash_dates_are_day_first(self):
        """en-GB: 01/02/2026 is 1 February. The same string is 2 January under
        en-US, which is exactly why this is locale knowledge."""
        dates = [e for e in RULES.extract_entities("Valid from 01/02/2026.", ALL_KINDS)
                 if e.kind is EntityKind.DATE]
        assert [d.normalized for d in dates] == ["2026-02-01"]

    @pytest.mark.parametrize(
        "text,iso",
        [
            ("Valid from 1 February 2026.", "2026-02-01"),
            ("Valid from 1st February 2026.", "2026-02-01"),
            ("Valid from 2026-02-01.", "2026-02-01"),
        ],
    )
    def test_date_spellings_normalise_alike(self, text, iso):
        dates = [e for e in RULES.extract_entities(text, ALL_KINDS) if e.kind is EntityKind.DATE]
        assert [d.normalized for d in dates] == [iso]

    def test_an_impossible_date_is_not_extracted(self):
        assert not [e for e in RULES.extract_entities("Ends on 31/02/2026.", ALL_KINDS)
                    if e.kind is EntityKind.DATE]

    @pytest.mark.parametrize(
        "text,iso",
        [
            ("The term is 24 months.", "P24M"),
            ("A 24-month contract is available.", "P24M"),
            ("Within 14 days.", "P14D"),
            ("Valid for 2 years.", "P2Y"),
        ],
    )
    def test_durations(self, text, iso):
        durations = [e for e in RULES.extract_entities(text, ALL_KINDS)
                     if e.kind is EntityKind.DURATION]
        assert [d.normalized for d in durations] == [iso]

    def test_comma_thousands_separator(self):
        assert kinds_of("You get 1,000 minutes.") == [(EntityKind.NUMBER, "1,000", "1000")]

    @pytest.mark.parametrize(
        "text,raw,normalized",
        [
            ("It is due on the 1st of the month.", "1st", "1"),
            ("Take the 2nd option.", "2nd", "2"),
            ("It is the 3rd working day.", "3rd", "3"),
            ("Reached the 21st milestone.", "21st", "21"),
            ("Reached the 102nd milestone.", "102nd", "102"),
        ],
    )
    def test_ordinals_extract_as_numbers(self, text, raw, normalized):
        """English marks an ordinal with a letter suffix instead of German's
        bare full stop. The suffix is not part of the entity: only the
        numeral is, matching what German produces for the same fact."""
        numbers = [e for e in RULES.extract_entities(text, ALL_KINDS) if e.kind is EntityKind.NUMBER]
        assert [(n.raw, n.normalized) for n in numbers] == [(raw, normalized)]

    def test_price_with_decimals_is_not_truncated_by_ordinal_handling(self):
        """`29.99` must stay a price of `29.99`, never collapse to the
        ordinal-style numeral `29`."""
        prices = [e for e in RULES.extract_entities("It costs £29.99.", ALL_KINDS) if e.kind is EntityKind.PRICE]
        assert [p.normalized for p in prices] == ["29.99 GBP"]
        assert not [e for e in RULES.extract_entities("It costs £29.99.", ALL_KINDS) if e.kind is EntityKind.NUMBER]

    def test_bare_number_followed_by_non_ordinal_letter_still_does_not_match(self):
        """A digit run continuing into an unrelated word must still be
        rejected, exactly as before ordinals were recognised."""
        assert not [e for e in RULES.extract_entities("Model 29x is not available.", ALL_KINDS)
                    if e.kind is EntityKind.NUMBER]
        assert not [e for e in RULES.extract_entities("abc3rdxyz has no meaning.", ALL_KINDS)
                    if e.kind is EntityKind.NUMBER]

    def test_de_en_ordinal_mirror_extracts_the_same_number(self):
        """Regression guard for the actual defect: the DE/EN billing-date pair
        must extract the same NUMBER set so the cross-locale grounding guard
        stays coherent, with no padding required in the English prose."""
        de = get_rules(Locale.DE_DE).extract_entities(
            "Wir stellen Ihre Rechnung am 3. Werktag des Folgemonats aus.",
            frozenset({EntityKind.NUMBER}),
        )
        en = RULES.extract_entities(
            "We issue your bill on the 3rd working day of the following month.",
            frozenset({EntityKind.NUMBER}),
        )
        assert {m.normalized for m in de} == {m.normalized for m in en} == {"3"}

    def test_spans_index_the_original_text(self):
        text = "From 01/02/2026 the tariff costs £19.99 for 24 months."
        for entity in RULES.extract_entities(text, ALL_KINDS):
            lo, hi = entity.span
            assert text[lo:hi] == entity.raw

    def test_entities_never_overlap(self):
        text = "From 01/02/2026 the tariff costs £19.99 for 24 months, i.e. 1,000 minutes."
        spans = [e.span for e in RULES.extract_entities(text, ALL_KINDS)]
        for (a_lo, a_hi), (b_lo, b_hi) in zip(spans, spans[1:]):
            assert a_hi <= b_lo


class TestCrossLocale:
    def test_the_same_quantity_normalises_to_the_same_value(self):
        """1234.50 in either notation is the same number -- which is the whole
        point of normalising before the grounding guard compares anything."""
        de = get_rules(Locale.DE_DE).extract_entities("1.234,50 Minuten", ALL_KINDS)
        en = get_rules(Locale.EN_GB).extract_entities("1,234.50 minutes", ALL_KINDS)
        assert de[0].normalized == en[0].normalized == "1234.5"

    def test_trailing_zeros_do_not_defeat_a_number_comparison(self):
        """A document writing `1.234,50` and a reply writing `1.234,5` state the
        same fact. Numbers drop trailing zeros so the grounding guard does not
        call that a mismatch."""
        rules = get_rules(Locale.DE_DE)
        long = rules.extract_entities("1.234,50 Minuten", ALL_KINDS)[0]
        short = rules.extract_entities("1.234,5 Minuten", ALL_KINDS)[0]
        assert long.normalized == short.normalized

    def test_prices_keep_two_decimals_instead(self):
        """Currency has fixed minor units, so prices quantise rather than strip:
        `29 EUR` and `29,00 EUR` must compare equal, and they do because both
        sides go through the same rule. The two kinds normalise differently on
        purpose."""
        rules = get_rules(Locale.DE_DE)
        bare = rules.extract_entities("29 €", ALL_KINDS)[0]
        padded = rules.extract_entities("29,00 €", ALL_KINDS)[0]
        assert bare.normalized == padded.normalized == "29.00 EUR"

    def test_currency_is_part_of_the_normal_form(self):
        de = get_rules(Locale.DE_DE).extract_entities("19,99 €", ALL_KINDS)
        en = get_rules(Locale.EN_GB).extract_entities("£19.99", ALL_KINDS)
        assert de[0].normalized != en[0].normalized


# --- Currency: the English channel sells the same EUR tariffs -------------


def test_extracts_eur_price_with_symbol():
    (mention,) = [
        m for m in RULES.extract_entities("The tariff costs €29.99 per month.", ALL_KINDS)
        if m.kind is EntityKind.PRICE
    ]
    assert mention.normalized == "29.99 EUR"


def test_extracts_eur_price_with_code():
    (mention,) = [
        m for m in RULES.extract_entities("The tariff costs 29.99 EUR per month.", ALL_KINDS)
        if m.kind is EntityKind.PRICE
    ]
    assert mention.normalized == "29.99 EUR"


def test_gbp_still_normalises_as_gbp():
    """Adding EUR support must not accidentally break GBP -- the normalized
    currency must follow whichever symbol actually matched."""
    (mention,) = [
        m for m in RULES.extract_entities("It costs £19.99.", ALL_KINDS)
        if m.kind is EntityKind.PRICE
    ]
    assert mention.normalized == "19.99 GBP"


class TestCommitments:
    @pytest.mark.parametrize(
        "text,commitment_id,raw",
        [
            ("We will refund the amount.", "refund", "refund"),
            ("We will reimburse you in full.", "refund", "reimburse"),
            ("We will waive the fee for you.", "waive_fee", "waive the fee"),
            ("There is no charge for this.", "waive_fee", "no charge"),
            ("We will credit your account.", "credit", "credit your account"),
            ("We can grant a discount here.", "discount", "grant a discount"),
            ("I will call you back tomorrow.", "schedule_callback", "call you back"),
            ("We will send you a confirmation.", "send_confirmation_email", "send you a confirmation"),
        ],
    )
    def test_each_configured_phrase_is_found(self, text, commitment_id, raw):
        hits = RULES.find_commitments(text)
        assert [(h.commitment_id, h.raw) for h in hits] == [(commitment_id, raw)]

    def test_matching_is_case_insensitive(self):
        hits = RULES.find_commitments("We will REFUND the amount.")
        assert [h.commitment_id for h in hits] == ["refund"]

    def test_spans_index_the_original_text(self):
        text = "Certainly. We will refund the amount."
        (hit,) = RULES.find_commitments(text)
        lo, hi = hit.span
        assert text[lo:hi] == hit.raw == "refund"

    def test_a_reply_with_no_promise_finds_nothing(self):
        assert RULES.find_commitments("Your contract ends on 1 February 2026.") == ()

    def test_multiple_commitments_in_one_reply_are_all_found(self):
        text = "We will refund the amount and call you back tomorrow."
        hits = RULES.find_commitments(text)
        assert {h.commitment_id for h in hits} == {"refund", "schedule_callback"}
