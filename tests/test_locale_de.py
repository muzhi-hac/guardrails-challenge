"""German language rules.

Weighted towards negative cases. A register detector that fires on `durch`
because it contains `du`, or a segmenter that splits `1.000` into two
sentences, produces findings that look plausible and are wrong -- the most
expensive kind of bug in a guard layer.
"""

import pytest

from guardrails.locale import get_rules
from guardrails.locale.base import count_words
from guardrails.types import AddressForm, EntityKind, Locale

RULES = get_rules(Locale.DE_DE)
ALL_KINDS = frozenset(EntityKind)


def kinds_of(text, kinds=ALL_KINDS):
    return [(e.kind, e.raw, e.normalized) for e in RULES.extract_entities(text, kinds)]


class TestSegmentation:
    def test_plain_sentences(self):
        text = "Ihr Vertrag läuft weiter. Die Kündigung ist möglich. Danke!"
        assert [s.text for s in RULES.segment_sentences(text)] == [
            "Ihr Vertrag läuft weiter.",
            "Die Kündigung ist möglich.",
            "Danke!",
        ]

    @pytest.mark.parametrize(
        "text",
        [
            "Das gilt z.B. für Neukunden.",
            "Der Betrag ist inkl. MwSt. angegeben.",
            "Sie erreichen uns unter Tel. 030 123456 täglich.",
            "Der Vertrag verlängert sich bzw. endet automatisch.",
        ],
    )
    def test_abbreviations_do_not_split(self, text):
        assert len(RULES.segment_sentences(text)) == 1

    def test_thousands_separator_does_not_split(self):
        """German writes thousands with a full stop: 1.000 is one number."""
        assert len(RULES.segment_sentences("Sie erhalten 1.000 Freiminuten monatlich.")) == 1

    def test_ordinal_before_a_month_does_not_split(self):
        assert len(RULES.segment_sentences("Der Wechsel erfolgt am 1. Januar 2026.")) == 1

    def test_ordinal_not_before_a_month_still_splits(self):
        """The narrow rule: only a month suppresses the break."""
        got = RULES.segment_sentences("Das kostet 1. Danach beginnt der neue Vertrag.")
        assert len(got) == 2

    def test_spans_index_the_original_text(self):
        text = "  Erster Satz. Zweiter Satz.  "
        for sentence in RULES.segment_sentences(text):
            lo, hi = sentence.span
            assert text[lo:hi] == sentence.text


class TestWordCount:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Der 5G-Tarif ist verfügbar", 4),
            ("Mindestvertragslaufzeit beträgt 24 Monate", 4),
            ("Ja, natürlich!", 2),
        ],
    )
    def test_hyphenated_and_compound_words_count_once(self, text, expected):
        assert count_words(text) == expected


class TestAddressForm:
    def test_informal_pronouns_are_flagged_for_a_formal_brand(self):
        hits = RULES.check_address_form("Kannst du mir deine Nummer geben?", AddressForm.FORMAL)
        assert [h.token.lower() for h in hits] == ["du", "deine"]
        assert {h.marker for h in hits} == {"informal_pronoun"}

    def test_capitalised_informal_still_counts(self):
        """`Du` and `Dein` are the polite written variants -- still informal."""
        hits = RULES.check_address_form("Hallo! Du hast Deine Rechnung erhalten.", AddressForm.FORMAL)
        assert len(hits) == 2

    @pytest.mark.parametrize(
        "text",
        [
            "Der Vertrag läuft durch das ganze Jahr.",     # 'durch' contains 'du'
            "Ihre Bildung ist uns wichtig.",                # 'Bildung' contains 'du'
            "Der Direktor meldet sich bei Ihnen.",          # 'Direktor' contains 'dir'
            "Wir arbeiten dienstags durchgehend.",
        ],
    )
    def test_substrings_of_longer_words_do_not_match(self, text):
        assert RULES.check_address_form(text, AddressForm.FORMAL) == ()

    def test_a_correct_formal_reply_produces_nothing(self):
        text = "Gern prüfe ich das für Sie. Ihre Rechnung senden wir Ihnen zu."
        assert RULES.check_address_form(text, AddressForm.FORMAL) == ()

    def test_formal_detection_skips_sentence_initial_sie(self):
        """Sentence-initial `Sie` is indistinguishable from `sie` = they."""
        hits = RULES.check_address_form("Sie kommen morgen. Wir rufen Sie an.", AddressForm.INFORMAL)
        assert [h.span for h in hits] == [(29, 32)]

    def test_ihnen_is_unambiguous(self):
        hits = RULES.check_address_form("Wir senden Ihnen das zu.", AddressForm.INFORMAL)
        assert [h.token for h in hits] == ["Ihnen"]

    def test_grammatical_support_is_declared(self):
        assert RULES.supports_grammatical_address_form is True


class TestEntities:
    @pytest.mark.parametrize(
        "text,normalized",
        [
            ("Das kostet 19,99 €.", "19.99 EUR"),
            ("Der Preis beträgt EUR 19,99.", "19.99 EUR"),
            ("Nur 19,99 Euro im Monat.", "19.99 EUR"),
            ("Einmalig 29 € Anschlussgebühr.", "29.00 EUR"),
            ("Insgesamt 1.234,50 €.", "1234.50 EUR"),
        ],
    )
    def test_price_spellings_normalise_alike(self, text, normalized):
        prices = [e for e in RULES.extract_entities(text, ALL_KINDS) if e.kind is EntityKind.PRICE]
        assert [p.normalized for p in prices] == [normalized]

    def test_a_price_is_not_also_a_number(self):
        """Overlap resolution: specific kinds claim their span first."""
        found = kinds_of("Das kostet 19,99 €.")
        assert found == [(EntityKind.PRICE, "19,99 €", "19.99 EUR")]

    @pytest.mark.parametrize(
        "text,iso",
        [
            ("Gültig ab 01.02.2026.", "2026-02-01"),
            ("Gültig ab 1.2.2026.", "2026-02-01"),
            ("Gültig ab 1. Februar 2026.", "2026-02-01"),
        ],
    )
    def test_date_spellings_normalise_alike(self, text, iso):
        dates = [e for e in RULES.extract_entities(text, ALL_KINDS) if e.kind is EntityKind.DATE]
        assert [d.normalized for d in dates] == [iso]

    def test_month_and_year_yields_a_partial_date(self):
        dates = [e for e in RULES.extract_entities("Ab Februar 2026 gilt der neue Preis.", ALL_KINDS)
                 if e.kind is EntityKind.DATE]
        assert [d.normalized for d in dates] == ["2026-02"]

    def test_an_impossible_date_is_not_extracted(self):
        """A regex match is not a fact: 31 February matches and does not exist."""
        assert not [e for e in RULES.extract_entities("Am 31.02.2026 endet es.", ALL_KINDS)
                    if e.kind is EntityKind.DATE]

    def test_a_date_is_not_split_into_numbers(self):
        assert kinds_of("Gültig ab 01.02.2026.") == [(EntityKind.DATE, "01.02.2026", "2026-02-01")]

    @pytest.mark.parametrize(
        "text,iso",
        [
            ("Die Laufzeit beträgt 24 Monate.", "P24M"),
            ("Nach 24 Monaten verlängert sich der Vertrag.", "P24M"),
            ("Ein 24-Monats-Vertrag ist möglich.", "P24M"),
            ("Die Frist beträgt 14 Tage.", "P14D"),
            ("Gültig für 2 Jahre.", "P2Y"),
            ("Innerhalb von 4 Wochen.", "P4W"),
        ],
    )
    def test_durations(self, text, iso):
        durations = [e for e in RULES.extract_entities(text, ALL_KINDS)
                     if e.kind is EntityKind.DURATION]
        assert [d.normalized for d in durations] == [iso]

    def test_thousands_separator_in_a_bare_number(self):
        found = kinds_of("Sie erhalten 1.000 Freiminuten.")
        assert found == [(EntityKind.NUMBER, "1.000", "1000")]

    def test_filtering_does_not_change_number_extraction(self):
        """Disabling PRICE must not turn a price into a number."""
        text = "Das kostet 19,99 € pro Monat."
        assert kinds_of(text, frozenset({EntityKind.NUMBER})) == []

    def test_spans_index_the_original_text(self):
        text = "Ab 01.02.2026 kostet der Tarif 19,99 € für 24 Monate."
        for entity in RULES.extract_entities(text, ALL_KINDS):
            lo, hi = entity.span
            assert text[lo:hi] == entity.raw

    def test_entities_never_overlap(self):
        text = "Ab 01.02.2026 kostet der Tarif 19,99 € für 24 Monate, also 1.000 Minuten."
        spans = [e.span for e in RULES.extract_entities(text, ALL_KINDS)]
        for (a_lo, a_hi), (b_lo, b_hi) in zip(spans, spans[1:]):
            assert a_hi <= b_lo


class TestCommitments:
    @pytest.mark.parametrize(
        "text,commitment_id,raw",
        [
            ("Ich erstatte Ihnen den Betrag.", "refund", "erstatte"),
            ("Sie erhalten eine Erstattung.", "refund", "Erstattung"),
            ("Wir werden den Betrag zurückerstatten.", "refund", "zurückerstatten"),
            ("Die Gebühr erlassen wir Ihnen.", "waive_fee", "Gebühr erlassen"),
            ("Dafür fällt keine Gebühr an.", "waive_fee", "keine Gebühr"),
            ("Die Gebühr entfällt in diesem Fall.", "waive_fee", "Gebühr entfällt"),
            ("Sie erhalten eine Gutschrift.", "credit", "Gutschrift"),
            ("Den Betrag schreibe ich Ihnen gut.", "credit", "schreibe ich Ihnen gut"),
            ("Wir werden Ihnen einen Rabatt gewähren.", "discount", "Rabatt gewähren"),
            ("Sie bekommen einen Nachlass.", "discount", "Nachlass"),
            ("Ich rufe Sie zurück.", "schedule_callback", "rufe Sie zurück"),
            ("Ein Rückruf ist möglich.", "schedule_callback", "Rückruf"),
            ("Ich sende Ihnen eine Bestätigung.", "send_confirmation_email", "sende Ihnen eine Bestätigung"),
            ("Sie erhalten eine Bestätigung per E-Mail.", "send_confirmation_email", "Bestätigung per E-Mail"),
        ],
    )
    def test_each_configured_phrase_is_found(self, text, commitment_id, raw):
        hits = RULES.find_commitments(text)
        assert [(h.commitment_id, h.raw) for h in hits] == [(commitment_id, raw)]

    def test_matching_is_case_insensitive(self):
        hits = RULES.find_commitments("ich ERSTATTE ihnen den betrag.")
        assert [h.commitment_id for h in hits] == ["refund"]

    def test_spans_index_the_original_text(self):
        text = "Gerne. Ich erstatte Ihnen den Betrag."
        (hit,) = RULES.find_commitments(text)
        lo, hi = hit.span
        assert text[lo:hi] == hit.raw == "erstatte"

    def test_a_reply_with_no_promise_finds_nothing(self):
        assert RULES.find_commitments("Ihr Vertrag läuft zum 01.02.2026 aus.") == ()

    def test_prose_containing_a_phrase_as_a_substring_of_another_word_does_not_match(self):
        """`Rückruf` must not fire inside an unrelated compound that merely
        contains the same letters -- word boundaries, not substring search."""
        assert RULES.find_commitments("Der Rückrufservice ist derzeit nicht verfügbar.") == ()

    def test_multiple_commitments_in_one_reply_are_all_found(self):
        text = "Ich erstatte Ihnen den Betrag und rufe Sie zurück."
        hits = RULES.find_commitments(text)
        assert {h.commitment_id for h in hits} == {"refund", "schedule_callback"}
