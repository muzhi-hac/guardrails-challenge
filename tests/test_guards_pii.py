"""PII guard: outbound leak detection, and the module-level redaction helper.

The central theme, mirroring the other guard test modules: the corpus and
ordinary replies are full of numbers, dates and capitalised nouns that must
not be mistaken for personal data, and the customer's own data coming back
to them in a confirmation is not a leak. Every false-positive case from the
task brief has its own test, the corpus is checked chunk by chunk, and every
entity type is tested both for detection (a stranger's data in the reply)
and for the echo case (the customer's own data, unremarkable).
"""

import asyncio
from pathlib import Path

from guardrails import findings
from guardrails.config import load_profile
from guardrails.guards import PiiGuard, registered_guards
from guardrails.guards.base import GuardContext
from guardrails.guards.pii import redact
from guardrails.locale import get_rules
from guardrails.retrieval.chunks import chunk_document
from guardrails.retrieval.documents import load_documents
from guardrails.types import Mode, Outcome, Severity, Stage

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ROOT / "profiles"
GUARD = PiiGuard()

# Invented for these tests -- not a real account. DE + checksum digits + 18
# BBAN digits chosen so the mod-97 checksum is valid.
VALID_IBAN = "DE02120300000000202051"
# Same digits, one digit flipped, so the checksum fails.
INVALID_CHECKSUM_IBAN = "DE02120300000000202052"

PHONE = "0170 1234567"
CUSTOMER_ID = "KD-12345678"
BIRTHDATE_CONTEXT = "geboren am 12.05.1980"
STREET_ADDRESS = "Musterstraße 12"
POSTCODE_CITY = "10115 Berlin"


def context(
    user_message: str = "",
    reply: str = "",
    *,
    profile: str = "telco_de",
    mode: Mode = Mode.CHAT,
) -> GuardContext:
    resolved = load_profile(PROFILES / f"{profile}.yaml").resolve(mode)
    return GuardContext(
        profile=resolved,
        rules=get_rules(resolved.locale),
        user_message=user_message,
        reply=reply,
    )


def run(user_message: str = "", reply: str = "", **kwargs):
    return asyncio.run(GUARD.check(context(user_message, reply, **kwargs)))


def kinds(verdict) -> list[str]:
    return [e.kind for e in verdict.evidence]


class TestContract:
    def test_declared_severities_match_the_finding_vocabulary(self):
        assert set(PiiGuard.DEFAULT_SEVERITY) == findings.PII_FINDINGS

    def test_identity(self):
        assert PiiGuard.name == "pii"
        assert PiiGuard.stage is Stage.OUTPUT
        assert PiiGuard.tier == 0
        assert "pii" in registered_guards()

    def test_a_clean_turn_passes(self):
        verdict = run("Wie hoch ist die monatliche Grundgebühr?", "Sie beträgt 29,99 EUR.")
        assert verdict.outcome is Outcome.PASS
        assert verdict.severity is Severity.NONE
        assert verdict.evidence == ()
        assert verdict.tier == 0
        assert verdict.cost_usd == 0.0


class TestOutboundLeak:
    def test_an_iban_in_the_reply_alone_is_a_leak(self):
        verdict = run("Wie hoch ist meine Erstattung?", f"Die Erstattung geht auf {VALID_IBAN}.")
        assert findings.OUTBOUND_LEAK in kinds(verdict)
        assert verdict.outcome is Outcome.FAIL
        assert verdict.severity is Severity.CRITICAL

    def test_a_phone_number_in_the_reply_alone_is_a_leak(self):
        verdict = run("Wie erreiche ich Sie?", f"Wir erreichen Sie unter {PHONE}.")
        assert findings.OUTBOUND_LEAK in kinds(verdict)

    def test_a_customer_id_in_the_reply_alone_is_a_leak(self):
        verdict = run("Wie lautet mein Vertrag?", f"Ihre Kundennummer ist {CUSTOMER_ID}.")
        assert findings.OUTBOUND_LEAK in kinds(verdict)

    def test_a_birthdate_in_the_reply_alone_is_a_leak(self):
        verdict = run("Wann wurde der Vertrag angelegt?", f"Der Kunde ist {BIRTHDATE_CONTEXT}.")
        assert findings.OUTBOUND_LEAK in kinds(verdict)

    def test_a_street_address_in_the_reply_alone_is_a_leak(self):
        verdict = run("Wo wohnt der Kunde?", f"Die Adresse lautet {STREET_ADDRESS}, {POSTCODE_CITY}.")
        assert findings.OUTBOUND_LEAK in kinds(verdict)

    def test_evidence_never_carries_the_matched_value(self):
        """The whole point of this guard is defeated if its own trace
        contains the value it flagged -- see the module docstring."""
        verdict = run("Wie hoch ist meine Erstattung?", f"Die Erstattung geht auf {VALID_IBAN}.")
        for evidence in verdict.evidence:
            assert VALID_IBAN not in evidence.detail


class TestEchoIsNotALeak:
    def test_an_iban_the_customer_typed_is_not_reported_when_echoed(self):
        verdict = run(
            f"Meine IBAN ist {VALID_IBAN}, bitte prüfen.",
            f"Wir haben die IBAN {VALID_IBAN} für die Erstattung notiert.",
        )
        assert verdict.outcome is Outcome.PASS
        assert findings.OUTBOUND_LEAK not in kinds(verdict)

    def test_a_phone_number_the_customer_typed_is_not_reported_when_echoed(self):
        verdict = run(
            f"Meine Nummer ist {PHONE}.",
            f"Wir rufen Sie unter {PHONE} zurueck.",
        )
        assert verdict.outcome is Outcome.PASS
        assert findings.OUTBOUND_LEAK not in kinds(verdict)

    def test_differently_formatted_echo_still_counts_as_the_same_value(self):
        """The customer writes their IBAN without spaces; the reply repeats
        it space-grouped the way banks print it. Same account, different
        formatting -- normalisation is what keeps this from misfiring."""
        solid = VALID_IBAN
        grouped = " ".join(solid[i : i + 4] for i in range(0, len(solid), 4))
        verdict = run(f"Meine IBAN ist {solid}.", f"Wir haben {grouped} notiert.")
        assert findings.OUTBOUND_LEAK not in kinds(verdict)

    def test_a_second_iban_not_mentioned_by_the_customer_is_still_a_leak(self):
        """Echoing back the customer's own IBAN must not blanket-suppress
        every IBAN in the reply -- only the ones the customer actually
        provided."""
        other_iban = "DE12500105170648489890"  # different account, valid checksum
        assert _iban_checksum_ok(other_iban)
        verdict = run(
            f"Meine IBAN ist {VALID_IBAN}.",
            f"Wir haben {VALID_IBAN} notiert. Zusätzlich sehen wir {other_iban} auf dem Konto.",
        )
        assert findings.OUTBOUND_LEAK in kinds(verdict)


def _iban_checksum_ok(iban: str) -> bool:
    from guardrails.guards.pii import _iban_checksum_valid

    return _iban_checksum_valid(iban)


class TestEntityListRespected:
    def test_an_entity_type_absent_from_the_profile_is_not_checked(self):
        ctx = context("Wie erreiche ich Sie?", f"Wir erreichen Sie unter {PHONE}.")
        narrowed = ctx.profile.guards.pii.model_copy(
            update={"entities": ("iban", "customer_id", "birthdate", "address")}
        )
        profile = ctx.profile.model_copy(
            update={"guards": ctx.profile.guards.model_copy(update={"pii": narrowed})}
        )
        verdict = asyncio.run(
            GUARD.check(
                GuardContext(
                    profile=profile,
                    rules=ctx.rules,
                    user_message=ctx.user_message,
                    reply=ctx.reply,
                )
            )
        )
        assert verdict.outcome is Outcome.PASS
        assert findings.OUTBOUND_LEAK not in kinds(verdict)


class TestFalsePositives:
    """Each case is drawn verbatim from the task brief: ordinary telco prose
    that must not be mistaken for personal data."""

    def test_a_bare_number_passes(self):
        verdict = run(reply="Die Kündigungsfrist beträgt 1 Monat.")
        assert verdict.outcome is Outcome.PASS
        assert verdict.evidence == ()

    def test_a_price_passes(self):
        verdict = run(reply="Tarif M kostet 29,99 EUR pro Monat.")
        assert verdict.outcome is Outcome.PASS
        assert verdict.evidence == ()

    def test_a_dateless_of_birth_context_passes(self):
        verdict = run(reply="Stand: 01.01.2026")
        assert verdict.outcome is Outcome.PASS
        assert verdict.evidence == ()

    def test_a_duration_passes(self):
        verdict = run(reply="Die Entstörfrist beträgt 48 Stunden.")
        assert verdict.outcome is Outcome.PASS
        assert verdict.evidence == ()

    def test_ordinary_prose_passes(self):
        verdict = run(reply='Ihre Rechnung finden Sie unter „Rechnungen".')
        assert verdict.outcome is Outcome.PASS
        assert verdict.evidence == ()

    def test_a_five_digit_count_followed_by_a_capitalised_noun_is_not_an_address(self):
        """German capitalises every noun, so a bare count ('über 50000
        Verträge') has the same surface shape as a postcode + city. Only a
        preceding comma or line start anchors the postcode/city pattern."""
        verdict = run(reply="Im letzten Jahr wurden über 50000 Verträge verlängert.")
        assert findings.ADDRESS not in kinds(verdict)
        assert verdict.outcome is Outcome.PASS


class TestInvalidIban:
    def test_a_checksum_invalid_iban_is_not_reported(self):
        verdict = run(reply=f"Ihre Referenz lautet {INVALID_CHECKSUM_IBAN}.")
        assert findings.OUTBOUND_LEAK not in kinds(verdict)
        assert verdict.outcome is Outcome.PASS


class TestCorpusHasNoFalsePositives:
    def test_every_real_corpus_chunk_passes_as_a_reply(self):
        profile = load_profile(PROFILES / "telco_de.yaml").resolve(Mode.CHAT)
        rules = get_rules(profile.locale)
        chunks = [
            c
            for d in load_documents(ROOT / "kb")
            for c in chunk_document(d)
            if c.locale is profile.locale
        ]
        assert len(chunks) > 0

        flagged = []
        for chunk in chunks:
            verdict = asyncio.run(
                GUARD.check(GuardContext(profile=profile, rules=rules, reply=chunk.text))
            )
            if verdict.outcome is Outcome.FAIL:
                flagged.append((chunk.chunk_id, kinds(verdict)))

        assert flagged == []


class TestRedact:
    def test_each_entity_type_is_replaced_with_its_placeholder(self):
        text = (
            f"IBAN {VALID_IBAN}, Telefon {PHONE}, Kundennummer {CUSTOMER_ID}, "
            f"{BIRTHDATE_CONTEXT}, Adresse {STREET_ADDRESS}, {POSTCODE_CITY}."
        )
        rules = get_rules(load_profile(PROFILES / "telco_de.yaml").locale)
        redacted, evidence = redact(
            text, ("iban", "phone", "customer_id", "birthdate", "address"), rules
        )

        assert "[IBAN]" in redacted
        assert "[PHONE]" in redacted
        assert "[CUSTOMER_ID]" in redacted
        assert "[BIRTHDATE]" in redacted
        assert "[ADDRESS]" in redacted

        assert VALID_IBAN not in redacted
        assert PHONE not in redacted
        assert CUSTOMER_ID not in redacted
        assert "12.05.1980" not in redacted
        assert STREET_ADDRESS not in redacted

        assert {e.kind for e in evidence} == {
            findings.IBAN,
            findings.PHONE,
            findings.CUSTOMER_ID,
            findings.BIRTHDATE,
            findings.ADDRESS,
        }

    def test_redaction_evidence_never_carries_the_value(self):
        rules = get_rules(load_profile(PROFILES / "telco_de.yaml").locale)
        _, evidence = redact(f"Meine IBAN ist {VALID_IBAN}.", ("iban",), rules)
        assert len(evidence) == 1
        assert VALID_IBAN not in evidence[0].detail

    def test_text_outside_the_matched_spans_is_preserved(self):
        rules = get_rules(load_profile(PROFILES / "telco_de.yaml").locale)
        text = f"Guten Tag, Ihre IBAN {VALID_IBAN} wurde erfasst. Vielen Dank."
        redacted, _ = redact(text, ("iban",), rules)
        assert redacted == "Guten Tag, Ihre IBAN [IBAN] wurde erfasst. Vielen Dank."

    def test_a_checksum_invalid_iban_is_not_redacted(self):
        rules = get_rules(load_profile(PROFILES / "telco_de.yaml").locale)
        text = f"Referenz {INVALID_CHECKSUM_IBAN}."
        redacted, evidence = redact(text, ("iban",), rules)
        assert redacted == text
        assert evidence == ()

    def test_an_entity_type_not_requested_is_left_alone(self):
        rules = get_rules(load_profile(PROFILES / "telco_de.yaml").locale)
        text = f"Telefon {PHONE}, IBAN {VALID_IBAN}."
        redacted, evidence = redact(text, ("phone",), rules)
        assert "[PHONE]" in redacted
        assert VALID_IBAN in redacted
        assert {e.kind for e in evidence} == {findings.PHONE}

    def test_birthdate_redaction_keeps_the_surrounding_sentence(self):
        """Only the date is removed, not the keyword that gave it away --
        the sentence stays legible about *what* was redacted."""
        rules = get_rules(load_profile(PROFILES / "telco_de.yaml").locale)
        redacted, _ = redact("Der Kunde ist geboren am 12.05.1980.", ("birthdate",), rules)
        assert redacted == "Der Kunde ist geboren am [BIRTHDATE]."
