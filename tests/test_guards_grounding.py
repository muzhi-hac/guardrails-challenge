"""Grounding guard: entity checks against retrieved chunks, and commitment
authorisation against the profile's own policy.

Two themes, mirroring ``test_guards_persona.py``. First, that grounding is
about the *normalised* value, not the raw spelling -- a price quoted with a
different currency notation, or a different locale's decimal convention,
must still ground. Second, that an unsupported commitment is a distinct and
more serious failure than an ungrounded fact: it is checked against the
profile's authorisation list, not against retrieval, and the profile's own
severity override for it is asserted directly rather than assumed.
"""

import asyncio
from pathlib import Path

from guardrails import findings
from guardrails.config import load_profile
from guardrails.guards import GroundingGuard, registered_guards
from guardrails.guards.base import GuardContext
from guardrails.locale import get_rules
from guardrails.types import EntityKind, Mode, Outcome, Severity, Stage

PROFILES = Path(__file__).resolve().parents[1] / "profiles"
GUARD = GroundingGuard()


def context(
    reply: str,
    *,
    retrieved: tuple[str, ...] = (),
    profile: str = "telco_de",
    mode: Mode = Mode.CHAT,
) -> GuardContext:
    resolved = load_profile(PROFILES / f"{profile}.yaml").resolve(mode)
    return GuardContext(
        profile=resolved, rules=get_rules(resolved.locale), reply=reply, retrieved=retrieved
    )


def run(reply: str, **kwargs):
    return asyncio.run(GUARD.check(context(reply, **kwargs)))


def kinds(verdict) -> list[str]:
    return [e.kind for e in verdict.evidence]


class TestContract:
    def test_declared_severities_match_the_finding_vocabulary(self):
        """Adding a finding without a default severity, or removing one and
        leaving the entry behind, both fail here rather than at runtime."""
        assert set(GroundingGuard.DEFAULT_SEVERITY) == findings.GROUNDING_FINDINGS

    def test_identity(self):
        assert GroundingGuard.name == "grounding"
        assert GroundingGuard.stage is Stage.OUTPUT
        assert GroundingGuard.tier == 0
        assert "grounding" in registered_guards()

    def test_a_reply_with_no_entities_and_no_commitments_passes(self):
        verdict = run("Gern helfe ich Ihnen weiter.")
        assert verdict.outcome is Outcome.PASS
        assert verdict.severity is Severity.NONE
        assert verdict.evidence == ()
        assert verdict.tier == 0
        assert verdict.cost_usd == 0.0


class TestEntityGrounding:
    def test_a_price_that_appears_in_a_retrieved_chunk_passes(self):
        verdict = run(
            "Tarif M kostet 29,99 EUR im Monat.",
            retrieved=("| Tarif M | 29,99 EUR | 20 GB |",),
        )
        assert verdict.outcome is Outcome.PASS
        assert verdict.evidence == ()

    def test_a_price_that_appears_nowhere_fails_naming_the_price(self):
        verdict = run(
            "Tarif M kostet 24,99 EUR im Monat.",
            retrieved=("| Tarif M | 29,99 EUR | 20 GB |",),
        )
        assert kinds(verdict) == [findings.UNGROUNDED_PRICE]
        assert verdict.outcome is Outcome.FAIL
        (evidence,) = verdict.evidence
        assert "24,99" in evidence.detail
        lo, hi = evidence.span
        assert "24,99 EUR" == "Tarif M kostet 24,99 EUR im Monat."[lo:hi]

    def test_normalized_comparison_crosses_notation_within_one_locale(self):
        """`normalized` earns its keep on notation variance *within* one
        locale's grammar, not across locales -- retrieval is locale-
        partitioned (see the module docstring on GroundingGuard), so a German
        turn never sees an English chunk or vice versa. What genuinely varies
        within German is currency placement and spelling: `29,99 EUR`,
        `29,99 €`, `EUR 29,99`, and `29,99 Euro` must all ground each other."""
        de_spellings = ("29,99 EUR", "29,99 €", "EUR 29,99", "29,99 Euro")
        for chunk_spelling in de_spellings:
            for reply_spelling in de_spellings:
                verdict = run(
                    f"Tarif M kostet {reply_spelling} im Monat.",
                    retrieved=(f"Tarif M kostet {chunk_spelling} im Monat.",),
                )
                assert verdict.outcome is Outcome.PASS, (chunk_spelling, reply_spelling)
                assert verdict.evidence == (), (chunk_spelling, reply_spelling)

        en_spellings = ("29.99 EUR", "€29.99")
        for chunk_spelling in en_spellings:
            for reply_spelling in en_spellings:
                verdict = run(
                    f"Tariff M costs {reply_spelling} per month.",
                    retrieved=(f"Tariff M costs {chunk_spelling} per month.",),
                    profile="telco_en",
                )
                assert verdict.outcome is Outcome.PASS, (chunk_spelling, reply_spelling)
                assert verdict.evidence == (), (chunk_spelling, reply_spelling)

    def test_regression_a_foreign_locales_grammar_must_not_ground_a_fabricated_price(self):
        """Before the fix, chunk entities were extracted with *every*
        registered locale's rules, unioned, not just the turn's own
        (`ctx.rules`). Retrieval is locale-partitioned -- a de-DE turn never
        retrieves an en-GB chunk -- so that union was reachable only through
        the guard's own extraction, never through real retrieval, and it
        backfired: parsing the German chunk `"Tarif M kostet 29,99 EUR pro
        Monat."` with *English* rules does not fail or return nothing: English
        reads `,` as a thousands separator, drops the leading `29`, and
        confidently returns a wrong-but-well-formed `99.00 EUR`. That spurious
        entity landed in the grounded set, so a reply fabricating `99,00 EUR`
        against a corpus that actually says `29,99 EUR` passed as grounded.
        A guard that accepts an invented price is wrong in the direction that
        matters most; this asserts the fabrication is caught."""
        chunk = "Tarif M kostet 29,99 EUR pro Monat."
        verdict = run(
            "Tarif M kostet 99,00 EUR pro Monat.",
            retrieved=(chunk,),
        )
        assert verdict.outcome is Outcome.FAIL
        assert findings.UNGROUNDED_PRICE in kinds(verdict)
        (evidence,) = [e for e in verdict.evidence if e.kind == findings.UNGROUNDED_PRICE]
        assert "99,00" in evidence.detail

    def test_an_entity_kind_absent_from_check_entities_is_not_checked(self):
        ctx = context(
            "Die Laufzeit beträgt 24 Monate.",
            retrieved=("Die Mindestlaufzeit beträgt 12 Monate.",),
        )
        narrowed = ctx.profile.guards.grounding.model_copy(
            update={"check_entities": (EntityKind.PRICE,)}
        )
        profile = ctx.profile.model_copy(
            update={"guards": ctx.profile.guards.model_copy(update={"grounding": narrowed})}
        )
        verdict = asyncio.run(
            GUARD.check(GuardContext(profile=profile, rules=ctx.rules, reply=ctx.reply, retrieved=ctx.retrieved))
        )
        assert verdict.outcome is Outcome.PASS
        assert findings.UNGROUNDED_DURATION not in kinds(verdict)

    def test_empty_retrieved_makes_every_checked_entity_unsupported(self):
        verdict = run("Der Tarif kostet 19,99 EUR und läuft 24 Monate.", retrieved=())
        assert set(kinds(verdict)) == {findings.UNGROUNDED_PRICE, findings.UNGROUNDED_DURATION}

    def test_evidence_names_the_kind_correctly(self):
        verdict = run(
            "Die Kündigungsfrist beträgt 14 Tage, gültig ab 01.02.2026, Kundennummer 42.",
            retrieved=(),
        )
        assert set(kinds(verdict)) == {
            findings.UNGROUNDED_DURATION,
            findings.UNGROUNDED_DATE,
            findings.UNGROUNDED_NUMBER,
        }


class TestCommitments:
    def test_an_allowed_commitment_passes(self):
        """telco_de.yaml permits schedule_callback and send_confirmation_email."""
        verdict = run("Ich rufe Sie zurück.")
        assert verdict.outcome is Outcome.PASS
        assert findings.UNSUPPORTED_COMMITMENT not in kinds(verdict)

    def test_a_disallowed_commitment_fails(self):
        verdict = run("Ich erstatte Ihnen den Betrag von 29,99 EUR.", retrieved=("29,99 EUR",))
        assert findings.UNSUPPORTED_COMMITMENT in kinds(verdict)
        offender = [e for e in verdict.evidence if e.kind == findings.UNSUPPORTED_COMMITMENT][0]
        assert "refund" in offender.detail

    def test_the_severity_override_in_telco_de_makes_it_critical(self):
        verdict = run("Ich erstatte Ihnen den Betrag.")
        assert findings.UNSUPPORTED_COMMITMENT in kinds(verdict)
        assert verdict.severity is Severity.CRITICAL

    def test_an_unsupported_commitment_severity_defaults_high_without_override(self):
        """Strip telco_de's override so the guard's own DEFAULT_SEVERITY
        applies, distinguishing "the guard thinks this is HIGH" from "this
        client dialled it up to CRITICAL"."""
        ctx = context("Ich erstatte Ihnen den Betrag.")
        stripped = ctx.profile.guards.grounding.model_copy(update={"severity_overrides": {}})
        profile = ctx.profile.model_copy(
            update={"guards": ctx.profile.guards.model_copy(update={"grounding": stripped})}
        )
        verdict = asyncio.run(GUARD.check(GuardContext(profile=profile, rules=ctx.rules, reply=ctx.reply)))
        assert findings.UNSUPPORTED_COMMITMENT in kinds(verdict)
        assert verdict.severity is Severity.HIGH

    def test_all_four_disallowed_commitment_ids_fire(self):
        cases = {
            "refund": "Ich erstatte Ihnen den Betrag.",
            "waive_fee": "Die Gebühr entfällt für Sie.",
            "credit": "Das ist eine Gutschrift für Sie.",
            "discount": "Ich möchte Ihnen einen Rabatt gewähren.",
        }
        for commitment_id, reply in cases.items():
            verdict = run(reply)
            offenders = [e for e in verdict.evidence if e.kind == findings.UNSUPPORTED_COMMITMENT]
            assert offenders, f"{commitment_id} should have fired for {reply!r}"
            assert commitment_id in offenders[0].detail

    def test_the_two_permitted_commitment_ids_never_fire(self):
        callback = run("Ich rufe Sie zurück.")
        confirmation = run("Ich sende Ihnen eine Bestätigung per E-Mail.")
        assert findings.UNSUPPORTED_COMMITMENT not in kinds(callback)
        assert findings.UNSUPPORTED_COMMITMENT not in kinds(confirmation)

    def test_evidence_span_locates_the_promise(self):
        text = "Gerne. Ich erstatte Ihnen den Betrag."
        verdict = run(text)
        (evidence,) = [e for e in verdict.evidence if e.kind == findings.UNSUPPORTED_COMMITMENT]
        lo, hi = evidence.span
        assert text[lo:hi] == "erstatte"


class TestEvidenceOrdering:
    def test_evidence_is_ordered_by_position(self):
        text = "Ich erstatte Ihnen 24,99 EUR."
        verdict = run(text)
        spans = [e.span for e in verdict.evidence]
        assert spans == sorted(spans)
