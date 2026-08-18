"""Brand-voice guard.

Two themes. First, false positives: a guard that flags a correct reply is worse
than one that misses, because it turns working answers into handovers. Most
cases here assert that something is *not* flagged. Second, the separation of
finding from policy: a client can configure a finding to be routed as
`continue`, and the trace still records that it happened.
"""

import asyncio
from pathlib import Path

import pytest

from guardrails import findings
from guardrails.config import load_profile
from guardrails.guards import PersonaGuard, registered_guards
from guardrails.guards.base import GuardContext, effective_severity
from guardrails.locale import get_rules
from guardrails.types import Mode, Outcome, Severity, Stage

PROFILES = Path(__file__).resolve().parents[1] / "profiles"
GUARD = PersonaGuard()


def context(reply: str, *, profile: str = "telco_de", mode: Mode = Mode.CHAT) -> GuardContext:
    resolved = load_profile(PROFILES / f"{profile}.yaml").resolve(mode)
    return GuardContext(profile=resolved, rules=get_rules(resolved.locale), reply=reply)


def run(reply: str, **kwargs):
    return asyncio.run(GUARD.check(context(reply, **kwargs)))


def kinds(verdict) -> list[str]:
    return [e.kind for e in verdict.evidence]


class TestContract:
    def test_declared_severities_match_the_finding_vocabulary(self):
        """Adding a finding without a default severity, or removing one and
        leaving the entry behind, both fail here rather than at runtime."""
        assert set(PersonaGuard.DEFAULT_SEVERITY) == findings.PERSONA_FINDINGS

    def test_identity(self):
        assert PersonaGuard.name == "persona"
        assert PersonaGuard.stage is Stage.OUTPUT
        assert "persona" in registered_guards()

    def test_a_clean_reply_passes(self):
        text = "Gern prüfe ich das für Sie. Ihre Rechnung senden wir Ihnen heute zu."
        verdict = run(text)
        assert verdict.outcome is Outcome.PASS
        assert verdict.severity is Severity.NONE
        assert verdict.evidence == ()
        assert verdict.tier == 0
        assert verdict.cost_usd == 0.0

    def test_evidence_spans_index_the_reply(self):
        text = "Hallo! Kannst du mir deine Kundennummer nennen? 😊"
        verdict = run(text)
        assert verdict.evidence
        for item in verdict.evidence:
            lo, hi = item.span
            assert text[lo:hi]

    def test_evidence_is_ordered_by_position(self):
        verdict = run("Kannst du 😊 mir deine Nummer geben?")
        spans = [e.span for e in verdict.evidence]
        assert spans == sorted(spans)


class TestAddressForm:
    def test_informal_pronouns_are_flagged(self):
        verdict = run("Kannst du mir deine Kundennummer nennen?")
        assert kinds(verdict) == [findings.ADDRESS_FORM, findings.ADDRESS_FORM]
        assert verdict.severity is Severity.MEDIUM

    def test_the_detail_records_how_strong_the_signal_is(self):
        """German is grammatical; English is not, and the evidence says so
        rather than presenting both as equivalent."""
        de = run("Kannst du das machen?")
        en = run("We can't do that.", profile="telco_en")
        assert "grammatical" in de.evidence[0].detail
        assert "does not mark the distinction grammatically" in en.evidence[0].detail


class TestEmoji:
    def test_emoji_is_flagged(self):
        assert kinds(run("Das freut mich 😊")) == [findings.EMOJI]

    @pytest.mark.parametrize(
        "text",
        [
            "Der Tarif kostet 19,99 €.",          # currency sign
            "Die Laufzeit beträgt 12–24 Monate.",  # en dash
            "Er sagte „Guten Tag“ zu mir.",        # German quotation marks
            "Die Temperatur beträgt 21°C.",
        ],
    )
    def test_ordinary_symbols_are_not_emoji(self, text):
        """A false positive here rewrites a perfectly good reply."""
        assert run(text).outcome is Outcome.PASS

    @pytest.mark.parametrize(
        "emoji", ["👍🏽", "❤️", "👨‍👩‍👧‍👦", "1️⃣"]
    )
    def test_the_span_covers_the_whole_sequence(self, emoji):
        """Skin-tone modifiers, variation selectors and ZWJ joins are part of
        the grapheme. A span stopping at the base code point would leave
        invisible characters behind when the span is later deleted."""
        text = f"Alles klar {emoji} bis bald"
        verdict = run(text)
        assert len(verdict.evidence) == 1
        lo, hi = verdict.evidence[0].span
        assert text[lo:hi] == emoji

    def test_a_profile_that_permits_emoji_gets_no_finding(self):
        ctx = context("Alles klar 😊")
        permissive = ctx.profile.model_copy(
            update={
                "guards": ctx.profile.guards.model_copy(
                    update={
                        "persona": ctx.profile.guards.persona.model_copy(
                            update={
                                "persona": ctx.profile.guards.persona.persona.model_copy(
                                    update={"emoji_allowed": True}
                                )
                            }
                        )
                    }
                )
            }
        )
        verdict = asyncio.run(GUARD.check(GuardContext(profile=permissive, rules=ctx.rules, reply=ctx.reply)))
        assert verdict.outcome is Outcome.PASS


class TestForbiddenPhrases:
    def test_configured_phrase_is_flagged(self):
        assert kinds(run("Kein Ding, das erledige ich.")) == [findings.FORBIDDEN_PHRASE]

    def test_matching_respects_word_boundaries(self):
        """`mate` must not fire inside `automated`."""
        assert run("This is an automated message.", profile="telco_en").outcome is Outcome.PASS


class TestSentenceLength:
    def test_over_long_sentence_is_flagged(self):
        long = "Der Vertrag " + "sehr " * 30 + "lang."
        verdict = run(long)
        assert findings.SENTENCE_TOO_LONG in kinds(verdict)

    def test_the_span_is_the_offending_sentence(self):
        text = "Kurz. " + "Wort " * 30 + "Ende."
        verdict = run(text)
        offenders = [e for e in verdict.evidence if e.kind == findings.SENTENCE_TOO_LONG]
        assert len(offenders) == 1
        lo, hi = offenders[0].span
        assert text[lo:hi].startswith("Wort")


class TestTtsSafety:
    MARKUP = "Ihr Tarif **Basic** kostet 19,99 EUR."

    def test_not_checked_outside_voice(self):
        assert findings.TTS_UNSAFE not in kinds(run(self.MARKUP, mode=Mode.CHAT))

    def test_checked_in_voice(self):
        assert findings.TTS_UNSAFE in kinds(run(self.MARKUP, mode=Mode.VOICE))

    @pytest.mark.parametrize(
        "reply",
        [
            "Details unter https://example.com/tarife finden Sie dort.",
            "- Erster Punkt\n- Zweiter Punkt",
            "Rufen Sie 4930123456789 an.",
            "Siehe [die Tarifseite](https://example.com).",
        ],
    )
    def test_things_a_speech_synthesiser_reads_out_literally(self, reply):
        assert findings.TTS_UNSAFE in kinds(run(reply, mode=Mode.VOICE))

    def test_plain_prose_passes_in_voice(self):
        text = "Ihr Tarif kostet 19,99 Euro im Monat. Die Laufzeit beträgt 24 Monate."
        assert run(text, mode=Mode.VOICE).outcome is Outcome.PASS

    def test_a_profile_without_tts_safe_skips_it_even_in_voice(self):
        """gesundheit_de is text-only, so it defines no voice mode; telco_en does,
        with tts_safe on. This asserts the flag drives the check, not the mode."""
        ctx = context(self.MARKUP, mode=Mode.VOICE)
        spec = ctx.profile.guards.persona.persona.model_copy(update={"tts_safe": False})
        profile = ctx.profile.model_copy(
            update={
                "guards": ctx.profile.guards.model_copy(
                    update={"persona": ctx.profile.guards.persona.model_copy(update={"persona": spec})}
                )
            }
        )
        verdict = asyncio.run(GUARD.check(GuardContext(profile=profile, rules=ctx.rules, reply=self.MARKUP)))
        assert findings.TTS_UNSAFE not in kinds(verdict)


class TestSeverityIsPolicyAndOutcomeIsFact:
    def test_severity_is_the_maximum_across_findings(self):
        verdict = run("Kannst du 😊 mir deine Nummer geben?")
        assert {findings.ADDRESS_FORM, findings.EMOJI} <= set(kinds(verdict))
        assert verdict.severity is Severity.MEDIUM  # address_form, not emoji's LOW

    def test_a_profile_override_changes_severity(self):
        """telco_en dials address_form down to LOW because English cannot
        support the check as strongly."""
        assert run("Kannst du das machen?").severity is Severity.MEDIUM
        assert run("We can't do that.", profile="telco_en").severity is Severity.LOW

    def test_overriding_to_none_still_records_the_finding(self):
        """A client can route a finding as `continue`. It cannot configure the
        finding out of the trace -- the record says what happened, the severity
        says what to do about it."""
        ctx = context("Alles klar 😊")
        persona_cfg = ctx.profile.guards.persona.model_copy(
            update={"severity_overrides": {findings.EMOJI: Severity.NONE}}
        )
        profile = ctx.profile.model_copy(
            update={"guards": ctx.profile.guards.model_copy(update={"persona": persona_cfg})}
        )
        verdict = asyncio.run(GUARD.check(GuardContext(profile=profile, rules=ctx.rules, reply=ctx.reply)))
        assert verdict.outcome is Outcome.FAIL
        assert verdict.severity is Severity.NONE
        assert kinds(verdict) == [findings.EMOJI]
        assert profile.action_for(verdict.severity).value == "continue"


class TestEffectiveSeverity:
    def test_override_wins(self):
        assert effective_severity("emoji", {"emoji": Severity.LOW}, {"emoji": Severity.HIGH}) is Severity.HIGH

    def test_default_applies_without_an_override(self):
        assert effective_severity("emoji", {"emoji": Severity.LOW}, {}) is Severity.LOW

    def test_an_undeclared_kind_is_visible_rather_than_discarded(self):
        assert effective_severity("mystery", {}, {}) is Severity.MEDIUM
