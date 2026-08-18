"""Tests for the profile schema.

The theme is silent failure. Nearly every case here is about a profile that
would otherwise load successfully while doing something other than what its
author intended: a mistyped key, an unroutable severity, a judge that cannot
fit in its budget, an override naming a finding that does not exist.
"""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from guardrails.config import (
    AddressForm,
    Locale,
    Profile,
    load_profile,
)
from guardrails.types import Action, Mode, Severity

PROFILES = Path(__file__).resolve().parents[1] / "profiles"


def raw_telco() -> dict:
    return yaml.safe_load((PROFILES / "telco_de.yaml").read_text(encoding="utf-8"))


class TestShippedProfilesLoad:
    @pytest.mark.parametrize("filename", ["telco_de.yaml", "gesundheit_de.yaml"])
    def test_profile_loads(self, filename):
        profile = load_profile(PROFILES / filename)
        assert profile.locale is Locale.DE_DE
        assert profile.guards.persona.persona.address_form is AddressForm.FORMAL

    def test_second_client_differs_only_in_configuration(self):
        """The multi-tenancy claim, asserted rather than described."""
        telco = load_profile(PROFILES / "telco_de.yaml").resolve(Mode.CHAT)
        health = load_profile(PROFILES / "gesundheit_de.yaml").resolve(Mode.CHAT)

        assert telco.action_for(Severity.MEDIUM) is Action.REWRITE
        assert health.action_for(Severity.MEDIUM) is Action.HANDOVER
        assert telco.guards.grounding.allowed_commitments != ()
        assert health.guards.grounding.allowed_commitments == ()


class TestSeverityNames:
    def test_names_are_accepted_for_routing_keys_and_values(self):
        profile = load_profile(PROFILES / "telco_de.yaml")
        assert profile.routing[Severity.HIGH] is Action.HANDOVER
        assert profile.guards.pii.on_timeout is Severity.CRITICAL

    def test_integers_still_work(self):
        raw = raw_telco()
        raw["guards"]["pii"]["on_timeout"] = 4
        assert Profile.model_validate(raw).guards.pii.on_timeout is Severity.CRITICAL

    def test_unknown_name_is_rejected(self):
        raw = raw_telco()
        raw["guards"]["pii"]["on_timeout"] = "catastrophic"
        with pytest.raises(ValidationError, match="unknown severity"):
            Profile.model_validate(raw)


class TestTyposAreRejected:
    def test_unknown_top_level_key(self):
        raw = raw_telco()
        raw["fallback_msg"] = "oops"
        with pytest.raises(ValidationError):
            Profile.model_validate(raw)

    def test_unknown_persona_key(self):
        raw = raw_telco()
        raw["guards"]["persona"]["persona"]["emojis_allowed"] = True  # note plural
        with pytest.raises(ValidationError):
            Profile.model_validate(raw)

    def test_unknown_finding_kind_in_severity_overrides(self):
        raw = raw_telco()
        raw["guards"]["grounding"]["severity_overrides"] = {"ungounded_price": "high"}
        with pytest.raises(ValidationError, match="unknown finding kind"):
            Profile.model_validate(raw)

    def test_finding_kind_from_another_guard_is_rejected(self):
        """`iban` is a real finding, just not one the grounding guard emits."""
        raw = raw_telco()
        raw["guards"]["grounding"]["severity_overrides"] = {"iban": "high"}
        with pytest.raises(ValidationError, match="unknown finding kind"):
            Profile.model_validate(raw)


class TestRoutingIsTotal:
    def test_missing_severity_is_rejected(self):
        raw = raw_telco()
        del raw["routing"]["critical"]
        with pytest.raises(ValidationError, match="missing: critical"):
            Profile.model_validate(raw)

    def test_every_severity_resolves_to_an_action(self):
        resolved = load_profile(PROFILES / "telco_de.yaml").resolve(Mode.CHAT)
        for severity in Severity:
            assert isinstance(resolved.action_for(severity), Action)


class TestTierFitsBudget:
    def test_tier_one_below_judge_budget_is_rejected(self):
        raw = raw_telco()
        raw["modes"]["voice"]["max_tier"] = 1  # 150 ms budget
        with pytest.raises(ValidationError, match="Every judge call would time out"):
            Profile.model_validate(raw)

    def test_lowering_the_declared_judge_latency_permits_it(self):
        """The threshold is configuration, not a constant: a faster judge
        legitimately changes the answer."""
        raw = raw_telco()
        raw["modes"]["voice"]["max_tier"] = 1
        raw["models"]["judge_min_budget_ms"] = 120
        assert Profile.model_validate(raw).modes[Mode.VOICE].max_tier == 1

    def test_tier_two_is_rejected(self):
        raw = raw_telco()
        raw["modes"]["chat"]["max_tier"] = 2
        with pytest.raises(ValidationError):
            Profile.model_validate(raw)


class TestResolution:
    def test_mode_routing_partially_overrides_the_profile_table(self):
        profile = load_profile(PROFILES / "telco_de.yaml")
        chat = profile.resolve(Mode.CHAT)
        voice = profile.resolve(Mode.VOICE)

        assert chat.action_for(Severity.MEDIUM) is Action.REWRITE
        assert voice.action_for(Severity.MEDIUM) is Action.HANDOVER
        # untouched levels are inherited
        assert voice.action_for(Severity.HIGH) is Action.HANDOVER
        assert voice.action_for(Severity.NONE) is Action.CONTINUE

    def test_mode_budget_and_tier_are_flattened(self):
        profile = load_profile(PROFILES / "telco_de.yaml")
        assert profile.resolve(Mode.VOICE).budget_ms == 150
        assert profile.resolve(Mode.VOICE).max_tier == 0
        assert profile.resolve(Mode.CHAT).max_tier == 1

    def test_guard_failure_policy_is_never_none_after_resolution(self):
        """The type is Optional for YAML's sake; resolution makes it concrete.
        This test is the guarantee that a Resolved* class hierarchy would
        otherwise provide."""
        for mode in (Mode.VOICE, Mode.CHAT):
            resolved = load_profile(PROFILES / "telco_de.yaml").resolve(mode)
            for name, guard in resolved.guards.all_guards().items():
                assert guard.on_timeout is not None, name
                assert guard.on_error is not None, name

    def test_guard_override_beats_the_mode_default(self):
        voice = load_profile(PROFILES / "telco_de.yaml").resolve(Mode.VOICE)
        assert voice.on_timeout is Severity.NONE            # mode fails open
        assert voice.guards.persona.on_timeout is Severity.NONE
        assert voice.guards.pii.on_timeout is Severity.CRITICAL  # except for PII

    def test_undefined_mode_raises_rather_than_defaulting(self):
        profile = load_profile(PROFILES / "gesundheit_de.yaml")
        with pytest.raises(KeyError, match="no mode 'voice'"):
            profile.resolve(Mode.VOICE)


class TestImmutability:
    def test_resolved_profile_is_frozen(self):
        resolved = load_profile(PROFILES / "telco_de.yaml").resolve(Mode.CHAT)
        with pytest.raises(ValidationError):
            resolved.budget_ms = 99
