"""The orchestrator.

Fake guards borrow the names of real ones (`grounding`, `pii`, `injection`) so
that configuration lookup, per-guard failure overrides and tier gating are
exercised through the real path rather than a test-only shortcut.
"""

import asyncio
from pathlib import Path

import pytest

from guardrails.config import load_profile
from guardrails.guards.base import GuardContext
from guardrails.locale import get_rules
from guardrails.pipeline import GuardrailPipeline
from guardrails.types import (
    Action,
    Evidence,
    Mode,
    Outcome,
    Severity,
    Stage,
    Verdict,
)

PROFILES = Path(__file__).resolve().parents[1] / "profiles"


def resolved(profile: str = "telco_de", mode: Mode = Mode.CHAT, **overrides):
    p = load_profile(PROFILES / f"{profile}.yaml").resolve(mode)
    return p.model_copy(update=overrides) if overrides else p


def context(profile=None, reply: str = "Guten Tag.") -> GuardContext:
    p = profile or resolved()
    return GuardContext(profile=p, rules=get_rules(p.locale), reply=reply)


def run(pipeline: GuardrailPipeline, ctx: GuardContext, stage: Stage = Stage.OUTPUT):
    return asyncio.run(pipeline.run(ctx, stage, trace_id="t-1", turn_index=0))


# --- fakes -----------------------------------------------------------------


class _Fake:
    stage = Stage.OUTPUT
    tier = 0
    DEFAULT_SEVERITY: dict[str, Severity] = {}

    def __init__(self, name: str):
        self.name = name


class PassGuard(_Fake):
    async def check(self, ctx):
        return Verdict(guard=self.name, stage=self.stage, outcome=Outcome.PASS)


class FailGuard(_Fake):
    def __init__(self, name: str, severity: Severity, *, cost: float = 0.0):
        super().__init__(name)
        self.severity, self.cost = severity, cost

    async def check(self, ctx):
        return Verdict(
            guard=self.name,
            stage=self.stage,
            outcome=Outcome.FAIL,
            severity=self.severity,
            evidence=(Evidence(kind="synthetic", detail="planted by the test"),),
            cost_usd=self.cost,
        )


class BoomGuard(_Fake):
    async def check(self, ctx):
        raise RuntimeError("secret-bearing message that must not reach the trace")


class SlowGuard(_Fake):
    def __init__(self, name: str, delay: float):
        super().__init__(name)
        self.delay, self.cancelled = delay, False

    async def check(self, ctx):
        try:
            await asyncio.sleep(self.delay)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return Verdict(guard=self.name, stage=self.stage, outcome=Outcome.PASS)


class TierOneGuard(_Fake):
    tier = 1

    async def check(self, ctx):
        return Verdict(guard=self.name, stage=self.stage, outcome=Outcome.PASS, tier=1)


class InputGuard(_Fake):
    stage = Stage.INPUT

    async def check(self, ctx):
        return Verdict(guard=self.name, stage=self.stage, outcome=Outcome.PASS)


# --- tests -----------------------------------------------------------------


class TestBasicOutcomes:
    def test_single_pass(self):
        result = run(GuardrailPipeline([PassGuard("pii")]), context())
        assert result.verdicts[0].outcome is Outcome.PASS
        assert result.action is Action.CONTINUE
        assert result.reason == "no findings"
        assert result.budget_exceeded is False

    def test_single_fail_routes_through_the_profile(self):
        result = run(GuardrailPipeline([FailGuard("pii", Severity.HIGH)]), context())
        assert result.action is Action.HANDOVER
        assert "pii found synthetic" in result.reason

    def test_severity_is_the_maximum_across_guards(self):
        pipeline = GuardrailPipeline(
            [
                FailGuard("pii", Severity.LOW),
                FailGuard("grounding", Severity.HIGH),
                FailGuard("injection", Severity.MEDIUM),
            ]
        )
        assert run(pipeline, context()).action is Action.HANDOVER

    def test_fail_at_severity_none_keeps_its_evidence(self):
        """Configuration decides the consequence; the record still happened."""
        result = run(GuardrailPipeline([FailGuard("pii", Severity.NONE)]), context())
        assert result.verdicts[0].outcome is Outcome.FAIL
        assert result.verdicts[0].evidence
        assert result.action is Action.CONTINUE

    def test_no_guards_for_the_stage(self):
        result = run(GuardrailPipeline([InputGuard("pii")]), context(), Stage.OUTPUT)
        assert result.verdicts == ()
        assert result.action is Action.CONTINUE
        assert "no guards registered" in result.reason


class TestStageSelection:
    def test_only_the_requested_stage_runs(self):
        pipeline = GuardrailPipeline([InputGuard("pii"), PassGuard("grounding")])
        assert [v.guard for v in run(pipeline, context(), Stage.INPUT).verdicts] == ["pii"]
        assert [v.guard for v in run(pipeline, context(), Stage.OUTPUT).verdicts] == ["grounding"]

    def test_the_default_registry_splits_injection_across_input_and_retrieval(self):
        """Pins the registry-level shape of the injection/document split:
        the user-turn channel is registered at INPUT, the document channel
        at RETRIEVAL -- not both bundled at RETRIEVAL the way a single
        combined guard used to be.
        """
        from guardrails.guards import DocumentGuard, InjectionGuard

        pipeline = GuardrailPipeline()
        assert [g.name for g in pipeline.guards_for(Stage.INPUT)] == [InjectionGuard.name]
        assert [g.name for g in pipeline.guards_for(Stage.RETRIEVAL)] == [DocumentGuard.name]


class TestSkipping:
    def test_disabled_guard_is_skipped_not_run(self):
        profile = resolved()
        disabled = profile.guards.model_copy(
            update={"pii": profile.guards.pii.model_copy(update={"enabled": False})}
        )
        result = run(
            GuardrailPipeline([BoomGuard("pii")]),  # would raise if it ran
            context(profile.model_copy(update={"guards": disabled})),
        )
        assert result.verdicts[0].outcome is Outcome.SKIPPED
        assert "disabled" in result.verdicts[0].error

    def test_tier_above_the_mode_cap_is_skipped(self):
        """Voice caps the cascade at tier 0, so a judge is never invoked -- the
        gate is before the call, not a wasted round trip."""
        result = run(GuardrailPipeline([TierOneGuard("pii")]), context(resolved(mode=Mode.VOICE)))
        assert result.verdicts[0].outcome is Outcome.SKIPPED
        assert "tier 1 above the voice cap of 0" in result.verdicts[0].error

    def test_the_same_guard_runs_in_chat(self):
        result = run(GuardrailPipeline([TierOneGuard("pii")]), context(resolved(mode=Mode.CHAT)))
        assert result.verdicts[0].outcome is Outcome.PASS

    def test_all_skipped_continues(self):
        result = run(GuardrailPipeline([TierOneGuard("pii")]), context(resolved(mode=Mode.VOICE)))
        assert result.action is Action.CONTINUE
        assert result.budget_exceeded is False


class TestErrors:
    def test_an_exception_becomes_an_error_verdict(self):
        result = run(GuardrailPipeline([BoomGuard("grounding")]), context())
        verdict = result.verdicts[0]
        assert verdict.outcome is Outcome.ERROR
        assert verdict.severity is Severity.HIGH  # chat fails closed
        assert verdict.confidence == 0.0

    def test_the_trace_records_the_type_not_the_message(self):
        """A trace is archived and shown to operators. An unhandled exception
        message can carry request content or a credential straight into it."""
        result = run(GuardrailPipeline([BoomGuard("grounding")]), context())
        assert result.verdicts[0].error == "RuntimeError"
        assert "secret-bearing" not in result.model_dump_json()

    def test_voice_fails_open_on_error(self):
        result = run(GuardrailPipeline([BoomGuard("grounding")]), context(resolved(mode=Mode.VOICE)))
        assert result.verdicts[0].severity is Severity.NONE
        assert result.action is Action.CONTINUE

    def test_a_per_guard_override_beats_the_mode_default(self):
        """telco_de keeps the PII guard fail-closed even where voice fails open:
        a PII check that did not run is not a PII check that passed."""
        result = run(GuardrailPipeline([BoomGuard("pii")]), context(resolved(mode=Mode.VOICE)))
        assert result.verdicts[0].severity is Severity.CRITICAL
        assert result.action is Action.SAFE_FALLBACK


class TestTimeouts:
    def test_a_slow_guard_times_out_and_is_cancelled(self):
        slow = SlowGuard("grounding", delay=5.0)
        result = run(GuardrailPipeline([slow]), context(resolved(budget_ms=30)))
        assert result.verdicts[0].outcome is Outcome.TIMEOUT
        assert result.verdicts[0].error == "budget exhausted"
        assert slow.cancelled is True
        assert result.budget_exceeded is True

    def test_timeout_severity_follows_the_mode_policy(self):
        chat = run(GuardrailPipeline([SlowGuard("grounding", 5.0)]), context(resolved(budget_ms=30)))
        voice = run(
            GuardrailPipeline([SlowGuard("grounding", 5.0)]),
            context(resolved(mode=Mode.VOICE, budget_ms=30)),
        )
        assert (chat.verdicts[0].severity, chat.action) == (Severity.HIGH, Action.HANDOVER)
        assert (voice.verdicts[0].severity, voice.action) == (Severity.NONE, Action.CONTINUE)

    def test_a_fast_guard_still_reports_when_a_slow_one_times_out(self):
        pipeline = GuardrailPipeline([PassGuard("pii"), SlowGuard("grounding", 5.0)])
        result = run(pipeline, context(resolved(budget_ms=40)))
        assert [v.outcome for v in result.verdicts] == [Outcome.PASS, Outcome.TIMEOUT]

    def test_budget_exceeded_is_not_set_by_wall_clock_alone(self):
        """Cancellation and trace assembly cost a little time of their own;
        only an actual timeout counts."""
        result = run(GuardrailPipeline([PassGuard("pii")]), context(resolved(budget_ms=1)))
        assert result.budget_exceeded is False


class TestConcurrencyAndOrder:
    def test_guards_run_concurrently(self):
        delay = 0.06
        pipeline = GuardrailPipeline([SlowGuard(n, delay) for n in ("pii", "grounding", "injection")])
        result = run(pipeline, context(resolved(budget_ms=2000)))
        assert all(v.outcome is Outcome.PASS for v in result.verdicts)
        assert result.total_latency_ms < delay * 3 * 1000 * 0.8

    def test_verdicts_follow_registration_order_not_completion_order(self):
        pipeline = GuardrailPipeline(
            [SlowGuard("pii", 0.05), PassGuard("grounding"), SlowGuard("injection", 0.02)]
        )
        result = run(pipeline, context(resolved(budget_ms=2000)))
        assert [v.guard for v in result.verdicts] == ["pii", "grounding", "injection"]

    def test_one_deadline_is_shared_across_guards(self):
        """A 60 ms budget is what the caller waits, not what each guard gets."""
        pipeline = GuardrailPipeline([SlowGuard(n, 5.0) for n in ("pii", "grounding", "injection")])
        result = run(pipeline, context(resolved(budget_ms=60)))
        assert all(v.outcome is Outcome.TIMEOUT for v in result.verdicts)
        assert result.total_latency_ms < 400


class TestAccounting:
    def test_total_latency_is_wall_clock_not_the_sum_of_guards(self):
        delay = 0.05
        pipeline = GuardrailPipeline([SlowGuard(n, delay) for n in ("pii", "grounding")])
        result = run(pipeline, context(resolved(budget_ms=2000)))
        assert result.total_latency_ms >= delay * 1000
        assert result.total_latency_ms < delay * 2 * 1000

    def test_cost_is_summed_across_guards(self):
        pipeline = GuardrailPipeline(
            [
                FailGuard("pii", Severity.LOW, cost=0.0004),
                FailGuard("grounding", Severity.LOW, cost=0.0011),
            ]
        )
        assert run(pipeline, context()).total_cost_usd == pytest.approx(0.0015)


class TestRegistryContract:
    def test_a_guard_without_a_config_field_fails_loudly(self):
        with pytest.raises(KeyError, match="no field on GuardsConfig"):
            run(GuardrailPipeline([PassGuard("nonexistent")]), context())
