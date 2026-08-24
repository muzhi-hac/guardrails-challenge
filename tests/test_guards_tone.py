"""Tier-1 tone judge: structured decisions, tier gating and failure policy."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from guardrails.guards.base import GuardContext
from guardrails.guards.tone import ToneGuard
from guardrails.locale import get_rules
from guardrails.pipeline import GuardrailPipeline
from guardrails.provider.base import StructuredCompletionResult
from guardrails.types import Mode, Outcome, Severity, Stage
from utils import load_profile

PROFILES = Path(__file__).resolve().parents[1] / "profiles"


class StubJudge:
    def __init__(self, payload=None):
        self.payload = payload
        self.calls = []

    async def complete_structured(
        self, *, system, messages, max_tokens, tool_name, input_schema
    ):
        self.calls.append(
            {
                "system": system,
                "messages": tuple(messages),
                "max_tokens": max_tokens,
                "tool_name": tool_name,
                "input_schema": input_schema,
            }
        )
        dimensions = input_schema["properties"]["assessments"]["items"]["properties"]["dimension"]["enum"]
        payload = self.payload or {
            "assessments": [
                {"dimension": dimension, "passed": True, "reason": "Matches the requested tone."}
                for dimension in dimensions
            ],
            "confidence": 0.93,
        }
        return StructuredCompletionResult(
            input=payload,
            model="stub-judge",
            input_tokens=20,
            output_tokens=12,
            latency_ms=7.5,
            stop_reason="tool_use",
        )


def profile(mode: Mode = Mode.CHAT):
    return load_profile(PROFILES / "telco_de.yaml").resolve(mode)


def context(*, resolved=None, judge=None, reply="Gern helfe ich Ihnen dabei."):
    resolved = resolved or profile()
    return GuardContext(
        profile=resolved,
        rules=get_rules(resolved.locale),
        user_message="Können Sie mir helfen?",
        reply=reply,
        judge=judge,
    )


def run(guard, ctx):
    return asyncio.run(guard.check(ctx))


def test_all_dimensions_pass():
    judge = StubJudge()
    verdict = run(ToneGuard(), context(judge=judge))

    assert verdict.outcome is Outcome.PASS
    assert verdict.evidence == ()
    assert verdict.tier == 1
    assert verdict.confidence == pytest.approx(0.93)
    assert verdict.latency_ms == pytest.approx(7.5)


def test_failed_dimension_carries_the_judges_reason():
    judge = StubJudge(
        {
            "assessments": [
                {"dimension": "formal", "passed": True, "reason": "Uses a formal register."},
                {"dimension": "empathetic", "passed": False, "reason": "Dismisses the customer's concern."},
                {"dimension": "precise", "passed": True, "reason": "States the answer precisely."},
            ],
            "confidence": 0.87,
        }
    )
    verdict = run(ToneGuard(), context(judge=judge, reply="Das ist doch offensichtlich."))

    assert verdict.outcome is Outcome.FAIL
    assert verdict.severity is Severity.LOW
    assert [item.kind for item in verdict.evidence] == ["tone"]
    assert verdict.evidence[0].detail == "empathetic: Dismisses the customer's concern."


def test_voice_pipeline_skips_the_registered_tier_one_guard_before_calling_it():
    judge = StubJudge()
    pipeline = GuardrailPipeline()
    guards = pipeline.guards_for(Stage.OUTPUT)
    assert any(guard.name == "tone" and guard.tier == 1 for guard in guards)

    resolved = profile(Mode.VOICE)
    result = asyncio.run(
        pipeline.run(
            context(resolved=resolved, judge=judge),
            Stage.OUTPUT,
            trace_id="voice-tone",
            turn_index=0,
        )
    )
    tone = next(verdict for verdict in result.verdicts if verdict.guard == "tone")
    assert tone.outcome is Outcome.SKIPPED
    assert "voice cap of 0" in tone.error
    assert judge.calls == []


def test_missing_judge_raises_and_pipeline_applies_on_error():
    guard = ToneGuard()
    with pytest.raises(RuntimeError, match="client is missing"):
        run(guard, context(judge=None))

    result = asyncio.run(
        GuardrailPipeline([guard]).run(
            context(judge=None),
            Stage.OUTPUT,
            trace_id="missing-tone",
            turn_index=0,
        )
    )
    verdict = result.verdicts[0]
    assert verdict.outcome is Outcome.ERROR
    assert verdict.severity is Severity.HIGH
    assert verdict.error == "RuntimeError"


def test_runtime_budget_check_names_the_misconfiguration():
    too_small = profile().model_copy(update={"budget_ms": 3999})
    with pytest.raises(
        ValueError,
        match=r"tone judge misconfiguration: budget_ms=3999.*judge_min_budget_ms=4000",
    ):
        run(ToneGuard(), context(resolved=too_small, judge=StubJudge()))


def test_dimensions_come_from_the_persona_spec():
    resolved = profile()
    spec = resolved.guards.persona.persona.model_copy(update={"tone": ("reassuring",)})
    persona = resolved.guards.persona.model_copy(update={"persona": spec})
    guards = resolved.guards.model_copy(update={"persona": persona})
    resolved = resolved.model_copy(update={"guards": guards})
    judge = StubJudge()

    verdict = run(ToneGuard(), context(resolved=resolved, judge=judge))

    assert verdict.outcome is Outcome.PASS
    enum = judge.calls[0]["input_schema"]["properties"]["assessments"]["items"]["properties"]["dimension"]["enum"]
    assert enum == ["reassuring"]
