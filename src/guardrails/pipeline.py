"""The orchestrator: runs a stage's guards under a budget and decides one action.

This is where every policy decision lives, and that concentration is the point.
Guards report findings; this module decides what a finding is worth, what a
timeout means, what to do when a guard raises, and which single action the turn
gets. Splitting those decisions across the guards would make them unauditable
and would tie each client's escalation rules to guard code.

Three properties the rest of the system leans on:

**One deadline per stage, not per guard.** ``budget_ms`` is the total a caller
waits, so five guards under a 150 ms budget still finish in 150 ms rather than
750. The deadline is absolute and shared.

**Verdicts come back in registry order.** Guards run concurrently, but the
result is ordered by registration, not by completion. Otherwise a trace would
differ between runs and between machines for no reason anyone cares about.

**Who produces which outcome is fixed.** ``PASS`` and ``FAIL`` come from
guards. ``ERROR``, ``TIMEOUT`` and ``SKIPPED`` are constructed here, from the
resolved profile — the fail-open and fail-closed policy never reaches a guard.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence

from guardrails.config import GuardConfig, ResolvedProfile
from guardrails.guards import default_guards
from guardrails.guards.base import Guard, GuardContext
from guardrails.types import (
    Action,
    Outcome,
    PipelineResult,
    Severity,
    Stage,
    Verdict,
    aggregate_severity,
)

__all__ = ["GuardrailPipeline"]


class GuardrailPipeline:
    """Runs guards for one stage of one turn."""

    def __init__(self, guards: Sequence[Guard] | None = None) -> None:
        self._guards: tuple[Guard, ...] = (
            tuple(guards) if guards is not None else default_guards()
        )

    def guards_for(self, stage: Stage) -> tuple[Guard, ...]:
        return tuple(guard for guard in self._guards if guard.stage is stage)

    async def run(
        self,
        ctx: GuardContext,
        stage: Stage,
        *,
        trace_id: str,
        turn_index: int,
    ) -> PipelineResult:
        """Run every guard registered for ``stage`` and return one decision.

        Stages are invoked separately rather than as one concurrent sweep,
        because they are genuinely sequential: input guards must run before the
        model is called and output guards after it. Passing the same
        ``trace_id`` and ``turn_index`` to each stage stitches them back
        together in the trace.
        """
        profile = ctx.profile
        started = time.perf_counter()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + profile.budget_ms / 1000

        scheduled: list[tuple[Guard, GuardConfig, asyncio.Task[Verdict] | None]] = []
        for guard in self.guards_for(stage):
            config = self._config_for(profile, guard)
            skip = self._skip_reason(guard, config, profile)
            task = None if skip else asyncio.create_task(guard.check(ctx))
            scheduled.append((guard, config, task))

        pending_tasks = [task for _, _, task in scheduled if task is not None]
        if pending_tasks:
            remaining = max(0.0, deadline - loop.time())
            _, pending = await asyncio.wait(pending_tasks, timeout=remaining)
            for task in pending:
                task.cancel()
            if pending:
                # Drain the cancellations so no task outlives the call and the
                # event loop has nothing to warn about.
                await asyncio.gather(*pending, return_exceptions=True)

        verdicts = tuple(
            self._collect(guard, config, task, profile)
            for guard, config, task in scheduled
        )

        severity = aggregate_severity(verdicts)
        action = profile.action_for(severity)
        return PipelineResult(
            trace_id=trace_id,
            turn_index=turn_index,
            mode=profile.mode,
            action=action,
            reason=_describe(verdicts, severity, stage),
            verdicts=verdicts,
            total_latency_ms=(time.perf_counter() - started) * 1000,
            total_cost_usd=sum(verdict.cost_usd for verdict in verdicts),
            budget_exceeded=any(v.outcome is Outcome.TIMEOUT for v in verdicts),
        )

    # -- scheduling ---------------------------------------------------------

    @staticmethod
    def _config_for(profile: ResolvedProfile, guard: Guard) -> GuardConfig:
        try:
            return profile.guards.all_guards()[guard.name]
        except KeyError as exc:
            raise KeyError(
                f"guard {guard.name!r} is registered but has no field on "
                f"GuardsConfig; add one so it can be configured per client"
            ) from exc

    @staticmethod
    def _skip_reason(guard: Guard, config: GuardConfig, profile: ResolvedProfile) -> str | None:
        if not config.enabled:
            return "disabled for this client"
        if guard.tier > profile.max_tier:
            return f"tier {guard.tier} above the {profile.mode.value} cap of {profile.max_tier}"
        return None

    # -- outcome construction ----------------------------------------------

    def _collect(
        self,
        guard: Guard,
        config: GuardConfig,
        task: asyncio.Task[Verdict] | None,
        profile: ResolvedProfile,
    ) -> Verdict:
        if task is None:
            return self._non_result(
                guard, Outcome.SKIPPED, Severity.NONE, self._skip_reason(guard, config, profile)
            )
        if task.cancelled():
            return self._non_result(
                guard, Outcome.TIMEOUT, _severity_of(config.on_timeout), "budget exhausted"
            )
        if (exc := task.exception()) is not None:
            # Only the exception *type*. A trace is archived and shown to
            # operators; an unhandled message can carry request content or a
            # credential straight into it.
            return self._non_result(guard, Outcome.ERROR, _severity_of(config.on_error), type(exc).__name__)
        return task.result()

    @staticmethod
    def _non_result(guard: Guard, outcome: Outcome, severity: Severity, error: str | None) -> Verdict:
        return Verdict(
            guard=guard.name,
            stage=guard.stage,
            outcome=outcome,
            severity=severity,
            confidence=0.0,  # not meaningful: the guard reached no judgement
            tier=guard.tier,
            error=error,
        )


def _severity_of(configured: Severity | None) -> Severity:
    """After resolution these are always concrete; the fallback is defensive."""
    return Severity.NONE if configured is None else configured


def _describe(verdicts: tuple[Verdict, ...], severity: Severity, stage: Stage) -> str:
    """A one-line explanation, written for the human agent a handover reaches."""
    if not verdicts:
        return f"no guards registered for the {stage.value} stage"

    parts: list[str] = []
    for verdict in verdicts:
        if verdict.outcome is Outcome.FAIL:
            kinds = ", ".join(dict.fromkeys(item.kind for item in verdict.evidence))
            parts.append(f"{verdict.guard} found {kinds}")
        elif verdict.outcome in (Outcome.ERROR, Outcome.TIMEOUT):
            parts.append(f"{verdict.guard} {verdict.outcome.value} ({verdict.error})")

    if not parts:
        return "no findings"
    return f"{'; '.join(parts)} — severity {severity}"
