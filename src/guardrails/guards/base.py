"""The guard interface and the helpers every guard shares.

A guard answers one question about one stage of the pipeline and reports what
it found. It does not decide what should happen as a result, it does not decide
what a timeout means, and it does not catch its own exceptions. Those are the
orchestrator's, because they are policy and policy belongs to the profile.

Concretely, a guard only ever returns ``PASS`` or ``FAIL``. ``ERROR``,
``TIMEOUT`` and ``SKIPPED`` are constructed by the orchestrator from the
resolved fail-open / fail-closed configuration. A guard that caught its own
exception and decided to report ``PASS`` would have made a policy decision
invisibly.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar, Protocol, runtime_checkable

from guardrails.config import ResolvedProfile
from guardrails.locale.base import LocaleRules
from guardrails.types import Evidence, Outcome, Severity, Stage, Verdict

__all__ = ["Guard", "GuardContext", "build_verdict", "effective_severity"]


@dataclass(frozen=True, slots=True)
class GuardContext:
    """Everything a guard may need for one turn.

    A dataclass rather than a Pydantic model, and the distinction is deliberate:
    :class:`~guardrails.types.Verdict` is a record that goes into the trace and
    needs validation and JSON; this is a call argument that does neither. It
    also holds a ``LocaleRules`` protocol object, which a Pydantic model would
    only accept under ``arbitrary_types_allowed`` — a sign the wrong tool is
    being used.

    ``retrieved`` and ``history`` are unused by the persona guard. They are
    declared now so that adding the grounding and injection guards does not
    change the protocol.
    """

    profile: ResolvedProfile
    rules: LocaleRules
    user_message: str = ""
    reply: str = ""
    retrieved: tuple[str, ...] = ()
    history: tuple[str, ...] = ()


@runtime_checkable
class Guard(Protocol):
    """One check over one stage."""

    name: ClassVar[str]
    stage: ClassVar[Stage]

    DEFAULT_SEVERITY: ClassVar[Mapping[str, Severity]]
    """How serious each finding this guard can emit is, before the profile has
    its say. The guard knows the semantics — reading markup aloud is worse than
    one emoji — and the profile knows what this client cares about."""

    async def check(self, ctx: GuardContext) -> Verdict:
        """Inspect the context and report what was found.

        Coroutine even for guards that never await, so the orchestrator can run
        every guard through one ``asyncio.wait`` with a deadline instead of
        maintaining separate paths for synchronous and asynchronous checks.

        Raises rather than swallowing: exception handling is the orchestrator's,
        driven by the guard's resolved ``on_error`` policy.
        """
        ...


def effective_severity(
    kind: str,
    defaults: Mapping[str, Severity],
    overrides: Mapping[str, Severity],
) -> Severity:
    """The severity of a finding for this client.

    Unknown kinds fall back to ``MEDIUM`` rather than to ``NONE``: a finding
    nobody assigned a severity to should be visible, not silently discarded.
    Profile loading rejects unregistered kinds, so reaching that fallback means
    a guard emitted something it never declared.
    """
    return overrides.get(kind, defaults.get(kind, Severity.MEDIUM))


def build_verdict(
    *,
    guard: str,
    stage: Stage,
    evidence: tuple[Evidence, ...],
    defaults: Mapping[str, Severity],
    overrides: Mapping[str, Severity],
    latency_ms: float,
    tier: int = 0,
    cost_usd: float = 0.0,
    confidence: float = 1.0,
) -> Verdict:
    """Assemble a verdict from findings.

    Outcome and severity are decided independently. Any evidence at all makes
    the outcome ``FAIL``, even where the profile has overridden that finding to
    ``Severity.NONE``. A client can configure a finding to be routed as
    ``continue``; it cannot configure it out of the record. Otherwise the trace
    would report that nothing was found, when what happened is that something
    was found and deliberately allowed.
    """
    severity = max(
        (effective_severity(item.kind, defaults, overrides) for item in evidence),
        default=Severity.NONE,
    )
    return Verdict(
        guard=guard,
        stage=stage,
        outcome=Outcome.FAIL if evidence else Outcome.PASS,
        severity=severity,
        confidence=confidence,
        evidence=evidence,
        latency_ms=latency_ms,
        cost_usd=cost_usd,
        tier=tier,
    )
