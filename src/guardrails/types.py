"""Shared vocabulary for the guardrail pipeline.

This module deliberately has **no intra-project imports**: guards, the
orchestrator, the configuration layer and the evaluation harness all depend on
it, and it depends on none of them. That constraint is what keeps the
dependency graph acyclic, and it is the test for whether something belongs
here — if it needs to know about a guard or a profile, it belongs elsewhere.

The central design decision is the split between what a *guard* produces and
what the *pipeline* produces:

* A guard returns a :class:`Verdict` — a finding, its severity, and the
  evidence for it. A guard never decides what to do about the finding.
* The orchestrator maps the aggregate of all verdicts onto a single
  :class:`Action`, using the active client profile.

Keeping routing policy out of the guards is what makes one engine serve
several clients with different escalation rules, and it keeps detection
auditable independently of policy.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum, StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

__all__ = [
    "Action",
    "Evidence",
    "Mode",
    "Outcome",
    "PipelineResult",
    "Severity",
    "Stage",
    "Verdict",
    "aggregate_severity",
]


class Stage(StrEnum):
    """Which segment of the pipeline a guard inspects."""

    INPUT = "input"
    """The end user's message."""

    RETRIEVAL = "retrieval"
    """Documents returned by retrieval, treated as untrusted data."""

    OUTPUT = "output"
    """The assistant's generated reply."""


class Outcome(StrEnum):
    """What happened when the guard ran.

    ``ERROR``, ``TIMEOUT`` and ``SKIPPED`` are kept distinct from ``PASS`` on
    purpose: a guard that never produced an answer must not be indistinguishable
    from one that looked and found nothing. Collapsing them is how a timeout
    silently becomes an approval.
    """

    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


class Severity(IntEnum):
    """How bad a finding is.

    An ``IntEnum`` so that aggregation is just ``max()``. It serialises as its
    lowercase name rather than its integer value, because traces are read by
    people as well as by the evaluation harness.
    """

    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

    def __str__(self) -> str:
        return self.name.lower()


class Action(StrEnum):
    """The orchestrator's decision for a turn.

    ``BLOCK`` returns nothing to the user. It is part of the vocabulary because
    some channels genuinely require silent suppression, but for a customer
    service assistant it is almost always the wrong answer — a profile is
    expected to route to ``SAFE_FALLBACK`` or ``HANDOVER`` instead.
    """

    CONTINUE = "continue"
    REWRITE = "rewrite"
    """Repair the offending spans and re-verify, rather than discarding the reply."""

    SAFE_FALLBACK = "safe_fallback"
    HANDOVER = "handover"
    """Escalate to a human agent, carrying the reason and a conversation summary."""

    BLOCK = "block"


class Mode(StrEnum):
    """Deployment mode, which selects the latency budget and the enabled tiers."""

    VOICE = "voice"
    CHAT = "chat"


_RECORD = ConfigDict(frozen=True, extra="forbid")


class Evidence(BaseModel):
    """One concrete finding, with enough detail to act on it.

    ``span`` is what makes :attr:`Action.REWRITE` possible: knowing the exact
    character range lets a repair step rewrite one sentence instead of
    discarding the whole reply.
    """

    model_config = _RECORD

    kind: str
    """Machine-readable finding type, e.g. ``ungrounded_number``, ``du_form``."""

    detail: str
    """One line of human-readable explanation."""

    span: tuple[int, int] | None = None
    """Character offsets into the inspected text, when the finding is localised."""

    source_ref: str | None = None
    """Supporting knowledge-base chunk id; ``None`` means no support was found."""


class Verdict(BaseModel):
    """One guard's record of one check. Immutable: this is a statement of fact."""

    model_config = _RECORD

    guard: str
    stage: Stage
    outcome: Outcome
    severity: Severity = Severity.NONE
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    """Deterministic guards report 1.0; LLM judges report their own calibration.

    Used as the tier-0 to tier-1 escalation criterion, and to check calibration
    during evaluation. Not meaningful when :attr:`conclusive` is false.
    """

    evidence: tuple[Evidence, ...] = ()
    latency_ms: float = Field(default=0.0, ge=0.0)
    cost_usd: float = Field(default=0.0, ge=0.0)
    """Per-guard, not just per-request: an aggregate cannot show which guard is
    the expensive one."""

    tier: int = Field(default=0, ge=0)
    """0 for deterministic checks, 1 for LLM judges. Makes "what fraction of
    requests reached tier 1" computable straight from the trace."""

    error: str | None = None

    @property
    def conclusive(self) -> bool:
        """Whether the guard actually reached a judgement about the content."""
        return self.outcome in (Outcome.PASS, Outcome.FAIL)

    @field_serializer("severity")
    def _serialise_severity(self, value: Severity) -> str:
        return value.name.lower()

    @field_validator("severity", mode="before")
    @classmethod
    def _accept_severity_name(cls, value: Any) -> Any:
        """Accept both ``3`` and ``"high"`` so traces round-trip."""
        if isinstance(value, str):
            try:
                return Severity[value.upper()]
            except KeyError as exc:
                raise ValueError(f"unknown severity {value!r}") from exc
        return value


class PipelineResult(BaseModel):
    """The orchestrator's output for one turn, and one line of the trace log."""

    model_config = _RECORD

    trace_id: str
    turn_index: int = Field(ge=0)
    mode: Mode
    action: Action
    reason: str
    """Why this action was chosen — shown to the human agent on handover."""

    verdicts: tuple[Verdict, ...] = ()
    total_latency_ms: float = Field(default=0.0, ge=0.0)
    total_cost_usd: float = Field(default=0.0, ge=0.0)
    budget_exceeded: bool = False


def aggregate_severity(verdicts: Sequence[Verdict]) -> Severity:
    """Highest severity among conclusive verdicts; :attr:`Severity.NONE` if none.

    Verdicts that are not conclusive (``ERROR``, ``TIMEOUT``, ``SKIPPED``)
    contribute no severity. They are not "no problem found" — they are "no
    answer" — and the orchestrator handles them through each guard's
    fail-open/fail-closed policy instead.
    """
    return max(
        (verdict.severity for verdict in verdicts if verdict.conclusive),
        default=Severity.NONE,
    )
