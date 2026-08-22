"""The protocol for chat completion.

Chat only. The forced tool use, structured output, price table, and
budget-aware timeout a judge needs all belong to M7 — their shape has to be
designed together with the tone / entailment judge, and settling it early
would settle it wrong.

``CompletionResult`` already carries token counts; the USD conversion is left
for M7. That way adding the price table later doesn't require changing this
signature, and ``Verdict.cost_usd``, which has existed since M1, gets a real
source to draw from once it lands.

``async`` even though the fixture implementation never awaits — the same
reason ``Guard.check()`` is async even when fully deterministic: it lets the
orchestrator maintain exactly one code path.

---
Relay behaviour observations, verified 2026-08-22, apply only to the
third-party relay endpoint, account routing, calling convention and model
identifier configured at that time; they do not represent the official
endpoint, other channels, or future behaviour:

- A request carrying an invalid ``effort`` value received no parameter error,
  and no verifiable constraining effect was observed with valid values either.
  This project therefore does not rely on that field to enforce judge effort.
- The json-schema output format did not, within the tested scope, force a
  schema-conforming result — it returned plain text instead. M7 therefore
  uses forced tool choice, verified on the same day within the same scope.
- A request carrying ``max_tokens=32`` returned roughly 700 characters of
  text with ``stop_reason`` of ``"end_turn"`` — meaning this relay endpoint
  did not truncate at 32 tokens. The path where "thinking exhausts the
  budget, the body comes back empty" therefore cannot be reproduced against
  this endpoint with a live call; its correctness rests on construction (the
  protocol requires ``stop_reason`` to be filled in) and unit tests, not on
  evidence observed from an actual call.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple, Protocol

__all__ = ["Completion", "CompletionResult", "Turn"]


class Turn(NamedTuple):
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class CompletionResult:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    """Wall-clock latency in milliseconds. Required — no default. A default
    would make "forgot to measure it" and "genuinely fast" indistinguishable
    to a type checker, to a test, and to whoever reads the trace, and a
    latency figure in a trace that has been distorted this way is lying to
    the person reading it."""
    stop_reason: str
    """Why the model stopped generating (e.g. ``"end_turn"``,
    ``"max_tokens"``). Required — no default, for the same reason as
    ``latency_ms``: omitting it would make "forgot to pass it" and "ended
    normally, legitimately" indistinguishable.

    Its purpose is letting a consumer distinguish "the reply was truncated"
    from "the reply is complete" — specifically, distinguishing
    ``text == ""`` with ``stop_reason == "max_tokens"`` (the budget ran out,
    thinking consumed the whole allowance, and the model never got to speak)
    from ``text == ""`` with ``stop_reason == "end_turn"`` (the model ended
    normally and genuinely had nothing to say) — two cases that look
    identical on the surface and mean completely different things.

    The provider's job is only to report this field honestly; mapping it to
    "retry", "surface an error to the user", or "treat as empty text" is the
    orchestrator's policy decision and does not belong at this layer — see
    the note at the top of this module about the provider reporting facts,
    never making policy judgements."""


class Completion(Protocol):
    async def complete(
        self, *, system: str, messages: Sequence[Turn], max_tokens: int
    ) -> CompletionResult: ...
