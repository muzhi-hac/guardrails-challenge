"""Protocols for ordinary chat and forced-tool structured completion.

The two call shapes are deliberately separate protocols. Most callers need
plain text and should not acquire optional ``tools`` arguments they never use;
judges require a tool call whose parsed input is their result. Keeping that
concern in :class:`StructuredCompletion` also prevents an empty text result
from silently discarding a ``tool_use`` block.

``CompletionResult`` already carries token counts; USD conversion is left to a
future dated price table. Adding that table later does not require changing this
signature, and ``Verdict.cost_usd``, which has existed since M1, gets a real
source to draw from once it lands.

Both protocols are asynchronous for the same reason ``Guard.check()`` is: the
orchestrator needs one path for real network implementations and test stubs.

---
Relay behaviour observations, verified 2026-08-22, apply only to the
third-party relay endpoint, account routing, calling convention and model
identifier configured at that time; they do not represent the official
endpoint, other channels, or future behaviour:

- A request carrying an invalid ``effort`` value received no parameter error,
  and no verifiable constraining effect was observed with valid values either.
  This project therefore does not rely on that field to enforce judge effort.
- The json-schema output format did not, within the tested scope, force a
  schema-conforming result — it returned plain text instead. The judge therefore
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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple, Protocol, runtime_checkable

__all__ = [
    "Completion",
    "CompletionResult",
    "StructuredCompletion",
    "StructuredCompletionResult",
    "Turn",
]


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


@dataclass(frozen=True, slots=True)
class StructuredCompletionResult:
    """Parsed tool input plus the same provider facts as a text completion.

    The provider parses the tool block but does not validate a judge-specific
    schema. The guard that owns the schema performs that validation.
    """

    input: Mapping[str, Any]
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    stop_reason: str


class Completion(Protocol):
    async def complete(
        self, *, system: str, messages: Sequence[Turn], max_tokens: int
    ) -> CompletionResult: ...


@runtime_checkable
class StructuredCompletion(Protocol):
    """A model call whose result is one named, forced tool invocation.

    This is a second protocol rather than optional parameters on
    :class:`Completion`: ordinary chat callers have no reason to know about
    tool schemas, while a judge must never mistake the absence of text blocks
    for an empty answer.
    """

    async def complete_structured(
        self,
        *,
        system: str,
        messages: Sequence[Turn],
        max_tokens: int,
        tool_name: str,
        input_schema: Mapping[str, Any],
    ) -> StructuredCompletionResult: ...
