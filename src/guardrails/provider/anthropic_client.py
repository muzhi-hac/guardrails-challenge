"""The real Anthropic call.

Credentials and the base URL are read entirely from environment variables
(``ANTHROPIC_API_KEY`` / ``ANTHROPIC_BASE_URL``). No key or endpoint hostname
appears anywhere in this repository — this has an executable acceptance
check (a repository-wide grep), see README.

``thinking`` is not passed, leaving the model's default. Per Anthropic's
official documentation, disabling thinking on the current generation of
models has two failure modes: a tool call can land in the visible text, and
internal tags can leak; lowering effort is the supported latency control
instead. **That is a judgement drawn from the documentation, not something
this project measured** — the two must not be conflated.

In practice: a chat call's latency is not inside a guard budget — guards run
around it — so no latency control is needed here.

---
Behaviour observations about the third-party relay endpoint this project is
configured against, verified 2026-08-22, apply only to that relay endpoint,
account routing, this module's calling convention, and the model identifier
in effect at that time; they do not represent the official endpoint, other
channels, or future behaviour (the same observations are also recorded in
the module docstring of ``guardrails.provider.base``, because they are a
premise of the tier-1 judge design):

- A request carrying an invalid effort value received no parameter error,
  and no verifiable constraining effect was observed with valid values
  either. This project therefore does not rely on that field to enforce
  judge effort.
- The json-schema output format did not, within the tested scope, force a
  schema-conforming result — it returned plain text instead. The tone judge
  module therefore switches to forced tool choice — verified as working on
  the same day, within the same scope.
- A request carrying ``max_tokens=32`` returned roughly 700 characters of
  text with ``stop_reason`` of ``"end_turn"`` — this relay endpoint did not
  truncate at 32 tokens. The path where "thinking exhausts the budget, the
  body comes back empty" therefore cannot be reproduced against this
  endpoint with a live call; it rests instead on construction (the
  ``stop_reason`` field is required — see the handling of ``None`` in
  ``complete`` below) and on unit tests.
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping, Sequence
from typing import Any

from anthropic import AsyncAnthropic

from guardrails.provider.base import CompletionResult, StructuredCompletionResult, Turn

__all__ = ["AnthropicCompletion"]


def _require_api_key() -> str:
    """Read ``ANTHROPIC_API_KEY``, failing with guidance instead of a bare
    ``KeyError``.

    Every other failure path in this codebase names the offending file and
    what to do about it (``load_profile``, ``documents._parse``); a first-time reader running the
    documented command without a key deserves the same, not a traceback
    ending in ``KeyError: 'ANTHROPIC_API_KEY'`` with no next step.
    """
    try:
        return os.environ["ANTHROPIC_API_KEY"]
    except KeyError as exc:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Export it to your Anthropic API "
            "key before running this command. ANTHROPIC_BASE_URL is "
            "optional and only needed to route through a non-default "
            "endpoint."
        ) from exc


class AnthropicCompletion:
    """Anthropic implementation of both completion protocols.

    Plain chat concatenates text blocks. Structured completion instead
    requires one forced ``tool_use`` block and returns its parsed input; the
    paths stay separate so a judge result cannot be reduced to ``text=""``.
    """

    def __init__(self, model: str, client: AsyncAnthropic | None = None) -> None:
        self._model = model
        self._client = client or AsyncAnthropic(
            api_key=_require_api_key(),
            base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
        )

    async def complete(
        self, *, system: str, messages: Sequence[Turn], max_tokens: int
    ) -> CompletionResult:
        started = time.perf_counter()
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": t.role, "content": t.content} for t in messages],
        )
        latency_ms = (time.perf_counter() - started) * 1000

        # The response may also contain a thinking block (Opus 5 has thinking
        # on by default) — only blocks with type == "text" are concatenated;
        # reading content[0].text directly would get the wrong content, or
        # raise, whenever thinking is on.
        text = "".join(
            block.text for block in response.content if block.type == "text"
        )

        # The SDK's type is ``stop_reason: StopReason | None`` — per the docs
        # it is ``None`` only in the intermediate events of a streamed
        # response, and a completed ``Message`` should in principle always
        # have a value. But ``CompletionResult.stop_reason`` is a required
        # ``str`` (see base.py for why), and this layer has no standing to
        # escalate "the SDK violated its own documented contract" into an
        # exception — that is a policy decision, not this layer's job. So it
        # falls back to a sentinel string that explicitly spells out
        # "unknown", rather than letting ``None`` quietly slip into a field
        # whose type promises ``str``.
        stop_reason = response.stop_reason if response.stop_reason is not None else "unknown"

        return CompletionResult(
            text=text,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=latency_ms,
            stop_reason=stop_reason,
        )

    async def complete_structured(
        self,
        *,
        system: str,
        messages: Sequence[Turn],
        max_tokens: int,
        tool_name: str,
        input_schema: Mapping[str, Any],
    ) -> StructuredCompletionResult:
        """Return the input of exactly one forced invocation of ``tool_name``."""
        started = time.perf_counter()
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": t.role, "content": t.content} for t in messages],
            tools=[{"name": tool_name, "input_schema": dict(input_schema)}],
            tool_choice={"type": "tool", "name": tool_name},
        )
        latency_ms = (time.perf_counter() - started) * 1000

        matches = [
            block
            for block in response.content
            if block.type == "tool_use" and block.name == tool_name
        ]
        if len(matches) != 1:
            raise ValueError(
                f"expected exactly one {tool_name!r} tool_use block; got {len(matches)}"
            )
        payload = matches[0].input
        if not isinstance(payload, Mapping):
            raise ValueError(f"tool {tool_name!r} returned a non-object input")

        stop_reason = response.stop_reason if response.stop_reason is not None else "unknown"
        return StructuredCompletionResult(
            input=dict(payload),
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=latency_ms,
            stop_reason=stop_reason,
        )
