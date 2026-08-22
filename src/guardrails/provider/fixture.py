"""Deterministic replay.

The hash key covers schema_version + system + canonical_messages + max_tokens
+ model. Leaving out any one of these would let different requests wrongly
hit the same recording, and that bug's symptom is "the test passes, but it
is testing something else."

A miss raises rather than falling back: a silent fallback means the test is
lying.

``latency_ms`` is written into the JSON record at recording time and returned
unchanged at replay time, rather than being measured here. The honest
latency for a replayed call is the latency of the real call that was
recorded: the network round trip, queueing, and model generation time at
recording time are the numbers a later evaluation actually needs. What
happens here is one dictionary lookup and one dataclass construction —
measuring that would capture microsecond-scale dispatch overhead, and
writing that into the trace as "latency" would be pouring fabricated data
into the evaluation record.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from guardrails.provider.base import CompletionResult, Turn

__all__ = ["FixtureCompletion", "SCHEMA_VERSION", "fixture_key"]

SCHEMA_VERSION = 3
"""The fixture format version. Bump it when the format changes, so old
records are explicitly invalidated instead of silently mismatched."""

_REQUIRED_FIELDS = (
    "text", "model", "input_tokens", "output_tokens", "latency_ms", "stop_reason",
)


def fixture_key(
    *, system: str, messages: Sequence[Turn], max_tokens: int, model: str
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "system": system,
        "messages": [{"role": t.role, "content": t.content} for t in messages],
        "max_tokens": max_tokens,
        "model": model,
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False,
                           separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


class FixtureCompletion:
    def __init__(self, path: Path, model: str) -> None:
        self._path = path
        self._model = model
        self._store: dict[str, dict[str, object]] = json.loads(
            path.read_text(encoding="utf-8")
        )

    async def complete(
        self, *, system: str, messages: Sequence[Turn], max_tokens: int
    ) -> CompletionResult:
        key = fixture_key(system=system, messages=messages,
                          max_tokens=max_tokens, model=self._model)
        try:
            record = self._store[key]
        except KeyError as exc:
            raise KeyError(
                f"no recorded completion for key {key} in {self._path}; "
                "record it with the live provider or fix the request"
            ) from exc
        return self._parse_record(record, key=key)

    def _parse_record(self, record: dict[str, object], *, key: str) -> CompletionResult:
        for field in _REQUIRED_FIELDS:
            if field not in record:
                raise ValueError(
                    f"malformed fixture record for key {key} in {self._path}: "
                    f"missing required field {field!r}"
                )

        text = record["text"]
        if not isinstance(text, str):
            raise ValueError(
                f"malformed fixture record for key {key} in {self._path}: "
                f"field 'text' must be a string, got {text!r}"
            )
        model = record["model"]
        if not isinstance(model, str):
            raise ValueError(
                f"malformed fixture record for key {key} in {self._path}: "
                f"field 'model' must be a string, got {model!r}"
            )
        input_tokens = record["input_tokens"]
        if not isinstance(input_tokens, int) or isinstance(input_tokens, bool):
            raise ValueError(
                f"malformed fixture record for key {key} in {self._path}: "
                f"field 'input_tokens' must be an int, got {input_tokens!r}"
            )
        output_tokens = record["output_tokens"]
        if not isinstance(output_tokens, int) or isinstance(output_tokens, bool):
            raise ValueError(
                f"malformed fixture record for key {key} in {self._path}: "
                f"field 'output_tokens' must be an int, got {output_tokens!r}"
            )
        latency_ms = record["latency_ms"]
        if not isinstance(latency_ms, (int, float)) or isinstance(latency_ms, bool):
            raise ValueError(
                f"malformed fixture record for key {key} in {self._path}: "
                f"field 'latency_ms' must be a number, got {latency_ms!r}"
            )
        stop_reason = record["stop_reason"]
        if not isinstance(stop_reason, str):
            raise ValueError(
                f"malformed fixture record for key {key} in {self._path}: "
                f"field 'stop_reason' must be a string, got {stop_reason!r}"
            )

        return CompletionResult(
            text=text,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=float(latency_ms),
            stop_reason=stop_reason,
        )
