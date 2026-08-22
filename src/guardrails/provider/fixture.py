"""确定性回放。

哈希键覆盖 schema_version + system + canonical_messages + max_tokens + model。
少任何一项都会让不同的请求错误命中同一条回放，而那种 bug 的表现是「测试通过但测的
是别的东西」。

未命中时抛异常而不是回退：静默回退等于测试在骗人。
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from pathlib import Path

from guardrails.provider.base import CompletionResult, Turn

__all__ = ["FixtureCompletion", "SCHEMA_VERSION", "fixture_key"]

SCHEMA_VERSION = 1
"""fixture 格式版本。改格式时递增，旧记录因此显式失效而不是静默错配。"""


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
        started = time.perf_counter()
        key = fixture_key(system=system, messages=messages,
                          max_tokens=max_tokens, model=self._model)
        try:
            record = self._store[key]
        except KeyError as exc:
            raise KeyError(
                f"no recorded completion for key {key} in {self._path}; "
                "record it with the live provider or fix the request"
            ) from exc
        return CompletionResult(
            text=str(record["text"]),
            model=str(record["model"]),
            input_tokens=int(record["input_tokens"]),  # type: ignore[arg-type]
            output_tokens=int(record["output_tokens"]),  # type: ignore[arg-type]
            latency_ms=(time.perf_counter() - started) * 1000,
        )
