"""确定性回放。

哈希键覆盖 schema_version + system + canonical_messages + max_tokens + model。
少任何一项都会让不同的请求错误命中同一条回放，而那种 bug 的表现是「测试通过但测的
是别的东西」。

未命中时抛异常而不是回退：静默回退等于测试在骗人。

``latency_ms`` 由录制时写入 JSON 记录，回放时原样返回，而不是在这里测量。一次回放调用
诚实的延迟就是被回放的那次真实调用的延迟：录制时的网络往返、排队、模型生成耗时才是
后续评估要看的数字；这里只有一次字典查找和一次 dataclass 构造，测出来的是微秒级的
分发开销，把它当作延迟写进 trace 就是在往评估记录里灌虚构数据。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from guardrails.provider.base import CompletionResult, Turn

__all__ = ["FixtureCompletion", "SCHEMA_VERSION", "fixture_key"]

SCHEMA_VERSION = 2
"""fixture 格式版本。改格式时递增，旧记录因此显式失效而不是静默错配。"""

_REQUIRED_FIELDS = ("text", "model", "input_tokens", "output_tokens", "latency_ms")


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

        return CompletionResult(
            text=text,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=float(latency_ms),
        )
