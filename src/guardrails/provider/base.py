"""聊天补全的协议。

只做 chat。judge 需要的强制 tool use、结构化输出、价格表和预算感知的超时都属于 M7 ——
它们的形状要跟 tone / entailment judge 一起设计，提前定型会定错。

``CompletionResult`` 现在就带 token 数，USD 换算留 M7。这样加价格表时不用改签名，
而 M1 就存在的 ``Verdict.cost_usd`` 将来有真实来源。

``async`` 即使 fixture 实现从不 await —— 和 ``Guard.check()`` 即使全确定性也是 async
是同一个理由：让编排器只维护一条路径。

---
中转行为观察，验证于 2026-08-22，仅适用于当时配置的第三方中转端点、账户路由、
调用方式和模型标识；不代表官方端点、其他渠道或未来行为：

- 请求携带非法 effort 值时未收到参数错误，使用有效值时也未观察到可验证的约束效果。
  因此本项目不依赖该字段强制控制 judge effort。
- json-schema format 在该测试范围内未强制返回符合 schema 的结果，而是返回普通文本。
  因此 M7 使用同日、同范围内验证过的强制 tool choice。
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
    latency_ms: float = 0.0


class Completion(Protocol):
    async def complete(
        self, *, system: str, messages: Sequence[Turn], max_tokens: int
    ) -> CompletionResult: ...
