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
- 携带 ``max_tokens=32`` 的请求返回了约 700 字符的文本，``stop_reason`` 为
  ``"end_turn"``——也就是说该中转端点没有在 32 个 token 处截断。因此「思考耗尽预算、
  正文为空」这条路径无法在该端点上用实时调用复现验证；它的正确性靠构造（协议要求
  ``stop_reason`` 必填）和单元测试兜底，而不是靠一次真实调用观察到的证据。
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
    """壁钟延迟（毫秒）。必须提供 —— 没有默认值。默认值会让"忘记测量"和"合法地快"
    这两种情况在类型检查器、测试和 trace 阅读者眼里都无法区分，而 trace 里的延迟
    数字一旦失真就是在骗看 trace 的人。"""
    stop_reason: str
    """模型停止生成的原因（如 ``"end_turn"``、``"max_tokens"``）。必须提供 —— 没有
    默认值，理由与 ``latency_ms`` 相同：省略它会让"忘记传"和"合法地正常结束"两种
    情况无法区分。

    它存在的目的是让消费者能区分「回复被截断」和「回复完整」——尤其是区分
    ``text == ""`` 且 ``stop_reason == "max_tokens"``（预算耗尽、思考过程占满了配额，
    模型还没来得及说话）与 ``text == ""`` 且 ``stop_reason == "end_turn"``（模型正常
    结束、确实无话可说）这两种表面相同、含义完全不同的情况。

    provider 只负责如实报告这个字段；把它映射成"重试"、"报错给用户"还是"按空文本
    处理"是编排器的策略决定，不在这一层做——见本模块开头关于 provider 只报告事实、
    不做策略判断的说明。"""


class Completion(Protocol):
    async def complete(
        self, *, system: str, messages: Sequence[Turn], max_tokens: int
    ) -> CompletionResult: ...
