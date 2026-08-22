"""真实的 Anthropic 调用。

凭据与 base_url 全部从环境变量读（``ANTHROPIC_API_KEY`` / ``ANTHROPIC_BASE_URL``）。
仓库内不出现任何 key 或端点主机名 —— 这条有一个可执行的验收项（仓库级 grep），见 README。

不传 ``thinking``，保持模型默认。据 Anthropic 官方文档，关闭 thinking 在当前一代模型
上有两个失效模式：把工具调用写进可见文本、以及泄漏内部标签；降 effort 才是受支持的
延迟控制手段。**那是文档来源的判断，不是本项目测出来的**，两者不要混。

实践上：chat 调用的延迟不在守卫预算内 —— 守卫是绕着它跑的 —— 所以这里不需要延迟控制。

---
本项目所配置的第三方中转端点的行为观察，验证于 2026-08-22，仅适用于当时配置的该
中转端点、账户路由、本模块的调用方式和当时的模型标识；不代表官方端点、其他渠道或
未来行为（同一份观察也记录在 ``guardrails.provider.base`` 的模块文档里，因为它是
M7 judge 设计的前提）：

- 请求携带非法 effort 值时未收到参数错误，使用有效值时也未观察到可验证的约束效果。
  因此本项目不依赖该字段强制控制 judge effort。
- json-schema output format 在该测试范围内未强制返回符合 schema 的结果，而是返回
  普通文本。因此后续的 judge 模块改用强制 tool choice —— 同日、同范围内验证过可用。
- 携带 ``max_tokens=32`` 的请求返回了约 700 字符的文本，``stop_reason`` 为
  ``"end_turn"``——该中转端点没有在 32 个 token 处截断。因此「思考耗尽预算、正文
  为空」这条路径无法在该端点上用实时调用复现验证；它靠构造（``stop_reason`` 字段
  必填、见下方 ``complete`` 里对 ``None`` 的处理）和单元测试兜底。
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence

from anthropic import AsyncAnthropic

from guardrails.provider.base import CompletionResult, Turn

__all__ = ["AnthropicCompletion"]


class AnthropicCompletion:
    """``Completion`` 协议的真实实现，走 Anthropic Messages API。"""

    def __init__(self, model: str, client: AsyncAnthropic | None = None) -> None:
        self._model = model
        self._client = client or AsyncAnthropic(
            api_key=os.environ["ANTHROPIC_API_KEY"],
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

        # 响应里可能同时含 thinking block（Opus 5 默认开 thinking）——
        # 只拼接 type == "text" 的块，直接取 content[0].text 在开了 thinking 时
        # 会拿到错误内容甚至报错。
        text = "".join(
            block.text for block in response.content if block.type == "text"
        )

        # SDK 类型是 ``stop_reason: StopReason | None`` —— 按文档它只在流式响应的
        # 中间事件里为 ``None``，一个已完成的 ``Message`` 理论上总有值。但
        # ``CompletionResult.stop_reason`` 是必填 ``str``（理由见 base.py），
        # 而这里没有立场把"SDK 违反了自己的文档承诺"升级成异常——那是策略判断，
        # 不是这一层该做的事。所以退化成一个明确写出"未知"的哨兵字符串，而不是让
        # ``None`` 悄悄混进一个类型上承诺是 ``str`` 的字段。
        stop_reason = response.stop_reason if response.stop_reason is not None else "unknown"

        return CompletionResult(
            text=text,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=latency_ms,
            stop_reason=stop_reason,
        )
