"""真实端点调用。默认跳过 —— 需要 -m live 才跑。

凭据与端点只从环境变量读，不出现在仓库里的任何文件中。
"""

from __future__ import annotations

import os

import pytest

from guardrails.provider.anthropic_client import AnthropicCompletion
from guardrails.provider.base import Turn

pytestmark = pytest.mark.live


@pytest.fixture
def provider():
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY 未设置")
    return AnthropicCompletion(model="claude-opus-5")


async def test_returns_text_and_token_counts(provider):
    result = await provider.complete(
        system="Antworte auf Deutsch in genau einem Satz.",
        messages=(Turn("user", "Was ist eine Kündigungsfrist?"),),
        max_tokens=128,
    )
    assert result.text.strip()
    assert result.input_tokens > 0
    assert result.output_tokens > 0
    assert result.latency_ms > 0
    assert isinstance(result.stop_reason, str) and result.stop_reason
