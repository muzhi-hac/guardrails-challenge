"""Calls the real endpoint. Skipped by default -- needs -m live to run.

Credentials and endpoints are read only from environment variables; they never appear
in any file in the repository.
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
        pytest.skip("ANTHROPIC_API_KEY is not set")
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
