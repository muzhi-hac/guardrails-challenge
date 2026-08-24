"""The model-calling layer."""

from __future__ import annotations

from guardrails.provider.anthropic_client import AnthropicCompletion
from guardrails.provider.base import (
    Completion,
    CompletionResult,
    StructuredCompletion,
    StructuredCompletionResult,
    Turn,
)
from guardrails.provider.fixture import FixtureCompletion, fixture_key

__all__ = [
    "AnthropicCompletion",
    "Completion",
    "CompletionResult",
    "StructuredCompletion",
    "StructuredCompletionResult",
    "FixtureCompletion",
    "Turn",
    "fixture_key",
]
