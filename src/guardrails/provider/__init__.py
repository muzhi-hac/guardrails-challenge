"""模型调用层。"""

from __future__ import annotations

from guardrails.provider.base import Completion, CompletionResult, Turn
from guardrails.provider.fixture import FixtureCompletion, fixture_key

__all__ = ["Completion", "CompletionResult", "FixtureCompletion", "Turn", "fixture_key"]
