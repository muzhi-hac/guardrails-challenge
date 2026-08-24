"""Provider result semantics and forced-tool structured completion."""

from __future__ import annotations

from types import SimpleNamespace

from guardrails.provider.anthropic_client import AnthropicCompletion
from guardrails.provider.base import CompletionResult, Turn


def test_stop_reason_distinguishes_truncated_from_complete_empty_reply():
    """Regression guard: this is exactly the problem this fix addresses -- when
    text is empty, the caller must be able to use stop_reason to tell "budget ran
    out, thinking consumed the whole allowance" apart from "the model finished
    normally and genuinely had nothing to say", instead of reading both as
    "the model said nothing"."""
    truncated = CompletionResult(
        text="", model="m", input_tokens=10, output_tokens=0,
        latency_ms=1.0, stop_reason="max_tokens",
    )
    complete = CompletionResult(
        text="", model="m", input_tokens=10, output_tokens=0,
        latency_ms=1.0, stop_reason="end_turn",
    )

    assert truncated.text == complete.text == ""
    assert truncated.stop_reason != complete.stop_reason
    assert truncated.stop_reason == "max_tokens"
    assert complete.stop_reason == "end_turn"


async def test_structured_completion_forces_and_returns_the_named_tool_input():
    class Messages:
        def __init__(self):
            self.kwargs = None

        async def create(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(
                model="judge-model",
                content=[
                    SimpleNamespace(type="thinking", text="private"),
                    SimpleNamespace(
                        type="tool_use",
                        name="record_result",
                        input={"passed": True},
                    ),
                ],
                usage=SimpleNamespace(input_tokens=12, output_tokens=4),
                stop_reason="tool_use",
            )

    messages_api = Messages()
    client = SimpleNamespace(messages=messages_api)
    provider = AnthropicCompletion(model="judge-model", client=client)

    result = await provider.complete_structured(
        system="Judge.",
        messages=(Turn("user", "Reply data"),),
        max_tokens=100,
        tool_name="record_result",
        input_schema={
            "type": "object",
            "properties": {"passed": {"type": "boolean"}},
            "required": ["passed"],
        },
    )

    assert result.input == {"passed": True}
    assert result.model == "judge-model"
    assert result.input_tokens == 12
    assert result.output_tokens == 4
    assert result.latency_ms >= 0
    assert result.stop_reason == "tool_use"
    assert messages_api.kwargs["tool_choice"] == {
        "type": "tool",
        "name": "record_result",
    }
    assert messages_api.kwargs["tools"][0]["name"] == "record_result"
