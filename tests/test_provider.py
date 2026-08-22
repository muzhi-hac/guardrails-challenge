"""Provider 层。

哈希键的宽度是这里唯一真正微妙的东西：只用 (system, messages) 的话，同一段提示词在
不同 max_tokens 或不同模型下会错误命中同一条回放 —— 那种 bug 表现为「测试通过但测的
是别的东西」，是最难发现的一类。所以键里有 max_tokens 和 model，并且有**负向测试**
钉住这一点。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from guardrails.provider.base import CompletionResult, Turn
from guardrails.provider.fixture import FixtureCompletion, fixture_key

MESSAGES = (Turn(role="user", content="Was kostet Tarif M?"),)
SYSTEM = "Du bist ein Kundenservice-Assistent."


def test_key_is_stable_across_calls():
    a = fixture_key(system=SYSTEM, messages=MESSAGES, max_tokens=512, model="m")
    b = fixture_key(system=SYSTEM, messages=MESSAGES, max_tokens=512, model="m")
    assert a == b


def test_key_changes_with_max_tokens():
    a = fixture_key(system=SYSTEM, messages=MESSAGES, max_tokens=512, model="m")
    b = fixture_key(system=SYSTEM, messages=MESSAGES, max_tokens=1024, model="m")
    assert a != b


def test_key_changes_with_model():
    a = fixture_key(system=SYSTEM, messages=MESSAGES, max_tokens=512, model="m1")
    b = fixture_key(system=SYSTEM, messages=MESSAGES, max_tokens=512, model="m2")
    assert a != b


def test_key_changes_with_schema_version(monkeypatch):
    import guardrails.provider.fixture as mod
    a = fixture_key(system=SYSTEM, messages=MESSAGES, max_tokens=512, model="m")
    monkeypatch.setattr(mod, "SCHEMA_VERSION", mod.SCHEMA_VERSION + 1)
    b = fixture_key(system=SYSTEM, messages=MESSAGES, max_tokens=512, model="m")
    assert a != b


def test_key_ignores_irrelevant_whitespace_in_serialisation():
    """规范化序列化：字段有序、无空白差异。"""
    a = fixture_key(system=SYSTEM, messages=MESSAGES, max_tokens=512, model="m")
    b = fixture_key(system=SYSTEM, messages=(Turn("user", "Was kostet Tarif M?"),),
                    max_tokens=512, model="m")
    assert a == b


async def test_replays_recorded_response(tmp_path: Path):
    key = fixture_key(system=SYSTEM, messages=MESSAGES, max_tokens=512, model="m")
    store = tmp_path / "completions.json"
    store.write_text(json.dumps({
        key: {"text": "Tarif M kostet 29,99 EUR pro Monat.",
              "model": "m", "input_tokens": 42, "output_tokens": 13,
              "latency_ms": 842.7, "stop_reason": "end_turn"}
    }), encoding="utf-8")

    provider = FixtureCompletion(store, model="m")
    result = await provider.complete(system=SYSTEM, messages=MESSAGES, max_tokens=512)

    assert result.text == "Tarif M kostet 29,99 EUR pro Monat."
    assert result.input_tokens == 42
    assert result.output_tokens == 13
    assert result.latency_ms == 842.7
    assert result.stop_reason == "end_turn"


async def test_miss_raises_rather_than_falling_back(tmp_path: Path):
    """静默回退等于测试在骗人。"""
    store = tmp_path / "completions.json"
    store.write_text("{}", encoding="utf-8")
    provider = FixtureCompletion(store, model="m")
    with pytest.raises(KeyError, match="no recorded completion"):
        await provider.complete(system=SYSTEM, messages=MESSAGES, max_tokens=512)


async def test_different_max_tokens_does_not_hit_the_same_recording(tmp_path: Path):
    """负向测试：这是宽哈希键存在的全部理由。"""
    key = fixture_key(system=SYSTEM, messages=MESSAGES, max_tokens=512, model="m")
    store = tmp_path / "completions.json"
    store.write_text(json.dumps({key: {"text": "x", "model": "m",
                                       "input_tokens": 1, "output_tokens": 1,
                                       "latency_ms": 1.0, "stop_reason": "end_turn"}}),
                     encoding="utf-8")
    provider = FixtureCompletion(store, model="m")
    with pytest.raises(KeyError):
        await provider.complete(system=SYSTEM, messages=MESSAGES, max_tokens=1024)


async def test_malformed_record_missing_field_names_file_key_and_field(tmp_path: Path):
    """字段格式检查器测试之一：缺字段的错误信息要能定位问题，而不是裸 KeyError。"""
    key = fixture_key(system=SYSTEM, messages=MESSAGES, max_tokens=512, model="m")
    store = tmp_path / "completions.json"
    store.write_text(json.dumps({
        key: {"model": "m", "input_tokens": 1, "output_tokens": 1, "latency_ms": 1.0}
    }), encoding="utf-8")

    provider = FixtureCompletion(store, model="m")
    with pytest.raises(ValueError) as excinfo:
        await provider.complete(system=SYSTEM, messages=MESSAGES, max_tokens=512)

    message = str(excinfo.value)
    assert str(store) in message
    assert key in message
    assert "text" in message


async def test_non_integer_token_count_is_rejected_not_coerced(tmp_path: Path):
    """`int("42")` 会成功，从而掩盖 fixture 编写错误 —— 拒绝而不是强转。"""
    key = fixture_key(system=SYSTEM, messages=MESSAGES, max_tokens=512, model="m")
    store = tmp_path / "completions.json"
    store.write_text(json.dumps({
        key: {"text": "x", "model": "m", "input_tokens": "42",
              "output_tokens": 1, "latency_ms": 1.0, "stop_reason": "end_turn"}
    }), encoding="utf-8")

    provider = FixtureCompletion(store, model="m")
    with pytest.raises(ValueError) as excinfo:
        await provider.complete(system=SYSTEM, messages=MESSAGES, max_tokens=512)

    message = str(excinfo.value)
    assert str(store) in message
    assert key in message
    assert "input_tokens" in message


async def test_missing_stop_reason_raises_naming_file_key_and_field(tmp_path: Path):
    """`stop_reason` 是新加的必填字段——缺它必须和缺其他字段一样，报错能定位问题。"""
    key = fixture_key(system=SYSTEM, messages=MESSAGES, max_tokens=512, model="m")
    store = tmp_path / "completions.json"
    store.write_text(json.dumps({
        key: {"text": "x", "model": "m", "input_tokens": 1,
              "output_tokens": 1, "latency_ms": 1.0}
    }), encoding="utf-8")

    provider = FixtureCompletion(store, model="m")
    with pytest.raises(ValueError) as excinfo:
        await provider.complete(system=SYSTEM, messages=MESSAGES, max_tokens=512)

    message = str(excinfo.value)
    assert str(store) in message
    assert key in message
    assert "stop_reason" in message


def test_stop_reason_distinguishes_truncated_from_complete_empty_reply():
    """回归守卫：这就是本次修复要修的那个问题——text 为空时，调用方必须能靠
    stop_reason 区分「预算耗尽、思考占满了配额」和「模型正常结束、确实无话可说」，
    而不是把两者都读成"模型什么都没说"。"""
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
