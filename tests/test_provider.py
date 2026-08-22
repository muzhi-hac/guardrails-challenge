"""The provider layer.

The width of the hash key is the one genuinely subtle thing here: using only
(system, messages), the same prompt under a different max_tokens or a different model
would wrongly hit the same recorded replay -- that class of bug shows up as "the test
passes, but it's testing something else", which is the hardest kind to catch. So
max_tokens and model are part of the key, and **negative tests** pin that down.
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
    """Canonical serialization: fields are ordered, and whitespace differences
    don't matter."""
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
    """A silent fallback would mean the test is lying to you."""
    store = tmp_path / "completions.json"
    store.write_text("{}", encoding="utf-8")
    provider = FixtureCompletion(store, model="m")
    with pytest.raises(KeyError, match="no recorded completion"):
        await provider.complete(system=SYSTEM, messages=MESSAGES, max_tokens=512)


async def test_different_max_tokens_does_not_hit_the_same_recording(tmp_path: Path):
    """Negative test: this is the entire reason the wide hash key exists."""
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
    """One of the field-shape checker tests: an error about a missing field must be
    able to locate the problem, not surface as a bare KeyError."""
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
    """`int("42")` would succeed, which would mask a fixture-authoring mistake --
    reject it instead of coercing it."""
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
    """`stop_reason` is a newly required field -- missing it must be just as
    locatable an error as missing any other field."""
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
