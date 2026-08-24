"""Coverage for ``chatbot.main`` -- the CLI entry point.

Guards branch-review Finding 4: before this file, nothing in ``tests/``
imported or invoked ``main()``. The two bugs found by hand while building the
CLI -- ``EOFError`` on piped stdin, and a missing ``PYTHONPATH`` breaking
``python -m chatbot`` -- are exactly the kind of thing a test would have
caught, and neither had a regression guard.

Every test here monkeypatches the Anthropic client so no network call ever
happens, and redirects ``TraceWriter`` under ``tmp_path`` so nothing is
written into the repository's own ``runs/`` directory.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import guardrails.provider.anthropic_client as anthropic_client_module
import utils as utils_module
from chatbot import main
from guardrails.provider.base import CompletionResult, Turn
from guardrails.types import Action, Mode, Severity
from utils import TraceWriter, load_profile

PROFILES = Path(__file__).resolve().parents[1] / "profiles"


class _StubCompletion:
    """A drop-in ``Completion`` that records every call and never touches
    the network. One instance is created per ``main()`` call (``main``
    constructs exactly one ``AnthropicCompletion``, reused for every REPL
    turn), so ``instances[-1]`` after a call is the one that served it."""

    instances: list["_StubCompletion"] = []

    def __init__(self, model: str, client: object | None = None) -> None:
        self.model = model
        self.calls: list[tuple[str, tuple[Turn, ...]]] = []
        _StubCompletion.instances.append(self)

    async def complete(self, *, system: str, messages, max_tokens: int) -> CompletionResult:
        self.calls.append((system, tuple(messages)))
        return CompletionResult(
            text=f"Antwort {len(self.calls)}",
            model=self.model,
            input_tokens=1,
            output_tokens=1,
            latency_ms=1.0,
            stop_reason="end_turn",
        )


@pytest.fixture(autouse=True)
def _no_network_and_no_repo_writes(monkeypatch, tmp_path):
    """Stub the Anthropic client and redirect trace output for every test in
    this module."""
    _StubCompletion.instances.clear()
    monkeypatch.setattr(anthropic_client_module, "AnthropicCompletion", _StubCompletion)

    def _tmp_rooted_trace_writer(root: Path, run_id: str) -> TraceWriter:
        del root  # main() always passes <repo>/runs; redirect it under tmp_path.
        return TraceWriter(tmp_path / "runs", run_id)

    monkeypatch.setattr(utils_module, "TraceWriter", _tmp_rooted_trace_writer)


def _last_stub() -> _StubCompletion:
    assert _StubCompletion.instances, "AnthropicCompletion was never constructed"
    return _StubCompletion.instances[-1]


def _read_trace_records(tmp_path: Path) -> list[dict]:
    files = list((tmp_path / "runs").glob("*.jsonl"))
    assert len(files) == 1, f"expected exactly one trace file, found {files}"
    lines = files[0].read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def test_one_shot_question_returns_zero_and_writes_one_trace_record(tmp_path):
    rc = main(["--profile", "telco_de", "--question", "Was kostet Tarif M?"])
    assert rc == 0

    records = _read_trace_records(tmp_path)
    assert len(records) == 1
    record = records[0]
    for field in (
        "profile", "mode", "query", "retrieved", "model", "input_tokens",
        "output_tokens", "latency_ms", "stop_reason", "prompt_hash",
        "error_type", "schema_version", "run_id", "ts",
    ):
        assert field in record, field
    assert record["profile"] == "telco_de"
    assert record["mode"] == "chat"
    assert record["error_type"] is None


def test_trace_record_carries_stop_reason(tmp_path):
    """Guards Finding 2: ``stop_reason`` is required on ``CompletionResult``
    precisely so a trace reader can tell a truncated empty reply from a
    genuinely empty one -- it must actually reach the trace record."""
    main(["--profile", "telco_de", "--question", "Was kostet Tarif M?"])
    record = _read_trace_records(tmp_path)[0]
    assert record["stop_reason"] == "end_turn"


def test_repl_accumulates_history_across_turns(tmp_path, monkeypatch):
    """Guards Finding 1: the REPL must not pass an empty history on every
    turn -- the second call should see the first exchange.

    Run with ``--no-guards``, and that is not a way of dodging the guard
    layer: the stub answers "Antwort 1" / "Antwort 2", whose bare digit is an
    ungrounded number, so with guards on this turn legitimately routes to
    rewrite and the assistant text the REPL threads back is the repair /
    fallback rather than the stub's own string. That is correct behaviour, and
    it is asserted directly in
    ``test_repl_threads_the_guarded_reply_into_history`` below. What *this*
    test isolates is the REPL loop's own plumbing -- that history is
    accumulated at all -- so it holds the guard layer still. Every assertion
    below is unchanged.
    """
    answers = iter(["Erste Frage", "Zweite Frage"])

    def fake_input(prompt: str = "") -> str:
        try:
            return next(answers)
        except StopIteration:
            raise EOFError from None

    monkeypatch.setattr("builtins.input", fake_input)

    rc = main(["--profile", "telco_de", "--no-guards"])
    assert rc == 0

    stub = _last_stub()
    assert len(stub.calls) == 2
    _, first_messages = stub.calls[0]
    _, second_messages = stub.calls[1]

    # First turn: no history yet, just the current (rendered) user turn.
    assert len(first_messages) == 1

    # Second turn: the first exchange is threaded in ahead of the new user
    # turn, in order, as raw Turn objects (not the rendered document block).
    assert second_messages[0] == Turn("user", "Erste Frage")
    assert second_messages[1] == Turn("assistant", "Antwort 1")
    assert len(second_messages) == 3


def test_piped_eof_stdin_exits_cleanly(tmp_path, monkeypatch):
    """Guards the ``EOFError`` bug found by hand in Task 13: piping empty
    stdin into the REPL must return 0, not raise."""

    def fake_input(prompt: str = "") -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", fake_input)

    rc = main(["--profile", "telco_de"])
    assert rc == 0
    assert _StubCompletion.instances[-1].calls == []


def test_unknown_profile_names_the_profile_path(tmp_path):
    expected_path = PROFILES / "does-not-exist.yaml"
    with pytest.raises(FileNotFoundError, match=re.escape(str(expected_path))):
        main(["--profile", "does-not-exist", "--question", "Hallo"])


def test_mode_accepts_documented_values(tmp_path):
    for mode in ("chat", "voice"):
        rc = main(["--profile", "telco_de", "--mode", mode, "--question", "Hallo"])
        assert rc == 0


def test_mode_rejects_undocumented_values(tmp_path):
    with pytest.raises(SystemExit):
        main(["--profile", "telco_de", "--mode", "sms", "--question", "Hallo"])


# --------------------------------------------------------------------------
# The guard layer, from the CLI
# --------------------------------------------------------------------------
#
# The stub answers "Antwort 1", whose bare digit is an ungrounded number
# (MEDIUM under telco_de -> rewrite). The repair answers "Antwort 2", which is
# ungrounded for exactly the same reason, so the guarded one-shot path here
# reliably exercises the full sequence: three stages, one repair call, a
# re-verification that fails, and the fallback. That is convenient rather than
# contrived -- a stub that always answers with a digit is precisely the kind of
# thing the grounding guard exists to catch.

IBAN = "DE89 3704 0044 0532 0130 00"


def test_no_guards_flag_runs_the_turn_without_the_pipeline(tmp_path):
    rc = main(["--profile", "telco_de", "--no-guards", "--question", "Was kostet Tarif M?"])
    assert rc == 0

    record = _read_trace_records(tmp_path)[0]
    assert record["guards_enabled"] is False
    assert record["guards_ran"] is False
    assert record["stages"] == []
    assert record["action"] == "continue"
    assert len(_last_stub().calls) == 1


def test_the_same_question_is_guarded_without_the_flag(tmp_path):
    """The before/after pair, at the CLI level: same profile, same question,
    same stub -- and with the guard layer on, the ungrounded reply does not
    reach the customer."""
    rc = main(["--profile", "telco_de", "--question", "Was kostet Tarif M?"])
    assert rc == 0

    record = _read_trace_records(tmp_path)[0]
    assert record["guards_enabled"] is True
    assert record["guards_ran"] is True
    assert record["action"] == "safe_fallback"
    assert record["rewrite"] == {"attempted": True, "succeeded": False}


def test_trace_record_summarises_every_stage_that_ran(tmp_path):
    main(["--profile", "telco_de", "--question", "Was kostet Tarif M?"])
    record = _read_trace_records(tmp_path)[0]

    stages = record["stages"]
    assert [stage["stage"] for stage in stages] == ["input", "retrieval", "output", "output"]
    for stage in stages:
        assert stage["action"] in {a.value for a in Action}
        assert isinstance(stage["findings"], list)

    kinds = {finding["kind"] for finding in stages[2]["findings"]}
    severities = {finding["severity"] for finding in stages[2]["findings"]}
    assert "ungrounded_number" in kinds
    assert severities <= {s.name.lower() for s in Severity}


def test_trace_record_carries_no_reply_text(tmp_path):
    """A trace is archived and read by operators; the reply is the one thing a
    guard finding says should not be handed on verbatim."""
    main(["--profile", "telco_de", "--question", "Was kostet Tarif M?"])
    raw = json.dumps(_read_trace_records(tmp_path)[0], ensure_ascii=False)
    assert "Antwort" not in raw


def test_trace_record_writes_the_redacted_question(tmp_path):
    main(["--profile", "telco_de", "--question", f"Meine IBAN {IBAN} stimmt nicht."])
    record = _read_trace_records(tmp_path)[0]

    assert IBAN not in json.dumps(record, ensure_ascii=False)
    assert "[IBAN]" in record["query"]
    assert record["redacted"] == ["iban"]


def test_no_guards_leaves_the_question_unredacted(tmp_path):
    """The honest baseline: without the guard layer the customer's IBAN goes
    to the provider and into the log in the clear. This test exists so that
    the 'after' half above is measured against what actually happens without
    it, not against a half-disabled version of it."""
    main(["--profile", "telco_de", "--no-guards", "--question", f"Meine IBAN {IBAN} stimmt nicht."])
    record = _read_trace_records(tmp_path)[0]

    assert IBAN in record["query"]
    assert record["redacted"] == []
    _, messages = _last_stub().calls[0]
    assert IBAN in messages[-1].content


def test_trace_record_counts_the_repair_call_in_the_token_totals(tmp_path):
    """A repaired turn made two model calls and was billed for both. Reporting
    only the first would make the repair path look free in exactly the
    aggregate someone would use to decide whether it is affordable."""
    main(["--profile", "telco_de", "--question", "Was kostet Tarif M?"])
    record = _read_trace_records(tmp_path)[0]

    assert len(_last_stub().calls) == 2
    assert record["input_tokens"] == 2
    assert record["output_tokens"] == 2


def test_repl_threads_the_guarded_reply_into_history(tmp_path, monkeypatch):
    """The counterpart to ``test_repl_accumulates_history_across_turns``: with
    guards on, what gets threaded into the next turn's history is the reply the
    customer actually received -- the fallback -- not the draft the guard
    rejected. Threading the rejected draft would quietly reintroduce it as
    context the model treats as its own prior answer.
    """
    answers = iter(["Erste Frage", "Zweite Frage"])

    def fake_input(prompt: str = "") -> str:
        try:
            return next(answers)
        except StopIteration:
            raise EOFError from None

    monkeypatch.setattr("builtins.input", fake_input)

    profile = load_profile(PROFILES / "telco_de.yaml").resolve(Mode.CHAT)
    assert main(["--profile", "telco_de"]) == 0

    stub = _last_stub()
    # Two turns, each a generation plus one repair.
    assert len(stub.calls) == 4
    _, second_turn_messages = stub.calls[2]
    assert second_turn_messages[0] == Turn("user", "Erste Frage")
    assert second_turn_messages[1] == Turn("assistant", profile.fallback_message)
