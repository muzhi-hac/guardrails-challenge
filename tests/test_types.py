"""Tests for the shared vocabulary.

These pin the three properties the rest of the system relies on: verdicts are
immutable records, severity aggregation ignores non-conclusive verdicts, and
everything round-trips through JSON so traces can be replayed.
"""

import json

import pytest
from pydantic import ValidationError

from guardrails.types import (
    Action,
    Evidence,
    Mode,
    Outcome,
    PipelineResult,
    Severity,
    Stage,
    Verdict,
    aggregate_severity,
)


def make_verdict(**overrides) -> Verdict:
    defaults = dict(guard="persona", stage=Stage.OUTPUT, outcome=Outcome.PASS)
    return Verdict(**{**defaults, **overrides})


class TestImmutability:
    def test_field_assignment_is_rejected(self):
        verdict = make_verdict()
        with pytest.raises(ValidationError):
            verdict.severity = Severity.HIGH

    def test_evidence_collection_cannot_be_mutated_in_place(self):
        verdict = make_verdict(evidence=(Evidence(kind="du_form", detail="informal"),))
        assert isinstance(verdict.evidence, tuple)
        with pytest.raises(AttributeError):
            verdict.evidence.append(Evidence(kind="x", detail="y"))

    def test_verdicts_are_hashable(self):
        # Would raise if any field were a list.
        assert len({make_verdict(), make_verdict()}) == 1

    def test_unknown_fields_are_rejected(self):
        with pytest.raises(ValidationError):
            make_verdict(typo_field=1)


class TestSeverityAggregation:
    def test_empty_sequence_is_none(self):
        assert aggregate_severity([]) is Severity.NONE

    def test_takes_the_maximum(self):
        verdicts = [
            make_verdict(outcome=Outcome.FAIL, severity=Severity.LOW),
            make_verdict(outcome=Outcome.FAIL, severity=Severity.HIGH),
            make_verdict(outcome=Outcome.FAIL, severity=Severity.MEDIUM),
        ]
        assert aggregate_severity(verdicts) is Severity.HIGH

    @pytest.mark.parametrize("outcome", [Outcome.ERROR, Outcome.TIMEOUT])
    def test_a_guard_that_never_answered_still_counts(self, outcome):
        """Guards only return PASS or FAIL, so an ERROR or TIMEOUT here was
        built by the orchestrator and already carries the client's fail-open or
        fail-closed severity. Excluding it would make fail-closed a no-op."""
        verdicts = [
            make_verdict(outcome=outcome, severity=Severity.CRITICAL),
            make_verdict(outcome=Outcome.FAIL, severity=Severity.LOW),
        ]
        assert aggregate_severity(verdicts) is Severity.CRITICAL

    def test_skipped_verdicts_carry_no_severity_of_their_own(self):
        verdicts = [
            make_verdict(outcome=Outcome.SKIPPED),
            make_verdict(outcome=Outcome.FAIL, severity=Severity.LOW),
        ]
        assert aggregate_severity(verdicts) is Severity.LOW

    def test_only_pass_and_fail_are_conclusive(self):
        assert make_verdict(outcome=Outcome.PASS).conclusive
        assert make_verdict(outcome=Outcome.FAIL).conclusive
        assert not make_verdict(outcome=Outcome.TIMEOUT).conclusive


class TestConfidenceBounds:
    @pytest.mark.parametrize("value", [-0.1, 1.1])
    def test_out_of_range_is_rejected(self, value):
        with pytest.raises(ValidationError):
            make_verdict(confidence=value)

    @pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
    def test_in_range_is_accepted(self, value):
        assert make_verdict(confidence=value).confidence == value


class TestSerialisation:
    def test_severity_serialises_as_its_name(self):
        payload = json.loads(
            make_verdict(outcome=Outcome.FAIL, severity=Severity.HIGH).model_dump_json()
        )
        assert payload["severity"] == "high"

    def test_severity_parses_from_both_name_and_int(self):
        assert make_verdict(severity="high").severity is Severity.HIGH
        assert make_verdict(severity=3).severity is Severity.HIGH

    def test_unknown_severity_name_is_rejected(self):
        with pytest.raises(ValidationError):
            make_verdict(severity="catastrophic")

    def test_verdict_round_trips(self):
        original = make_verdict(
            outcome=Outcome.FAIL,
            severity=Severity.MEDIUM,
            confidence=0.62,
            tier=1,
            cost_usd=0.00031,
            latency_ms=412.5,
            evidence=(
                Evidence(
                    kind="ungrounded_number",
                    detail="'19,99 EUR' not present in retrieved context",
                    span=(41, 50),
                    source_ref=None,
                ),
            ),
        )
        assert Verdict.model_validate_json(original.model_dump_json()) == original

    def test_pipeline_result_round_trips(self):
        original = PipelineResult(
            trace_id="t-001",
            turn_index=2,
            mode=Mode.VOICE,
            action=Action.HANDOVER,
            reason="ungrounded pricing claim; profile escalates HIGH to a human",
            verdicts=(make_verdict(outcome=Outcome.FAIL, severity=Severity.HIGH),),
            total_latency_ms=118.0,
            total_cost_usd=0.0,
        )
        restored = PipelineResult.model_validate_json(original.model_dump_json())
        assert restored == original
        assert isinstance(restored.verdicts, tuple)
