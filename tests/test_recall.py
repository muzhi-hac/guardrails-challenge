"""Retrieval recall@k.

This is the "retrieval recall (measured separately)" row in the §5 numbers table.
It is measured separately rather than folded into one overall score because retrieval
quality **upper-bounds** the grounding guard: a retrieval miss makes the guard judge a
correct answer as unsupported, and that false positive is attributable to retrieval,
not to the guard.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fixtures.recall_queries import RECALL_QUERIES, RecallCase
from guardrails.retrieval.knowledge_base import KnowledgeBase

KB_ROOT = Path(__file__).resolve().parents[1] / "kb"


@pytest.fixture(scope="module")
def kb():
    return KnowledgeBase.load(KB_ROOT)


def hits(kb: KnowledgeBase, case: RecallCase, k: int) -> bool:
    results = kb.for_locale(case.locale).search(case.query, k=k)
    return bool({r.item.doc_id for r in results} & case.expected_doc_ids)


def recall_at_k(kb: KnowledgeBase, cases, k: int) -> float:
    cases = tuple(cases)
    return sum(hits(kb, c, k) for c in cases) / len(cases)


@pytest.mark.parametrize("case", RECALL_QUERIES, ids=lambda c: c.query[:32])
def test_exact_term_queries_hit_at_5(kb, case):
    """Non-limitation queries must hit within k=5. Known-limitation cases have no
    threshold -- only the numbers are reported."""
    if case.known_limitation:
        pytest.skip("known-limitation query; the number is recorded by test_recall_report")
    assert hits(kb, case, 5), f"no hit: {case.query}"


def test_recall_report(kb, capsys):
    """Prints recall@1/@3/@5 for transcription into the §5 numbers table. The
    assertion only sets a floor, to avoid brittleness."""
    exact = [c for c in RECALL_QUERIES if not c.known_limitation]
    limitation = [c for c in RECALL_QUERIES if c.known_limitation]
    kinds = sorted({c.known_limitation for c in limitation})
    lines = ["", "retrieval recall (denominator in parentheses)"]
    rows = [("exact-term", exact)]
    rows += [(f"limitation-{kind}", [c for c in limitation if c.known_limitation == kind])
             for kind in kinds]
    rows += [("overall", RECALL_QUERIES)]
    for label, cases in rows:
        row = " ".join(f"@{k}={recall_at_k(kb, cases, k):.2f}" for k in (1, 3, 5))
        lines.append(f"  {label} (n={len(tuple(cases))}): {row}")
    lines.append("  per case (limitation):")
    for case in limitation:
        row = " ".join(f"@{k}={recall_at_k(kb, [case], k):.2f}" for k in (1, 3, 5))
        lines.append(f"    [{case.known_limitation}] {case.query}: {row}")
    with capsys.disabled():
        print("\n".join(lines))

    assert recall_at_k(kb, exact, 5) >= 0.9
    assert recall_at_k(kb, RECALL_QUERIES, 5) >= 0.7
