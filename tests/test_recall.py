"""检索 recall@k。

这是 §5 数字表里「检索 recall（单独测）」那一行。单独测量而不是折进一个总分，是因为
检索质量**上界了**溯源守卫：检索 miss 会让守卫把一个正确回答判成无依据，那是可归因
于检索、不是可归因于守卫的假阳性。
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
    """非局限型查询必须在 k=5 内命中。已知局限不设门槛，只报告数字。"""
    if case.known_limitation:
        pytest.skip("已知局限型查询，数字由 test_recall_report 记录")
    assert hits(kb, case, 5), f"未命中: {case.query}"


def test_recall_report(kb, capsys):
    """打印 recall@1/@3/@5，供 §5 数字表抄录。断言只设下限，避免脆弱。"""
    exact = [c for c in RECALL_QUERIES if not c.known_limitation]
    limitation = [c for c in RECALL_QUERIES if c.known_limitation]
    kinds = sorted({c.known_limitation for c in limitation})
    lines = ["", "检索 recall（分母见括号）"]
    rows = [("精确词", exact)]
    rows += [(f"局限-{kind}", [c for c in limitation if c.known_limitation == kind])
             for kind in kinds]
    rows += [("全部", RECALL_QUERIES)]
    for label, cases in rows:
        row = " ".join(f"@{k}={recall_at_k(kb, cases, k):.2f}" for k in (1, 3, 5))
        lines.append(f"  {label} (n={len(tuple(cases))}): {row}")
    with capsys.disabled():
        print("\n".join(lines))

    assert recall_at_k(kb, exact, 5) >= 0.9
    assert recall_at_k(kb, RECALL_QUERIES, 5) >= 0.7
