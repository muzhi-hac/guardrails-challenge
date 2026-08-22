"""知识库的结构性约束。

内容对不对靠人读；这里测的是那些一旦破了、下游全部静默失真的性质：唯一键、
德英事实一致、以及实体密度 —— 语料实体稀疏等于溯源守卫没有东西可测。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from guardrails.locale import get_rules
from guardrails.retrieval.documents import load_documents
from guardrails.types import EntityKind, Locale

KB_ROOT = Path(__file__).resolve().parents[1] / "kb"

MIRRORED_DOC_IDS = frozenset({
    "tarife-mobilfunk",
    "vertragslaufzeit-kuendigung",
    "roaming-eu",
    "rechnung-zahlungsarten",
    "stoerung-entstoerfrist",
    "umzug",
})


@pytest.fixture(scope="module")
def documents():
    return load_documents(KB_ROOT)


def test_loads_expected_counts(documents):
    by_locale = {loc: [d for d in documents if d.locale is loc] for loc in Locale}
    assert len(by_locale[Locale.DE_DE]) == 12
    assert len(by_locale[Locale.EN_GB]) == 6


def test_unique_key_is_locale_and_doc_id(documents):
    keys = [(d.locale, d.doc_id) for d in documents]
    assert len(keys) == len(set(keys))


def test_english_mirrors_reuse_german_doc_ids(documents):
    """镜像共用逻辑 doc_id —— 区分靠 locale，不靠改名。"""
    english = {d.doc_id for d in documents if d.locale is Locale.EN_GB}
    german = {d.doc_id for d in documents if d.locale is Locale.DE_DE}
    assert english == MIRRORED_DOC_IDS
    assert english <= german


def test_front_matter_complete(documents):
    for doc in documents:
        assert doc.title.strip(), doc.source_path
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", doc.version), doc.source_path


def test_every_document_has_at_least_three_checkable_entities(documents):
    """用 extract_entities 计数，不靠人工判断。"""
    for doc in documents:
        rules = get_rules(doc.locale)
        mentions = rules.extract_entities(doc.body, tuple(EntityKind))
        assert len(mentions) >= 3, f"{doc.source_path}: {len(mentions)}"


def test_mirrored_prices_match_across_locales(documents):
    """德英同一 doc_id 的价格集合必须逐项相等，否则跨 locale 评测测的是内容差异。"""
    def prices(doc):
        rules = get_rules(doc.locale)
        return {
            m.normalized
            for m in rules.extract_entities(doc.body, (EntityKind.PRICE,))
        }

    by_key = {(d.locale, d.doc_id): d for d in documents}
    for doc_id in MIRRORED_DOC_IDS:
        de = prices(by_key[(Locale.DE_DE, doc_id)])
        en = prices(by_key[(Locale.EN_GB, doc_id)])
        assert de == en, doc_id


def test_no_document_contains_credentials_or_endpoints(documents):
    """语料是要交付的内容，凭据与端点名不能漏进来。"""
    for doc in documents:
        assert "sk-" not in doc.body, doc.source_path
        assert "shubiaobiao" not in doc.body, doc.source_path
