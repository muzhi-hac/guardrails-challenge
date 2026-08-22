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

# 事实表按 doc_id 钉死 —— 溯源守卫要拿回复里的价格去跟这些字面值核对，所以语料
# 里的每一个数字都是规格（spec），不是「凑够 >=3 个实体」就算过关的细节。
#
# 只比较「德语价格集合 == 英语价格集合」（见 test_mirrored_prices_match_across_
# locales）测不出「两边被同时改坏」：把 19,99 EUR 和 19.99 EUR 一起改成 18,99 /
# 18.99，德英仍然相等，测试仍然绿。这里把从文档里读出来的期望值写成字面量，
# 任何一侧偏离这些字面量都会失败，不管另一侧有没有同步偏离。
#
# 每个值都是从对应文档正文里读出来的，不是编出来的：
#   tarife-mobilfunk           费率表三档月费：19,99 / 29,99 / 49,99 EUR
#   rechnung-zahlungsarten     「Zahlungsverzug und Mahngebühr」：首次催缴费 5,00 EUR
#   roaming-eu                 「Roaming außerhalb der EU」：欧盟外每 MB 数据 0,49 EUR
#   umzug                      「Umzugsservice」：搬迁一次性费用 29,90 EUR
#   datenvolumen-drosselung    「Datenautomatik」：追加 1 GB 数据自动计费 3,00 EUR
#                              （仅德语，没有英文镜像——不在 MIRRORED_DOC_IDS 里）
PRICE_EXPECTATIONS: dict[str, frozenset[str]] = {
    "tarife-mobilfunk": frozenset({"19.99 EUR", "29.99 EUR", "49.99 EUR"}),
    "rechnung-zahlungsarten": frozenset({"5.00 EUR"}),
    "roaming-eu": frozenset({"0.49 EUR"}),
    "umzug": frozenset({"29.90 EUR"}),
    "datenvolumen-drosselung": frozenset({"3.00 EUR"}),
}


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


def test_price_fact_table_matches_documents(documents):
    """把事实表的具体数值钉死在测试里，而不只是比较德英两侧是否相等。

    ``test_mirrored_prices_match_across_locales`` 只能发现「德英不一致」，发现不了
    「德英被同步改坏」——把 19,99 EUR 连同它的英文镜像一起改成 18,99 EUR，德语
    集合仍然等于英语集合，那个测试仍然是绿的。语料里的事实表是溯源守卫比对的
    规格本身，所以这里把从文档里读出来的期望值写成字面量常量
    （``PRICE_EXPECTATIONS``），任何一侧偏离字面量都必须失败，不管另一侧有没
    有同步偏离。
    """

    def prices(doc):
        rules = get_rules(doc.locale)
        return {
            m.normalized
            for m in rules.extract_entities(doc.body, (EntityKind.PRICE,))
        }

    by_doc_id: dict[str, list] = {}
    for doc in documents:
        by_doc_id.setdefault(doc.doc_id, []).append(doc)

    for doc_id, expected in PRICE_EXPECTATIONS.items():
        # 期望集合本身不能是空的——空集合会让下面的相等断言在「文档根本没有
        # 价格实体」时也通过，重现 finding 1 描述的「vacuous comparison」问题。
        assert expected, f"{doc_id}: expected price set must not be empty"
        matching = by_doc_id.get(doc_id, [])
        assert matching, f"{doc_id}: no document with this doc_id was loaded"
        for doc in matching:
            assert prices(doc) == expected, (doc.source_path, doc_id)


def test_no_document_contains_credentials_or_endpoints(documents):
    """语料是要交付的内容，凭据与端点名不能漏进来。"""
    for doc in documents:
        assert "sk-" not in doc.body, doc.source_path
        assert "shubiaobiao" not in doc.body, doc.source_path


def test_unterminated_front_matter_names_the_file(tmp_path):
    """front matter 开了没关：报错必须点名是哪个文件，而不是裸的 unpack 报错。"""
    bad = tmp_path / "broken.md"
    bad.write_text(
        "---\ndoc_id: x\ntitle: X\nlocale: de-DE\nversion: 2026-01-01\n"
        "no closing delimiter here\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=re.escape(str(bad))):
        load_documents(tmp_path)


def test_invalid_locale_names_the_file(tmp_path):
    """locale 拼错（比如 de_DE 而不是 de-DE）：报错必须点名是哪个文件。"""
    bad = tmp_path / "typo-locale.md"
    bad.write_text(
        "---\ndoc_id: x\ntitle: X\nlocale: de_DE\nversion: 2026-01-01\n---\nBody.\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=re.escape(str(bad))):
        load_documents(tmp_path)


def test_malformed_yaml_names_the_file(tmp_path):
    """YAML 本身格式错（比如冒号后缺值导致的缩进/语法错误）：报错必须点名文件。"""
    bad = tmp_path / "bad-yaml.md"
    bad.write_text(
        "---\ndoc_id: x\ntitle: [unclosed\nlocale: de-DE\nversion: 2026-01-01\n---\nBody.\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=re.escape(str(bad))):
        load_documents(tmp_path)
