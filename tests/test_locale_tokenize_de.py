"""德语分词：六步流水线。

流水线顺序是设计的一部分：表面词元 -> casefold -> 连字符成分 -> 复合词成分
-> 对全部产出加 ae/oe/ue 别名 -> 词典验证的词干别名。

第 5 步是**闭包**：别名施加于全部产出而非仅表面词。少了这一步，查询 ``Kuendigung``
命中不了只写了 ``Kündigungsfrist`` 的文档 —— 那正是折叠要解决的问题本身。
"""

from __future__ import annotations

import pytest

from guardrails.locale import get_rules
from guardrails.types import Locale


@pytest.fixture(scope="module")
def rules():
    return get_rules(Locale.DE_DE)


def tok(rules, text: str) -> set[str]:
    return set(rules.tokenize(text))


# --- casefold 与 ß --------------------------------------------------------

def test_eszett_casefolds_to_ss(rules):
    assert "strasse" in tok(rules, "Straße")


def test_eszett_spelling_variants_meet(rules):
    """Straße 与 Strasse 必须落到同一个词元，否则两种写法互相检索不到。"""
    assert tok(rules, "Straße") & tok(rules, "Strasse")


# --- 变音符别名 -----------------------------------------------------------

def test_umlaut_alias_on_surface_word(rules):
    assert {"kündigung", "kuendigung"} <= tok(rules, "Kündigung")


def test_umlaut_alias_is_additive_not_replacing(rules):
    assert "kündigung" in tok(rules, "Kündigung")


def test_no_bare_vowel_folding(rules):
    """只做 ü->ue，不做 ü->u。更宽的折叠会把本不该合并的词并到一起。"""
    assert "kundigung" not in tok(rules, "Kündigung")


# --- 复合词分解 -----------------------------------------------------------

def test_compound_yields_parts_and_whole(rules):
    tokens = tok(rules, "Kündigungsfrist")
    assert {"kündigungsfrist", "kündigung", "frist"} <= tokens


def test_compound_without_linking_morpheme(rules):
    assert {"mobilfunkvertrag", "mobilfunk", "vertrag"} <= tok(rules, "Mobilfunkvertrag")


def test_unknown_compound_degrades_to_whole_word(rules):
    """未登录复合词退化为整词匹配 —— 失效方向保守，不产生错误召回。"""
    tokens = tok(rules, "Quastenflossergehege")
    assert tokens == {"quastenflossergehege"}


# --- 别名闭包（第 5 步的直接验收）-----------------------------------------

def test_alias_closure_covers_compound_parts(rules):
    """成分也必须生成别名，否则折叠等于白做。"""
    assert "kuendigung" in tok(rules, "Kündigungsfrist")


def test_ascii_query_retrieves_umlaut_compound(rules):
    """成对断言：查询侧产出别名 + 文档侧成分也产出同一别名。

    单独断言任何一边都证明不了两种写法可以互相检索。
    """
    assert "kuendigung" in tok(rules, "Kuendigung")
    assert "kuendigung" in tok(rules, "Kündigungsfrist")


# --- 连字符 ---------------------------------------------------------------

def test_hyphenated_parts(rules):
    assert {"eu-roaming", "eu", "roaming"} <= tok(rules, "EU-Roaming")


# --- 词干：按词典验证，不按长度 -------------------------------------------

def test_stem_produced_when_in_lexicon(rules):
    assert "frist" in tok(rules, "Fristen")


def test_stem_produced_for_genitive_in_lexicon(rules):
    assert "preis" in tok(rules, "Preises")


def test_stem_suppressed_when_not_in_lexicon(rules):
    """长度规则会造出 'nutz' 这种假词干；词典验证不会。"""
    tokens = tok(rules, "Nutzer")
    assert "nutzer" in tokens
    assert "nutz" not in tokens


def test_inflection_pair_meets(rules):
    assert tok(rules, "Fristen") & tok(rules, "Frist")


# --- 数字 -----------------------------------------------------------------

def test_price_stays_one_token(rules):
    assert "29,99" in tok(rules, "Der Tarif kostet 29,99 EUR pro Monat.")


# --- 词频纪律 -------------------------------------------------------------

def test_variants_of_one_word_do_not_inflate_tf(rules):
    """一个 Kündigung 展开出多个变体，但每个变体只计一次。"""
    tokens = rules.tokenize("Kündigung")
    assert len(tokens) == len(set(tokens))


def test_genuine_repetition_is_preserved(rules):
    """文档里真出现两次就该是两次 —— 否则 BM25 的 TF 不再可解释。"""
    tokens = rules.tokenize("Kündigung Kündigung")
    assert tokens.count("kündigung") == 2


# --- 回溯：连接语素的剥离必须可回退 -----------------------------------------

def test_linking_morpheme_strip_is_backtrackable(rules):
    """`ruf` 之后的 `n` 是 `nummer` 的首字母，不是连接语素。

    贪心地把它当语素吃掉会让整词无法分解。剥离必须是可回退的尝试。
    """
    assert {"ruf", "nummer", "mitnahme"} <= tok(rules, "Rufnummernmitnahme")


def test_whole_word_survives_backtracking(rules):
    assert "rufnummernmitnahme" in tok(rules, "Rufnummernmitnahme")


# --- 末尾成分允许带屈折 -----------------------------------------------------

def test_final_component_may_carry_plural(rules):
    assert {"service", "zeit"} <= tok(rules, "Servicezeiten")


def test_final_component_plural_after_linking_morpheme(rules):
    assert {"zahlung", "art"} <= tok(rules, "Zahlungsarten")


def test_inflected_compound_keeps_the_whole_word(rules):
    assert "servicezeiten" in tok(rules, "Servicezeiten")


# --- 词典缺口 ---------------------------------------------------------------

def test_entstoerfrist_decomposes(rules):
    assert {"entstör", "frist"} <= tok(rules, "Entstörfrist")


# --- 回溯不得放松「全有或全无」---------------------------------------------

def test_backtracking_does_not_admit_partial_splits(rules):
    """一个词典词 + 无法识别的残渣，仍须整体退化为整词。

    回溯是为了找到**完整**的分解，不是为了接受不完整的分解。
    """
    tokens = tok(rules, "Vertragxyzq")
    assert tokens == {"vertragxyzq"}


def test_unknown_compound_still_degrades(rules):
    assert tok(rules, "Quastenflossergehege") == {"quastenflossergehege"}
