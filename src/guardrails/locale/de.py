"""German language rules.

The address-form check is the reason locale is a first-class dimension of a
profile. German marks the T-V distinction grammatically, so "this brand uses the
formal register" is a closed set of pronouns and costs a regex.

It also has a useful asymmetry. Every ambiguity in German register detection
sits on the *formal* side: lowercase ``sie`` is "she"/"they", a sentence-initial
``Sie`` is indistinguishable from it, and ``Ihr`` is both the polite possessive
and "her"/"their". None of that matters for a client whose brand voice is
formal, because detecting a *violation* only requires finding informal
pronouns — and those are unambiguous. The high-precision path is the one the
common case takes.
"""

from __future__ import annotations

import re
from datetime import date
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from typing import ClassVar

from guardrails.locale.base import (
    AddressFormHit,
    EntityMention,
    Sentence,
    count_words,
    format_decimal,
    split_hyphenated,
    surface_tokens,
)
from guardrails.locale.de_lexicon import INFLECTION_SUFFIXES, LEXICON, LINKING_MORPHEMES
from guardrails.types import AddressForm, EntityKind, Locale

__all__ = ["GermanRules"]

# --- sentence segmentation -------------------------------------------------

_ABBREVIATIONS: frozenset[str] = frozenset(
    {
        "z.b.", "d.h.", "u.a.", "s.o.", "s.u.", "i.d.r.", "u.u.", "o.ä.", "ggf.",
        "ggfs.", "bzw.", "bzgl.", "ca.", "inkl.", "exkl.", "zzgl.", "evtl.",
        "vgl.", "max.", "min.", "nr.", "str.", "tel.", "usw.", "etc.", "abs.",
        "art.", "mind.", "ggü.", "jhrl.", "mtl.",
    }
)

_MONTHS: dict[str, int] = {
    "januar": 1, "februar": 2, "märz": 3, "april": 4, "mai": 5, "juni": 6,
    "juli": 7, "august": 8, "september": 9, "oktober": 10, "november": 11,
    "dezember": 12,
}
_MONTH_ALT = "|".join(m.capitalize() for m in _MONTHS)

# --- register markers ------------------------------------------------------

_INFORMAL_WORDS = (
    "deinem", "deinen", "deiner", "deines", "deine", "dein",
    "eurem", "euren", "eurer", "eures", "eure", "euer",
    "dich", "dir", "du", "euch",
)
_INFORMAL_RE = re.compile(
    r"\b(?:" + "|".join(sorted(_INFORMAL_WORDS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

_FORMAL_UNAMBIGUOUS_RE = re.compile(r"\bIhnen\b")
"""Capitalised ``Ihnen`` has no homograph: lowercase ``ihnen`` is "to them"."""

_FORMAL_SIE_RE = re.compile(r"\bSie\b")
"""Ambiguous at the start of a sentence, where ``Sie`` may be "she"/"they".
Sentence-initial matches are discarded rather than guessed at."""

# --- quantities ------------------------------------------------------------

_AMOUNT = r"\d{1,3}(?:\.\d{3})+(?:,\d{1,2})?|\d+(?:,\d{1,2})?"
_CURRENCY = r"€|EUR\b|Euro\b"

_PRICE_RE = re.compile(
    rf"(?:(?P<pre>{_CURRENCY})\s*(?P<a1>{_AMOUNT}))"
    rf"|(?:(?P<a2>{_AMOUNT})\s*(?P<post>{_CURRENCY}))"
)
_DATE_NUMERIC_RE = re.compile(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b")
_DATE_LONG_RE = re.compile(rf"\b(\d{{1,2}})\.\s*({_MONTH_ALT})\s+(\d{{4}})\b")
_DATE_MONTH_YEAR_RE = re.compile(rf"\b({_MONTH_ALT})\s+(\d{{4}})\b")
_DURATION_RE = re.compile(
    r"\b(\d+)\s*-?\s*(Monat(?:en|es|s|e)?|Jahr(?:en|es|e)?|Tag(?:en|es|e)?|Woche(?:n)?)\b",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(rf"(?<![\w.,]){_AMOUNT}(?![\w])")

_DURATION_UNITS = {"monat": "M", "jahr": "Y", "tag": "D", "woche": "W"}


def _parse_amount(raw: str) -> Decimal | None:
    """``1.234,56`` -> ``Decimal("1234.56")``. German uses ``.`` for thousands."""
    try:
        return Decimal(raw.replace(".", "").replace(",", "."))
    except InvalidOperation:
        return None


# --- 检索分词 --------------------------------------------------------------

_UMLAUT_ALIASES: dict[str, str] = {"ä": "ae", "ö": "oe", "ü": "ue"}


def _ascii_alias(token: str) -> str | None:
    """``kündigung`` -> ``kuendigung``；无变音符时返回 ``None``。

    只折叠 ä/ö/ü。不做 ü->u：收益是假想的，代价是把本该区分的词并到一起。
    ß 不在这里处理 —— ``casefold()`` 已经把它变成 ``ss``。
    """
    if not any(ch in token for ch in _UMLAUT_ALIASES):
        return None
    out = token
    for umlaut, replacement in _UMLAUT_ALIASES.items():
        out = out.replace(umlaut, replacement)
    return out


def _split_compound(token: str) -> tuple[str, ...]:
    """把复合词拆成词典中的成分；拆不动就返回空元组。

    递归下降 + 回溯。贪心版本会在两处失败，而两处都不是罕见构词：

    - 连接语素的剥离必须**可回退**。``rufnummernmitnahme`` 在 ``ruf`` 之后剩下
      ``nummernmitnahme``，贪心地把 ``n`` 当连接语素吃掉，而它是 ``nummer`` 的首
      字母。所以每个位置都要同时尝试「剥」与「不剥」两条路。
    - **最后一个成分可以带屈折后缀**。``servicezeiten`` = service + zeit + en。

    仍然是全有或全无：只有当整个词都被消耗完时才返回成分，否则返回空元组。部分
    匹配（一个词典词加一段残渣）会产生错误召回，比不拆更糟。
    """
    parts = _decompose(token)
    return parts if parts is not None and len(parts) > 1 else ()


# 缓存 _decompose 以阻止回溯随词典增长而指数爆炸。有界是因为 tokenize 也在长期
# 运行的聊天进程中的用户查询上执行，而非仅在索引时的固定语料。无界缓存会随用户
# 输入无限增长。每个分解的工作集远小于 2048，所以驱逐不会影响指数阻止的效果。
@lru_cache(maxsize=2048)
def _decompose(rest: str) -> tuple[str, ...] | None:
    """把 ``rest`` 完整拆成词典成分；拆不完返回 ``None``。

    ``None`` 与 ``()`` 的区别是有意义的：``()`` 表示「空串已成功消耗完」，是递归的
    成功基例；``None`` 表示「这条路走不通」，触发回溯。
    """
    if not rest:
        return ()
    for end in range(len(rest), 2, -1):
        head = rest[:end]
        if head not in LEXICON:
            continue
        tail = rest[end:]
        sub = _decompose(tail)
        if sub is not None:
            return (head, *sub)
        for morpheme in LINKING_MORPHEMES:
            if not tail.startswith(morpheme):
                continue
            sub = _decompose(tail[len(morpheme):])
            if sub is not None:
                return (head, *sub)
    for suffix in INFLECTION_SUFFIXES:
        if rest.endswith(suffix) and rest[: -len(suffix)] in LEXICON:
            return (rest[: -len(suffix)],)
    return None


def _stem_alias(token: str) -> str | None:
    """剥词形后缀，**仅当结果命中词典**时才产出。

    长度规则（"剩余 >= 4 字符就剥"）会造出 ``Nutzer`` -> ``nutz`` 这种假词干。
    词典验证把「能不能剥」变成一个有据可查的问题。
    """
    if token in LEXICON:
        return None
    for suffix in INFLECTION_SUFFIXES:
        if not token.endswith(suffix):
            continue
        stem = token[: -len(suffix)]
        if stem in LEXICON:
            return stem
    return None


class GermanRules:
    """Language rules for German."""

    locale: ClassVar[Locale] = Locale.DE_DE
    supports_grammatical_address_form: ClassVar[bool] = True

    # -- sentences ----------------------------------------------------------

    def segment_sentences(self, text: str) -> tuple[Sentence, ...]:
        boundaries = [i for i, ch in enumerate(text) if ch in ".!?" and self._is_boundary(text, i)]

        sentences: list[Sentence] = []
        start = 0
        for end in [*boundaries, len(text) - 1]:
            if end < start:
                continue
            sentences.append(self._make_sentence(text, start, end + 1))
            start = end + 1
        return tuple(s for s in sentences if s.text)

    @staticmethod
    def _make_sentence(text: str, start: int, end: int) -> Sentence:
        chunk = text[start:end]
        lead = len(chunk) - len(chunk.lstrip())
        trail = len(chunk) - len(chunk.rstrip())
        lo, hi = start + lead, end - trail
        body = text[lo:hi]
        return Sentence(text=body, span=(lo, hi), word_count=count_words(body))

    def _is_boundary(self, text: str, i: int) -> bool:
        after = text[i + 1 :]
        if after and not after[0].isspace():
            # Mid-token full stop: `z.B`, `1.000`, `i.d.R`.
            return False
        following = after.lstrip()
        if following and not (following[0].isupper() or following[0] in "\"'«„("):
            return False
        if text[i] != ".":
            return True
        if self._is_abbreviation(text, i):
            return False
        return not self._is_date_ordinal(text, i, following)

    @staticmethod
    def _is_abbreviation(text: str, i: int) -> bool:
        j = i
        while j > 0 and (text[j - 1].isalpha() or text[j - 1] == "."):
            j -= 1
        return text[j : i + 1].lower() in _ABBREVIATIONS

    @staticmethod
    def _is_date_ordinal(text: str, i: int, following: str) -> bool:
        """``am 1. Januar`` is one sentence; ``Das kostet 1. Danach ...`` is two.

        Deliberately narrow: only a digit followed by a month name suppresses the
        break. The general ordinal case (``der 1. Platz``) still splits — a rule
        wide enough to catch it would swallow real sentence boundaries, which is
        the more damaging error.
        """
        if i == 0 or not text[i - 1].isdigit():
            return False
        word = re.match(r"[^\W\d_]+", following)
        return bool(word) and word.group().lower() in _MONTHS

    # -- register -----------------------------------------------------------

    def check_address_form(self, text: str, expected: AddressForm) -> tuple[AddressFormHit, ...]:
        if expected is AddressForm.FORMAL:
            return tuple(
                AddressFormHit(
                    form=AddressForm.INFORMAL,
                    marker="informal_pronoun",
                    token=m.group(),
                    span=m.span(),
                )
                for m in _INFORMAL_RE.finditer(text)
            )

        starts = {s.span[0] for s in self.segment_sentences(text)}
        hits = [
            AddressFormHit(AddressForm.FORMAL, "formal_pronoun", m.group(), m.span())
            for m in _FORMAL_UNAMBIGUOUS_RE.finditer(text)
        ]
        hits += [
            AddressFormHit(AddressForm.FORMAL, "formal_pronoun", m.group(), m.span())
            for m in _FORMAL_SIE_RE.finditer(text)
            if m.start() not in starts
        ]
        return tuple(sorted(hits, key=lambda h: h.span))

    # -- quantities ---------------------------------------------------------

    def extract_entities(
        self, text: str, kinds: frozenset[EntityKind]
    ) -> tuple[EntityMention, ...]:
        """Specific kinds claim their spans first; ``NUMBER`` takes what is left.

        All kinds are extracted regardless of ``kinds`` and the result is
        filtered at the end, so that the set of ``NUMBER`` mentions does not
        change depending on whether ``PRICE`` happens to be enabled.
        """
        found: list[EntityMention] = []
        taken: list[tuple[int, int]] = []

        def claim(mention: EntityMention) -> None:
            found.append(mention)
            taken.append(mention.span)

        for m in _PRICE_RE.finditer(text):
            amount = _parse_amount(m.group("a1") or m.group("a2"))
            if amount is not None:
                claim(
                    EntityMention(
                        EntityKind.PRICE,
                        m.group(),
                        f"{amount.quantize(Decimal('0.01'))} EUR",
                        m.span(),
                    )
                )

        for m in _DATE_NUMERIC_RE.finditer(text):
            day, month, year = (int(g) for g in m.groups())
            if (iso := _iso_date(year, month, day)) and not _overlaps(m.span(), taken):
                claim(EntityMention(EntityKind.DATE, m.group(), iso, m.span()))

        for m in _DATE_LONG_RE.finditer(text):
            day, month, year = int(m.group(1)), _MONTHS[m.group(2).lower()], int(m.group(3))
            if (iso := _iso_date(year, month, day)) and not _overlaps(m.span(), taken):
                claim(EntityMention(EntityKind.DATE, m.group(), iso, m.span()))

        for m in _DATE_MONTH_YEAR_RE.finditer(text):
            if not _overlaps(m.span(), taken):
                month, year = _MONTHS[m.group(1).lower()], int(m.group(2))
                claim(EntityMention(EntityKind.DATE, m.group(), f"{year:04d}-{month:02d}", m.span()))

        for m in _DURATION_RE.finditer(text):
            value = int(m.group(1))
            unit = _duration_unit(m.group(2))
            if value > 0 and unit and not _overlaps(m.span(), taken):
                claim(EntityMention(EntityKind.DURATION, m.group(), f"P{value}{unit}", m.span()))

        for m in _NUMBER_RE.finditer(text):
            amount = _parse_amount(m.group())
            if amount is not None and not _overlaps(m.span(), taken):
                claim(EntityMention(EntityKind.NUMBER, m.group(), format_decimal(amount), m.span()))

        return tuple(sorted((f for f in found if f.kind in kinds), key=lambda e: e.span))

    def tokenize(self, text: str) -> tuple[str, ...]:
        """六步流水线，全程叠加不替换。见 de_lexicon 的模块文档。"""
        out: list[str] = []
        for surface in surface_tokens(text):
            out.extend(_expand(surface.casefold()))
        return tuple(out)


def _expand(folded: str) -> tuple[str, ...]:
    """一个已 casefold 的表面词元展开成它的全部检索词元。

    顺序不能重排：成分要先拆完，别名才能施加于**全部**产出（闭包）。只对表面词
    加别名的话，查询 ``Kuendigung`` 命中不了只写了 ``Kündigungsfrist`` 的文档。

    去重发生在**单个表面词元内部**，所以文档中真实重复的词仍然重复计入 TF。
    """
    produced: list[str] = []

    # 3. 连字符成分
    base = list(split_hyphenated(folded))
    # 4. 复合词成分
    for token in tuple(base):
        base.extend(_split_compound(token))
    produced.extend(base)
    # 5. 别名闭包 —— 施加于以上全部产出
    for token in tuple(produced):
        alias = _ascii_alias(token)
        if alias is not None:
            produced.append(alias)
    # 6. 词典验证的词干别名
    for token in tuple(produced):
        stem = _stem_alias(token)
        if stem is not None:
            produced.append(stem)

    return tuple(dict.fromkeys(produced))


def _duration_unit(word: str) -> str | None:
    lowered = word.lower()
    for stem, code in _DURATION_UNITS.items():
        if lowered.startswith(stem):
            return code
    return None


def _iso_date(year: int, month: int, day: int) -> str | None:
    """A regex match is not a date. ``31.02.2026`` matches and does not exist."""
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def _overlaps(span: tuple[int, int], taken: list[tuple[int, int]]) -> bool:
    return any(span[0] < hi and lo < span[1] for lo, hi in taken)
