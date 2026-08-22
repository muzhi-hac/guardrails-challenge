"""The language-rule interface and its shared data types.

A :class:`LocaleRules` implementation holds the knowledge that is specific to a
*language*: where sentences end, which pronouns carry which register, and how
quantities are written. It deliberately does not hold country-specific formats
— IBANs, telephone numbering plans, postcodes — because those vary
independently of language and belong with the guard that consumes them.

Three properties every implementation must hold to, because guards downstream
depend on them:

* **Spans index the original text.** Never normalise, lowercase or strip the
  input before computing offsets. Bounded repair rewrites the exact span a
  finding points at; an offset computed against a cleaned copy rewrites the
  wrong characters.
* **Entities do not overlap.** ``19,99 €`` is one price, not a price and a
  number. Specific kinds are extracted first and claim their span; ``NUMBER``
  only takes what is left.
* **Normalisation validates.** A pattern match is not a fact. ``31.02.2026``
  matches every date regex ever written and is not a date.
"""

from __future__ import annotations

import re
from decimal import Decimal
from typing import ClassVar, NamedTuple, Protocol, runtime_checkable

from guardrails.types import AddressForm, EntityKind, Locale

__all__ = [
    "AddressFormHit",
    "EntityMention",
    "LocaleRules",
    "Sentence",
    "count_words",
    "format_decimal",
    "surface_tokens",
    "split_hyphenated",
    "simple_tokenize",
]


class Sentence(NamedTuple):
    """One sentence, located in the original text."""

    text: str
    span: tuple[int, int]
    word_count: int


class AddressFormHit(NamedTuple):
    """A register marker that contradicts the configured address form."""

    form: AddressForm
    """The form the marker indicates — i.e. the one the profile did *not* ask for."""

    marker: str
    """What kind of evidence this is: ``informal_pronoun``, ``formal_pronoun``,
    ``contraction``, ``casual_greeting``. Kept distinct so evidence stays honest
    about *why* a language flagged something — a German pronoun and an English
    contraction are not the same strength of signal."""

    token: str
    span: tuple[int, int]


class EntityMention(NamedTuple):
    """A quantity found in text, in both its written and comparable forms."""

    kind: EntityKind
    raw: str
    """Exactly as written, for the evidence message."""

    normalized: str
    """Canonical form, for comparison against retrieved context. A reply saying
    ``19,99 €`` and a document saying ``EUR 19,99`` state the same fact; without
    normalisation the grounding guard would call a correct answer unsupported."""

    span: tuple[int, int]


@runtime_checkable
class LocaleRules(Protocol):
    """Language-specific text analysis."""

    locale: ClassVar[Locale]

    supports_grammatical_address_form: ClassVar[bool]
    """Whether the language marks the address form grammatically.

    ``True`` for German, where ``du`` versus ``Sie`` is a closed class of
    pronouns and detection is near-exact. ``False`` for English, where the check
    degrades to register heuristics. Exposed as data rather than described in
    prose so that a guard can weigh the finding accordingly instead of treating
    both languages as equally reliable.
    """

    def segment_sentences(self, text: str) -> tuple[Sentence, ...]:
        """Split into sentences, honouring the language's abbreviations and
        number formats."""
        ...

    def check_address_form(
        self, text: str, expected: AddressForm
    ) -> tuple[AddressFormHit, ...]:
        """Return markers of the form that was *not* asked for."""
        ...

    def extract_entities(
        self, text: str, kinds: frozenset[EntityKind]
    ) -> tuple[EntityMention, ...]:
        """Extract and normalise quantities, without overlaps, ordered by span."""
        ...

    def tokenize(self, text: str) -> tuple[str, ...]:
        """检索用词元。

        与 :meth:`extract_entities` 不同，这里的产出是给 BM25 索引用的，允许一个
        表面词展开出多个词元（德语的复合词成分与变音符别名）。展开一律**叠加**，
        原词永远保留 —— 精确词匹配是这个检索器的主要能力，任何把原词换掉的规范化
        都会削弱它。
        """
        ...


_WORD = re.compile(r"[^\W_]+(?:[-'’][^\W_]+)*", re.UNICODE)


def count_words(text: str) -> int:
    """Count words for the sentence-length rule.

    A word is a run of letters or digits, with internal hyphens and apostrophes
    joining rather than splitting: ``5G-Tarif`` is one word, as is a German
    compound. Fixed here rather than left to each implementation so that
    ``max_sentence_words`` means the same thing in every language.
    """
    return len(_WORD.findall(text))


def format_decimal(value: Decimal) -> str:
    """Render a decimal without exponent notation or trailing-zero surprises."""
    if value == value.to_integral_value():
        return str(value.quantize(Decimal(1)))
    return format(value.normalize(), "f")


_TOKEN_RE = re.compile(r"[0-9]+(?:[.,][0-9]+)*|[^\W\d_]+(?:-[^\W\d_]+)*", re.UNICODE)


def surface_tokens(text: str) -> tuple[str, ...]:
    """按语言无关的规则切出表面词元。

    数字保持完整（``29,99`` 与 ``29.99`` 都是一个词元），带连字符的词先整体切出，
    连字符成分的展开由各语言的 ``tokenize`` 决定。
    """
    return tuple(m.group(0) for m in _TOKEN_RE.finditer(text))


def split_hyphenated(token: str) -> tuple[str, ...]:
    """``eu-roaming`` -> ``(eu-roaming, eu, roaming)``；无连字符时原样返回。"""
    if "-" not in token:
        return (token,)
    return (token, *(part for part in token.split("-") if part))


def simple_tokenize(text: str) -> tuple[str, ...]:
    """没有形态学的语言够用的分词：切词、casefold、拆连字符。

    英语的最终实现就是这一份。德语在此之上再叠复合词分解与别名闭包，所以德语**不**
    调用它 —— 见 ``locale/de.py``。

    ``dict.fromkeys`` 是**每个原始词元内部**去重：一个表面词展开出的多个变体各计一次，
    但文档里真实重复出现的词仍然重复计入，BM25 的 TF 因此保持可解释。
    """
    out: list[str] = []
    for token in surface_tokens(text):
        out.extend(dict.fromkeys(split_hyphenated(token.casefold())))
    return tuple(out)
