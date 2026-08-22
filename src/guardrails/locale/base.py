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
        """Tokens for retrieval.

        Unlike :meth:`extract_entities`, this output feeds the BM25 index and
        may expand one surface word into several tokens (German compound
        constituents and umlaut aliases). Expansion is always **additive** —
        the original word is always kept — because exact word matching is
        this retriever's primary capability, and any normalisation that
        replaces the original word instead of adding to it would weaken it.
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
    """Split into surface tokens by language-independent rules.

    Numbers stay intact (``29,99`` and ``29.99`` are each one token), and a
    hyphenated word is cut out whole first; whether its hyphenated
    constituents are also expanded is decided by each language's
    ``tokenize``.
    """
    return tuple(m.group(0) for m in _TOKEN_RE.finditer(text))


def split_hyphenated(token: str) -> tuple[str, ...]:
    """``eu-roaming`` -> ``(eu-roaming, eu, roaming)``; returned unchanged when
    there is no hyphen."""
    if "-" not in token:
        return (token,)
    return (token, *(part for part in token.split("-") if part))


def simple_tokenize(text: str) -> tuple[str, ...]:
    """Tokenization sufficient for a language with no relevant morphology: split
    into words, casefold, split hyphens.

    This is English's final implementation. German layers compound
    decomposition and alias closure on top and therefore **does not** call
    this — see ``locale/de.py``.

    ``dict.fromkeys`` deduplicates **within each original token**: the several
    variants produced by expanding one surface word each count once, but a
    word that genuinely repeats in the document still counts again, so
    BM25's TF stays interpretable.
    """
    out: list[str] = []
    for token in surface_tokens(text):
        out.extend(dict.fromkeys(split_hyphenated(token.casefold())))
    return tuple(out)
