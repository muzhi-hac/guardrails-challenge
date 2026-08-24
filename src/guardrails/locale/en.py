"""English (en-GB) language rules.

The address-form check is deliberately weaker here, and says so. English has no
grammaticalised T-V distinction, so there is no equivalent of finding a ``du``.
What remains are register heuristics — contractions, casual greetings — which
indicate informality without proving it. ``supports_grammatical_address_form``
is ``False`` so that a guard can weigh the finding accordingly rather than
treating a contraction as equivalent to a German pronoun violation.

Reporting this as the same check with a lower confidence would be dishonest;
the marker on each hit names what was actually found.
"""

from __future__ import annotations

import re
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import ClassVar

from guardrails.locale.base import (
    AddressFormHit,
    CommitmentHit,
    EntityMention,
    Sentence,
    count_words,
    find_commitments_by_phrase,
    format_decimal,
    simple_tokenize,
)
from guardrails.types import AddressForm, EntityKind, Locale

__all__ = ["EnglishRules"]

_ABBREVIATIONS: frozenset[str] = frozenset(
    {
        "mr.", "mrs.", "ms.", "dr.", "prof.", "st.", "e.g.", "i.e.", "etc.",
        "vs.", "inc.", "ltd.", "plc.", "no.", "approx.", "est.", "dept.",
        "min.", "max.", "fig.", "cf.", "p.a.",
    }
)

_MONTHS: dict[str, int] = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
}
_MONTH_ALT = "|".join(m.capitalize() for m in _MONTHS)

_CONTRACTION_RE = re.compile(
    r"\b\w+(?:'|’)(?:s|t|re|ve|ll|d|m)\b|\b(?:can|won|don|doesn|isn|aren|wasn|weren|"
    r"couldn|shouldn|wouldn|didn|hasn|haven|hadn)(?:'|’)t\b",
    re.IGNORECASE,
)
_CASUAL_GREETING_RE = re.compile(r"(?:^|(?<=[.!?]\s))\s*(Hi|Hey|Hiya)\b")
_FORMAL_GREETING_RE = re.compile(r"\bDear (?:Sir|Madam|Mr|Mrs|Ms|Dr)\b")

_AMOUNT = r"\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?"
_CURRENCY = r"£|GBP\b|€|EUR\b"

_PRICE_RE = re.compile(
    rf"(?:(?P<pre>{_CURRENCY})\s*(?P<a1>{_AMOUNT}))"
    rf"|(?:(?P<a2>{_AMOUNT})\s*(?P<post>{_CURRENCY}))"
)
_DATE_SLASH_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")
"""en-GB is day-first: ``01/02/2026`` is 1 February, not 2 January. The same
string means a different date under en-US, which is why this lives in a
locale module rather than in a shared date helper."""

_DATE_ISO_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_DATE_LONG_RE = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_ALT})\s+(\d{{4}})\b")
_DATE_MONTH_YEAR_RE = re.compile(rf"\b({_MONTH_ALT})\s+(\d{{4}})\b")
_DURATION_RE = re.compile(
    r"\b(\d+)[\s-]*(months?|years?|days?|weeks?)\b", re.IGNORECASE
)

_ORDINAL_SUFFIX = r"(?:st|nd|rd|th)"
_ORDINAL_INT = r"\d{1,3}(?:,\d{3})+|\d+"
"""Integer only -- an ordinal never carries a decimal fraction, so this is
``_AMOUNT`` without the ``.\\d{1,2}`` tail. Keeping it separate stops the
ordinal branch from ever nibbling at a price like ``29.99``."""

# German already extracts a numeral out of its own ordinal form: `am 3.
# Werktag` yields NUMBER `3` because the ordinal marker there is a bare full
# stop. English marks the same grammatical thing with a letter suffix instead
# (`3rd`, `21st`, `102nd`), which the plain amount pattern below does not
# recognise at all -- the trailing `(?![\w])` guard exists to reject a digit
# run that a word continues (`29x`), and a bare ordinal suffix is exactly such
# a continuation as far as that guard can tell. Left alone, English therefore
# extracts nothing from a sentence where German extracts `3`, and a mirror
# whose two sides extract different NUMBER sets cannot support the grounding
# guard that later compares reply entities against retrieved chunks. So the
# ordinal branch below recognises the suffix explicitly and strips it: the
# numeral is the entity, the suffix is not part of it and never appears in
# ``normalized``. It still requires the *whole* token to end at the suffix
# (`(?![\w])` after it), so a genuine word continuation (`3rdclass`,
# `abc3rdxyz`) is rejected exactly as a bare number followed by a letter
# would be.
_NUMBER_RE = re.compile(
    rf"(?<![\w.,])(?:(?P<ordinal>{_ORDINAL_INT}){_ORDINAL_SUFFIX}(?![\w])"
    rf"|(?P<amount>{_AMOUNT})(?![\w]))"
)

_DURATION_UNITS = {"month": "M", "year": "Y", "day": "D", "week": "W"}

_CURRENCY_CODES: dict[str, str] = {"£": "GBP", "gbp": "GBP", "€": "EUR", "eur": "EUR"}


def _currency_code(marker: str) -> str:
    """Map a matched currency marker to its ISO code.

    Currency is really *region*-shaped knowledge, not *language*-shaped — the
    same English channel might sell in EUR or in GBP. The correct home for
    this is a future ``region`` field on ``Profile``; until that lands, the
    least this function can do is not hard-code the currency.
    """
    return _CURRENCY_CODES[marker.casefold()]


def _parse_amount(raw: str) -> Decimal | None:
    try:
        return Decimal(raw.replace(",", ""))
    except InvalidOperation:
        return None


# --- commitments -------------------------------------------------------

_COMMITMENT_PHRASES: tuple[tuple[str, str], ...] = (
    ("refund", "refund"),
    ("refund", "reimburse"),
    ("waive_fee", "waive the fee"),
    ("waive_fee", "no charge"),
    ("credit", "credit your account"),
    ("discount", "grant a discount"),
    ("schedule_callback", "call you back"),
    ("send_confirmation_email", "send you a confirmation"),
)
"""Mirrors ``de.py``'s ``_COMMITMENT_PHRASES`` -- same commitment ids, worded
for English. Kept as a same-shaped table rather than a shared cross-language
mapping because the wording, not the id, is what genuinely differs; the ids
are the language-independent part and already live in the profile schema."""


class EnglishRules:
    """Language rules for British English."""

    locale: ClassVar[Locale] = Locale.EN_GB
    supports_grammatical_address_form: ClassVar[bool] = False

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
            return False
        following = after.lstrip()
        if following and not (following[0].isupper() or following[0] in "\"'“‘("):
            return False
        if text[i] != ".":
            return True
        j = i
        while j > 0 and (text[j - 1].isalpha() or text[j - 1] == "."):
            j -= 1
        return text[j : i + 1].lower() not in _ABBREVIATIONS

    def check_address_form(self, text: str, expected: AddressForm) -> tuple[AddressFormHit, ...]:
        if expected is AddressForm.FORMAL:
            hits = [
                AddressFormHit(AddressForm.INFORMAL, "contraction", m.group(), m.span())
                for m in _CONTRACTION_RE.finditer(text)
            ]
            hits += [
                AddressFormHit(
                    AddressForm.INFORMAL, "casual_greeting", m.group(1), m.span(1)
                )
                for m in _CASUAL_GREETING_RE.finditer(text)
            ]
        else:
            hits = [
                AddressFormHit(AddressForm.FORMAL, "formal_greeting", m.group(), m.span())
                for m in _FORMAL_GREETING_RE.finditer(text)
            ]
        return tuple(sorted(hits, key=lambda h: h.span))

    def extract_entities(
        self, text: str, kinds: frozenset[EntityKind]
    ) -> tuple[EntityMention, ...]:
        found: list[EntityMention] = []
        taken: list[tuple[int, int]] = []

        def claim(mention: EntityMention) -> None:
            found.append(mention)
            taken.append(mention.span)

        for m in _PRICE_RE.finditer(text):
            amount = _parse_amount(m.group("a1") or m.group("a2"))
            if amount is not None:
                marker = m.group("pre") or m.group("post")
                claim(
                    EntityMention(
                        EntityKind.PRICE,
                        m.group(),
                        f"{amount.quantize(Decimal('0.01'))} {_currency_code(marker)}",
                        m.span(),
                    )
                )

        for m in _DATE_ISO_RE.finditer(text):
            year, month, day = (int(g) for g in m.groups())
            if (iso := _iso_date(year, month, day)) and not _overlaps(m.span(), taken):
                claim(EntityMention(EntityKind.DATE, m.group(), iso, m.span()))

        for m in _DATE_SLASH_RE.finditer(text):
            day, month, year = (int(g) for g in m.groups())  # day-first
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
            value, unit = int(m.group(1)), _duration_unit(m.group(2))
            if value > 0 and unit and not _overlaps(m.span(), taken):
                claim(EntityMention(EntityKind.DURATION, m.group(), f"P{value}{unit}", m.span()))

        for m in _NUMBER_RE.finditer(text):
            amount = _parse_amount(m.group("ordinal") or m.group("amount"))
            if amount is not None and not _overlaps(m.span(), taken):
                claim(EntityMention(EntityKind.NUMBER, m.group(), format_decimal(amount), m.span()))

        return tuple(sorted((f for f in found if f.kind in kinds), key=lambda e: e.span))

    def find_commitments(self, text: str) -> tuple[CommitmentHit, ...]:
        return find_commitments_by_phrase(text, _COMMITMENT_PHRASES)

    def tokenize(self, text: str) -> tuple[str, ...]:
        """English has no compounding of the kind German has; the trivial
        tokenizer is the final implementation."""
        return simple_tokenize(text)


def _duration_unit(word: str) -> str | None:
    lowered = word.lower().rstrip("s")
    return _DURATION_UNITS.get(lowered)


def _iso_date(year: int, month: int, day: int) -> str | None:
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def _overlaps(span: tuple[int, int], taken: list[tuple[int, int]]) -> bool:
    return any(span[0] < hi and lo < span[1] for lo, hi in taken)
