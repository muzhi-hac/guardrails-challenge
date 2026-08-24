"""Brand-voice guard: does the reply still sound like this client?

Five deterministic checks, all tier 0 — no model call, no network, microseconds.
Two of them are worth singling out.

The **address form** check is the reason locale is a profile dimension rather
than a string. In German it is near-exact; in English the same configuration
key buys a much weaker heuristic. The locale layer reports which it is, and the
profile calibrates severity accordingly.

The **TTS safety** check has, as far as I can tell, no equivalent in the
published guardrail frameworks, and it catches a genuine production incident:
markdown, bullet points, URLs and unseparated digit runs are invisible defects
in a chat transcript and are read out literally over a phone line.
"""

from __future__ import annotations

import re
import time
from collections.abc import Mapping
from typing import ClassVar

from guardrails import findings
from guardrails.config import PersonaSpec
from guardrails.guards.base import GuardContext, build_verdict
from guardrails.locale.base import count_words
from guardrails.types import Evidence, Mode, Severity, Stage, Verdict

__all__ = ["PersonaGuard"]

# --- emoji -----------------------------------------------------------------
#
# Conservative ranges. `€` (U+20AC), en dashes and German quotation marks must
# never match; a false positive here rewrites a correct reply.
#
# Matching covers the whole sequence, not the leading code point: skin-tone
# modifiers, variation selectors and ZWJ joins are all part of one grapheme.
# A span that stopped at the base character would leave invisible control
# characters behind once bounded repair starts deleting spans.

_EMOJI_BASE = (
    "["
    "\U0001F000-\U0001FAFF"   # pictographs, emoticons, transport, extended-A
    "\u2600-\u26FF"           # miscellaneous symbols
    "\u2700-\u27BF"           # dingbats
    "\u2B00-\u2BFF"           # symbols and arrows
    "\u203C\u2049"            # double exclamation, exclamation question
    "]"
)
_SKIN_TONE = "[\U0001F3FB-\U0001F3FF]"
_VARIATION = "\uFE0F"          # variation selector-16
_ZWJ = "\u200D"                # zero-width joiner
_KEYCAP_MARK = "\u20E3"
_EMOJI_ATOM = rf"{_EMOJI_BASE}{_VARIATION}?{_SKIN_TONE}?"
_KEYCAP = rf"[0-9#*]{_VARIATION}?{_KEYCAP_MARK}"
_EMOJI_RE = re.compile(rf"(?:{_KEYCAP})|(?:{_EMOJI_ATOM}(?:{_ZWJ}{_EMOJI_ATOM})*)")

# --- text that a speech synthesiser reads out literally ---------------------

_TTS_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("bold or emphasis markup", re.compile(r"\*\*[^*\n]+\*\*|__[^_\n]+__|(?<!\*)\*[^*\n]+\*(?!\*)")),
    ("markdown link", re.compile(r"\[[^\]\n]+\]\([^)\n]+\)")),
    ("heading marker", re.compile(r"(?m)^\s{0,3}#{1,6}\s")),
    ("table row", re.compile(r"(?m)^.*\|.*\|.*$")),
    ("bullet or numbered list", re.compile(r"(?m)^\s*(?:[-*•‣]|\d+\.)\s+")),
    ("URL", re.compile(r"https?://\S+|\bwww\.\S+")),
    ("unseparated digit run", re.compile(r"\d{7,}")),
)


def _overlaps(span: tuple[int, int], taken: list[tuple[int, int]]) -> bool:
    return any(span[0] < hi and lo < span[1] for lo, hi in taken)


class PersonaGuard:
    """Checks a reply against the client's :class:`PersonaSpec`."""

    name: ClassVar[str] = "persona"
    stage: ClassVar[Stage] = Stage.OUTPUT
    tier: ClassVar[int] = 0

    DEFAULT_SEVERITY: ClassVar[Mapping[str, Severity]] = {
        findings.ADDRESS_FORM: Severity.MEDIUM,
        findings.EMOJI: Severity.LOW,
        findings.FORBIDDEN_PHRASE: Severity.MEDIUM,
        findings.SENTENCE_TOO_LONG: Severity.LOW,
        findings.TTS_UNSAFE: Severity.MEDIUM,
    }

    async def check(self, ctx: GuardContext) -> Verdict:
        started = time.perf_counter()
        config = ctx.profile.guards.persona
        spec = config.persona
        text = ctx.reply

        evidence: list[Evidence] = [
            *self._address_form(text, spec, ctx),
            *self._emoji(text, spec),
            *self._forbidden_phrases(text, spec),
            *self._sentence_length(text, spec, ctx),
            *self._tts_safety(text, spec, ctx.profile.mode),
        ]

        return build_verdict(
            guard=self.name,
            stage=self.stage,
            evidence=tuple(sorted(evidence, key=lambda e: (e.span or (0, 0)))),
            defaults=self.DEFAULT_SEVERITY,
            overrides=config.severity_overrides,
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    # -- checks -------------------------------------------------------------

    @staticmethod
    def _address_form(text: str, spec: PersonaSpec, ctx: GuardContext) -> list[Evidence]:
        strength = (
            "grammatical"
            if ctx.rules.supports_grammatical_address_form
            else "heuristic; this language does not mark the distinction grammatically"
        )
        return [
            Evidence(
                kind=findings.ADDRESS_FORM,
                detail=(
                    f"{hit.marker} {hit.token!r} indicates the "
                    f"{hit.form.value} register; this client uses "
                    f"{spec.address_form.value} ({strength})"
                ),
                span=hit.span,
            )
            for hit in ctx.rules.check_address_form(text, spec.address_form)
        ]

    @staticmethod
    def _emoji(text: str, spec: PersonaSpec) -> list[Evidence]:
        if spec.emoji_allowed:
            return []
        return [
            Evidence(
                kind=findings.EMOJI,
                detail=f"emoji {match.group()!r} is not permitted in this brand voice",
                span=match.span(),
            )
            for match in _EMOJI_RE.finditer(text)
        ]

    @staticmethod
    def _forbidden_phrases(text: str, spec: PersonaSpec) -> list[Evidence]:
        out: list[Evidence] = []
        for phrase in spec.forbidden_phrases:
            pattern = re.compile(rf"\b{re.escape(phrase)}\b", re.IGNORECASE)
            out += [
                Evidence(
                    kind=findings.FORBIDDEN_PHRASE,
                    detail=f"phrase {phrase!r} is on this client's forbidden list",
                    span=match.span(),
                )
                for match in pattern.finditer(text)
            ]
        return out

    @staticmethod
    def _sentence_length(text: str, spec: PersonaSpec, ctx: GuardContext) -> list[Evidence]:
        limit = spec.max_sentence_words
        if limit is None:
            return []
        return [
            Evidence(
                kind=findings.SENTENCE_TOO_LONG,
                detail=f"sentence runs to {sentence.word_count} words, limit is {limit}",
                span=sentence.span,
            )
            for sentence in ctx.rules.segment_sentences(text)
            if sentence.word_count > limit
        ]

    @staticmethod
    def _tts_safety(text: str, spec: PersonaSpec, mode: Mode) -> list[Evidence]:
        """Only in voice, where the failure is audible rather than cosmetic.

        Gating this sub-check by mode inside the guard rather than in
        configuration is the one place the design deliberately offers less
        configurability: a whole sub-check-level configuration layer for a
        single case is more surface than it earns.
        """
        if not spec.tts_safe or mode is not Mode.VOICE:
            return []

        out: list[Evidence] = []
        taken: list[tuple[int, int]] = []
        for label, pattern in _TTS_PATTERNS:
            for match in pattern.finditer(text):
                if _overlaps(match.span(), taken):
                    continue
                taken.append(match.span())
                out.append(
                    Evidence(
                        kind=findings.TTS_UNSAFE,
                        detail=f"{label} would be read out literally by speech synthesis",
                        span=match.span(),
                    )
                )
        return out
