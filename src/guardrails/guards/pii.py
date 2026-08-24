"""PII guard: does the reply expose personal data the customer did not
themselves provide?

This is the fourth tier-0 guard and the last of the "does this turn violate
something structural" checks, running at :data:`~guardrails.types.Stage.OUTPUT`.
Its job is **outbound leak prevention**, not PII detection in general. Those
sound similar and are not: a customer who types their own IBAN to ask "kann
mir jemand diese Nummer bestätigen?" gets a reply that repeats it back, and a
guard that scanned the reply alone would flag that repetition as a leak on
every such turn. It is not a leak — it is the customer's own data coming
back to them, the ordinary shape of a confirmation. The dangerous direction
is the other one: an IBAN, phone number, customer ID, birthdate or address
in the reply that the customer *never mentioned*, which means either
retrieval surfaced another customer's record, or the model fabricated a
plausible-looking value. So the check implemented here is a **difference**,
not a scan: every entity :func:`extract <_DETECTORS>` finds in ``ctx.reply``
is compared against the same entities extracted from ``ctx.user_message``,
and only the ones absent from the user's own turn are reported, always under
the single finding :data:`~guardrails.findings.OUTBOUND_LEAK`. A guard that
fired on every echoed value would trip on legitimate confirmation turns
constantly enough that a client would disable it — which is worse than the
narrower check, because a disabled guard also misses the leaks it exists for.

**Where the region-specific patterns live, and why.** ``IBAN`` prefixes,
German phone numbering, and postcode shape are not language facts — they are
account-format and numbering-plan facts that vary by country independently
of the language spoken. :class:`~guardrails.types.Locale`'s own docstring
makes this argument for why ``locale/`` holds *only* language rules: "a
German-language client in Switzerland keeps every rule in ``de`` and none of
the German account formats." Putting IBAN/phone/postcode regexes in
``locale/de.py`` would be exactly that mistake — a Swiss German-speaking
deployment would inherit patterns for account formats it does not use, and
there would be no clean way to turn them off without also turning off German
sentence segmentation. So these patterns live here, with the guard that
consumes them, conceptually keyed by region rather than by locale.

In practice there is only one region implemented: the profile schema has no
``region`` field yet (:class:`~guardrails.config.Profile` carries only
``locale``), and this deployment (``telco_de``) is a German company serving
German customers, so every pattern below is hardcoded to Germany's formats.
The upgrade trigger is the day a second region needs the same language —
e.g. a German-language client based in Switzerland — at which point
``Profile`` needs a ``region`` field, this module's patterns need to be
looked up by it instead of living as bare module constants, and this
docstring's justification is exactly the argument for why that lookup
belongs here and not in ``locale/``.

**Redaction is a function, not a guard.** ``PiiGuardConfig.redact_inbound``
exists because the customer's own data — typed in their own turn — should
not reach logs or a third-party model provider in the clear, even though it
is not a leak. But a guard's contract (see ``guards/base.py``) is to report
findings and never take action; rewriting text is an action, and actions
belong to the orchestrator. So :func:`redact` is a plain module-level
function, not part of :class:`PiiGuard`. **Nothing calls it yet** — this is
the detection half of the PII story; the chatbot wiring task is what will
call ``redact`` on the user's turn before it reaches a log line or a
provider request. It is written and tested here so that wiring is a single
call, not a design decision made under a different task's time pressure.

Every evidence ``detail`` produced in this module, by both the guard and
``redact``, deliberately never includes the matched value itself — only the
entity kind and, for the guard, that the value did not appear in the user's
own turn. A PII guard whose own trace output contained the IBAN it just
flagged would defeat its own purpose the moment the trace was logged or
shipped to a third party.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import date
from typing import ClassVar

from guardrails import findings
from guardrails.guards.base import GuardContext, build_verdict
from guardrails.locale.base import LocaleRules
from guardrails.types import Evidence, Severity, Stage, Verdict

__all__ = ["PiiGuard", "redact"]


class _Hit:
    """One matched span, before it becomes either an ``OUTBOUND_LEAK``
    (guard) or a typed finding (``redact``). Not :class:`Evidence` itself:
    ``raw`` must never reach a trace (see the module docstring), but the
    detectors below need it internally, to compare a reply's value against
    the user's own turn and to know exactly how many characters to replace."""

    __slots__ = ("kind", "raw", "span")

    def __init__(self, kind: str, raw: str, span: tuple[int, int]) -> None:
        self.kind = kind
        self.raw = raw
        self.span = span


# -- Germany-specific formats, keyed by region (see module docstring) ------

# IBAN: "DE" + 2 check digits + 18-digit BBAN, optionally space-grouped in
# blocks of 4 the way banks print them ("DE89 3704 0044 0532 0130 00") or
# written solid. 20 digits after the letters either way.
_IBAN_RE = re.compile(r"\bDE\d{2}(?:[ ]?\d{4}){4}[ ]?\d{2}\b")


def _iban_checksum_valid(candidate: str) -> bool:
    """Mod-97 checksum from ISO 7064: move the first 4 characters to the
    end, replace each letter with its alphabet position + 9 (A=10 .. Z=35),
    and the result must be congruent to 1 mod 97.

    Validating this, rather than reporting any ``DE`` + 20 digits, is the
    same lesson the grounding guard's date validation already encodes:
    ``31.02.2026`` matches every date regex and is not a date. Order
    references, contract numbers and other 20-digit-ish strings in this
    domain do not carry a valid IBAN checksum by chance (1-in-97 odds), so
    skipping the check would mean a guard that cries wolf on ordinary
    reference numbers — exactly the kind of guard that gets disabled.
    """
    compact = candidate.replace(" ", "").upper()
    rearranged = compact[4:] + compact[:4]
    digits = "".join(str(ord(ch) - 55) if ch.isalpha() else ch for ch in rearranged)
    return int(digits) % 97 == 1


def _find_ibans(text: str) -> list[_Hit]:
    return [
        _Hit(findings.IBAN, match.group(0), match.span())
        for match in _IBAN_RE.finditer(text)
        if _iban_checksum_valid(match.group(0))
    ]


# Phone: "+49" / "0049" international prefix, or a bare "0" trunk prefix,
# followed by an area/mobile code and a subscriber number, separated by
# spaces, slashes or hyphens (or nothing). The subscriber part requires 6-9
# digits specifically so this does not fire on a 3-4 digit area code alone
# ("0170" as a tariff or product code, on its own, is not a phone number) —
# every real German number has enough digits after the trunk/area code to
# clear that floor.
_PHONE_RE = re.compile(
    r"(?:\+49|0049)[ /-]?\d{2,5}[ /-]?\d{6,9}\b"
    r"|\b0\d{2,5}[ /-]?\d{6,9}\b"
)


def _find_phones(text: str) -> list[_Hit]:
    return [_Hit(findings.PHONE, m.group(0), m.span()) for m in _PHONE_RE.finditer(text)]


# Customer ID: client-specific shape. "KD-" (Kundennummer) followed by 8
# digits is a placeholder for this exercise; in production this pattern (and
# the prefix) comes from the client's own configuration, since every telco
# invents its own customer-number format and there is no industry standard
# to hardcode.
_CUSTOMER_ID_RE = re.compile(r"\bKD-\d{8}\b")


def _find_customer_ids(text: str) -> list[_Hit]:
    return [
        _Hit(findings.CUSTOMER_ID, m.group(0), m.span()) for m in _CUSTOMER_ID_RE.finditer(text)
    ]


# Birthdate: a German date (TT.MM.JJJJ) is only a birthdate in context — the
# corpus and normal replies are full of dates ("Stand: 01.01.2026", contract
# start dates, notice deadlines) that are not anyone's birthday. Requiring
# one of "geboren", "Geburtsdatum" or "geb." immediately before the date is
# what makes a bare date pass and a birth-context date fail; only the date
# itself is captured as the match span, so redaction removes the value and
# leaves the surrounding sentence ("geboren am [BIRTHDATE]") legible.
_BIRTH_CONTEXT = r"(?:geboren(?:e[nr]?)?(?:\s+am)?|geburtsdatum|geb\.)"
_BIRTHDATE_RE = re.compile(
    rf"{_BIRTH_CONTEXT}\s*:?\s*(?P<date>\d{{1,2}}\.\d{{1,2}}\.\d{{4}})",
    re.IGNORECASE,
)


def _valid_calendar_date(day: int, month: int, year: int) -> bool:
    """A regex match is not a date — ``31.02.2026`` matches the pattern and
    does not exist on a calendar. Mirrors ``locale/de.py``'s ``_iso_date``:
    the same lesson, applied here rather than imported from there, because
    that helper is private to the date-*extraction* concern in ``locale``
    and this module deliberately does not depend on ``locale/de`` for
    region-specific matching (see the module docstring)."""
    try:
        date(year, month, day)
    except ValueError:
        return False
    return True


def _find_birthdates(text: str) -> list[_Hit]:
    hits: list[_Hit] = []
    for m in _BIRTHDATE_RE.finditer(text):
        day, month, year = (int(g) for g in m.group("date").split("."))
        if _valid_calendar_date(day, month, year):
            hits.append(_Hit(findings.BIRTHDATE, m.group("date"), m.span("date")))
    return hits


# Address: two independent shapes.
#
# 1. A street line — a capitalised word ending in "straße"/"strasse"/"str."/
#    "weg"/"platz" followed by a house number ("Hauptstraße 12",
#    "Bahnhofstr. 5", "Kirchweg 3", "Marktplatz 1"). Anchoring on the
#    suffix-then-number pair is deliberate: "unterwegs" or "Umweg" end in
#    "weg" too, but German prose essentially never follows one of those
#    words directly with a bare number, so the pair is the safe signal, not
#    the suffix alone.
_STREET_SUFFIX = r"(?:straße|strasse|str\.|weg|platz)"
_STREET_RE = re.compile(
    rf"\b([A-ZÄÖÜ][A-Za-zÄÖÜäöüß]*{_STREET_SUFFIX})\s+(\d{{1,4}}[a-zA-Z]?)\b"
)

# 2. A postcode + city ("10115 Berlin"). Anchored to only fire right after a
#    comma or at the start of a line/text, not anywhere a bare 5-digit
#    number happens to precede a capitalised word. That anchor matters
#    because German capitalises every noun: "über 50000 Verträge" is a
#    5-digit count followed by a capitalised noun with exactly the same
#    shape as a postcode and a city, and it appears in ordinary policy
#    prose. The comma is what a written address actually looks like
#    ("Musterstraße 12, 10115 Berlin") and a sentence about a quantity does
#    not produce by coincidence.
_POSTCODE_CITY_RE = re.compile(
    r"(?:^|,\s*)(\d{5})\s+([A-ZÄÖÜ][a-zäöüß]+(?:-[A-ZÄÖÜ][a-zäöüß]+)*)\b",
    re.MULTILINE,
)


def _find_addresses(text: str) -> list[_Hit]:
    hits = [_Hit(findings.ADDRESS, m.group(0), m.span()) for m in _STREET_RE.finditer(text)]
    hits.extend(
        # span covers postcode..city only (group 1 through the end of group
        # 2), not the leading comma/whitespace the anchor consumed -- that
        # separator is not part of the address.
        _Hit(findings.ADDRESS, text[m.start(1) : m.end(2)], (m.start(1), m.end(2)))
        for m in _POSTCODE_CITY_RE.finditer(text)
    )
    return hits


_DETECTORS: Mapping[str, Callable[[str], list[_Hit]]] = {
    findings.IBAN: _find_ibans,
    findings.PHONE: _find_phones,
    findings.CUSTOMER_ID: _find_customer_ids,
    findings.BIRTHDATE: _find_birthdates,
    findings.ADDRESS: _find_addresses,
}

_LEAK_DETAIL: Mapping[str, str] = {
    findings.IBAN: "an IBAN",
    findings.PHONE: "a phone number",
    findings.CUSTOMER_ID: "a customer ID",
    findings.BIRTHDATE: "a birthdate",
    findings.ADDRESS: "an address",
}

_PLACEHOLDERS: Mapping[str, str] = {
    findings.IBAN: "[IBAN]",
    findings.PHONE: "[PHONE]",
    findings.CUSTOMER_ID: "[CUSTOMER_ID]",
    findings.BIRTHDATE: "[BIRTHDATE]",
    findings.ADDRESS: "[ADDRESS]",
}


def _normalize(raw: str) -> str:
    """Collapse formatting so the same IBAN written with and without spaces,
    or the same phone number with different separators, compare equal.
    Punctuation and whitespace carry no identity for any of these entities —
    only the digits (and, for the address suffix, the letters) do."""
    return re.sub(r"[\s/\-.,]", "", raw).casefold()


class PiiGuard:
    """Detects personal data in a reply that the customer's own turn did not
    already contain — see the module docstring for why that difference,
    rather than a bare scan of the reply, is the check that does not fire on
    every confirmation turn.
    """

    name: ClassVar[str] = "pii"
    stage: ClassVar[Stage] = Stage.OUTPUT
    tier: ClassVar[int] = 0

    DEFAULT_SEVERITY: ClassVar[Mapping[str, Severity]] = {
        # The finding this guard's own check() emits. CRITICAL because the
        # telco_de profile routes CRITICAL to safe_fallback specifically so
        # a reply carrying leaked personal data never reaches the customer
        # or a human agent's queue verbatim -- see profiles/telco_de.yaml's
        # routing comment, which names outbound PII (alongside injection) as
        # the two CRITICAL categories.
        findings.OUTBOUND_LEAK: Severity.CRITICAL,
        # These five are never emitted by check() -- they are emitted by the
        # module-level redact() function, which has no DEFAULT_SEVERITY of
        # its own to declare them in because it is a function, not a guard.
        # Declared here anyway, the same way InjectionGuard declares
        # JAILBREAK_CLASSIFIED for a tier-1 judge that shares its name: the
        # finding vocabulary and its severities are one contract, not one
        # per emitter. MEDIUM because detecting the customer's own data in
        # their own turn is not itself a violation -- it is what
        # redact_inbound exists to launder before logging, not a finding
        # that should drive routing on its own.
        findings.IBAN: Severity.MEDIUM,
        findings.PHONE: Severity.MEDIUM,
        findings.CUSTOMER_ID: Severity.MEDIUM,
        findings.BIRTHDATE: Severity.MEDIUM,
        findings.ADDRESS: Severity.MEDIUM,
    }

    async def check(self, ctx: GuardContext) -> Verdict:
        started = time.perf_counter()
        config = ctx.profile.guards.pii
        entities = [name for name in config.entities if name in _DETECTORS]

        user_values = {
            _normalize(hit.raw)
            for name in entities
            for hit in _DETECTORS[name](ctx.user_message)
        }

        evidence = [
            Evidence(
                kind=findings.OUTBOUND_LEAK,
                detail=(
                    f"{_LEAK_DETAIL[hit.kind]} in the reply does not appear in "
                    f"the customer's own message this turn"
                ),
                span=hit.span,
            )
            for name in entities
            for hit in _DETECTORS[name](ctx.reply)
            if _normalize(hit.raw) not in user_values
        ]

        return build_verdict(
            guard=self.name,
            stage=self.stage,
            evidence=tuple(sorted(evidence, key=lambda e: (e.span or (0, 0)))),
            defaults=self.DEFAULT_SEVERITY,
            overrides=config.severity_overrides,
            latency_ms=(time.perf_counter() - started) * 1000,
        )


def redact(text: str, entities: Sequence[str], rules: LocaleRules) -> tuple[str, tuple[Evidence, ...]]:
    """Replace detected personal data with typed placeholders.

    A module-level function rather than guard behaviour: a guard's contract
    is to report findings, never to act, and rewriting text is an action —
    see the module docstring's "Redaction is a function, not a guard"
    section. ``Chatbot`` calls it on the current user message and on user turns
    retained in history before either text reaches a provider request or trace,
    as directed by ``PiiGuardConfig.redact_inbound``.

    ``rules`` is accepted but not used by any pattern below — every entity
    here is a region-specific format (IBAN, phone, postcode), not a language
    rule, so nothing in this function needs ``segment_sentences`` or
    ``tokenize``. It is part of the signature anyway so every call site
    passes the same two pieces of turn context (``entities`` from the
    profile, ``rules`` from the locale) that :class:`PiiGuard` itself
    receives via ``GuardContext``, and so a future entity that *does* need
    language-aware segmentation has somewhere to get it without changing
    this function's signature a second time.

    Returns the redacted text and the evidence of what was removed — kind
    and span, never the value — so a trace can record that redaction
    happened and of what kind without recording the personal data itself.
    """
    del rules

    hits: list[_Hit] = []
    for name in entities:
        detector = _DETECTORS.get(name)
        if detector is not None:
            hits.extend(detector(text))
    hits.sort(key=lambda h: h.span[0])

    pieces: list[str] = []
    evidence: list[Evidence] = []
    cursor = 0
    claimed: list[tuple[int, int]] = []
    for hit in hits:
        lo, hi = hit.span
        # Two detectors should never claim overlapping spans given how
        # distinct these patterns are, but redaction is exactly the wrong
        # place to find that out the hard way -- an overlap here would
        # corrupt the rebuilt string. Skip any span already claimed rather
        # than risk it.
        if any(lo < claimed_hi and claimed_lo < hi for claimed_lo, claimed_hi in claimed):
            continue
        claimed.append((lo, hi))
        pieces.append(text[cursor:lo])
        pieces.append(_PLACEHOLDERS[hit.kind])
        cursor = hi
        evidence.append(
            Evidence(
                kind=hit.kind,
                detail=f"{_LEAK_DETAIL[hit.kind]} redacted from inbound text before logging",
                span=(lo, hi),
            )
        )
    pieces.append(text[cursor:])

    return "".join(pieces), tuple(evidence)
