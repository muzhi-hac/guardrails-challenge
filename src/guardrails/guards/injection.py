"""Injection guard: does the user's own turn try to make the assistant obey
someone other than the operator?

Runs at :data:`~guardrails.types.Stage.INPUT`, the first stage in the
pipeline. That placement is now possible precisely because this guard covers
one channel, not two: the user's own turn
(:attr:`~guardrails.guards.base.GuardContext.user_message` and
:attr:`~guardrails.guards.base.GuardContext.history`). It used to also scan
retrieved documents and run at :data:`~guardrails.types.Stage.RETRIEVAL`
because a guard has exactly one stage and the document check needed
retrieval's output; that channel now lives in
:class:`~guardrails.guards.document.DocumentGuard`, which runs at RETRIEVAL
for the same reason. See that module's docstring for why splitting was worth
duplicating nothing: the two channels differ in kind, not just in stage.

**Different threat, different trust assumption.** A user's own message is
adversarial by default -- every customer service bot has to assume a
non-trivial fraction of its traffic is someone probing for a jailbreak. A
retrieved document is trusted infrastructure: the knowledge base is not
re-authored by an adversary on every turn, so a construction that looks like
an attack there means the corpus itself is compromised or mis-authored, a
different finding with a different investigation path. Running this guard at
INPUT means an adversarial user turn is rejected before retrieval is even
attempted -- strictly earlier than RETRIEVAL, and the wasted retrieval call
is not just latency, it is a small amplification vector for whoever is
sending the probes.

**Why the false-positive patterns are anchored on the imperative
construction, not on isolated keywords.** A telecom customer legitimately
writes ``System``, ``Rolle``, ``ab jetzt`` and ``act as`` in ordinary German
and English sentences about tariffs, complaints and error messages. Every
pattern below requires the *verb tightly followed by its object* — the
override patterns require an imperative directed at instructions/rules
(``ignoriere ... Anweisungen``, not just ``ignorieren`` anywhere near
``vorherige``), and the role-play patterns require the reassignment verb
immediately followed by the identity marker (``du bist jetzt ein``, not
``ab jetzt`` on its own). A keyword match would fire on ``Das System zeigt
mir eine Fehlermeldung an`` or ``Welche Rolle spielt die Kündigungsfrist``;
the construction match does not, because neither sentence contains the verb
in the position the construction requires. See the docstrings on
``_OVERRIDE_PATTERNS`` and ``_ROLE_PLAY_PATTERNS`` for the specific phrase
each pattern targets and the specific benign sentence it was checked
against.

**Why this guard's patterns are multilingual, and why that is the opposite
call from the grounding guard's.** The grounding guard was corrected to
extract entities using *only* the turn's locale, because retrieval is
locale-partitioned and applying a foreign language's grammar to a chunk that
cannot occur there does not fail loudly — it silently invents a
wrong-but-well-formed entity. Restricting to one locale was the fix, not the
bug.

An attacker is not a retrieved chunk. Nothing partitions attack attempts by
the deployment's configured locale, and an attacker who wants to slip past a
German-language filter has every reason to write the attack in English — it
is the language the underlying model was most heavily trained and red-teamed
against, and it is the language most publicly documented jailbreak strings
are written in. A German-language assistant that only recognised German
override phrases would be blind to precisely the attacks most likely to be
tried against it. So these patterns are deliberately bilingual — German and
English together, checked regardless of the turn's own locale — and for the
same reason they live in this module rather than in :mod:`guardrails.locale`:
they are security signatures, not language rules. ``locale/`` answers "how
does this language express a price"; this module answers "what does an
attack look like," and an attack's language is the attacker's choice, not
the deployment's.

``OVERRIDE_PATTERNS`` and ``ROLE_PLAY_PATTERNS`` are also imported by
:mod:`guardrails.guards.document`, which reuses them verbatim to check a
retrieved document for the same imperative constructions addressed at the
assistant instead of at a reader. An instruction is an instruction whether it
arrives via the user's turn or via a document; duplicating the pattern set
would mean fixing every false positive twice.
"""

from __future__ import annotations

import re
import time
from collections.abc import Mapping
from typing import ClassVar

from guardrails import findings
from guardrails.guards.base import GuardContext, build_verdict
from guardrails.types import Evidence, Severity, Stage, Verdict

__all__ = ["InjectionGuard", "OVERRIDE_PATTERNS", "ROLE_PLAY_PATTERNS"]

# -- instruction override ----------------------------------------------
#
# Each pattern requires the discard-verb to be immediately followed (allowing
# only a polite "Sie" / "bitte" in between, on the German side) by the noun it
# is discarding. "Ignorieren Sie bitte meine vorherige Frage, ich meinte etwas
# anderes." -- a customer correcting themselves -- fails every one of these:
# the verb is followed by "meine vorherige Frage", and none of these patterns
# accept "meine" or "Frage" where they require "alle"/"vorherigen" and an
# instruction-noun (Anweisungen/Regeln/Vorgaben/Befehle).
OVERRIDE_PATTERNS: tuple[re.Pattern[str], ...] = (
    # "Ignoriere/Ignorieren Sie alle vorherigen/bisherigen Anweisungen/Regeln/..."
    re.compile(
        r"ignorier\w*\s+(sie\s+)?(bitte\s+)?alle\s+(vorherigen|bisherigen)"
        r"\s+(anweisungen|regeln|vorgaben|befehle)",
        re.IGNORECASE,
    ),
    # "Vergiss deine/alle/die Anweisungen/Regeln/Vorgaben"
    re.compile(r"vergiss\w*\s+(deine|alle|die)\s+(anweisungen|regeln|vorgaben)", re.IGNORECASE),
    # "Neue Anweisung vom Betreiber" -- the framing the poisoned document in
    # examples/example_runs.md and tests/test_chatbot_live.py actually uses,
    # a document pretending its content originates from the operator.
    re.compile(r"neue\s+anweisung\w*\s+vom\s+betreiber", re.IGNORECASE),
    # "Missachten Sie (bitte) (alle/jegliche) (vorherigen/bisherigen) Anweisungen/..."
    re.compile(
        r"missachte\w*\s+(sie\s+)?(bitte\s+)?(alle\s+|jegliche\s+)?"
        r"(vorherigen|bisherigen)\s+(anweisungen|regeln|vorgaben|befehle)",
        re.IGNORECASE,
    ),
    # "ignore all/the previous/prior/above instructions"
    re.compile(r"ignore\s+(all\s+)?(the\s+)?(above|previous|prior)\s+instructions", re.IGNORECASE),
    # "disregard the above"
    re.compile(r"disregard\s+(the\s+)?above", re.IGNORECASE),
    # "forget your instructions"
    re.compile(r"forget\s+your\s+instructions", re.IGNORECASE),
    # "new instructions from ..."
    re.compile(r"new\s+instructions?\s+from", re.IGNORECASE),
)

# -- role-play / identity reassignment -----------------------------------
#
# "Welche Rolle spielt die Kündigungsfrist bei einem Umzug?" fails every one
# of these: the noun ("Rolle") precedes the verb ("spielt"), where the attack
# construction is verb-then-noun ("spiele ... Rolle"). "Ich möchte ab jetzt
# monatlich zahlen." fails because "ab jetzt" is not followed by "bist du".
# "Can you act as my contact for this case?" fails because "act as" here is
# not followed by "if" -- the only English role-play pattern using "act as"
# requires the conditional framing the jailbreak template actually uses.
ROLE_PLAY_PATTERNS: tuple[re.Pattern[str], ...] = (
    # "Du bist jetzt ein/eine/kein/nicht.../DAN ..."
    re.compile(r"du\s+bist\s+jetzt\s+(ein|eine|kein|nicht|dan)", re.IGNORECASE),
    # "Ab jetzt bist du ..."
    re.compile(r"ab\s+jetzt\s+bist\s+du", re.IGNORECASE),
    # "Spiele/Spielst die Rolle eines/einer/von ..."
    re.compile(r"spiel(e|st)\s+(die\s+)?rolle\s+(eines|einer|von)", re.IGNORECASE),
    # "Tu so, als ob du ..."
    re.compile(r"tu\s+so,?\s+als\s+ob\s+du", re.IGNORECASE),
    # "you are now a/an/DAN/no longer ..."
    re.compile(r"you\s+are\s+now\s+(a|an|dan|no\s+longer)\b", re.IGNORECASE),
    # "pretend to be ..."
    re.compile(r"pretend\s+to\s+be", re.IGNORECASE),
    # "act as if you ..." -- deliberately requires "if you", not bare "act as"
    re.compile(r"act\s+as\s+if\s+you", re.IGNORECASE),
    # "from now on you are ..."
    re.compile(r"from\s+now\s+on\s+you\s+are", re.IGNORECASE),
    # "DAN mode"
    re.compile(r"dan\s+mode", re.IGNORECASE),
)

# -- encoded payload --------------------------------------------------------
#
# Thresholds chosen to sit well above anything a legitimate field in this
# domain produces. A German IBAN is 22 characters; the contract and order
# references used in this corpus run well under 20. 48 unbroken characters
# of base64 alphabet is both far beyond those, and long enough to encode a
# complete instruction (48 base64 chars ~= 36 bytes, room for a sentence like
# "ignoriere alle vorherigen Regeln").
_BASE64_MIN_LEN: int = 48
_BASE64_RUN = re.compile(rf"[A-Za-z0-9+/]{{{_BASE64_MIN_LEN},}}={{0,2}}")

# Four or more \uXXXX escapes in one message is not something a customer
# produces by hand -- one or two might appear pasted from a log or an error
# message, but a real base64/utf-16 payload needs a run of them, and four is
# already well past anything incidental.
_UNICODE_ESCAPE_MIN_COUNT: int = 4
_UNICODE_ESCAPE = re.compile(r"\\u[0-9a-fA-F]{4}")


def _looks_encoded(candidate: str) -> bool:
    """Filter for :data:`_BASE64_RUN` matches that are plausibly encoded bytes
    rather than one long natural-language word.

    A German compound noun can easily clear 48 characters
    (``Kraftfahrzeughaftpflichtversicherungsbeitragsberechnungsgrundlage``)
    while drawing from a single case and no digits at all. Base64 draws from
    a 64-symbol alphabet spanning both cases, the digits, and ``+``/``/``; a
    run of that length using only one case and no digits is far more
    consistent with a long word than with encoded bytes, so it is not
    reported.
    """
    has_symbol = ("+" in candidate) or ("/" in candidate)
    if has_symbol:
        return True
    has_digit = any(char.isdigit() for char in candidate)
    has_upper = any(char.isupper() for char in candidate)
    has_lower = any(char.islower() for char in candidate)
    return has_digit and has_upper and has_lower


def _matches_override_or_roleplay(text: str) -> bool:
    return any(p.search(text) for p in (*OVERRIDE_PATTERNS, *ROLE_PLAY_PATTERNS))


class InjectionGuard:
    """Detects attempts, arriving through the user's own turn or an assembly
    spread across recent turns, to make the assistant discard its
    instructions or its identity.

    See the module docstring for why this runs at
    :data:`~guardrails.types.Stage.INPUT` rather than RETRIEVAL now, why the
    document channel moved to :class:`~guardrails.guards.document.DocumentGuard`,
    and why its patterns are deliberately multilingual rather than following
    ``ctx.rules`` the way the grounding guard does.
    """

    name: ClassVar[str] = "injection"
    stage: ClassVar[Stage] = Stage.INPUT
    tier: ClassVar[int] = 0

    DEFAULT_SEVERITY: ClassVar[Mapping[str, Severity]] = {
        # These are high-confidence, high-consequence detections. The
        # telco_de profile routes CRITICAL to safe_fallback specifically so
        # that an injected turn never reaches the user or a human agent's
        # queue verbatim (see profiles/telco_de.yaml's routing comment,
        # which names injection as one of the two CRITICAL categories) --
        # that is the routing these findings are meant to trigger.
        findings.INSTRUCTION_OVERRIDE: Severity.CRITICAL,
        findings.ROLE_PLAY_FRAMING: Severity.CRITICAL,
        findings.CROSS_TURN_ASSEMBLY: Severity.CRITICAL,
        # A long encoded run is a weaker signal on its own -- obfuscation
        # without a decoded payload proves intent to hide something, not
        # what it says -- so it is routed to a human via HIGH/handover
        # rather than auto-blocked via CRITICAL/safe_fallback.
        findings.ENCODED_PAYLOAD: Severity.HIGH,
        # Tier-1 classifier finding; this tier-0 guard never emits it, but a
        # severity is declared here for the same reason GroundingGuard
        # declares one for UNSUPPORTED_CLAIM: the finding vocabulary and its
        # severities are one contract, and a tier-1 judge sharing this
        # guard's name should not need a second place to look up "how bad".
        findings.JAILBREAK_CLASSIFIED: Severity.CRITICAL,
    }

    async def check(self, ctx: GuardContext) -> Verdict:
        started = time.perf_counter()
        config = ctx.profile.guards.injection
        text = ctx.user_message

        override_evidence = self._instruction_override(text)
        role_play_evidence = self._role_play_framing(text)
        current_turn_matched = bool(override_evidence or role_play_evidence)

        evidence: list[Evidence] = [
            *override_evidence,
            *role_play_evidence,
            *self._encoded_payload(text),
        ]

        evidence.extend(
            self._cross_turn_assembly(
                text, ctx.history, config.cross_turn_window, current_turn_matched
            )
        )

        return build_verdict(
            guard=self.name,
            stage=self.stage,
            evidence=tuple(sorted(evidence, key=lambda e: (e.span or (0, 0)))),
            defaults=self.DEFAULT_SEVERITY,
            overrides=config.severity_overrides,
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    # -- checks ---------------------------------------------------------

    @staticmethod
    def _instruction_override(text: str) -> list[Evidence]:
        return [
            Evidence(
                kind=findings.INSTRUCTION_OVERRIDE,
                detail=f"instruction-override phrase matched: {match.group(0)!r}",
                span=match.span(),
            )
            for pattern in OVERRIDE_PATTERNS
            for match in pattern.finditer(text)
        ]

    @staticmethod
    def _role_play_framing(text: str) -> list[Evidence]:
        return [
            Evidence(
                kind=findings.ROLE_PLAY_FRAMING,
                detail=f"role-reassignment phrase matched: {match.group(0)!r}",
                span=match.span(),
            )
            for pattern in ROLE_PLAY_PATTERNS
            for match in pattern.finditer(text)
        ]

    @staticmethod
    def _encoded_payload(text: str) -> list[Evidence]:
        """A long base64-looking run, or a dense cluster of ``\\uXXXX``
        escapes -- content shaped to survive a plain-text scan of the user's
        own turn undetected. See the module-level threshold comments for why
        these two specific thresholds were chosen."""
        evidence: list[Evidence] = []

        for match in _BASE64_RUN.finditer(text):
            if _looks_encoded(match.group(0)):
                evidence.append(
                    Evidence(
                        kind=findings.ENCODED_PAYLOAD,
                        detail=(
                            f"unbroken {len(match.group(0))}-character base64-alphabet "
                            f"run, longer than any legitimate field in this domain"
                        ),
                        span=match.span(),
                    )
                )

        escapes = list(_UNICODE_ESCAPE.finditer(text))
        if len(escapes) >= _UNICODE_ESCAPE_MIN_COUNT:
            evidence.append(
                Evidence(
                    kind=findings.ENCODED_PAYLOAD,
                    detail=f"{len(escapes)} unicode escape sequences in one message",
                    span=(escapes[0].start(), escapes[-1].end()),
                )
            )

        return evidence

    @staticmethod
    def _cross_turn_assembly(
        text: str,
        history: tuple[str, ...],
        window: int,
        current_turn_matched: bool,
    ) -> list[Evidence]:
        """A payload split so that no single turn trips the override or
        role-play patterns, but the recent turns concatenated do.

        Only fires when the join matches *and* no individual turn already
        did -- checked via ``current_turn_matched`` (computed once in
        :meth:`check`, not recomputed here) and by re-checking each history
        entry alone. Skipping that guard would double-report every ordinary
        single-turn attack a second time as "cross-turn", which is not what
        this finding means: it means the attack was invisible until turns
        were joined, not merely that it happens to still be visible after
        joining.
        """
        recent = history[-window:] if window > 0 else ()
        if current_turn_matched or any(_matches_override_or_roleplay(turn) for turn in recent):
            return []

        joined = " ".join((*recent, text))
        if not _matches_override_or_roleplay(joined):
            return []

        return [
            Evidence(
                kind=findings.CROSS_TURN_ASSEMBLY,
                detail=(
                    f"the last {len(recent)} turn(s) concatenated with the current "
                    f"message match an instruction-override or role-play pattern "
                    f"that no single turn matched on its own"
                ),
                # No single-turn offset applies to a finding that only exists
                # once several turns are joined.
                span=None,
            )
        ]
