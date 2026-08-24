"""Grounding guard: does the reply state only what retrieval actually supports?

Two checks, both tier 0 — pattern matching and set membership, no model call.

The **entity check** answers a narrower question than "is this reply true":
it asks whether every number, price, date and duration the reply states also
appears, in some form, among the chunks retrieval returned for this turn. That
is deliberately weaker than entailment — a chunk that mentions ``29,99 EUR``
in an unrelated context "grounds" a reply that misuses the figure — and the
tier-1 entailment judge (:data:`~guardrails.findings.UNSUPPORTED_CLAIM`,
gated on :attr:`~guardrails.config.GroundingGuardConfig.escalate_below_confidence`)
exists precisely to catch what this cannot. But cheap, high-precision, no
network call, and it is exactly this weaker question — "did the model invent
a number" — that produces the large majority of ungrounded-fact incidents in
a retrieval-augmented assistant. Building the expensive check first and never
shipping the cheap one would leave every one of those incidents uncaught while
the harder problem is still being worked on.

Comparing normalised values rather than raw text is the whole point of
routing this through :meth:`~guardrails.locale.base.LocaleRules.extract_entities`
instead of a substring search: the German channel writes ``29,99 EUR`` and the
English one writes ``29.99 EUR``. Both extract to ``"29.99 EUR"`` — but only
if each string is parsed with the rules that understand its own punctuation
convention. Parsing ``29,99 EUR`` with the *English* comma-as-thousands
grammar does not fail loudly; it silently mis-extracts ``99 EUR`` (English
reads the comma as a thousands separator, drops the leading digits, and
happily returns a wrong-but-well-formed price). That is why retrieved
documents are extracted with **every registered locale's rules**, unioned,
rather than only the reply's own: retrieval is over untrusted external
content that is not guaranteed to share the reply's punctuation convention —
a mirrored document, a pasted note, a legacy record — and assuming it does
would turn a correctly-quoted foreign-format figure into a false
``ungrounded_price``. The reply itself is extracted only with the profile's
own locale rules, because the reply is this deployment's own output and is
expected to conform to the convention it was asked to write in; extracting it
under a locale it was never written in would manufacture entities that were
never actually said.

The **commitment check** is a different failure mode from the entity check
and is kept separate rather than folded into it. An ungrounded price is a
wrong statement; an unsupported commitment (a refund, a fee waiver, a credit)
is a promise the assistant had no authority to make — it creates an
obligation regardless of whether it happens to be numerically consistent with
the retrieved context. There is nothing to normalise and compare here, only
whether the assistant said it at all, so this check does not touch
``retrieved``; it checks the reply against
:attr:`~guardrails.config.GroundingGuardConfig.allowed_commitments`, the
client's own authorisation list.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import ClassVar

from guardrails import findings
from guardrails.guards.base import GuardContext, build_verdict
from guardrails.locale import get_rules, supported_locales
from guardrails.types import EntityKind, Evidence, Severity, Stage, Verdict

__all__ = ["GroundingGuard"]

_ENTITY_FINDING: Mapping[EntityKind, str] = {
    EntityKind.NUMBER: findings.UNGROUNDED_NUMBER,
    EntityKind.PRICE: findings.UNGROUNDED_PRICE,
    EntityKind.DATE: findings.UNGROUNDED_DATE,
    EntityKind.DURATION: findings.UNGROUNDED_DURATION,
}


class GroundingGuard:
    """Checks a reply's stated facts and promises against what this turn's
    retrieval and this client's policy actually support.

    ``escalate_below_confidence`` on :class:`~guardrails.config.GroundingGuardConfig`
    is not read here. It belongs to the tier-1 entailment judge that emits
    :data:`~guardrails.findings.UNSUPPORTED_CLAIM` — deciding whether a claim
    the entity check cannot characterise (no extractable quantity, but still
    a substantive assertion) is actually supported requires reading the
    chunks for meaning, which is exactly what a tier-0 guard does not do.
    """

    name: ClassVar[str] = "grounding"
    stage: ClassVar[Stage] = Stage.OUTPUT
    tier: ClassVar[int] = 0

    DEFAULT_SEVERITY: ClassVar[Mapping[str, Severity]] = {
        findings.UNGROUNDED_NUMBER: Severity.MEDIUM,
        findings.UNGROUNDED_PRICE: Severity.HIGH,
        findings.UNGROUNDED_DATE: Severity.HIGH,
        findings.UNGROUNDED_DURATION: Severity.HIGH,
        findings.UNSUPPORTED_COMMITMENT: Severity.HIGH,
        findings.UNSUPPORTED_CLAIM: Severity.HIGH,
    }

    async def check(self, ctx: GuardContext) -> Verdict:
        started = time.perf_counter()
        config = ctx.profile.guards.grounding
        text = ctx.reply

        evidence: list[Evidence] = [
            *self._entity_grounding(text, ctx),
            *self._commitments(text, ctx),
        ]

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
    def _entity_grounding(text: str, ctx: GuardContext) -> list[Evidence]:
        """Every checked-kind entity in the reply must reappear, normalised,
        among the entities extracted from the retrieved chunks.

        Extracting entities from each chunk separately and unioning their
        normalised forms — rather than concatenating chunk text and running
        extraction once — keeps every mention's provenance independent of
        chunk order and boundary placement; ``extract_entities`` already
        resolves overlaps within one string, and there is no reason to make
        that resolution span chunk boundaries it was never designed to see.

        Chunks are parsed with every registered locale's rules, not only
        ``ctx.rules`` — see the module docstring for why a single locale's
        grammar cannot be trusted to parse content it did not write.
        """
        kinds = frozenset(ctx.profile.guards.grounding.check_entities)
        if not kinds:
            return []

        grounded = {
            mention.normalized
            for chunk in ctx.retrieved
            for locale in supported_locales()
            for mention in get_rules(locale).extract_entities(chunk, kinds)
        }

        return [
            Evidence(
                kind=_ENTITY_FINDING[mention.kind],
                detail=(
                    f"{mention.kind.value} {mention.raw!r} does not appear in "
                    f"the retrieved context for this turn"
                ),
                span=mention.span,
            )
            for mention in ctx.rules.extract_entities(text, kinds)
            if mention.normalized not in grounded
        ]

    @staticmethod
    def _commitments(text: str, ctx: GuardContext) -> list[Evidence]:
        """A promise outside ``allowed_commitments`` is a finding regardless
        of whether the knowledge base would have supported it — authorisation
        to make a commitment is a policy fact, not a retrieval fact."""
        allowed = frozenset(ctx.profile.guards.grounding.allowed_commitments)
        return [
            Evidence(
                kind=findings.UNSUPPORTED_COMMITMENT,
                detail=(
                    f"commitment {hit.commitment_id!r} ({hit.raw!r}) is not on "
                    f"this client's allowed_commitments list"
                ),
                span=hit.span,
            )
            for hit in ctx.rules.find_commitments(text)
            if hit.commitment_id not in allowed
        ]
