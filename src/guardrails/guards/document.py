"""Document guard: does a retrieved knowledge-base chunk try to give the
assistant an instruction, rather than describe policy to a reader?

Runs at :data:`~guardrails.types.Stage.RETRIEVAL`, after retrieval has
returned but before generation. This check used to live inside
:class:`~guardrails.guards.injection.InjectionGuard`, because a guard has
exactly one stage and that guard needed to inspect both the user's own turn
and the document channel. It is a separate guard now because the two
channels are not the same threat.

**Different threat, different trust assumption.** The user's own message is
adversarial by default -- see
:mod:`~guardrails.guards.injection`'s docstring. A retrieved document is
trusted infrastructure in this system's threat model: retrieval is not
re-checked by a human per turn, and the corpus does not change on every
request the way an attacker's phrasing does. A chunk that reuses an
instruction-override or role-play construction, or opens a line with a bare
``SYSTEM:`` marker, therefore means something different from the same
construction in a user's turn -- either the document has been compromised
(a supply-chain concern: who can write to the knowledge base, and when) or
it was mis-authored by whoever wrote the policy text. Both are findings
worth routing differently than "a customer typed a jailbreak attempt," which
is why this is its own guard with its own finding
(:data:`~guardrails.findings.DOCUMENT_INSTRUCTION`) and its own
:class:`~guardrails.config.DocumentGuardConfig`, rather than a shared
severity table with the user-turn checks.

This is the guard-side half of the defence documented in
examples/example_runs.md #6, where the model-side half is the per-turn nonce
delimiter (see ``Chatbot._render_user_turn``) that keeps a document from
being able to close its own untrusted region early.

Reuses :data:`guardrails.guards.injection.OVERRIDE_PATTERNS` and
:data:`~guardrails.guards.injection.ROLE_PLAY_PATTERNS` rather than a second
pattern set: an instruction is an instruction whether it addresses "you"
from the user's turn or from a document, and duplicating every pattern would
mean fixing every false positive twice. Only the document-specific marker
patterns (a bare ``SYSTEM:`` line, an explicit "tell the customer") are
declared here.
"""

from __future__ import annotations

import re
import time
from collections.abc import Mapping
from typing import ClassVar

from guardrails import findings
from guardrails.guards.base import GuardContext, build_verdict
from guardrails.guards.injection import OVERRIDE_PATTERNS, ROLE_PLAY_PATTERNS
from guardrails.types import Evidence, Severity, Stage, Verdict

__all__ = ["DocumentGuard"]

# -- document-channel markers ---------------------------------------------
#
# Applied only to retrieved documents. A legitimate knowledge-base chunk
# describes policy in the third person -- "Der Kunde muss ..." -- it never
# opens a line with a bare "SYSTEM:" role marker or tells the assistant what
# to relay to "the customer" in the second person; only an attempt to
# smuggle a directive past the retrieval boundary does that, so these two
# are safe to check with no further qualification.
_DOCUMENT_MARKER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bSYSTEM:\s"),
    re.compile(r"you\s+must\s+tell\s+the\s+customer", re.IGNORECASE),
)

_PATTERNS: tuple[re.Pattern[str], ...] = (
    *OVERRIDE_PATTERNS,
    *ROLE_PLAY_PATTERNS,
    *_DOCUMENT_MARKER_PATTERNS,
)


class DocumentGuard:
    """Detects a retrieved document that addresses the assistant directly
    instead of describing policy to a reader -- indirect prompt injection
    carried in through the corpus.

    See the module docstring for why this is a guard of its own rather than
    a second check inside :class:`~guardrails.guards.injection.InjectionGuard`.
    """

    name: ClassVar[str] = "document"
    stage: ClassVar[Stage] = Stage.RETRIEVAL
    tier: ClassVar[int] = 0

    DEFAULT_SEVERITY: ClassVar[Mapping[str, Severity]] = {
        # CRITICAL for the same reason InjectionGuard's user-turn findings
        # are: the telco_de profile routes CRITICAL to safe_fallback so that
        # a reply generated on a poisoned chunk's instructions never reaches
        # the user or a human agent's queue verbatim.
        findings.DOCUMENT_INSTRUCTION: Severity.CRITICAL,
    }

    async def check(self, ctx: GuardContext) -> Verdict:
        started = time.perf_counter()
        config = ctx.profile.guards.document

        evidence: list[Evidence] = []
        if config.scan_retrieved_documents:
            evidence.extend(self._document_instructions(ctx.retrieved))

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
    def _document_instructions(retrieved: tuple[str, ...]) -> list[Evidence]:
        """Imperative text inside a retrieved document that addresses the
        assistant rather than describing policy to a reader.

        See the module docstring for why this reuses the override/role-play
        patterns from :mod:`guardrails.guards.injection` rather than
        re-deriving them.
        """
        evidence: list[Evidence] = []
        for document in retrieved:
            for pattern in _PATTERNS:
                for match in pattern.finditer(document):
                    evidence.append(
                        Evidence(
                            kind=findings.DOCUMENT_INSTRUCTION,
                            detail=(
                                f"retrieved document addresses the assistant "
                                f"directly: {match.group(0)!r}"
                            ),
                            # Indexes into the chunk's own text, not the
                            # user's turn -- the finding originates in
                            # retrieved context, so there is no offset into
                            # ctx.user_message for it to mean.
                            span=match.span(),
                        )
                    )
        return evidence
