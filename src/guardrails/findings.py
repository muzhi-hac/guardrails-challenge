"""The finding vocabulary shared by the guards and the configuration layer.

A *finding kind* is the machine-readable label on a piece of
:class:`~guardrails.types.Evidence`. Guards emit them; profiles refer to them
when overriding severity for a particular client.

This module holds names only, no logic, which is what lets ``config`` validate
a profile's ``severity_overrides`` without importing any guard — the
dependency runs guards -> findings <- config, never guards <-> config.

Adding a finding kind is a deliberate act: register it here, and the profile
schema will accept it in ``severity_overrides``. A kind that is not registered
is rejected at load time rather than silently ignored.
"""

from __future__ import annotations

from typing import Final

# --- persona / brand voice -------------------------------------------------

ADDRESS_FORM: Final = "address_form"
"""Wrong grammatical register — in German, ``du`` where the brand uses ``Sie``."""

EMOJI: Final = "emoji"
FORBIDDEN_PHRASE: Final = "forbidden_phrase"
SENTENCE_TOO_LONG: Final = "sentence_too_long"

TTS_UNSAFE: Final = "tts_unsafe"
"""Markup, URLs, bullet lists or unformatted long numbers that a speech
synthesiser would read out literally. Only meaningful in voice mode."""

TONE: Final = "tone"
"""Tier-1 judge finding: formality, empathy or concision outside the spec."""

PERSONA_FINDINGS: Final = frozenset(
    {ADDRESS_FORM, EMOJI, FORBIDDEN_PHRASE, SENTENCE_TOO_LONG, TTS_UNSAFE}
)
TONE_FINDINGS: Final = frozenset({TONE})
"""Kept separate because ``ToneGuard`` owns this finding. Allowing it under
``persona.severity_overrides`` would accept a setting that ``PersonaGuard``
can never emit and therefore silently ignore the operator's policy."""

# --- grounding / policy facts ----------------------------------------------

UNGROUNDED_NUMBER: Final = "ungrounded_number"
UNGROUNDED_PRICE: Final = "ungrounded_price"
UNGROUNDED_DATE: Final = "ungrounded_date"
UNGROUNDED_DURATION: Final = "ungrounded_duration"

UNSUPPORTED_COMMITMENT: Final = "unsupported_commitment"
"""A promise the assistant is not authorised to make (refund, waiver, credit)."""

UNSUPPORTED_CLAIM: Final = "unsupported_claim"
"""Tier-1 entailment judge: a substantive claim the retrieved context does not
support."""

GROUNDING_FINDINGS: Final = frozenset(
    {
        UNGROUNDED_NUMBER,
        UNGROUNDED_PRICE,
        UNGROUNDED_DATE,
        UNGROUNDED_DURATION,
        UNSUPPORTED_COMMITMENT,
        UNSUPPORTED_CLAIM,
    }
)

# --- injection / jailbreak -------------------------------------------------

INSTRUCTION_OVERRIDE: Final = "instruction_override"
ROLE_PLAY_FRAMING: Final = "role_play_framing"
ENCODED_PAYLOAD: Final = "encoded_payload"

DOCUMENT_INSTRUCTION: Final = "document_instruction"
"""Imperative text inside a *retrieved document* — indirect prompt injection."""

CROSS_TURN_ASSEMBLY: Final = "cross_turn_assembly"
"""A payload that is benign per turn but forms an attack once the recent turns
are concatenated."""

JAILBREAK_CLASSIFIED: Final = "jailbreak_classified"
"""Tier-1 classifier finding."""

INJECTION_FINDINGS: Final = frozenset(
    {
        INSTRUCTION_OVERRIDE,
        ROLE_PLAY_FRAMING,
        ENCODED_PAYLOAD,
        DOCUMENT_INSTRUCTION,
        CROSS_TURN_ASSEMBLY,
        JAILBREAK_CLASSIFIED,
    }
)

# --- PII -------------------------------------------------------------------

IBAN: Final = "iban"
PHONE: Final = "phone"
CUSTOMER_ID: Final = "customer_id"
BIRTHDATE: Final = "birthdate"
ADDRESS: Final = "address"

OUTBOUND_LEAK: Final = "outbound_leak"
"""Personal data in the assistant's reply that was not in the user's own turn —
typically another customer's record surfaced through retrieval."""

PII_FINDINGS: Final = frozenset(
    {IBAN, PHONE, CUSTOMER_ID, BIRTHDATE, ADDRESS, OUTBOUND_LEAK}
)

ALL_FINDINGS: Final = (
    PERSONA_FINDINGS | TONE_FINDINGS | GROUNDING_FINDINGS | INJECTION_FINDINGS | PII_FINDINGS
)
