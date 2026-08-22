"""Client profiles: the configuration that makes one engine serve many clients.

A profile is the whole of what varies between deployments — persona, locale,
policy thresholds, latency budgets, and the severity-to-action routing table.
Nothing here describes *how* a check is performed; that lives in the guards.

Two layers:

``Profile``
    A direct mapping of the YAML file: base settings plus per-mode overrides.

``ResolvedProfile``
    A profile flattened for one :class:`~guardrails.types.Mode`. Guards and the
    orchestrator consume only this, so override resolution happens exactly once
    instead of being re-implemented in every guard.

Two deliberate choices worth knowing about:

*Fail-open and fail-closed are expressed as a severity*, not as a second policy
language. ``on_timeout: none`` routes through the ordinary table to
``continue`` (fail open); ``on_timeout: high`` routes to whatever the table
says for HIGH (fail closed). One mechanism covers both.

*Guard configs are named fields on* :class:`GuardsConfig`, *not a dict*. Adding
a guard therefore requires editing this module — friction on purpose, so a new
guard's configuration surface gets designed rather than accreted.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal, Self

import yaml
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    model_validator,
)

from guardrails import findings
from guardrails.types import Action, AddressForm, EntityKind, Locale, Mode, Severity

__all__ = [
    "AddressForm",
    "GroundingGuardConfig",
    "GuardConfig",
    "GuardsConfig",
    "InjectionGuardConfig",
    "Locale",
    "ModeConfig",
    "ModelConfig",
    "PersonaGuardConfig",
    "PersonaSpec",
    "PiiGuardConfig",
    "Profile",
    "ResolvedProfile",
    "load_profile",
]


def _parse_severity(value: Any) -> Any:
    """Accept ``"high"`` as well as ``3``.

    ``Severity`` is an ``IntEnum`` so that aggregation is ``max()``, but a
    profile author should never have to write the integer.
    """
    if isinstance(value, str):
        try:
            return Severity[value.upper()]
        except KeyError as exc:
            known = ", ".join(s.name.lower() for s in Severity)
            raise ValueError(f"unknown severity {value!r}; expected one of {known}") from exc
    return value


SeverityName = Annotated[Severity, BeforeValidator(_parse_severity)]
"""A severity that may be written by name in YAML. Used for values *and* for
the keys of the routing table."""


class ConfigModel(BaseModel):
    """Base for every configuration model.

    ``extra="forbid"`` is the point: a typo in a YAML key is the most dangerous
    kind of silent failure in a configuration system, because the setting the
    author believed they were applying simply is not applied.

    Mapping fields are annotated ``Mapping`` rather than ``dict`` so that static
    checkers reject item assignment. Frozen models still permit in-place
    mutation of a dict at runtime; the annotation is a partial guarantee, not a
    complete one.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


class PersonaSpec(ConfigModel):
    """The brand voice this client expects."""

    address_form: AddressForm
    emoji_allowed: bool = False
    forbidden_phrases: tuple[str, ...] = ()
    max_sentence_words: int | None = Field(default=None, gt=0)

    tts_safe: bool = False
    """Reject markup, URLs, lists and unformatted long numbers.

    Enforced by the persona guard when running in voice mode. Gating individual
    sub-checks by mode from configuration would mean another configuration
    layer for a single case; the intent is declared here and the guard applies
    it. This is the one place the design deliberately offers less
    configurability.
    """

    tone: tuple[str, ...] = ()
    """Dimensions the tier-1 judge scores, e.g. ``("formal", "empathetic")``."""


class GuardConfig(ConfigModel):
    """Settings common to every guard.

    ``on_timeout`` and ``on_error`` are ``None`` in a profile, meaning "use the
    mode default". :meth:`Profile.resolve` fills them in; after resolution they
    are never ``None``. Mirroring each guard config into a ``Resolved`` variant
    would buy that as a type-level guarantee at the cost of four more classes —
    a test pins it instead.
    """

    KNOWN_FINDINGS: ClassVar[frozenset[str]] = frozenset()

    enabled: bool = True
    on_timeout: SeverityName | None = None
    on_error: SeverityName | None = None
    severity_overrides: Mapping[str, SeverityName] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_override_keys(self) -> Self:
        unknown = set(self.severity_overrides) - self.KNOWN_FINDINGS
        if unknown:
            known = ", ".join(sorted(self.KNOWN_FINDINGS))
            raise ValueError(
                f"unknown finding kind(s) in severity_overrides: "
                f"{', '.join(sorted(unknown))}. Known kinds: {known}"
            )
        return self


class PersonaGuardConfig(GuardConfig):
    KNOWN_FINDINGS: ClassVar[frozenset[str]] = findings.PERSONA_FINDINGS

    persona: PersonaSpec


class GroundingGuardConfig(GuardConfig):
    KNOWN_FINDINGS: ClassVar[frozenset[str]] = findings.GROUNDING_FINDINGS

    check_entities: tuple[EntityKind, ...] = (
        EntityKind.NUMBER,
        EntityKind.PRICE,
        EntityKind.DATE,
        EntityKind.DURATION,
    )
    """Entity classes extracted from the reply and matched against retrieved
    context. This deterministic check is the primary mechanism; the tier-1
    entailment judge only handles what it cannot decide."""

    allowed_commitments: tuple[str, ...] = ()
    """Promises this client permits the assistant to make. Anything outside the
    list is a finding regardless of whether the knowledge base supports it."""

    escalate_below_confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class InjectionGuardConfig(GuardConfig):
    KNOWN_FINDINGS: ClassVar[frozenset[str]] = findings.INJECTION_FINDINGS

    scan_retrieved_documents: bool = True
    """Treat retrieved documents as untrusted input. In an assistant with
    retrieval the document channel, not the user turn, is the realistic
    injection vector."""

    cross_turn_window: int = Field(default=5, ge=1)
    """How many recent user turns are concatenated and re-scanned, to catch a
    payload assembled across turns."""


class PiiGuardConfig(GuardConfig):
    KNOWN_FINDINGS: ClassVar[frozenset[str]] = findings.PII_FINDINGS

    redact_inbound: bool = True
    """Redact before the text reaches logs and the model provider."""

    entities: tuple[str, ...] = ("iban", "phone", "customer_id", "birthdate")


class GuardsConfig(ConfigModel):
    """The guard set. Named fields rather than a dict, so that adding a guard
    is a deliberate edit here."""

    persona: PersonaGuardConfig
    grounding: GroundingGuardConfig
    injection: InjectionGuardConfig
    pii: PiiGuardConfig

    def _fields(self) -> tuple[str, ...]:
        return tuple(type(self).model_fields)

    def with_failure_defaults(self, *, on_timeout: Severity, on_error: Severity) -> GuardsConfig:
        """Fill each guard's unset ``on_timeout`` / ``on_error`` from the mode."""
        filled: dict[str, GuardConfig] = {}
        for name in self._fields():
            guard: GuardConfig = getattr(self, name)
            filled[name] = guard.model_copy(
                update={
                    "on_timeout": guard.on_timeout if guard.on_timeout is not None else on_timeout,
                    "on_error": guard.on_error if guard.on_error is not None else on_error,
                }
            )
        return GuardsConfig(**filled)

    def all_guards(self) -> Mapping[str, GuardConfig]:
        return {name: getattr(self, name) for name in self._fields()}


class ModeConfig(ConfigModel):
    """Per-mode settings. The latency budget is what turns "speed versus
    safety" from a paragraph into a measurable axis."""

    budget_ms: int = Field(gt=0)

    max_tier: int = Field(ge=0, le=1)
    """Caps the cascade rather than naming disabled guards: voice runs tier 0
    only. Upper bound is 1 because no tier 2 exists yet — raise it when one
    does, rather than silently accepting a value nothing implements."""

    on_timeout: SeverityName = Severity.NONE
    on_error: SeverityName = Severity.NONE
    routing: Mapping[SeverityName, Action] = Field(default_factory=dict)
    """Partial override of the profile routing table."""


class ModelConfig(ConfigModel):
    chat: str = "claude-opus-5"
    judge: str = "claude-opus-5"
    judge_effort: Literal["low", "medium", "high"] = "low"
    """Lower effort rather than disabled thinking: on current models disabling
    thinking can put a tool call in the visible text or leak internal tags, and
    effort is the supported latency control."""

    judge_min_budget_ms: int = Field(default=400, gt=0)
    """Below this, a tier-1 judge cannot complete. Configurable because a local
    or smaller judge moves the threshold."""


class ResolvedProfile(ConfigModel):
    """A profile flattened for one mode. Nothing here needs further resolution."""

    name: str
    brand_name: str
    locale: Locale
    mode: Mode
    models: ModelConfig

    budget_ms: int
    max_tier: int
    on_timeout: Severity
    on_error: Severity

    routing: Mapping[SeverityName, Action]
    guards: GuardsConfig

    fallback_message: str
    handover_message: str

    def action_for(self, severity: Severity) -> Action:
        """Map an aggregate severity onto the action this client wants."""
        return self.routing[severity]


class Profile(ConfigModel):
    """One client deployment, as written in YAML."""

    name: str
    """The profile's identifier: file naming, trace records, lookups by
    :func:`load_profile`'s caller. Never shown to a customer."""

    brand_name: str
    """What the assistant calls itself to a customer, e.g. in the system
    prompt. Distinct from ``name`` on purpose: ``name`` is ``telco_de``, an
    identifier a customer must never see; ``brand_name`` is the client's
    actual brand. Two profiles for the same client on different channels
    (``telco_de`` / ``telco_en``) share one ``brand_name``."""

    locale: Locale
    models: ModelConfig = Field(default_factory=ModelConfig)
    routing: Mapping[SeverityName, Action]
    guards: GuardsConfig
    modes: Mapping[Mode, ModeConfig]
    fallback_message: str
    handover_message: str

    @model_validator(mode="after")
    def _routing_is_total(self) -> Self:
        """Every severity must have an action.

        A missing level would silently default to letting the turn through,
        which is precisely the failure a routing table exists to prevent.
        """
        missing = set(Severity) - set(self.routing)
        if missing:
            names = ", ".join(sorted(s.name.lower() for s in missing))
            raise ValueError(f"profile routing table is missing: {names}")
        return self

    @model_validator(mode="after")
    def _tier1_fits_in_budget(self) -> Self:
        """Reject a mode that enables tier 1 inside a budget it cannot meet.

        Otherwise every judge call times out, and under a fail-open policy a
        timeout reads as approval — a guard layer that appears to run and
        silently approves everything.
        """
        for mode, cfg in self.modes.items():
            if cfg.max_tier >= 1 and cfg.budget_ms < self.models.judge_min_budget_ms:
                raise ValueError(
                    f"mode {mode.value!r} enables tier 1 with budget_ms="
                    f"{cfg.budget_ms}, below judge_min_budget_ms="
                    f"{self.models.judge_min_budget_ms}. Every judge call would "
                    f"time out. Raise the budget, set max_tier to 0, or lower "
                    f"judge_min_budget_ms if the judge is genuinely faster."
                )
        return self

    def resolve(self, mode: Mode) -> ResolvedProfile:
        if mode not in self.modes:
            available = ", ".join(sorted(m.value for m in self.modes))
            raise KeyError(f"profile {self.name!r} defines no mode {mode.value!r} (has: {available})")

        mode_cfg = self.modes[mode]
        routing = {**self.routing, **mode_cfg.routing}

        missing = set(Severity) - set(routing)
        if missing:  # pragma: no cover - unreachable while the profile table is total
            names = ", ".join(sorted(s.name.lower() for s in missing))
            raise ValueError(f"resolved routing table for {mode.value!r} is missing: {names}")

        return ResolvedProfile(
            name=self.name,
            brand_name=self.brand_name,
            locale=self.locale,
            mode=mode,
            models=self.models,
            budget_ms=mode_cfg.budget_ms,
            max_tier=mode_cfg.max_tier,
            on_timeout=mode_cfg.on_timeout,
            on_error=mode_cfg.on_error,
            routing=routing,
            guards=self.guards.with_failure_defaults(
                on_timeout=mode_cfg.on_timeout, on_error=mode_cfg.on_error
            ),
            fallback_message=self.fallback_message,
            handover_message=self.handover_message,
        )


def load_profile(path: str | Path) -> Profile:
    """Load and validate a profile from a YAML file."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level")
    return Profile.model_validate(raw)
