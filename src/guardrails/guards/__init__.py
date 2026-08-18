"""Guards, and the interface they share.

The registry is explicit: adding a guard means adding a line here and a field on
:class:`~guardrails.config.GuardsConfig`. Both are deliberate friction, so that
a new guard's configuration surface gets designed rather than accreted.
"""

from __future__ import annotations

from guardrails.guards.base import Guard, GuardContext, build_verdict, effective_severity
from guardrails.guards.persona import PersonaGuard

__all__ = [
    "Guard",
    "GuardContext",
    "PersonaGuard",
    "build_verdict",
    "effective_severity",
    "get_guard",
    "registered_guards",
]

_REGISTRY: dict[str, Guard] = {
    PersonaGuard.name: PersonaGuard(),
}


def get_guard(name: str) -> Guard:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        known = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"no guard named {name!r}; have: {known}") from exc


def registered_guards() -> tuple[str, ...]:
    return tuple(_REGISTRY)
