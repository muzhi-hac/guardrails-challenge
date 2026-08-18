"""Language rules, selected by locale.

The registry is explicit rather than discovered by import scanning: adding a
language means adding a line here, which is where a reviewer looks to find out
what is supported.
"""

from __future__ import annotations

from guardrails.locale.base import (
    AddressFormHit,
    EntityMention,
    LocaleRules,
    Sentence,
    count_words,
)
from guardrails.locale.de import GermanRules
from guardrails.locale.en import EnglishRules
from guardrails.types import Locale

__all__ = [
    "AddressFormHit",
    "EntityMention",
    "EnglishRules",
    "GermanRules",
    "LocaleRules",
    "Sentence",
    "count_words",
    "get_rules",
    "supported_locales",
]

_REGISTRY: dict[Locale, LocaleRules] = {
    Locale.DE_DE: GermanRules(),
    Locale.EN_GB: EnglishRules(),
}


def get_rules(locale: Locale) -> LocaleRules:
    """Return the rules for ``locale``, or raise naming what is available."""
    try:
        return _REGISTRY[locale]
    except KeyError as exc:
        known = ", ".join(sorted(loc.value for loc in _REGISTRY))
        raise KeyError(f"no language rules for {locale.value!r}; have: {known}") from exc


def supported_locales() -> tuple[Locale, ...]:
    return tuple(_REGISTRY)
