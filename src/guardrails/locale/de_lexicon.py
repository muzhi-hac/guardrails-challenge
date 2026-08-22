"""The domain lexicon for German retrieval.

This module is the central asset of German tokenization: the same word list
both drives compound decomposition and verifies stemming. Tying the two to
one shared piece of data is deliberate — a constituent that can be split out
is exactly the same thing as a stem that can be validated after stripping a
suffix; two separate word lists would drift apart from each other.

Coverage matches the current corpus. Extending the corpus requires extending
the lexicon in step, or a new compound degrades to whole-word matching (a
conservative failure direction — it costs recall, it does not produce false
positives).
"""

from __future__ import annotations

from typing import Final

LEXICON: Final[frozenset[str]] = frozenset(
    {
        # Contract and cancellation
        "vertrag", "kündigung", "frist", "laufzeit", "mindest", "sonder",
        "widerruf", "recht", "partner", "wechsel", "termin", "verlängerung",
        # Tariff and billing
        "tarif", "preis", "betrag", "rechnung", "zahlung", "art", "gebühr",
        "konto", "lastschrift", "verzug", "monat", "jahr", "woche", "tag",
        # Product
        "mobilfunk", "daten", "volumen", "netz", "abdeckung", "karte",
        "roaming", "gerät", "finanzierung", "anschluss", "nummer", "mitnahme",
        "ruf", "drosselung", "geschwindigkeit",
        # Service
        "service", "zeit", "kunde", "störung", "meldung", "entstörung", "entstör",
        "umzug", "auskunft", "schutz", "adresse",
    }
)

LINKING_MORPHEMES: Final[tuple[str, ...]] = ("es", "en", "s", "n")
"""Linking morphemes, tried longest first: Kündigung|s|frist, Rufnummer|n|mitnahme."""

INFLECTION_SUFFIXES: Final[tuple[str, ...]] = ("en", "es", "er", "e", "n", "s")
"""Inflectional suffixes. Used by ``_stem_alias`` (stripping inflection from a
standalone word) and by ``_decompose`` (the **last constituent** of a
compound may carry inflection). Both sites require that the stripped result
**hit the LEXICON** before it is produced."""
