"""Shared helpers: configuration loading and writing traces to disk.

The cost-to-USD conversion does not live here — it needs a price table, and
that belongs with a dated price table. Here token counts are written as-is.
"""

from __future__ import annotations

import json
import secrets
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import yaml

from guardrails.config import Profile

__all__ = ["SCHEMA_VERSION", "TraceWriter", "load_profile", "new_run_id"]

SCHEMA_VERSION = 1


def load_profile(path: Path) -> Profile:
    """Read and validate one profile YAML file.

    If any of the three steps — ``read_text`` → ``yaml.safe_load`` →
    ``Profile.model_validate`` — fails, the bare ``yaml.YAMLError`` or
    pydantic ``ValidationError`` raised does not name which file was at
    fault; debugging it would mean guessing from the traceback alone. The
    path is attached uniformly here, the same thing ``_parse`` does for
    failures in ``guardrails/retrieval/documents.py``. ``FileNotFoundError``
    already names the path itself, and wrapping it further would only make
    it vaguer without needing to — so it passes through unchanged.
    """
    try:
        return Profile.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except FileNotFoundError:
        raise
    except Exception as exc:
        raise ValueError(f"{path}: {exc}") from exc


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def new_run_id() -> str:
    """Generate one run's id: a UTC timestamp plus random bytes.

    Timestamp alone collides when two runs happen concurrently within the
    same second; random bytes alone leave files with no chronological order
    when sorted by name, forcing anyone digging through old runs to rely on
    file modification times. The two together: the timestamp gives a human a
    readable order, and the random suffix keeps runs within the same second
    from colliding.
    """
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    return f"{stamp}-{secrets.token_hex(3)}"


class TraceWriter:
    """One JSONL file per run.

    ``run_id`` goes into both the filename and every individual record —
    relying on the filename alone to express provenance means a record can
    no longer say which run it came from, the moment records get merged or
    copied elsewhere.
    """

    def __init__(self, root: Path, run_id: str) -> None:
        self._run_id = run_id
        root.mkdir(parents=True, exist_ok=True)
        self._path = root / f"{run_id}.jsonl"

    @property
    def path(self) -> Path:
        return self._path

    def write(self, record: Mapping[str, object], *, error: BaseException | None = None) -> None:
        """Append one JSONL record.

        ``record`` is whatever the caller wants written; ``schema_version``,
        ``run_id``, ``ts`` and ``error_type`` are the writer's own
        provenance fields, and the caller cannot override them even by
        passing a key of the same name. These fields exist for exactly one
        reason: to keep a record's provenance trustworthy.
        ``schema_version`` lets an evaluation script reject an old format
        instead of misreading it; ``run_id`` lets a record still say which
        run it came from after being merged or copied elsewhere; ``ts``
        makes the record itself, rather than a filesystem timestamp, the
        basis for time. ``error_type`` puts the writer, not the caller, in
        control of exception information, which is what prevents sensitive
        detail in an exception message from leaking out. So the dict
        literal places them after ``**record`` — where a key collides, the
        writer wins.

        ``error`` is an optional exception object: pass it, and the writer
        pulls the type name from ``type(error).__name__`` itself and writes
        it into ``error_type``, never touching ``str(error)`` — the same
        discipline as ``guardrails/pipeline.py``: a trace records only the
        exception *type*, never the message, because the message may carry
        user input, a prompt fragment, or a credential. ``error_type`` is
        entirely under the writer's control: a caller that passes an
        ``error_type`` key without also passing ``error=`` is rejected. This
        closes off the loophole of writing ``{"error_type": str(exc)}`` to
        route around the rule, forcing the caller to pass ``error=exc``
        instead. A record always includes an ``error_type`` field: the type
        name when there was an exception, ``None`` when there wasn't — so a
        reader can tell "this turn had no exception" apart from "this
        record came from an earlier version [that didn't write the field]".
        """
        if "error_type" in record:
            raise ValueError(
                "record already has 'error_type'; pass the exception via error= instead"
            )
        payload: dict[str, object] = {**record}
        payload["error_type"] = type(error).__name__ if error is not None else None
        payload["schema_version"] = SCHEMA_VERSION
        payload["run_id"] = self._run_id
        payload["ts"] = _utc_now()
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
