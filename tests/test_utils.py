"""Writing traces to disk.

schema_version lets an evaluation script reject an old format instead of misreading it;
run_id is a **field inside the record itself**, not just the filename, or a record would
lose its provenance once merged or copied elsewhere.

error_type records only the exception type name, never the message -- the same
discipline as M5's ERROR verdict, where a test likewise asserts that the message never
appears in the serialized result.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from utils import TraceWriter, load_profile, new_run_id


def test_run_id_is_unique():
    assert new_run_id() != new_run_id()


def test_writes_jsonl_with_schema_and_run_id(tmp_path: Path):
    writer = TraceWriter(tmp_path, run_id="r1")
    writer.write({"query": "Was kostet Tarif M?"})
    writer.write({"query": "Und Tarif L?"})

    lines = (tmp_path / "r1.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["schema_version"] == 1
    assert first["run_id"] == "r1"
    assert first["ts"].endswith("Z")
    assert first["query"] == "Was kostet Tarif M?"


def test_error_type_only_never_the_message(tmp_path: Path):
    writer = TraceWriter(tmp_path, run_id="r2")
    try:
        raise ValueError("streng geheim")
    except ValueError as exc:
        writer.write({"query": "x"}, error=exc)

    body = (tmp_path / "r2.jsonl").read_text(encoding="utf-8")
    assert "ValueError" in body
    assert "streng geheim" not in body


def test_caller_cannot_forge_provenance(tmp_path: Path):
    """schema_version/run_id/ts are the writer's own provenance fields; a caller
    passing a key of the same name cannot override them."""
    writer = TraceWriter(tmp_path, run_id="echte-run-id")
    writer.write(
        {
            "run_id": "gefaelscht",
            "schema_version": 999,
            "ts": "irgendwann",
            "query": "x",
        }
    )

    first = json.loads((tmp_path / "echte-run-id.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert first["run_id"] == "echte-run-id"
    assert first["schema_version"] == 1
    assert first["ts"] != "irgendwann"
    assert first["ts"].endswith("Z")
    assert first["query"] == "x"


def test_error_kwarg_records_type_not_message(tmp_path: Path):
    writer = TraceWriter(tmp_path, run_id="r3")
    try:
        raise ValueError("streng geheimes Passwort 12345")
    except ValueError as exc:
        writer.write({"query": "x"}, error=exc)

    body = (tmp_path / "r3.jsonl").read_text(encoding="utf-8")
    record = json.loads(body.strip().splitlines()[0])
    assert record["error_type"] == "ValueError"
    assert "geheim" not in body


def test_error_kwarg_conflicts_with_explicit_error_type(tmp_path: Path):
    writer = TraceWriter(tmp_path, run_id="r4")
    with pytest.raises(ValueError, match="error_type"):
        writer.write({"error_type": "ValueError"}, error=ValueError("y"))


def test_error_type_in_record_rejected_without_error_kwarg(tmp_path: Path):
    writer = TraceWriter(tmp_path, run_id="r4b")
    with pytest.raises(ValueError, match="error_type"):
        writer.write({"error_type": "ValueError: streng geheimes Passwort 12345"})


def test_write_without_error_still_works(tmp_path: Path):
    writer = TraceWriter(tmp_path, run_id="r5")
    writer.write({"query": "kein Fehler"})

    record = json.loads((tmp_path / "r5.jsonl").read_text(encoding="utf-8").strip().splitlines()[0])
    assert record["error_type"] is None
    assert record["query"] == "kein Fehler"


PROFILES = Path(__file__).resolve().parents[1] / "profiles"


def test_load_profile_loads_a_shipped_profile():
    profile = load_profile(PROFILES / "telco_de.yaml")
    assert profile.name


def test_load_profile_malformed_yaml_names_the_file(tmp_path: Path):
    bad = tmp_path / "bad-yaml.yaml"
    bad.write_text("name: [unclosed\n", encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape(str(bad))):
        load_profile(bad)


def test_load_profile_validation_failure_names_the_file(tmp_path: Path):
    """When parsing succeeds but Profile.model_validate rejects it, the file must
    still be named."""
    bad = tmp_path / "incomplete.yaml"
    bad.write_text("name: nur-ein-feld\n", encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape(str(bad))):
        load_profile(bad)
