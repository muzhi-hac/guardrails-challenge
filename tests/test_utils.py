"""Trace 落盘。

schema_version 让评测脚本能拒绝旧格式而不是误读；run_id 是**记录内的字段**而不只是
文件名，否则记录被合并或转存后就失去了归属。

error_type 只记异常类型名不记 message —— 和 M5 的 ERROR verdict 同一条纪律，那里也有
一条测试断言 message 不出现在序列化结果里。
"""

from __future__ import annotations

import json
from pathlib import Path

from utils import TraceWriter, new_run_id


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
        writer.write({"error_type": type(exc).__name__})

    body = (tmp_path / "r2.jsonl").read_text(encoding="utf-8")
    assert "ValueError" in body
    assert "streng geheim" not in body
