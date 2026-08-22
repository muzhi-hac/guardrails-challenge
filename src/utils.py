"""共享辅助：配置加载与 trace 落盘。

成本 USD 换算不在这里 —— 它需要价格表，属于 M7。这里只把 token 数原样落盘。
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
    return Profile.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def new_run_id() -> str:
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    return f"{stamp}-{secrets.token_hex(3)}"


class TraceWriter:
    """一次运行一个 JSONL 文件。

    ``run_id`` 同时进文件名和每条记录 —— 只靠文件名表达归属的话，记录一旦被合并或
    转存就再也说不清它来自哪次运行。
    """

    def __init__(self, root: Path, run_id: str) -> None:
        self._run_id = run_id
        root.mkdir(parents=True, exist_ok=True)
        self._path = root / f"{run_id}.jsonl"

    @property
    def path(self) -> Path:
        return self._path

    def write(self, record: Mapping[str, object]) -> None:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self._run_id,
            "ts": _utc_now(),
            **record,
        }
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
