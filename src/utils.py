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
    """读取并校验一份 profile YAML。

    ``read_text`` → ``yaml.safe_load`` → ``Profile.model_validate`` 三步任何一步
    失败，裸抛出的 ``yaml.YAMLError`` 或 pydantic ``ValidationError`` 都不会带上
    是哪个文件出的错 —— 排查时只能靠 traceback 猜。这里统一补上路径，和
    ``guardrails/retrieval/documents.py`` 里 ``_parse`` 对失败做的事一样。
    ``FileNotFoundError`` 本身已经点名了路径，不需要也不应该再包一层让它变得
    含糊，所以原样透传。
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
    """生成一次运行的 id：UTC 时间戳 + 随机字节。

    只用时间戳，同一秒内并发跑两次就会撞；只用随机字节，文件按名字排序时就没有
    时间顺序、翻旧账全靠改文件时间。两者拼在一起：时间戳给人读顺序，随机后缀
    在同一秒内也不会碰撞。
    """
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

    def write(self, record: Mapping[str, object], *, error: BaseException | None = None) -> None:
        """追加一条 JSONL 记录。

        ``record`` 是调用方想记的内容；``schema_version``、``run_id``、``ts``、
        ``error_type`` 是 writer 自己的出处字段，调用方即便传了同名 key 也不能覆盖它们 ——
        这些字段存在的意义就是让记录的出处可信：``schema_version`` 让评测脚本能拒绝
        旧格式而不是误读，``run_id`` 让记录被合并或转存后还能说清来自哪次运行，
        ``ts`` 让记录本身而不是文件系统时间戳成为时间依据。``error_type`` 让 writer
        而不是调用方控制异常信息，从而防止消息中的敏感信息泄露。所以字典字面量里把它们
        放在 ``**record`` 之后，同名 key 以 writer 为准。

        ``error`` 是可选的异常对象：传了它，writer 自己从 ``type(error).__name__``
        取出类型名写进 ``error_type``，绝不碰 ``str(error)`` —— 和
        ``guardrails/pipeline.py`` 里的纪律一致：trace 只记异常类型，不记消息，
        因为消息可能带着用户输入、prompt 片段或凭证。``error_type`` 完全由 writer
        掌控：调用方即便只传 ``error_type`` key 不传 ``error=``，也会被拒绝。这样可以
        堵死调用方写成 ``{"error_type": str(exc)}`` 绕过规则的旁路，逼着调用方改成
        传 ``error=exc``。记录总是包含 ``error_type`` 字段：有异常时是类型名，无异常时
        是 ``None``，让读者能区分"这一轮没有异常"和"这条记录来自更早的版本"。
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
