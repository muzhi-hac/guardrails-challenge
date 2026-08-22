"""知识库文档的加载。

Markdown + front-matter，一篇一个文件。选 Markdown 而不是 JSON：溯源守卫的
evidence 要引用原文，人可读的语料让评测结果可以肉眼复核。选 front-matter 而不是
单独的 manifest：locale 跟着文档走，加一篇文档不需要改第二个文件。

文档的唯一键是 ``(locale, doc_id)``。德英镜像共用逻辑 ``doc_id`` —— 它们是同一个
客户的两个渠道，不是两份内容。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from guardrails.types import Locale

__all__ = ["Document", "load_documents"]

_FRONT_MATTER = "---"


@dataclass(frozen=True, slots=True)
class Document:
    doc_id: str
    locale: Locale
    title: str
    version: str
    source_path: str
    body: str


def _parse(path: Path, root: Path) -> Document:
    text = path.read_text(encoding="utf-8")
    if not text.startswith(_FRONT_MATTER):
        raise ValueError(f"{path}: missing front matter")
    _, raw_meta, body = text.split(_FRONT_MATTER, 2)
    meta = yaml.safe_load(raw_meta) or {}
    missing = {"doc_id", "title", "locale", "version"} - set(meta)
    if missing:
        raise ValueError(f"{path}: front matter missing {sorted(missing)}")
    return Document(
        doc_id=str(meta["doc_id"]),
        locale=Locale(str(meta["locale"])),
        title=str(meta["title"]),
        version=str(meta["version"]),
        source_path=str(path.relative_to(root.parent)),
        body=body.strip(),
    )


def load_documents(root: Path) -> tuple[Document, ...]:
    """加载 ``root`` 下的全部文档，按 ``(locale, doc_id)`` 稳定排序。

    排序是刻意的：文件系统的遍历顺序在不同机器上不同，而 chunk 的构建顺序会流进
    trace。和 M5「verdict 按注册顺序而非完成顺序」是同一条原则。
    """
    docs = [_parse(path, root) for path in sorted(root.rglob("*.md"))]
    return tuple(sorted(docs, key=lambda d: (d.locale.value, d.doc_id)))
