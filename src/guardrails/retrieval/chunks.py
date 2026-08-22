"""把文档切成检索单元。

按 Markdown 的 ``##`` 小节切，不按固定 token 数。理由直接来自失效模式：把
``Tarif M`` 和 ``29,99 EUR`` 切进两个块，溯源守卫会把一个**正确**的回答判成无依据。
这是可归因于分块策略的假阳性，和检索 miss 导致的误报是同一类错误的不同来源。

``chunk_id`` 用 slug 而不是内容哈希：trace 要人读。哈希的稳定性优势在这里是假的 ——
改一个错别字会让内容哈希全变，而 slug 只在小节标题真的改了时才变，那时候 id 变化
是正确的。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Generic, NamedTuple, TypeVar

from guardrails.locale.base import count_words
from guardrails.retrieval.documents import Document
from guardrails.types import Locale

__all__ = ["Chunk", "Scored", "chunk_document", "slugify"]

T = TypeVar("T")

_HEADING_RE = re.compile(r"^##\s+(?P<title>.+?)\s*$", re.MULTILINE)
_UMLAUT_SLUG = {"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"}


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    locale: Locale
    title: str
    section: str
    source_path: str
    text: str


class Scored(NamedTuple, Generic[T]):
    item: T
    score: float


class _Atom(NamedTuple):
    """一个不可再分的原子，带上它的种类。

    ``_atoms`` 判断出一个块是「按行」（表格行、列表项）还是「按段」（散文）之后，
    这个种类必须活着传到 ``_pack``，否则拼接时就不知道该用 ``"\\n"`` 还是
    ``"\\n\\n"`` —— 用错了空行会把跨块的表格拆散，见 Defect 3。
    """
    text: str
    kind: str  # "line" | "block"


def slugify(text: str) -> str:
    """``Tarifübersicht`` -> ``tarifuebersicht``。

    变音符按德语惯例转写而不是丢弃 —— ``tarifbersicht`` 既不可读也不可反查。
    """
    lowered = text.casefold()
    for source, target in _UMLAUT_SLUG.items():
        lowered = lowered.replace(source, target)
    ascii_only = unicodedata.normalize("NFKD", lowered).encode("ascii", "ignore").decode()
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", ascii_only)).strip("-")


def _atoms(body: str) -> list[_Atom]:
    """段落、列表项、表格行 —— 二次切分不得穿过这些单位。

    一个表格行天然是「资费名 + 价格」的最小完整单位。每个原子带上它的种类
    （``line`` 还是 ``block``），供 ``_pack`` 决定用什么分隔符重新拼接。
    """
    atoms: list[_Atom] = []
    for block in re.split(r"\n\s*\n", body.strip()):
        lines = block.splitlines()
        if all(line.lstrip().startswith(("|", "-", "*")) for line in lines if line.strip()):
            atoms.extend(_Atom(line, "line") for line in lines if line.strip())
        else:
            stripped = block.strip()
            if stripped:
                atoms.append(_Atom(stripped, "block"))
    return atoms


def chunk_document(doc: Document, *, max_words: int = 180) -> tuple[Chunk, ...]:
    sections = _sections(doc.body)
    seen: dict[str, int] = {}
    chunks: list[Chunk] = []
    for n, (section_title, section_body) in enumerate(sections):
        # 空标题（首标题前的正文，见 _sections）落到位置化的 abschnitt-{n}；
        # 同一文档内重复出现的 slug（不管是不是靠 fallback 得来的）从第二次起
        # 追加 -2、-3……第一次出现保持不加后缀，语料里现有的 chunk_id 因此不变。
        slug = slugify(section_title) or f"abschnitt-{n}"
        seen[slug] = seen.get(slug, 0) + 1
        if seen[slug] > 1:
            slug = f"{slug}-{seen[slug]}"
        base_id = f"{doc.locale.value}:{doc.doc_id}#{slug}"
        heading = f"## {section_title}" if section_title else ""
        whole = f"{heading}\n\n{section_body}".strip()
        if count_words(whole) <= max_words:
            chunks.append(_make(doc, base_id, section_title, whole))
            continue
        for index, part in enumerate(_pack(_atoms(section_body), max_words, heading)):
            chunks.append(_make(doc, f"{base_id}#{index}", section_title, part))
    return tuple(chunks)


def _sections(body: str) -> list[tuple[str, str]]:
    matches = list(_HEADING_RE.finditer(body))
    if not matches:
        return [("", body.strip())]
    out: list[tuple[str, str]] = []
    # 第一个 ``##`` 之前的正文不是「文档的一部分标题」的续篇，而是自己的一段
    # 内容——corpus 目前的约定是 body 总以标题开头，但那只是约定，不是契约
    # （见 documents.py），丢掉它就是 Defect 2。给它一个空标题的「小节」，
    # 空标题在 chunk_document 里落到 abschnitt-0。
    lead = body[:matches[0].start()].strip()
    if lead:
        out.append(("", lead))
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        out.append((match.group("title"), body[match.end():end].strip()))
    return out


def _join(atoms: list[_Atom]) -> str:
    """按种类拼回文本：连续的 ``line`` 原子之间只用单换行，其余（``block`` 原子，
    以及 block 和 line 的交界处）用空行分隔。

    表格行是 ``line`` 原子（见 ``_atoms``）；一旦在两行表格行之间插入空行，
    GFM 就不再把它们渲成同一张表 —— 这正是 Defect 3。
    """
    pieces: list[str] = []
    for i, atom in enumerate(atoms):
        if i == 0:
            pieces.append(atom.text)
            continue
        sep = "\n" if atom.kind == "line" and atoms[i - 1].kind == "line" else "\n\n"
        pieces.append(sep)
        pieces.append(atom.text)
    return "".join(pieces)


def _pack(atoms: list[_Atom], max_words: int, heading: str) -> list[str]:
    """把原子装进不超过 ``max_words`` 的片，每片都带上小节标题前缀。

    标题前缀让片段自身可读，也让 BM25 仍能通过标题词命中这一片。
    """
    parts: list[str] = []
    current: list[_Atom] = []
    # 标题很长时 budget 可能非正；spec 要求原子（段落/表格行）永不被切分，
    # 所以第一个原子总是无条件接受 —— 这里宁可让单个片超预算，也不违反原子性。
    budget = max_words - count_words(heading)
    for atom in atoms:
        if current and count_words(_join(current)) + count_words(atom.text) > budget:
            parts.append(f"{heading}\n\n{_join(current)}".strip())
            current = []
        current.append(atom)
    if current:
        parts.append(f"{heading}\n\n{_join(current)}".strip())
    return parts


def _make(doc: Document, chunk_id: str, section: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc.doc_id,
        locale=doc.locale,
        title=doc.title,
        section=section,
        source_path=doc.source_path,
        text=text,
    )
