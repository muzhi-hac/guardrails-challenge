"""检索 recall 的固定查询集。

判定按 ``doc_id``，不绑定 ``chunk_id``，也不绑定完整排序：同一文档的哪个小节被召回
是分块策略的实现细节，写进断言会让分块的每次调整都误报成检索退化。

一个查询允许有多个正确文档 —— ``tarife-mobilfunk`` 与 ``vertragslaufzeit-kuendigung``
必然共享期限事实，强行指定唯一正确文档等于把内容组织方式写进断言。

**改写型提问是刻意放进来的**：那正是纯词汇检索的已知弱点。测出来记为局限，比回避
它更有用。
"""

from __future__ import annotations

from typing import NamedTuple

from guardrails.types import Locale


class RecallCase(NamedTuple):
    query: str
    expected_doc_ids: frozenset[str]
    locale: Locale
    paraphrase: bool = False
    """True 表示这条不含语料里的关键词，测的是词汇检索的弱点边界。"""


DE = Locale.DE_DE
EN = Locale.EN_GB

RECALL_QUERIES: tuple[RecallCase, ...] = (
    # --- 德语：精确词 ---
    RecallCase("Was kostet Tarif M?", frozenset({"tarife-mobilfunk"}), DE),
    RecallCase("Wie hoch ist die Kündigungsfrist?",
               frozenset({"vertragslaufzeit-kuendigung"}), DE),
    RecallCase("Mindestlaufzeit meines Vertrags",
               frozenset({"vertragslaufzeit-kuendigung", "tarife-mobilfunk"}), DE),
    RecallCase("Roaming außerhalb der EU Kosten", frozenset({"roaming-eu"}), DE),
    RecallCase("Wann kommt meine Rechnung?", frozenset({"rechnung-zahlungsarten"}), DE),
    RecallCase("Entstörfrist bei Störung", frozenset({"stoerung-entstoerfrist"}), DE),
    RecallCase("Rufnummernmitnahme Dauer", frozenset({"rufnummernmitnahme"}), DE),
    RecallCase("Wann beginnt die Drosselung meines Datenvolumens?",
               frozenset({"datenvolumen-drosselung"}), DE),
    RecallCase("Widerrufsfrist 14 Tage", frozenset({"widerrufsrecht"}), DE),
    RecallCase("Wann erreiche ich den Kundenservice?", frozenset({"servicezeiten"}), DE),
    # --- 德语：ASCII 变音符输入（真实用户行为）---
    RecallCase("Kuendigung Frist", frozenset({"vertragslaufzeit-kuendigung"}), DE),
    # --- 德语：改写型（已知弱点）---
    RecallCase("Wie komme ich aus meinem Vertrag heraus?",
               frozenset({"vertragslaufzeit-kuendigung"}), DE, paraphrase=True),
    RecallCase("Ich ziehe um, was passiert mit meinem Anschluss?",
               frozenset({"umzug"}), DE, paraphrase=True),
    RecallCase("Mein Internet ist seit gestern weg",
               frozenset({"stoerung-entstoerfrist"}), DE, paraphrase=True),
    # --- 英语 ---
    RecallCase("How much does Tariff M cost?", frozenset({"tarife-mobilfunk"}), EN),
    RecallCase("notice period for cancellation",
               frozenset({"vertragslaufzeit-kuendigung"}), EN),
    RecallCase("roaming outside the EU", frozenset({"roaming-eu"}), EN),
    RecallCase("when is my invoice issued", frozenset({"rechnung-zahlungsarten"}), EN),
    RecallCase("how long to fix a fault", frozenset({"stoerung-entstoerfrist"}), EN),
    RecallCase("I am moving house", frozenset({"umzug"}), EN, paraphrase=True),
)
