"""检索 recall 的固定查询集。

判定按 ``doc_id``，不绑定 ``chunk_id``，也不绑定完整排序：同一文档的哪个小节被召回
是分块策略的实现细节，写进断言会让分块的每次调整都误报成检索退化。

一个查询允许有多个正确文档 —— ``tarife-mobilfunk`` 与 ``vertragslaufzeit-kuendigung``
必然共享期限事实，强行指定唯一正确文档等于把内容组织方式写进断言。

**改写型提问是刻意放进来的**：``known_limitation`` 按查询与文档的词汇关系分类
——查询用的是不是文档自己的术语——而不是按"这条会不会命中"的预期难度分类。分类
只描述输入，命中与否留给测量去报告，这样测出来的数字才是发现，不是分类时就定
好的结论。

**派生形态是第二类已知局限，边界不同于改写。** 德语分词器对屈折变化做词典校验的
还原（如名词的格/数后缀），但不做派生形态还原：动词分词（``gedrosselt``）永远到
不了对应的名词（``Drosselung``），因为词干化只剥离已登记的名词屈折后缀，再拿结
果去词典核对——``gedrosselt`` 本身不是词典条目，也没有剥离动词 "ge-" 前缀的机制。
"Ab wann wird gedrosselt?" 和 "Wann beginnt die Drosselung meines Datenvolumens?"
是同一个客户意图的两种问法，前者是真实、自然的德语提问，只是恰好踩在这条派生形
态的边界上。两种问法都留在集合里，是故意的：只留下能命中的那个问法，测不出边界
在哪里；两个都留，边界就写在数字里。

**测量结果**：五条局限用例里，三条在 k=5 命中，其中两条命中在第 1 位（见
``test_recall_report`` 打印的逐条数字）。词法检索对口语化说法的鲁棒性比分类时
假设的要好，因为顾客的真实提问通常仍带着至少一个能在分词后存活的领域名词——
``Anschluss``（"Ich ziehe um, was passiert mit meinem Anschluss?" 命中
``kb/de/umzug.md``）、``moving``（"I am moving house" 命中 ``kb/en/umzug.md``
标题 "Moving home"）。两条完全未命中的用例，一条是起区分作用的词全都缺失
（"Wie komme ich aus meinem Vertrag heraus?"），另一条是隔着派生形态边界够
不到（``gedrosselt``）——这两种失配的原因不同，`known_limitation` 的两个取值
把它们分开正是为了让这个差异在数字里可见。
"""

from __future__ import annotations

from typing import NamedTuple

from guardrails.types import Locale


class RecallCase(NamedTuple):
    query: str
    expected_doc_ids: frozenset[str]
    locale: Locale
    known_limitation: str = ""
    """按查询与文档的词汇关系分类，不是按预期难度分类：

    空字符串 —— 查询用的是文档自身的术语：起区分作用的领域词在查询和文档里
    都出现。

    ``paraphrase`` —— 查询是顾客自己的说法，不是文档的术语，起区分作用的
    领域词缺失或很弱。不等于词汇零重叠：顾客提问通常仍带着至少一个共享词，
    这一类里有两条就是如此（``Anschluss``、``moving``）。

    ``derivation`` —— 查询用的形态变体是分词器跨不过去的边界：分词器对屈折
    变化做词典校验的还原，但不做派生形态还原。``gedrosselt``（动词分词）和
    ``Drosselung``（名词）因此分出互不相交的词。"""


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
    # --- 德语：改写型（已知局限：paraphrase，查询用顾客说法而非文档术语）---
    RecallCase("Wie komme ich aus meinem Vertrag heraus?",
               frozenset({"vertragslaufzeit-kuendigung"}), DE,
               known_limitation="paraphrase"),
    RecallCase("Ich ziehe um, was passiert mit meinem Anschluss?",
               frozenset({"umzug"}), DE, known_limitation="paraphrase"),
    RecallCase("Mein Internet ist seit gestern weg",
               frozenset({"stoerung-entstoerfrist"}), DE,
               known_limitation="paraphrase"),
    # --- 德语：派生形态型（已知局限：derivation，动词分词到不了对应名词）---
    RecallCase("Ab wann wird gedrosselt?",
               frozenset({"datenvolumen-drosselung"}), DE,
               known_limitation="derivation"),
    # --- 英语 ---
    RecallCase("How much does Tariff M cost?", frozenset({"tarife-mobilfunk"}), EN),
    RecallCase("notice period for cancellation",
               frozenset({"vertragslaufzeit-kuendigung"}), EN),
    RecallCase("roaming outside the EU", frozenset({"roaming-eu"}), EN),
    RecallCase("when is my invoice issued", frozenset({"rechnung-zahlungsarten"}), EN),
    RecallCase("how long to fix a fault", frozenset({"stoerung-entstoerfrist"}), EN),
    RecallCase("I am moving house", frozenset({"umzug"}), EN,
               known_limitation="paraphrase"),
)
