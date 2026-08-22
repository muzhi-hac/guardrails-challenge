"""德语检索的领域词典。

这个模块是德语分词的核心资产：同一份词表既驱动复合词分解，也验证词干化。把两者
绑在同一份数据上是有意的 —— 一个能被拆出来的成分，才是一个可以被剥后缀剥出来的
词干；两套词表会各自漂移。

覆盖范围是当前语料。扩语料必须同步扩词典，否则新复合词退化为整词匹配（失效方向
保守，不会产生错误召回，但 recall 会掉）。
"""

from __future__ import annotations

from typing import Final

LEXICON: Final[frozenset[str]] = frozenset(
    {
        # 合同与解约
        "vertrag", "kündigung", "frist", "laufzeit", "mindest", "sonder",
        "widerruf", "recht", "partner", "wechsel", "termin", "verlängerung",
        # 资费与账务
        "tarif", "preis", "betrag", "rechnung", "zahlung", "art", "gebühr",
        "konto", "lastschrift", "verzug", "monat", "jahr", "woche", "tag",
        # 产品
        "mobilfunk", "daten", "volumen", "netz", "abdeckung", "karte",
        "roaming", "gerät", "finanzierung", "anschluss", "nummer", "mitnahme",
        "ruf", "drosselung", "geschwindigkeit",
        # 服务
        "service", "zeit", "kunde", "störung", "meldung", "entstörung", "entstör",
        "umzug", "auskunft", "schutz", "adresse",
    }
)

LINKING_MORPHEMES: Final[tuple[str, ...]] = ("es", "en", "s", "n")
"""连接语素，按长度降序尝试：Kündigung|s|frist、Rufnummer|n|mitnahme。"""

INFLECTION_SUFFIXES: Final[tuple[str, ...]] = ("en", "es", "er", "e", "n", "s")
"""词形后缀。剥离结果**必须命中 LEXICON** 才产出，见 de.py 的 _stem_alias。"""
