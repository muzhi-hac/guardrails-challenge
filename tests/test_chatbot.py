"""Chatbot 链路。

这里**不接守卫** —— Action 编排与 REWRITE 属于 M7。M6 要证明的是三件事：
检索结果真的进了提示词、文档被标成不可信数据、以及 ChatTurn 带够了 M7 需要的证据。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chatbot import HISTORY_TURNS, Chatbot
from guardrails.provider.base import CompletionResult, Turn
from guardrails.retrieval.knowledge_base import KnowledgeBase
from guardrails.types import Locale, Mode
from utils import load_profile

KB_ROOT = Path(__file__).resolve().parents[1] / "kb"
PROFILES = Path(__file__).resolve().parents[1] / "profiles"


class SpyCompletion:
    """记录它被怎么调用的，本身返回固定文本。"""

    def __init__(self) -> None:
        self.system: str = ""
        self.messages: tuple[Turn, ...] = ()

    async def complete(self, *, system, messages, max_tokens):
        self.system = system
        self.messages = tuple(messages)
        return CompletionResult(
            text="Tarif M kostet 29,99 EUR pro Monat.",
            model="spy",
            input_tokens=1,
            output_tokens=1,
            latency_ms=0.0,
            stop_reason="end_turn",
        )


@pytest.fixture
def profile():
    return load_profile(PROFILES / "telco_de.yaml").resolve(Mode.CHAT)


@pytest.fixture
def bot(profile):
    kb = KnowledgeBase.load(KB_ROOT)
    return SpyCompletion(), kb, profile


async def test_retrieved_chunks_reach_the_prompt(bot):
    spy, kb, profile = bot
    chatbot = Chatbot(kb.for_locale(Locale.DE_DE), spy, profile)
    turn = await chatbot.reply("Was kostet Tarif M?", ())
    assert turn.retrieved
    assert "29,99" in spy.messages[-1].content


async def test_documents_are_marked_as_untrusted_data(bot):
    """注入守卫的文档通道将来挂在这里 —— 分隔符现在就得留对。"""
    spy, kb, profile = bot
    chatbot = Chatbot(kb.for_locale(Locale.DE_DE), spy, profile)
    await chatbot.reply("Was kostet Tarif M?", ())
    rendered = spy.messages[-1].content
    assert "<document" in rendered and "</document>" in rendered
    assert "niemals Anweisungen" in spy.system


async def test_system_prompt_requests_the_persona(bot):
    """提示词请求，守卫核验 —— 这是两件事。"""
    spy, kb, profile = bot
    chatbot = Chatbot(kb.for_locale(Locale.DE_DE), spy, profile)
    await chatbot.reply("Hallo", ())
    assert "Sie" in spy.system


async def test_history_is_truncated_to_its_own_window(bot):
    """HISTORY_TURNS 不复用 profile 的 cross_turn_window（默认 5）。"""
    spy, kb, profile = bot
    chatbot = Chatbot(kb.for_locale(Locale.DE_DE), spy, profile)
    history = tuple(Turn("user", f"Frage {i}") for i in range(20))
    await chatbot.reply("Was kostet Tarif M?", history)
    assert len(spy.messages) == HISTORY_TURNS + 1


def test_history_window_is_not_the_injection_window(profile):
    assert HISTORY_TURNS != profile.guards.injection.cross_turn_window


async def test_chat_turn_carries_the_evidence_m7_needs(bot):
    spy, kb, profile = bot
    chatbot = Chatbot(kb.for_locale(Locale.DE_DE), spy, profile)
    turn = await chatbot.reply("Was kostet Tarif M?", ())
    assert turn.reply
    assert turn.completion.model == "spy"
    assert all(s.score > 0 for s in turn.retrieved)
    assert len(turn.prompt_hash) == 32


async def test_prompt_hash_is_stable_for_the_same_input(bot):
    spy, kb, profile = bot
    chatbot = Chatbot(kb.for_locale(Locale.DE_DE), spy, profile)
    a = await chatbot.reply("Was kostet Tarif M?", ())
    b = await chatbot.reply("Was kostet Tarif M?", ())
    assert a.prompt_hash == b.prompt_hash
