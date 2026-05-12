"""LLM client tests."""

from __future__ import annotations

import pytest

from langgraph_research_pilot.llm import MockLLM, extract_json, get_llm


def test_extract_json_strips_fenced_block() -> None:
    text = "Sure, here you go:\n```json\n[{\"id\":1}]\n```\nLet me know."
    assert extract_json(text) == '[{"id":1}]'


def test_extract_json_passthrough_when_unfenced() -> None:
    text = '[{"id":1}]'
    assert extract_json(text) == text


def test_extract_json_handles_generic_fence() -> None:
    text = "```\nhello\n```"
    assert extract_json(text) == "hello"


def test_mock_llm_handler_takes_priority() -> None:
    def h(prompt: str, system: str | None) -> str:
        return "handler-said-this"

    m = MockLLM(canned={"anything": "canned"}, handler=h)
    assert m.complete("anything", system="x") == "handler-said-this"


def test_mock_llm_canned_substring_match() -> None:
    m = MockLLM(canned={"trigger": "yes"})
    assert m.complete("contains trigger word") == "yes"


def test_mock_llm_records_calls() -> None:
    m = MockLLM()
    m.complete("a", system="sys-a")
    m.complete("b")
    assert len(m.calls) == 2
    assert m.calls[0] == ("a", "sys-a")
    assert m.calls[1] == ("b", None)


def test_get_llm_returns_mock_when_explicit() -> None:
    assert isinstance(get_llm("mock"), MockLLM)


def test_get_llm_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        get_llm("not-a-real-backend")
