"""Verify node tests."""

from __future__ import annotations

import json

from langgraph_research_pilot.llm import MockLLM
from langgraph_research_pilot.nodes import _parse_verify, make_verify_node


def test_parse_verify_supported() -> None:
    raw = json.dumps({"supported": True, "issues": []})
    supported, issues = _parse_verify(raw)
    assert supported is True
    assert issues == []


def test_parse_verify_unsupported_with_issues() -> None:
    raw = json.dumps({"supported": False, "issues": ["claim about date not in any source"]})
    supported, issues = _parse_verify(raw)
    assert supported is False
    assert issues == ["claim about date not in any source"]


def test_parse_verify_handles_fenced_json() -> None:
    raw = '```json\n{"supported": true}\n```'
    supported, _ = _parse_verify(raw)
    assert supported is True


def test_parse_verify_falls_back_to_unsupported() -> None:
    raw = "this is not json"
    supported, issues = _parse_verify(raw)
    assert supported is False
    assert "unparseable" in issues[0]


def test_verify_node_passes_when_supported() -> None:
    llm = MockLLM(canned={"fact-check": json.dumps({"supported": True, "issues": []})})

    # The verify prompt has "fact-checker" only in the SYSTEM, not the user prompt.
    # Use a handler for cleaner control.
    def handler(prompt: str, system: str | None) -> str:
        if system and "fact-checker" in system.lower():
            return json.dumps({"supported": True, "issues": []})
        return "Mock answer."

    llm = MockLLM(handler=handler)
    verify = make_verify_node(llm)
    state = {
        "question": "Q?",
        "final_answer": "A",
        "search_results": {"1": [{"title": "T", "url": "u", "snippet": "A is the answer.", "score": 1.0}]},
    }
    delta = verify(state)
    assert delta["verified"] is True


def test_verify_node_regenerates_when_unsupported() -> None:
    calls: list[str] = []

    def handler(prompt: str, system: str | None) -> str:
        sys_text = (system or "").lower()
        calls.append(sys_text)
        if "fact-checker" in sys_text:
            return json.dumps({"supported": False, "issues": ["unsupported claim"]})
        if "strict research synthesis" in sys_text:
            return "unknown"
        return "Mock answer."

    llm = MockLLM(handler=handler)
    verify = make_verify_node(llm)
    state = {
        "question": "Q?",
        "final_answer": "A",
        "search_results": {"1": [{"title": "T", "url": "u", "snippet": "A is the answer.", "score": 1.0}]},
        "plan": [],
        "answers": {},
    }
    delta = verify(state)
    assert delta["verified"] is False
    assert delta["final_answer"] == "unknown"
    # Verify the regenerate path was taken (strict synth was called).
    assert any("strict research synthesis" in c for c in calls)


def test_verify_node_skips_when_no_final() -> None:
    llm = MockLLM()
    verify = make_verify_node(llm)
    delta = verify({"question": "Q?", "final_answer": ""})
    assert delta["verified"] is False
    assert delta["verification_notes"] == ["no final_answer"]
