"""State helper tests."""

from __future__ import annotations

from langgraph_research_pilot.state import (
    make_audit_entry,
    make_search_result,
    make_sub_question,
)


def test_sub_question_has_required_keys() -> None:
    sub = make_sub_question(id=3, text="hi", rationale="reason")
    assert sub == {"id": 3, "text": "hi", "rationale": "reason"}


def test_search_result_defaults_score() -> None:
    sr = make_search_result(title="t", url="u", snippet="s")
    assert sr["score"] == 0.0
    assert sr["title"] == "t"


def test_audit_entry_defaults_elapsed() -> None:
    a = make_audit_entry(node="plan", summary="ok")
    assert a["elapsed_ms"] == 0
