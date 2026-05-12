"""Per-node unit tests."""

from __future__ import annotations

from langgraph_research_pilot.nodes import (
    _parse_plan,
    make_answer_node,
    make_plan_node,
    make_search_node,
    make_synthesize_node,
)


def test_parse_plan_handles_clean_json() -> None:
    raw = '[{"id":1,"text":"q1","rationale":"r1"},{"id":2,"text":"q2","rationale":"r2"}]'
    items = _parse_plan(raw, fallback_question="fallback", max_items=4)
    assert [it["text"] for it in items] == ["q1", "q2"]


def test_parse_plan_handles_fenced_json() -> None:
    raw = "Sure:\n```json\n[{\"id\":1,\"text\":\"only\"}]\n```"
    items = _parse_plan(raw, fallback_question="fallback", max_items=4)
    assert items[0]["text"] == "only"


def test_parse_plan_falls_back_when_json_invalid() -> None:
    items = _parse_plan("not json at all", fallback_question="MAIN", max_items=4)
    assert len(items) == 1
    assert items[0]["text"] == "MAIN"


def test_parse_plan_skips_entries_without_text() -> None:
    raw = '[{"id":1},{"id":2,"text":"good"}]'
    items = _parse_plan(raw, fallback_question="fallback", max_items=4)
    assert len(items) == 1
    assert items[0]["text"] == "good"


def test_parse_plan_respects_max_items() -> None:
    raw = '[{"id":1,"text":"a"},{"id":2,"text":"b"},{"id":3,"text":"c"}]'
    items = _parse_plan(raw, fallback_question="fallback", max_items=2)
    assert len(items) == 2


def test_plan_node_populates_plan(llm, search) -> None:
    plan = make_plan_node(llm)
    delta = plan({"question": "Tell me about Marie Curie."})
    assert "plan" in delta
    assert len(delta["plan"]) == 2
    assert delta["audit_log"][-1]["node"] == "plan"


def test_search_node_fans_out(llm, search) -> None:
    plan = make_plan_node(llm)
    delta_plan = plan({"question": "Tell me about Marie Curie."})
    do_search = make_search_node(search, k=3)
    delta = do_search({"question": "Tell me about Marie Curie.", "plan": delta_plan["plan"]})
    assert set(delta["search_results"].keys()) == {"1", "2"}
    assert all(len(v) >= 1 for v in delta["search_results"].values())


def test_answer_node_returns_unknown_when_no_hits(llm, search) -> None:
    plan = make_plan_node(llm)
    delta_plan = plan({"question": "Tell me about Marie Curie."})
    answer = make_answer_node(llm)
    delta = answer(
        {
            "question": "Tell me about Marie Curie.",
            "plan": delta_plan["plan"],
            "search_results": {},
        }
    )
    assert delta["answers"]["1"] == "unknown"
    assert delta["answers"]["2"] == "unknown"


def test_synthesize_node_includes_question_in_prompt(llm, search) -> None:
    plan = make_plan_node(llm)
    delta_plan = plan({"question": "Tell me about Marie Curie."})
    do_search = make_search_node(search)
    delta_search = do_search({"question": "Tell me about Marie Curie.", "plan": delta_plan["plan"]})
    answer = make_answer_node(llm)
    delta_ans = answer(
        {
            "question": "Tell me about Marie Curie.",
            "plan": delta_plan["plan"],
            "search_results": delta_search["search_results"],
        }
    )
    synth = make_synthesize_node(llm)
    delta = synth(
        {
            "question": "Tell me about Marie Curie.",
            "plan": delta_plan["plan"],
            "search_results": delta_search["search_results"],
            "answers": delta_ans["answers"],
        }
    )
    assert "Marie Curie" in delta["final_answer"]
