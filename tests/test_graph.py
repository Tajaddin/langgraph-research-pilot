"""End-to-end graph tests using MockLLM + MockSearch."""

from __future__ import annotations

import uuid

from langgraph_research_pilot.graph import build_graph


def test_graph_compiles_without_interrupts(llm, search) -> None:
    graph = build_graph(llm=llm, search=search, interrupt_search=False, interrupt_answer=False)
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    state = graph.invoke({"question": "Tell me about Marie Curie."}, config=config)
    assert state["final_answer"]
    assert "Marie Curie" in state["final_answer"]
    assert len(state["plan"]) == 2
    assert state["answers"]
    assert any(entry["node"] == "synthesize" for entry in state["audit_log"])


def test_graph_audit_log_in_order(llm, search) -> None:
    graph = build_graph(llm=llm, search=search, interrupt_search=False, interrupt_answer=False)
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    state = graph.invoke({"question": "Tell me about Marie Curie."}, config=config)
    nodes = [entry["node"] for entry in state["audit_log"]]
    assert nodes == ["planner", "search", "answer", "synthesize", "verify"]
