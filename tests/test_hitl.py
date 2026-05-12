"""Human-in-the-loop interrupt tests."""

from __future__ import annotations

import uuid

from langgraph_research_pilot.graph import build_graph


def test_pauses_before_answer_node(llm, search) -> None:
    graph = build_graph(llm=llm, search=search, interrupt_search=True, interrupt_answer=False)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    state = graph.invoke({"question": "Tell me about Marie Curie."}, config=config)
    # Search has run, but answer/synthesize have not.
    assert "search_results" in state
    assert state.get("answers") is None or state.get("answers") == {}
    assert not state.get("final_answer")


def test_pauses_before_synthesize_node(llm, search) -> None:
    graph = build_graph(llm=llm, search=search, interrupt_search=False, interrupt_answer=True)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    state = graph.invoke({"question": "Tell me about Marie Curie."}, config=config)
    assert state.get("answers")
    assert not state.get("final_answer")


def test_resume_after_search_gate(llm, search) -> None:
    graph = build_graph(llm=llm, search=search, interrupt_search=True, interrupt_answer=False)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    paused = graph.invoke({"question": "Tell me about Marie Curie."}, config=config)
    assert not paused.get("final_answer")
    final = graph.invoke(None, config=config)
    assert final["final_answer"]
    assert "Marie Curie" in final["final_answer"]


def test_resume_after_both_gates(llm, search) -> None:
    graph = build_graph(llm=llm, search=search, interrupt_search=True, interrupt_answer=True)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    s1 = graph.invoke({"question": "Tell me about Marie Curie."}, config=config)
    assert not s1.get("answers")
    s2 = graph.invoke(None, config=config)
    assert s2.get("answers")
    assert not s2.get("final_answer")
    s3 = graph.invoke(None, config=config)
    assert s3["final_answer"]
