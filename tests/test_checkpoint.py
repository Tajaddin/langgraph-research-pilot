"""SQLite checkpoint resume tests.

Simulates a "kill mid-run + restart" by closing the saver after the first
invoke, opening a fresh saver against the same DB file, rebuilding the graph,
and resuming with only the thread_id.
"""

from __future__ import annotations

import os
import tempfile
import uuid

from langgraph.checkpoint.sqlite import SqliteSaver

from langgraph_research_pilot.graph import build_graph


def test_sqlite_resume_across_fresh_graph(llm, search) -> None:
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    question = "Tell me about Marie Curie."

    try:
        # First "run": go up to the search gate, then exit (saver closes).
        with SqliteSaver.from_conn_string(db_path) as saver1:
            graph1 = build_graph(
                llm=llm,
                search=search,
                checkpointer=saver1,
                interrupt_search=True,
                interrupt_answer=False,
            )
            paused = graph1.invoke({"question": question}, config=config)
            assert paused.get("search_results"), "Should have search results before gate"
            assert not paused.get("final_answer"), "Should not have final answer before gate"

        # Second "run": fresh saver on same file, fresh graph, resume by thread_id only.
        with SqliteSaver.from_conn_string(db_path) as saver2:
            graph2 = build_graph(
                llm=llm,
                search=search,
                checkpointer=saver2,
                interrupt_search=True,
                interrupt_answer=False,
            )
            snapshot = graph2.get_state(config)
            assert snapshot is not None
            assert snapshot.values.get("question") == question
            final = graph2.invoke(None, config=config)
            assert final.get("final_answer"), "Resume should produce a final answer"
            assert "Marie Curie" in final["final_answer"]
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_sqlite_state_persists_question_and_plan(llm, search) -> None:
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    try:
        with SqliteSaver.from_conn_string(db_path) as saver1:
            graph = build_graph(
                llm=llm,
                search=search,
                checkpointer=saver1,
                interrupt_search=True,
                interrupt_answer=True,
            )
            graph.invoke({"question": "Tell me about Marie Curie."}, config=config)

        with SqliteSaver.from_conn_string(db_path) as saver2:
            graph = build_graph(
                llm=llm,
                search=search,
                checkpointer=saver2,
                interrupt_search=True,
                interrupt_answer=True,
            )
            snapshot = graph.get_state(config)
            assert snapshot.values["question"] == "Tell me about Marie Curie."
            assert len(snapshot.values["plan"]) == 2
            assert snapshot.values["search_results"]
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)
