"""LangGraph build function with HITL gates, multi-query search,
fact-checking, and SQLite checkpointing.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from langgraph_research_pilot.llm import LLMClient, get_llm
from langgraph_research_pilot.nodes import (
    make_answer_node,
    make_plan_node,
    make_search_node,
    make_synthesize_node,
    make_verify_node,
)
from langgraph_research_pilot.search import SearchBackend, get_search
from langgraph_research_pilot.state import ResearchState


def build_graph(
    llm: LLMClient | None = None,
    search: SearchBackend | None = None,
    *,
    checkpointer: Any | None = None,
    interrupt_search: bool = True,
    interrupt_answer: bool = True,
    k: int = 4,
    verify: bool = True,
) -> Any:
    """Compile the research graph.

    Parameters
    ----------
    llm:
        LLM client. ``None`` resolves via :func:`langgraph_research_pilot.llm.get_llm`.
    search:
        Search backend. ``None`` resolves via :func:`langgraph_research_pilot.search.get_search`.
    checkpointer:
        Optional LangGraph checkpointer. If ``None``, a ``MemorySaver`` is used.
    interrupt_search / interrupt_answer:
        Whether to pause for human review before ``answer`` (after search) and before
        ``synthesize`` (after per-sub-question answers).
    k:
        Hits per sub-question (after merging multi-query results).
    verify:
        If True, append a post-synthesis fact-check node that may regenerate
        the final answer under a strict source-grounded prompt.
    """
    llm = llm or get_llm()
    search = search or get_search()

    plan = make_plan_node(llm)
    do_search = make_search_node(search, k=k)
    answer = make_answer_node(llm)
    synthesize = make_synthesize_node(llm)
    verify_node = make_verify_node(llm)

    graph: StateGraph = StateGraph(ResearchState)
    graph.add_node("planner", plan)
    graph.add_node("search", do_search)
    graph.add_node("answer", answer)
    graph.add_node("synthesize", synthesize)
    if verify:
        graph.add_node("verify", verify_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "search")
    graph.add_edge("search", "answer")
    graph.add_edge("answer", "synthesize")
    if verify:
        graph.add_edge("synthesize", "verify")
        graph.add_edge("verify", END)
    else:
        graph.add_edge("synthesize", END)

    if checkpointer is None:
        from langgraph.checkpoint.memory import MemorySaver

        checkpointer = MemorySaver()

    interrupt_before: list[str] = []
    if interrupt_search:
        interrupt_before.append("answer")
    if interrupt_answer:
        interrupt_before.append("synthesize")

    return graph.compile(checkpointer=checkpointer, interrupt_before=interrupt_before or None)


def make_sqlite_checkpointer(path: str) -> Any:
    """Open a SQLite checkpointer context manager at ``path``.

    The result is a context manager. Use it inside a ``with`` block so the
    SQLite connection closes cleanly when the run finishes. Useful for
    resume-from-crash demos.

    Example:
        from langgraph_research_pilot.graph import build_graph, make_sqlite_checkpointer

        with make_sqlite_checkpointer("checkpoints.db") as cp:
            graph = build_graph(checkpointer=cp)
            state = graph.invoke({"question": "..."}, config={"configurable": {"thread_id": "t1"}})
    """
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError as exc:  # pragma: no cover - explicit handling for missing extra
        raise RuntimeError(
            "SqliteSaver not available. Install with `pip install langgraph-checkpoint-sqlite`."
        ) from exc
    return SqliteSaver.from_conn_string(path)
