"""Typed state for the research graph.

All state types are plain ``TypedDict``s so they serialize cleanly through
LangGraph's checkpoint layer (msgpack/JSON). The few places that need richer
validation (LLM JSON output parsing) use Pydantic at the boundary and
convert to TypedDicts before placing values in state.
"""

from __future__ import annotations

from typing import TypedDict

from typing_extensions import NotRequired


class SubQuestion(TypedDict):
    """One decomposed sub-question the agent plans to answer."""

    id: int
    text: str
    rationale: str
    queries: NotRequired[list[str]]


class SearchResult(TypedDict):
    """One retrieved snippet from the search backend."""

    title: str
    url: str
    snippet: str
    score: float


class AuditEntry(TypedDict):
    """One row in the run audit log."""

    node: str
    summary: str
    elapsed_ms: int


class ResearchState(TypedDict, total=False):
    """Graph state. ``total=False`` because LangGraph builds state incrementally."""

    question: str
    plan: list[SubQuestion]
    search_results: dict[str, list[SearchResult]]
    answers: dict[str, str]
    final_answer: str
    audit_log: list[AuditEntry]
    user_approves_search: bool
    user_approves_answer: bool
    user_feedback: str
    verified: bool
    verification_notes: list[str]


def make_sub_question(
    id: int,
    text: str,
    rationale: str = "",
    queries: list[str] | None = None,
) -> SubQuestion:
    sub: SubQuestion = {"id": id, "text": text, "rationale": rationale}
    if queries:
        sub["queries"] = queries
    return sub


def make_search_result(title: str, url: str, snippet: str, score: float = 0.0) -> SearchResult:
    return {"title": title, "url": url, "snippet": snippet, "score": score}


def make_audit_entry(node: str, summary: str, elapsed_ms: int = 0) -> AuditEntry:
    return {"node": node, "summary": summary, "elapsed_ms": elapsed_ms}
