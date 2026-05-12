"""Shared pytest fixtures."""

from __future__ import annotations

import json

import pytest

from langgraph_research_pilot.llm import MockLLM
from langgraph_research_pilot.search import MockSearch
from langgraph_research_pilot.state import make_search_result


def _planner_response() -> str:
    return json.dumps(
        [
            {
                "id": 1,
                "text": "Where was Marie Curie born?",
                "rationale": "establish origin",
                "queries": ["Marie Curie birthplace", "Marie Sklodowska Curie Warsaw"],
            },
            {
                "id": 2,
                "text": "When did she win the Nobel Prize?",
                "rationale": "establish timeline",
                "queries": ["Marie Curie Nobel Prize 1903", "Marie Curie Physics Nobel"],
            },
        ]
    )


def _handler(prompt: str, system: str | None) -> str:
    system_text = (system or "").lower()
    if "research planning" in system_text:
        return _planner_response()
    if "fact-checker" in system_text:
        return json.dumps({"supported": True, "issues": []})
    if "focused research" in system_text:
        if "Where was" in prompt:
            return "Marie Curie was born in Warsaw, Poland."
        if "Nobel" in prompt:
            return "She won the Nobel Prize in Physics in 1903."
        return "unknown"
    if "research synthesis" in system_text:
        return "Marie Curie was born in Warsaw and won the Nobel Prize in 1903 (Marie Curie)."
    return "Mock answer."


@pytest.fixture
def llm() -> MockLLM:
    return MockLLM(handler=_handler)


@pytest.fixture
def search() -> MockSearch:
    return MockSearch(
        fixtures={
            "Marie Curie birthplace": [
                make_search_result(
                    title="Marie Curie",
                    url="https://en.wikipedia.org/wiki/Marie_Curie",
                    snippet="Marie Sklodowska Curie was born in Warsaw, Poland in 1867.",
                )
            ],
            "Marie Sklodowska Curie Warsaw": [
                make_search_result(
                    title="Marie Curie",
                    url="https://en.wikipedia.org/wiki/Marie_Curie",
                    snippet="Marie Sklodowska Curie was born in Warsaw, Poland in 1867.",
                )
            ],
            "Marie Curie Nobel Prize 1903": [
                make_search_result(
                    title="Marie Curie",
                    url="https://en.wikipedia.org/wiki/Marie_Curie",
                    snippet="Curie won the Nobel Prize in Physics in 1903 (shared) and in Chemistry in 1911.",
                )
            ],
            "Marie Curie Physics Nobel": [
                make_search_result(
                    title="Marie Curie",
                    url="https://en.wikipedia.org/wiki/Marie_Curie",
                    snippet="Curie won the Nobel Prize in Physics in 1903 (shared) and in Chemistry in 1911.",
                )
            ],
            # Keep older fixtures as fallback when sub-question text is used as the query.
            "Where was": [
                make_search_result(
                    title="Marie Curie",
                    url="https://en.wikipedia.org/wiki/Marie_Curie",
                    snippet="Marie Sklodowska Curie was born in Warsaw, Poland in 1867.",
                )
            ],
            "Nobel": [
                make_search_result(
                    title="Marie Curie",
                    url="https://en.wikipedia.org/wiki/Marie_Curie",
                    snippet="Curie won the Nobel Prize in Physics in 1903 (shared) and in Chemistry in 1911.",
                )
            ],
        }
    )
