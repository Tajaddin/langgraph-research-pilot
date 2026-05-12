"""Search backends. Wikipedia REST API for the live demo, Mock for tests."""

from __future__ import annotations

import os
from typing import Protocol

import httpx

from langgraph_research_pilot.state import SearchResult, make_search_result

WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_REST = "https://en.wikipedia.org/api/rest_v1/page/summary"
USER_AGENT = "langgraph-research-pilot/0.1 (https://github.com/Tajaddin/langgraph-research-pilot)"


class SearchBackend(Protocol):
    """Minimal search backend."""

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        """Return up to ``k`` results for the query."""
        ...


class MockSearch:
    """In-memory search for tests."""

    def __init__(self, fixtures: dict[str, list[SearchResult]] | None = None) -> None:
        self.fixtures = fixtures or {}
        self.calls: list[str] = []

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        self.calls.append(query)
        for needle, results in self.fixtures.items():
            if needle.lower() in query.lower():
                return results[:k]
        return [
            make_search_result(
                title=f"Mock result for {query[:40]}",
                url="https://example.com",
                snippet=f"Mock snippet about {query[:80]}.",
                score=0.5,
            )
        ]


class WikipediaSearch:
    """Wikipedia REST search. No API key required.

    Uses the MediaWiki search API to list candidates, then the REST summary
    endpoint to pull a clean extract for each hit. Aggressively short-circuits
    when the network is unreachable so unit tests stay fast.
    """

    def __init__(self, timeout: float = 10.0, max_summary_chars: int = 600) -> None:
        self.timeout = timeout
        self.max_summary_chars = max_summary_chars

    def _client(self) -> httpx.Client:
        return httpx.Client(timeout=self.timeout, headers={"User-Agent": USER_AGENT})

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        with self._client() as client:
            try:
                titles = self._search_titles(client, query, k)
            except httpx.HTTPError:
                return []
            results: list[SearchResult] = []
            for title in titles:
                try:
                    summary = self._get_summary(client, title)
                except httpx.HTTPError:
                    continue
                if summary is None:
                    continue
                results.append(summary)
            return results[:k]

    def _search_titles(self, client: httpx.Client, query: str, k: int) -> list[str]:
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": max(k, 3),
            "format": "json",
            "origin": "*",
        }
        resp = client.get(WIKI_API, params=params)
        resp.raise_for_status()
        data = resp.json()
        hits = data.get("query", {}).get("search", []) or []
        return [hit["title"] for hit in hits]

    def _get_summary(self, client: httpx.Client, title: str) -> SearchResult | None:
        url = f"{WIKI_REST}/{title.replace(' ', '_')}"
        resp = client.get(url)
        if resp.status_code != 200:
            return None
        data = resp.json()
        extract = (data.get("extract") or "").strip()
        if not extract:
            return None
        page_url = data.get("content_urls", {}).get("desktop", {}).get("page") or f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
        return make_search_result(
            title=data.get("title", title),
            url=page_url,
            snippet=extract[: self.max_summary_chars],
            score=1.0,
        )


def get_search(name: str | None = None) -> SearchBackend:
    """Resolve a search backend by name.

    Order:
    1. Explicit ``name`` (``mock``, ``wikipedia``)
    2. Env var ``RESEARCH_PILOT_SEARCH``
    3. Default ``wikipedia``.
    """
    name = (name or os.environ.get("RESEARCH_PILOT_SEARCH") or "wikipedia").lower().strip()
    if name == "mock":
        return MockSearch()
    if name in {"wikipedia", "wiki"}:
        return WikipediaSearch()
    raise ValueError(f"Unknown search backend: {name!r}")
