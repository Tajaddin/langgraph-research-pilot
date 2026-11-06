"""WikipediaSearch request construction, against a stubbed transport."""

from __future__ import annotations

from urllib.parse import unquote

import httpx
import pytest

from langgraph_research_pilot.search import WikipediaSearch

SUMMARY_PREFIX = "/api/rest_v1/page/summary/"


def _stub_wikipedia(monkeypatch, titles: list[str]) -> tuple[WikipediaSearch, list[str]]:
    """Serve ``titles`` as search hits and answer summary requests like Wikipedia:
    the title must be the single, percent-decoded last path segment."""
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/w/api.php":
            return httpx.Response(200, json={"query": {"search": [{"title": t} for t in titles]}})
        raw = request.url.raw_path.decode("ascii")
        requested.append(raw)
        segment = raw.removeprefix(SUMMARY_PREFIX)
        title = unquote(segment).replace("_", " ")
        if not raw.startswith(SUMMARY_PREFIX) or "/" in segment or "?" in segment or title not in titles:
            return httpx.Response(404)
        return httpx.Response(200, json={"title": title, "extract": f"About {title}."})

    backend = WikipediaSearch()
    monkeypatch.setattr(
        backend, "_client", lambda: httpx.Client(transport=httpx.MockTransport(handler))
    )
    return backend, requested


@pytest.mark.parametrize(
    ("title", "page_url"),
    [
        ("AC/DC", "https://en.wikipedia.org/wiki/AC/DC"),
        ("Why?", "https://en.wikipedia.org/wiki/Why%3F"),
    ],
)
def test_titles_with_reserved_characters_are_fetched(monkeypatch, title: str, page_url: str) -> None:
    backend, _ = _stub_wikipedia(monkeypatch, [title])

    results = backend.search("query", k=3)

    assert [(r["title"], r["url"]) for r in results] == [(title, page_url)]


def test_spaces_become_underscores(monkeypatch) -> None:
    backend, requested = _stub_wikipedia(monkeypatch, ["Marie Curie"])

    results = backend.search("query", k=3)

    assert [r["title"] for r in results] == ["Marie Curie"]
    assert requested == [SUMMARY_PREFIX + "Marie_Curie"]
