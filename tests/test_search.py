"""WikipediaSearch request construction, against a stubbed transport."""

from __future__ import annotations

import httpx

from langgraph_research_pilot.search import WikipediaSearch

SUMMARY_PREFIX = "/api/rest_v1/page/summary/"


def _stub_wikipedia(
    monkeypatch, hits: list[str], pages: list[str] | None = None
) -> tuple[WikipediaSearch, list[str]]:
    """Serve ``hits`` as search results and answer summary requests the way
    Wikipedia's REST API does: the title is the percent-decoded rest of the
    path, slashes included, so an unencoded ``?`` cuts it short and the
    summary of whatever page that prefix names comes back. ``pages`` are the
    titles that exist (default: the hits)."""
    pages = hits if pages is None else pages
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/w/api.php":
            return httpx.Response(200, json={"query": {"search": [{"title": t} for t in hits]}})
        requested.append(request.url.raw_path.decode("ascii"))
        title = request.url.path.removeprefix(SUMMARY_PREFIX).replace("_", " ")
        if title not in pages:
            return httpx.Response(404)
        return httpx.Response(200, json={"title": title, "extract": f"About {title}."})

    backend = WikipediaSearch()
    monkeypatch.setattr(
        backend, "_client", lambda: httpx.Client(transport=httpx.MockTransport(handler))
    )
    return backend, requested


def test_question_mark_title_is_not_cut_short(monkeypatch) -> None:
    backend, requested = _stub_wikipedia(
        monkeypatch, ["What Is Life?"], pages=["What Is Life?", "What Is Life"]
    )

    results = backend.search("query", k=3)

    assert [(r["title"], r["url"]) for r in results] == [
        ("What Is Life?", "https://en.wikipedia.org/wiki/What_Is_Life%3F")
    ]
    assert requested == [SUMMARY_PREFIX + "What_Is_Life%3F"]


def test_slash_in_title_is_encoded_as_one_segment(monkeypatch) -> None:
    backend, requested = _stub_wikipedia(monkeypatch, ["AC/DC"])

    results = backend.search("query", k=3)

    assert requested == [SUMMARY_PREFIX + "AC%2FDC"]
    assert [(r["title"], r["url"]) for r in results] == [
        ("AC/DC", "https://en.wikipedia.org/wiki/AC/DC")
    ]


def test_spaces_become_underscores(monkeypatch) -> None:
    backend, requested = _stub_wikipedia(monkeypatch, ["Marie Curie"])

    results = backend.search("query", k=3)

    assert [r["title"] for r in results] == ["Marie Curie"]
    assert requested == [SUMMARY_PREFIX + "Marie_Curie"]
