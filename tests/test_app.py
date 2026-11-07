"""The Space entry point must start on the dependency set that ships with it."""

from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parent.parent


def _pinned(requirements: str, package: str) -> str:
    match = re.search(rf"^{re.escape(package)}==([^\s;]+)", requirements, re.MULTILINE)
    assert match, f"{package} is not pinned in requirements.txt"
    return match.group(1)


def test_space_sdk_version_matches_requirements() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    sdk_version = re.search(r'^sdk_version:\s*"?([^"\s]+)"?\s*$', readme, re.MULTILINE)
    assert sdk_version, "README front matter has no sdk_version"
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert sdk_version.group(1) == _pinned(requirements, "gradio")


def test_app_launches_and_serves_the_ui(monkeypatch) -> None:
    pytest.importorskip("gradio")
    monkeypatch.setenv("RESEARCH_PILOT_LLM", "mock")
    monkeypatch.setenv("RESEARCH_PILOT_SEARCH", "mock")
    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "False")
    monkeypatch.syspath_prepend(str(ROOT))
    monkeypatch.delitem(sys.modules, "app", raising=False)
    app = importlib.import_module("app")

    _, local_url, _ = app.launch(prevent_thread_lock=True, server_name="127.0.0.1")
    try:
        resp = httpx.get(local_url, timeout=10)
        assert resp.status_code == 200
        assert "LangGraph Research Pilot" in resp.text
    finally:
        app.demo.close()
