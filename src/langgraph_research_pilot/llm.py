"""LLM backend abstraction. Free HF Inference API by default; mockable for tests."""

from __future__ import annotations

import json
import os
import re
from typing import Callable, Protocol


class LLMClient(Protocol):
    """Minimal LLM client interface."""

    def complete(self, prompt: str, system: str | None = None, max_tokens: int = 512) -> str:
        """Return a completion for the prompt."""
        ...


class MockLLM:
    """Deterministic LLM for tests and offline demos.

    Three resolution modes, tried in order:
    1. ``handler`` callable — full control, called with ``(prompt, system)``
    2. ``canned`` dict — substring match against the prompt
    3. Built-in heuristics for the planner/synthesizer prompts
    """

    def __init__(
        self,
        canned: dict[str, str] | None = None,
        handler: "Callable[[str, str | None], str] | None" = None,  # type: ignore[name-defined]
    ) -> None:
        self.canned = canned or {}
        self.handler = handler
        self.calls: list[tuple[str, str | None]] = []

    def complete(self, prompt: str, system: str | None = None, max_tokens: int = 512) -> str:
        self.calls.append((prompt, system))
        if self.handler is not None:
            return self.handler(prompt, system)
        for needle, answer in self.canned.items():
            if needle in prompt:
                return answer
        system_text = (system or "").lower()
        if "research planning" in system_text:
            return json.dumps(
                [
                    {"id": 1, "text": "What is the topic?", "rationale": "Identify subject"},
                    {"id": 2, "text": "Why is it relevant?", "rationale": "Establish context"},
                ]
            )
        if "research synthesis" in system_text:
            return "Synthesized answer from mock LLM."
        if "focused research" in system_text:
            return "Answer from mock LLM."
        return "Mock answer."


class HFInferenceClient:
    """Uses HuggingFace Inference API. Free tier with HF_TOKEN.

    Default model is small enough for the free tier. Override via env or kwarg.
    """

    def __init__(self, model: str | None = None, token: str | None = None, timeout: float = 60.0) -> None:
        from huggingface_hub import InferenceClient  # imported lazily so package is optional

        self.model = model or os.environ.get("HF_MODEL", "meta-llama/Llama-3.2-3B-Instruct")
        self.token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        self.client = InferenceClient(model=self.model, token=self.token, timeout=timeout)

    def complete(self, prompt: str, system: str | None = None, max_tokens: int = 512) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = self.client.chat_completion(messages=messages, max_tokens=max_tokens, temperature=0.2)
        return resp.choices[0].message.content or ""


class AnthropicClient:
    """Uses Anthropic API if ANTHROPIC_API_KEY is set."""

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        from anthropic import Anthropic  # imported lazily

        self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
        self.client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def complete(self, prompt: str, system: str | None = None, max_tokens: int = 512) -> str:
        kwargs: dict = {"model": self.model, "max_tokens": max_tokens, "messages": [{"role": "user", "content": prompt}]}
        if system:
            kwargs["system"] = system
        resp = self.client.messages.create(**kwargs)
        parts = [b.text for b in resp.content if getattr(b, "type", "") == "text"]
        return "".join(parts)


class GroqClient:
    """Uses Groq's free Inference API if GROQ_API_KEY is set.

    Honors a per-instance minimum interval between calls (default 2.5s for
    ~24 RPM, comfortably under the 30 RPM free-tier ceiling) and retries on
    429 responses with exponential backoff respecting the ``Retry-After``
    header if present.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        min_interval_secs: float | None = None,
        max_retries: int = 5,
    ) -> None:
        from groq import Groq  # imported lazily so the dependency stays optional

        self.model = model or os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
        self.client = Groq(api_key=api_key or os.environ.get("GROQ_API_KEY"))
        env_interval = os.environ.get("GROQ_MIN_INTERVAL_SECS")
        self.min_interval = (
            min_interval_secs
            if min_interval_secs is not None
            else (float(env_interval) if env_interval else 2.5)
        )
        self.max_retries = max_retries
        self._last_call_t: float = 0.0

    def _wait_for_slot(self) -> None:
        import time

        now = time.monotonic()
        delta = now - self._last_call_t
        if delta < self.min_interval:
            time.sleep(self.min_interval - delta)
        self._last_call_t = time.monotonic()

    def complete(self, prompt: str, system: str | None = None, max_tokens: int = 512) -> str:
        import time

        from groq import RateLimitError

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        backoff = 4.0
        for attempt in range(self.max_retries + 1):
            self._wait_for_slot()
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.2,
                )
                return resp.choices[0].message.content or ""
            except RateLimitError as exc:
                if attempt == self.max_retries:
                    raise
                # Honor Retry-After when the SDK exposes it.
                wait_s = backoff
                retry_after = getattr(getattr(exc, "response", None), "headers", {}) or {}
                ra = retry_after.get("retry-after") if hasattr(retry_after, "get") else None
                if ra:
                    try:
                        wait_s = max(wait_s, float(ra))
                    except (TypeError, ValueError):
                        pass
                time.sleep(wait_s)
                backoff = min(backoff * 2, 60.0)


def get_llm(name: str | None = None) -> LLMClient:
    """Resolve an LLM client by name.

    Order of resolution:
    1. Explicit ``name`` argument (``mock``, ``hf``, ``anthropic``, ``groq``)
    2. Env var ``RESEARCH_PILOT_LLM``
    3. Fallback: Groq > HF > Mock based on which token is set.
    """
    name = (name or os.environ.get("RESEARCH_PILOT_LLM") or "").lower().strip()
    if name == "mock":
        return MockLLM()
    if name == "hf":
        return HFInferenceClient()
    if name == "anthropic":
        return AnthropicClient()
    if name == "groq":
        return GroqClient()
    if not name:
        if os.environ.get("GROQ_API_KEY"):
            return GroqClient()
        if os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN"):
            return HFInferenceClient()
        return MockLLM()
    raise ValueError(f"Unknown LLM backend: {name!r}")


_JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> str:
    """Pull a JSON blob out of a possibly-markdown-wrapped LLM response.

    Returns the original text if no fenced block is found.
    """
    match = _JSON_BLOCK.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()
