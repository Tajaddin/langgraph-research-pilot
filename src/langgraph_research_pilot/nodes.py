"""LangGraph node implementations.

Each node is a pure function (state in, state delta out) so individual nodes
are unit-testable without the graph runtime.

Path 2 additions:
* The plan node also returns 2-3 entity-style Wikipedia search queries per
  sub-question. The search node fans out across those queries and merges
  results by title.
* A post-synthesis verify node checks the final answer against retrieved
  sources. If unsupported, the synthesize step is re-run with a strict
  source-grounded prompt and the result is accepted.
"""

from __future__ import annotations

import json
import time
from typing import Callable

from langgraph_research_pilot.llm import LLMClient, extract_json
from langgraph_research_pilot.search import SearchBackend
from langgraph_research_pilot.state import (
    ResearchState,
    SearchResult,
    SubQuestion,
    make_audit_entry,
    make_sub_question,
)

PLAN_SYSTEM = (
    "You are a research planning assistant. Decompose the user's question into 2-4 focused "
    "sub-questions whose answers, combined, will answer the main question. For each sub-question "
    "also produce 2 to 3 short Wikipedia search queries (2-6 words each) that would surface a "
    "relevant article. Queries must be entity-focused: drop question words like who/what/when/"
    "where, focus on the entities and the fact you need. "
    "Reply with a JSON array of objects with keys: id (int), text (string), rationale (string), "
    "queries (array of strings). No prose around the JSON."
)

SYNTH_SYSTEM = (
    "You are a research synthesis assistant. You MUST produce an answer. Begin your reply with "
    "the single most specific entity, date, year, or short phrase that answers the question "
    "(no preamble, no hedging — just the entity). Then write one short grounding sentence with "
    "source titles in parentheses. If the findings only partially answer the question, choose "
    "the best supported candidate and proceed; do not refuse. Never reply 'I cannot' or 'I don't "
    "have enough information'. If forced to guess, pick the most likely candidate from the findings."
)

SYNTH_STRICT_SYSTEM = (
    "You are a strict research synthesis assistant. Reply ONLY with information that appears in "
    "the provided source snippets. Begin with the single most specific entity or short phrase "
    "extracted directly from the snippets. Then one grounding sentence citing source titles. "
    "If no snippet contains the answer, reply with exactly 'unknown' on its own line."
)

ANSWER_SYSTEM = (
    "You are a focused research assistant. Extract the answer from the snippets in 1 to 8 words. "
    "Output only the answer — no sentence, no preamble. Use the most specific entity, date, or "
    "phrase that addresses the sub-question. Reply 'unknown' only if no snippet is even loosely "
    "relevant."
)

VERIFY_SYSTEM = (
    "You are a fact-checker. Decide whether the candidate final answer is supported by the "
    "retrieved source snippets. Reply with a JSON object: "
    '{"supported": true|false, "issues": ["...optional list of contradictions or missing-support notes..."]}. '
    "Reply with valid JSON only, no prose."
)


# ----------------------------- plan -----------------------------


def make_plan_node(llm: LLMClient, max_sub_questions: int = 4) -> Callable[[ResearchState], ResearchState]:
    """Build a plan node bound to a specific LLM client."""

    def plan(state: ResearchState) -> ResearchState:
        start = time.perf_counter()
        question = state["question"]
        prompt = (
            f"Question: {question}\n\n"
            "Decompose this into sub-questions, with 2-3 Wikipedia search queries each. "
            "Reply with a JSON array only."
        )
        raw = llm.complete(prompt, system=PLAN_SYSTEM, max_tokens=600)
        plan_items = _parse_plan(raw, fallback_question=question, max_items=max_sub_questions)
        elapsed = int((time.perf_counter() - start) * 1000)
        total_q = sum(len(s.get("queries") or []) for s in plan_items)
        audit = list(state.get("audit_log") or [])
        audit.append(
            make_audit_entry(
                node="plan",
                summary=f"Decomposed into {len(plan_items)} sub-questions with {total_q} reformulated queries",
                elapsed_ms=elapsed,
            )
        )
        return {"plan": plan_items, "audit_log": audit}

    return plan


def _parse_plan(raw: str, fallback_question: str, max_items: int) -> list[SubQuestion]:
    text = extract_json(raw)
    items: list[SubQuestion] = []
    try:
        data = json.loads(text)
        if isinstance(data, list):
            for entry in data[:max_items]:
                if not isinstance(entry, dict):
                    continue
                text_value = str(entry.get("text", "")).strip()
                if not text_value:
                    continue
                try:
                    sub_id = int(entry.get("id", len(items) + 1))
                except (TypeError, ValueError):
                    sub_id = len(items) + 1
                queries_raw = entry.get("queries") or []
                queries: list[str] = []
                if isinstance(queries_raw, list):
                    for q in queries_raw:
                        if isinstance(q, str) and q.strip():
                            queries.append(q.strip())
                items.append(
                    make_sub_question(
                        id=sub_id,
                        text=text_value,
                        rationale=str(entry.get("rationale", "")).strip(),
                        queries=queries,
                    )
                )
    except json.JSONDecodeError:
        pass
    if not items:
        items = [
            make_sub_question(
                id=1,
                text=fallback_question,
                rationale="Fallback: ask main question directly",
                queries=[fallback_question],
            )
        ]
    return items


# ----------------------------- search -----------------------------


def make_search_node(search: SearchBackend, k: int = 4) -> Callable[[ResearchState], ResearchState]:
    """Build a search node that fans out across all queries per sub-question.

    Each sub-question contributes its reformulated ``queries`` list (or, as a
    fallback, its ``text``) to a list of searches. Results are merged by title,
    re-scored by frequency-of-appearance, and capped at ``k``.
    """

    def run_search(state: ResearchState) -> ResearchState:
        start = time.perf_counter()
        results: dict[str, list[SearchResult]] = {}
        total_queries_run = 0
        for sub in state.get("plan", []):
            queries = list(sub.get("queries") or [])
            if not queries:
                queries = [sub["text"]]
            merged: dict[str, SearchResult] = {}
            merge_counts: dict[str, int] = {}
            for q in queries:
                hits = search.search(q, k=k)
                total_queries_run += 1
                for rank, hit in enumerate(hits):
                    title = hit["title"]
                    merge_counts[title] = merge_counts.get(title, 0) + 1
                    if title not in merged:
                        # Score: frequency_bonus + position_bonus
                        merged[title] = {
                            **hit,
                            "score": hit.get("score", 0.0) + (1.0 / (rank + 1)),
                        }
                    else:
                        merged[title]["score"] += 1.0 / (rank + 1)
            # Rerank by merged score (frequency boost helps); cap at k.
            ordered = sorted(merged.values(), key=lambda h: h["score"], reverse=True)[:k]
            results[str(sub["id"])] = ordered
        elapsed = int((time.perf_counter() - start) * 1000)
        audit = list(state.get("audit_log") or [])
        total_hits = sum(len(v) for v in results.values())
        audit.append(
            make_audit_entry(
                node="search",
                summary=f"Retrieved {total_hits} unique snippets across {total_queries_run} queries",
                elapsed_ms=elapsed,
            )
        )
        return {"search_results": results, "audit_log": audit}

    return run_search


# ----------------------------- answer -----------------------------


def make_answer_node(llm: LLMClient) -> Callable[[ResearchState], ResearchState]:
    """For each sub-question, ask the LLM to extract a one-sentence answer from its snippets."""

    def answer(state: ResearchState) -> ResearchState:
        start = time.perf_counter()
        results = state.get("search_results", {})
        answers: dict[str, str] = {}
        for sub in state.get("plan", []):
            hits = results.get(str(sub["id"]), [])
            if not hits:
                answers[str(sub["id"])] = "unknown"
                continue
            context = "\n\n".join(f"- ({h['title']}) {h['snippet']}" for h in hits)
            prompt = (
                f"Sub-question: {sub['text']}\n\nSnippets:\n{context}\n\nShort answer:"
            )
            answers[str(sub["id"])] = llm.complete(prompt, system=ANSWER_SYSTEM, max_tokens=120).strip()
        elapsed = int((time.perf_counter() - start) * 1000)
        audit = list(state.get("audit_log") or [])
        audit.append(
            make_audit_entry(
                node="answer",
                summary=f"Answered {len(answers)} sub-questions",
                elapsed_ms=elapsed,
            )
        )
        return {"answers": answers, "audit_log": audit}

    return answer


# ----------------------------- synthesize -----------------------------


def _synth_prompt(state: ResearchState) -> tuple[str, str]:
    question = state["question"]
    plan = state.get("plan", [])
    answers = state.get("answers", {})
    results = state.get("search_results", {})
    findings_lines: list[str] = []
    sources: list[str] = []
    for sub in plan:
        ans = answers.get(str(sub["id"]), "unknown")
        findings_lines.append(f"- Sub-question: {sub['text']}\n  Answer: {ans}")
        for hit in results.get(str(sub["id"]), []):
            if hit["title"] not in sources:
                sources.append(hit["title"])
    findings = "\n".join(findings_lines) or "(no findings)"
    sources_str = ", ".join(sources[:8]) or "(no sources)"
    prompt = (
        f"Main question: {question}\n\nFindings:\n{findings}\n\nSources: {sources_str}\n\n"
        "Write the final grounded answer. First line: the single most specific entity or phrase. "
        "Second line blank. Third line: one short grounding sentence citing source titles in parentheses."
    )
    return prompt, sources_str


def make_synthesize_node(llm: LLMClient) -> Callable[[ResearchState], ResearchState]:
    """Combine the per-sub-question answers into a final grounded reply."""

    def synthesize(state: ResearchState) -> ResearchState:
        start = time.perf_counter()
        prompt, _ = _synth_prompt(state)
        final = llm.complete(prompt, system=SYNTH_SYSTEM, max_tokens=350).strip()
        elapsed = int((time.perf_counter() - start) * 1000)
        audit = list(state.get("audit_log") or [])
        audit.append(
            make_audit_entry(
                node="synthesize",
                summary="Synthesized final answer",
                elapsed_ms=elapsed,
            )
        )
        return {"final_answer": final, "audit_log": audit}

    return synthesize


# ----------------------------- verify -----------------------------


def make_verify_node(llm: LLMClient) -> Callable[[ResearchState], ResearchState]:
    """Post-synthesis fact-check.

    Asks the LLM whether the final answer is supported by the retrieved
    snippets. If not, re-runs synthesize with a strict source-grounded prompt
    and accepts whatever that returns (including ``unknown``).
    """

    def verify(state: ResearchState) -> ResearchState:
        start = time.perf_counter()
        final = state.get("final_answer", "")
        if not final.strip():
            elapsed = int((time.perf_counter() - start) * 1000)
            audit = list(state.get("audit_log") or [])
            audit.append(
                make_audit_entry(node="verify", summary="No final answer to verify", elapsed_ms=elapsed)
            )
            return {"verified": False, "verification_notes": ["no final_answer"], "audit_log": audit}

        results = state.get("search_results", {})
        all_snippets: list[str] = []
        for hits in results.values():
            for hit in hits:
                all_snippets.append(f"- ({hit['title']}) {hit['snippet']}")
        snippet_block = "\n".join(all_snippets[:20]) or "(no snippets)"
        prompt = (
            f"Question: {state['question']}\n\n"
            f"Candidate final answer:\n{final}\n\n"
            f"Source snippets:\n{snippet_block}\n\n"
            "Is the candidate final answer fully supported by the snippets? Reply with the JSON object."
        )
        raw = llm.complete(prompt, system=VERIFY_SYSTEM, max_tokens=300)
        supported, issues = _parse_verify(raw)
        notes = list(state.get("verification_notes") or []) + issues

        if not supported:
            # One regenerate pass with strict prompt.
            re_prompt, _ = _synth_prompt(state)
            re_final = llm.complete(re_prompt, system=SYNTH_STRICT_SYSTEM, max_tokens=300).strip()
            elapsed = int((time.perf_counter() - start) * 1000)
            audit = list(state.get("audit_log") or [])
            audit.append(
                make_audit_entry(
                    node="verify",
                    summary=f"Fact-check failed: {len(issues)} issue(s). Regenerated under strict prompt.",
                    elapsed_ms=elapsed,
                )
            )
            return {
                "final_answer": re_final,
                "verified": False,
                "verification_notes": notes,
                "audit_log": audit,
            }

        elapsed = int((time.perf_counter() - start) * 1000)
        audit = list(state.get("audit_log") or [])
        audit.append(
            make_audit_entry(node="verify", summary="Fact-check passed", elapsed_ms=elapsed)
        )
        return {"verified": True, "verification_notes": notes, "audit_log": audit}

    return verify


def _parse_verify(raw: str) -> tuple[bool, list[str]]:
    text = extract_json(raw)
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            supported = bool(data.get("supported", False))
            issues = data.get("issues") or []
            if not isinstance(issues, list):
                issues = [str(issues)]
            return supported, [str(i) for i in issues if i]
    except json.JSONDecodeError:
        pass
    # Conservative fallback: if the JSON didn't parse but the raw mentions
    # supported/yes/true, accept; otherwise treat as unverified.
    lowered = raw.lower()
    if '"supported": true' in lowered or "supported: true" in lowered:
        return True, []
    return False, ["unparseable verify response"]
