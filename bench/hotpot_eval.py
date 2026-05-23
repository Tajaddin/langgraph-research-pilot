"""HotpotQA evaluation: single-prompt baseline vs multi-step agent.

Pulls 30 validation questions from HotpotQA, runs both conditions against the
same LLM, computes exact match and F1 token overlap, and dumps results to
``bench/results.json`` and ``bench/RESULTS.md``.

Run with the project venv active and an ANTHROPIC_API_KEY (or any other
``get_llm``-supported backend) in the environment:

    python bench/hotpot_eval.py --limit 30 --seed 7 --model claude-haiku-4-5-20251001
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import string
import sys
import time
from collections.abc import Iterable
from pathlib import Path

# Make src importable when running this script directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from langgraph_research_pilot.graph import build_graph  # noqa: E402
from langgraph_research_pilot.llm import LLMClient, get_llm  # noqa: E402
from langgraph_research_pilot.search import get_search  # noqa: E402

BENCH_DIR = Path(__file__).resolve().parent
RESULTS_JSON = BENCH_DIR / "results.json"
RESULTS_MD = BENCH_DIR.parent / "RESULTS.md"


def normalize(text: str) -> str:
    """SQuAD-style normalization: lowercase, strip articles + punctuation."""
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if normalize(pred) == normalize(gold) else 0.0


def f1_score(pred: str, gold: str) -> float:
    pred_tokens = normalize(pred).split()
    gold_tokens = normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common: dict[str, int] = {}
    for tok in pred_tokens:
        common[tok] = min(pred_tokens.count(tok), gold_tokens.count(tok))
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def gold_in_prediction(pred: str, gold: str) -> bool:
    """Looser match: does the normalized gold appear as a substring in pred?"""
    return normalize(gold) in normalize(pred)


def load_questions(limit: int, seed: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("hotpot_qa", "fullwiki", split="validation", trust_remote_code=False)
    rng = random.Random(seed)
    # Hotpot validation has yes/no comparison + bridge questions. Mix levels for fairness.
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    picks: list[dict] = []
    for idx in indices:
        item = ds[idx]
        if not item.get("question") or not item.get("answer"):
            continue
        picks.append(
            {
                "id": item["id"],
                "question": item["question"],
                "answer": item["answer"],
                "type": item.get("type"),
                "level": item.get("level"),
            }
        )
        if len(picks) >= limit:
            break
    return picks


BASELINE_SYSTEM = (
    "You are a research assistant. Answer the user's question in 1 to 5 words. "
    "Answer only with the entity or short phrase. Do not write a sentence. Do not explain."
)


def baseline_answer(llm: LLMClient, question: str) -> str:
    raw = llm.complete(question, system=BASELINE_SYSTEM, max_tokens=40)
    # Strip likely sentence padding.
    raw = raw.strip().split("\n")[0]
    raw = re.sub(r"^(answer:|the answer is)\s*", "", raw, flags=re.IGNORECASE).strip()
    return raw


def agent_answer(graph, question: str, thread_id: str) -> tuple[str, dict]:
    config = {"configurable": {"thread_id": thread_id}}
    state = graph.invoke({"question": question}, config=config)
    return state.get("final_answer", "") or "", state


def main() -> int:
    # Windows consoles default to cp1252 — HotpotQA has plenty of non-Latin titles.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=30, help="Number of questions")
    parser.add_argument("--seed", type=int, default=7, help="RNG seed for question selection")
    parser.add_argument("--llm", type=str, default=None, help="LLM backend override (anthropic, hf, mock)")
    parser.add_argument("--model", type=str, default=None, help="Specific model id (sets ANTHROPIC_MODEL or HF_MODEL)")
    parser.add_argument("--out", type=str, default=str(RESULTS_JSON))
    args = parser.parse_args()

    if args.model:
        backend = (args.llm or os.environ.get("RESEARCH_PILOT_LLM") or "").lower()
        if backend == "anthropic" or os.environ.get("ANTHROPIC_API_KEY"):
            os.environ["ANTHROPIC_MODEL"] = args.model
        if backend == "hf" or os.environ.get("HF_TOKEN"):
            os.environ["HF_MODEL"] = args.model

    llm = get_llm(args.llm)
    search = get_search()
    graph = build_graph(
        llm=llm, search=search, interrupt_search=False, interrupt_answer=False, k=6
    )

    questions = load_questions(args.limit, args.seed)
    rows: list[dict] = []
    print(f"Running {len(questions)} questions with backend={type(llm).__name__}")
    t_total_baseline = 0.0
    t_total_agent = 0.0
    for i, q in enumerate(questions, start=1):
        print(f"[{i}/{len(questions)}] {q['question'][:90]}...")
        t0 = time.perf_counter()
        try:
            base = baseline_answer(llm, q["question"])
        except Exception as exc:  # noqa: BLE001
            base = f"<ERROR: {exc.__class__.__name__}>"
        t_baseline = time.perf_counter() - t0
        t_total_baseline += t_baseline

        t0 = time.perf_counter()
        try:
            agent_final, agent_state = agent_answer(graph, q["question"], thread_id=f"hotpot-{q['id']}")
        except Exception as exc:  # noqa: BLE001
            agent_final = f"<ERROR: {exc.__class__.__name__}>"
            agent_state = {}
        t_agent = time.perf_counter() - t0
        t_total_agent += t_agent

        row = {
            "id": q["id"],
            "question": q["question"],
            "gold": q["answer"],
            "type": q["type"],
            "level": q["level"],
            "baseline": base,
            "agent": agent_final,
            "baseline_em": exact_match(base, q["answer"]),
            "agent_em": exact_match(agent_final, q["answer"]),
            "baseline_f1": f1_score(base, q["answer"]),
            "agent_f1": f1_score(agent_final, q["answer"]),
            "baseline_contains_gold": gold_in_prediction(base, q["answer"]),
            "agent_contains_gold": gold_in_prediction(agent_final, q["answer"]),
            "baseline_secs": round(t_baseline, 2),
            "agent_secs": round(t_agent, 2),
            "agent_plan_size": len(agent_state.get("plan", [])),
            "agent_hits": sum(len(v) for v in (agent_state.get("search_results") or {}).values()),
        }
        rows.append(row)
        print(
            f"   gold={q['answer'][:60]!r:64} base={base[:50]!r:54} agent={agent_final[:50]!r}"
        )

    summary = summarize(rows)
    summary["timing"] = {
        "baseline_total_secs": round(t_total_baseline, 1),
        "agent_total_secs": round(t_total_agent, 1),
        "baseline_avg_secs": round(t_total_baseline / max(len(rows), 1), 2),
        "agent_avg_secs": round(t_total_agent / max(len(rows), 1), 2),
    }
    summary["meta"] = {
        "backend": type(llm).__name__,
        "model": getattr(llm, "model", None),
        "n": len(rows),
        "seed": args.seed,
    }

    out_path = Path(args.out)
    out_path.write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nWrote {out_path}")
    print(json.dumps(summary, indent=2))

    write_markdown(summary, rows, RESULTS_MD)
    print(f"Wrote {RESULTS_MD}")
    return 0


def summarize(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    n = len(rows)

    def avg(key: str) -> float:
        return round(sum(r[key] for r in rows) / n, 4) if n else 0.0

    return {
        "baseline_em": avg("baseline_em"),
        "agent_em": avg("agent_em"),
        "baseline_f1": avg("baseline_f1"),
        "agent_f1": avg("agent_f1"),
        "baseline_contains_gold": round(sum(1 for r in rows if r["baseline_contains_gold"]) / n, 4) if n else 0.0,
        "agent_contains_gold": round(sum(1 for r in rows if r["agent_contains_gold"]) / n, 4) if n else 0.0,
    }


def write_markdown(summary: dict, rows: list[dict], path: Path) -> None:
    lines: list[str] = []
    lines.append("# HotpotQA benchmark — results\n")
    meta = summary.get("meta", {})
    timing = summary.get("timing", {})
    lines.append(
        f"Backend: `{meta.get('backend')}` | Model: `{meta.get('model')}` | n={meta.get('n')} | seed={meta.get('seed')}\n"
    )
    lines.append("## Headline\n")
    lines.append(
        f"| Metric | Single-prompt baseline | Multi-step agent | Lift |\n"
        f"|---|---:|---:|---:|\n"
        f"| Exact match | {summary['baseline_em']:.1%} | {summary['agent_em']:.1%} | {summary['agent_em']-summary['baseline_em']:+.1%} |\n"
        f"| F1 (token overlap) | {summary['baseline_f1']:.3f} | {summary['agent_f1']:.3f} | {summary['agent_f1']-summary['baseline_f1']:+.3f} |\n"
        f"| Contains gold (substring) | {summary['baseline_contains_gold']:.1%} | {summary['agent_contains_gold']:.1%} | {summary['agent_contains_gold']-summary['baseline_contains_gold']:+.1%} |\n"
    )
    lines.append("## Timing\n")
    lines.append(
        f"- Baseline: total {timing.get('baseline_total_secs')}s, avg {timing.get('baseline_avg_secs')}s/question\n"
        f"- Agent: total {timing.get('agent_total_secs')}s, avg {timing.get('agent_avg_secs')}s/question\n"
    )
    lines.append("\n## Per-question detail\n")
    lines.append("| # | Type | Level | Question | Gold | Baseline | Agent | Base F1 | Agent F1 |\n")
    lines.append("|---|---|---|---|---|---|---|---:|---:|\n")
    for i, r in enumerate(rows, start=1):
        q = r["question"].replace("|", "/")[:80]
        gold = r["gold"].replace("|", "/")[:60]
        base = r["baseline"].replace("|", "/").replace("\n", " ")[:60]
        agent = r["agent"].replace("|", "/").replace("\n", " ")[:60]
        lines.append(
            f"| {i} | {r.get('type')} | {r.get('level')} | {q} | {gold} | {base} | {agent} | {r['baseline_f1']:.2f} | {r['agent_f1']:.2f} |\n"
        )
    path.write_text("".join(lines), encoding="utf-8")  # explicit utf-8 for HotpotQA non-Latin titles


if __name__ == "__main__":
    sys.exit(main())
