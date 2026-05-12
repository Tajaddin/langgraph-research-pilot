"""Recompute metrics with first-line extraction from the existing results.json.

The agent intentionally outputs ``ENTITY\\n\\ngrounding sentence`` per the system
prompt. To compare apples-to-apples with the baseline (entity only), extract the
first non-empty line of the agent's answer before scoring.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hotpot_eval import exact_match, f1_score, gold_in_prediction, summarize, write_markdown

BENCH = Path(__file__).resolve().parent
JSON_PATH = BENCH / "results.json"
MD_PATH = BENCH.parent / "RESULTS.md"


def extract_short(text: str) -> str:
    if not text:
        return ""
    for line in text.splitlines():
        cleaned = line.strip().strip("*").strip()
        if cleaned:
            return cleaned
    return text.strip()


def main() -> None:
    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    rows = data["rows"]
    for r in rows:
        short = extract_short(r["agent"])
        r["agent_short"] = short
        r["agent_em"] = exact_match(short, r["gold"])
        r["agent_f1"] = f1_score(short, r["gold"])
        r["agent_contains_gold"] = gold_in_prediction(short, r["gold"])
    summary = summarize(rows)
    summary["timing"] = data["summary"].get("timing")
    summary["meta"] = data["summary"].get("meta")
    summary["meta"]["scoring"] = "first-line extraction of agent output"
    data["summary"] = summary
    data["rows"] = rows
    JSON_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(summary, rows, MD_PATH)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
