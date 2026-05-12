---
title: LangGraph Research Pilot
emoji: 🧭
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: "4.44.1"
app_file: app.py
pinned: false
license: mit
---

# LangGraph Research Pilot

> Production RAG agent for Wikipedia question answering with provable grounding, citations, and crash resumability. Trades 22× latency for full auditability.

[![ci](https://github.com/Tajaddin/langgraph-research-pilot/actions/workflows/ci.yml/badge.svg)](https://github.com/Tajaddin/langgraph-research-pilot/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![Demo](https://img.shields.io/badge/demo-HF%20Space-yellow)](https://huggingface.co/spaces/Tajaddin85/langgraph-research-pilot)

**Live demo:** https://huggingface.co/spaces/Tajaddin85/langgraph-research-pilot

## Hero metrics

What changes when you switch from a single-prompt LLM to this agent. Numbers from a 30-question HotpotQA-fullwiki run with Claude Haiku 4.5 (seed 7).

| | Single-prompt baseline | This agent | Delta |
|---|---:|---:|---:|
| **Sources cited per answer** | **0** | **1-3 named Wikipedia titles** | every answer becomes auditable |
| **Snippets retrieved per question** | 0 | **14.8 (across 2.5 sub-questions)** | every claim has an inspectable source |
| **State checkpoint coverage** | none | **100%** (every node persists via `SqliteSaver`) | run survives crash, deploy, browser close |
| **Verify-driven abstentions** | 0 | **11 / 30 (37%)** | when sources don't support a claim, agent says "unknown" instead of hallucinating |
| **LLM cost per query** (Haiku 4.5) | ~$0.0002 | ~$0.008 | 40× higher cost, full receipts on every answer |
| **Avg latency per query** | 0.8 s | 17.9 s | 22× slower in exchange for audit, grounding, resume |

The agent halts at two human-in-the-loop gates (after retrieval and before answer drafting, and after drafting and before synthesis). It writes a state checkpoint after every node. Every reasoning step is recoverable from a `thread_id`.

## Accuracy is comparable. The value is verifiability, not raw correctness.

Same 30-question run, first-line extraction so both backends are scored apples-to-apples:

| Metric | Baseline | Agent | Lift |
|---|---:|---:|---:|
| Exact match | 23.3% | **23.3%** | tied |
| F1 (token overlap) | 0.313 | **0.368** | **+17.4% relative** |
| Contains gold (substring) | 30.0% | 30.0% | tied |

The agent does not move EM by design. It returns the same accuracy class as the underlying model with cited Wikipedia sources on every claim. Reviewers verify, contest, or override any single answer. The single-prompt baseline gives them nothing to verify against.

Full per-question table: [`RESULTS.md`](RESULTS.md). Raw benchmark output: [`bench/results_haiku_path2.json`](bench/results_haiku_path2.json).

## Architecture

```
                       ┌─────────┐
   START ──────────────▶ planner │   (LLM: decompose + generate
                       └────┬────┘    2-3 entity-style Wikipedia
                            │         queries per sub-question)
                          │
                       ┌──▼─────┐
                       │ search │ (Wikipedia REST API,
                       └──┬─────┘  multi-query merge by title,
                          │        freq + position re-scoring)
                          │
                     ◆ HITL gate #1 (interrupt_before answer)
                          │
                       ┌──▼─────┐
                       │ answer │ (LLM: per sub-question
                       └──┬─────┘  extract from snippets)
                          │
                     ◆ HITL gate #2 (interrupt_before synthesize)
                          │
                  ┌───────▼──────┐
                  │ synthesize   │ (LLM: entity-first
                  └───────┬──────┘  + grounding sentence
                          │         with cited sources)
                          │
                       ┌──▼─────┐
                       │ verify │ (LLM: fact-check final
                       └──┬─────┘  against sources. If
                          │        unsupported, regenerate
                          │        under a strict prompt)
                          │
                         END
```

State lives in a typed `TypedDict`, persisted by either `MemorySaver` (in-process) or `SqliteSaver` (durable). Both honor the same `thread_id` config. A killed run resumes from disk with no replay.

## Install

Requires Python 3.10+.

```bash
git clone https://github.com/Tajaddin/langgraph-research-pilot.git
cd langgraph-research-pilot
pip install -e ".[anthropic,groq,ui,eval,dev]"
```

## Usage

### CLI

```bash
# Mock backend, no API key. Useful for inspecting the flow.
research-pilot --llm mock --search mock "Who composed Va, pensiero?"

# Real backend, no HITL gates (autonomous)
export ANTHROPIC_API_KEY=sk-ant-...
research-pilot --llm anthropic --no-hitl "Which Nobel Prize did Marie Curie win first?"

# Real backend with HITL gates (prompts in terminal between steps)
research-pilot --llm anthropic "Which language is most widely spoken at the mouth of the Danube?"
```

### Python

```python
from langgraph_research_pilot.graph import build_graph

graph = build_graph()  # uses get_llm() + get_search() defaults
config = {"configurable": {"thread_id": "session-1"}}

# Run until the first HITL gate (after plan + search)
state = graph.invoke({"question": "Who composed Va, pensiero?"}, config=config)
print("plan:", state["plan"])
print("hits:", state["search_results"])

# User reviews, then resume
state = graph.invoke(None, config=config)
print("drafts:", state["answers"])

# Resume again past the second gate. The final answer goes through verify.
state = graph.invoke(None, config=config)
print("final:", state["final_answer"], "verified:", state.get("verified"))
```

### Resume after process kill (SQLite)

```python
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph_research_pilot.graph import build_graph

# Process A: start a run, exit at the gate
with SqliteSaver.from_conn_string("./checkpoints.db") as cp:
    graph = build_graph(checkpointer=cp)
    graph.invoke({"question": "..."}, config={"configurable": {"thread_id": "t1"}})

# Process B (later, fresh interpreter): pick up exactly where we left off
with SqliteSaver.from_conn_string("./checkpoints.db") as cp:
    graph = build_graph(checkpointer=cp)
    final = graph.invoke(None, config={"configurable": {"thread_id": "t1"}})
```

Covered by [`tests/test_checkpoint.py`](tests/test_checkpoint.py). A fresh Python interpreter scope and a fresh `SqliteSaver` against the same DB file resume the run from `thread_id` alone.

## Demo

Live Gradio Space: https://huggingface.co/spaces/Tajaddin85/langgraph-research-pilot. The Space uses HF Inference Provider on the free tier. With no token set, it falls back to `MockLLM` so the UI flow stays inspectable.

Local:
```bash
pip install -e ".[anthropic,ui]"
python app.py
```

## Configuration

| Env var | Purpose |
|---|---|
| `RESEARCH_PILOT_LLM` | `mock`, `hf`, `anthropic`, `groq` |
| `RESEARCH_PILOT_SEARCH` | `mock`, `wikipedia` |
| `HF_TOKEN` | HuggingFace Inference token (free tier OK) |
| `HF_MODEL` | HF model id (default: `meta-llama/Llama-3.2-3B-Instruct`) |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `ANTHROPIC_MODEL` | Anthropic model id (default: `claude-haiku-4-5-20251001`) |
| `GROQ_API_KEY` | Groq API key |
| `GROQ_MODEL` | Groq model id (default: `llama-3.1-8b-instant`) |
| `GROQ_MIN_INTERVAL_SECS` | Pacing for Groq's 30 RPM free tier (default: 2.5) |

## Tests

35 tests covering parsing, individual nodes, end-to-end graph, HITL pause/resume, **SQLite kill-and-resume across a fresh Python process**, and verify-node fact-checking:

```bash
pip install -e ".[dev]"
pytest -ra
```

```
============================== 35 passed in 0.34s ==============================
```

## Reproduce the benchmark

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python bench/hotpot_eval.py --limit 30 --seed 7 --llm anthropic --model claude-haiku-4-5-20251001
python bench/recompute.py   # extract first-line entity for apples-to-apples scoring
```

Outputs go to `bench/results.json` and `RESULTS.md`. Uses public Wikipedia REST API for retrieval (no key needed) and ~210 LLM calls (~$0.24 on Haiku 4.5 for the agent run, ~$0.005 for the baseline).

## Deploy to HuggingFace Spaces

This README's frontmatter already configures the Space. To publish:

1. Create a new Space at https://huggingface.co/new-space (Gradio SDK)
2. In the Space's Settings → Variables and secrets, add `HF_TOKEN` so the Space reaches an Inference Provider
3. Push this repo to the Space:
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/langgraph-research-pilot
   git push space main
   ```
4. Spaces will install `requirements.txt` and serve `app.py` automatically.

Without `HF_TOKEN`, the Space starts in `MockLLM` mode and the UI flow is still inspectable.

## Caveats

Llama-3.1-8B regressed. A separate full 30-question run via Groq with `llama-3.1-8b-instant` produced a negative lift: agent EM 16.7 % vs baseline 23.3 % (−28 % relative), agent F1 0.270 vs baseline 0.299 (−10 % relative).

Root cause: Llama-3.1-8B's instruction-following ceiling. The agent depends on the synthesizer producing `entity-first\n\ngrounding sentence with citations`. Haiku 4.5 follows the format. Llama-3.1-8B routinely deviates. It names the question's subject instead of the answer (for example "King George V" for a "what year was he born" question, with the year buried in the grounding sentence), or it answers a different question type (date instead of comparison, company instead of rank).

A production fix is constrained decoding (Outlines, Guidance, or Instructor with a Pydantic schema) so the model fails closed on a non-conforming first line. Out of scope here. See [`bench/results_llama_groq.json`](bench/results_llama_groq.json) for the raw Llama run.

HotpotQA fullwiki is hard. State-of-the-art fullwiki accuracy with retrieval-only (no specialized re-rankers) sits in the 30-50 % EM range. The 23 % baseline and 23 % agent numbers reflect Haiku 4.5's parametric knowledge plus generic Wikipedia retrieval, not a fine-tuned QA system.

No streaming. Each node runs to completion before emitting a state delta. A token-streaming variant of synthesize is a natural extension.

## Project layout

```
.
├── app.py                                  # Gradio app (HF Space entrypoint)
├── src/langgraph_research_pilot/
│   ├── state.py                            # Typed state (TypedDicts)
│   ├── llm.py                              # Pluggable LLM: Mock / HF / Anthropic / Groq
│   ├── search.py                           # Wikipedia REST + Mock search
│   ├── nodes.py                            # plan / search / answer / synthesize / verify
│   ├── graph.py                            # StateGraph + interrupts + SqliteSaver
│   └── cli.py                              # research-pilot CLI
├── tests/                                  # 35 pytest cases
├── bench/
│   ├── hotpot_eval.py                      # benchmark runner
│   ├── recompute.py                        # apples-to-apples first-line scoring
│   ├── results_haiku_path2.json            # hero numbers (Haiku 4.5)
│   └── results_llama_groq.json             # Llama-3.1-8B comparison (regressed)
└── RESULTS.md                              # full per-question breakdown
```

## Contributing

PRs welcome for: streaming-mode synthesis, constrained decoding integration (Outlines/Instructor), additional search backends (e.g. arXiv, OpenFDA), and additional eval datasets.

## License

MIT. See [LICENSE](LICENSE).
