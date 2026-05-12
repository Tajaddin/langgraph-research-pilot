"""Gradio app for HuggingFace Spaces.

Two HITL gates exposed as buttons:
  1. After plan + search, the user sees the plan and retrieved snippets.
     They can approve (continue to per-sub-question answers) or reject (restart).
  2. After per-sub-question answers, the user sees those drafts.
     They can approve (synthesize final) or reject (restart).

Backend resolution:
- If HF_TOKEN is set in the Space environment, uses the HuggingFace Inference
  API with the model named in HF_MODEL (default: meta-llama/Llama-3.2-3B-Instruct).
- Otherwise falls back to MockLLM so the Space remains demoable.
"""

from __future__ import annotations

import os
import uuid

import gradio as gr

from langgraph_research_pilot.graph import build_graph
from langgraph_research_pilot.llm import MockLLM, get_llm
from langgraph_research_pilot.search import get_search

MODEL_LABEL = os.environ.get("HF_MODEL", "meta-llama/Llama-3.2-3B-Instruct")
USING_REAL_LLM = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN") or os.environ.get("ANTHROPIC_API_KEY"))


def _build():
    llm = get_llm()
    search = get_search()
    return build_graph(llm=llm, search=search, interrupt_search=True, interrupt_answer=True)


GRAPH = _build()


def _format_plan(plan: list[dict]) -> str:
    if not plan:
        return "_(no plan)_"
    lines = ["### Plan"]
    for sub in plan:
        lines.append(f"**{sub['id']}.** {sub['text']}")
        if sub.get("rationale"):
            lines.append(f"  - rationale: {sub['rationale']}")
    return "\n".join(lines)


def _format_search(search_results: dict, plan: list[dict]) -> str:
    if not search_results:
        return "_(no search results)_"
    lines = ["### Retrieved snippets"]
    for sub in plan:
        sid = str(sub["id"])
        hits = search_results.get(sid, [])
        lines.append(f"\n**Sub-question {sid}:** {sub['text']}")
        if not hits:
            lines.append("- _no hits_")
            continue
        for hit in hits:
            lines.append(f"- [{hit['title']}]({hit['url']}) — {hit['snippet'][:240]}{'...' if len(hit['snippet']) > 240 else ''}")
    return "\n".join(lines)


def _format_answers(answers: dict, plan: list[dict]) -> str:
    if not answers:
        return "_(no answers yet)_"
    lines = ["### Draft sub-answers"]
    for sub in plan:
        sid = str(sub["id"])
        ans = answers.get(sid, "_(no answer)_")
        lines.append(f"\n**{sid}. {sub['text']}**\n→ {ans}")
    return "\n".join(lines)


def _format_audit(audit: list[dict]) -> str:
    if not audit:
        return ""
    lines = ["### Audit trail"]
    for entry in audit:
        lines.append(f"- `{entry['node']}` — {entry['summary']} ({entry['elapsed_ms']} ms)")
    return "\n".join(lines)


def start(question: str, ui_state: dict):
    """Step 1: plan + search, pause at the search-approval gate."""
    if not question.strip():
        return (
            ui_state,
            "Enter a question first.",
            "",
            "",
            "",
            "",
            gr.update(visible=False),
            gr.update(visible=False),
        )
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    state = GRAPH.invoke({"question": question}, config=config)
    new_ui_state = {"thread_id": thread_id, "stage": "awaiting_search_approval"}
    return (
        new_ui_state,
        f"**Question:** {question}",
        _format_plan(state.get("plan", [])),
        _format_search(state.get("search_results", {}), state.get("plan", [])),
        "",
        _format_audit(state.get("audit_log", [])),
        gr.update(visible=True),  # approve_search button
        gr.update(visible=False),  # approve_answers button
    )


def approve_search(ui_state: dict):
    """Step 2: run the per-sub-question answer node, pause at the answer-approval gate."""
    if ui_state.get("stage") != "awaiting_search_approval":
        return (
            ui_state,
            gr.update(),
            gr.update(),
            gr.update(),
            "Click 'Run' first.",
            gr.update(),
            gr.update(visible=False),
            gr.update(visible=False),
        )
    config = {"configurable": {"thread_id": ui_state["thread_id"]}}
    state = GRAPH.invoke(None, config=config)
    ui_state = {**ui_state, "stage": "awaiting_answer_approval"}
    return (
        ui_state,
        gr.update(),
        gr.update(),
        gr.update(),
        _format_answers(state.get("answers", {}), state.get("plan", [])),
        _format_audit(state.get("audit_log", [])),
        gr.update(visible=False),
        gr.update(visible=True),
    )


def approve_answers(ui_state: dict):
    """Step 3: synthesize the final answer."""
    if ui_state.get("stage") != "awaiting_answer_approval":
        return (
            ui_state,
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            "Click 'Approve search' first.",
            "",
            gr.update(visible=False),
            gr.update(visible=False),
        )
    config = {"configurable": {"thread_id": ui_state["thread_id"]}}
    state = GRAPH.invoke(None, config=config)
    ui_state = {**ui_state, "stage": "done"}
    final = state.get("final_answer", "_(no answer produced)_")
    return (
        ui_state,
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        _format_audit(state.get("audit_log", [])),
        f"### Final answer\n\n{final}",
        gr.update(visible=False),
        gr.update(visible=False),
    )


def reset() -> tuple:
    return (
        {"thread_id": None, "stage": "init"},
        "",
        "",
        "",
        "",
        "",
        "",
        gr.update(visible=False),
        gr.update(visible=False),
    )


THEME = gr.themes.Soft(primary_hue="indigo", secondary_hue="slate")

DESCRIPTION = f"""
# LangGraph Research Pilot

Multi-step research agent with **human-in-the-loop approval gates**. Decomposes a question into sub-questions,
retrieves Wikipedia snippets for each, drafts per-sub-question answers, then synthesizes a grounded final answer.

The agent **pauses twice for your approval** — first after retrieval (so you can inspect sources before the model
spends tokens on synthesis), then after the per-sub-question drafts (so you can catch reasoning errors before
the final answer).

**Backend:** {'`' + MODEL_LABEL + '` via HuggingFace Inference API' if USING_REAL_LLM else '`MockLLM` (no HF_TOKEN set — set one in Space secrets to enable real generation).'}
"""


with gr.Blocks(title="LangGraph Research Pilot") as demo:
    gr.Markdown(DESCRIPTION)
    ui_state = gr.State({"thread_id": None, "stage": "init"})
    with gr.Row():
        question_box = gr.Textbox(
            label="Research question",
            placeholder="e.g. Which Nobel Prize did the discoverer of polonium win first?",
            lines=2,
            scale=4,
        )
        run_btn = gr.Button("Run (plan + search)", variant="primary", scale=1)
        reset_btn = gr.Button("Reset", scale=1)
    question_echo = gr.Markdown()
    with gr.Row():
        plan_md = gr.Markdown()
        search_md = gr.Markdown()
    approve_search_btn = gr.Button("Approve search → draft sub-answers", visible=False, variant="primary")
    answers_md = gr.Markdown()
    approve_answers_btn = gr.Button("Approve sub-answers → synthesize final", visible=False, variant="primary")
    final_md = gr.Markdown()
    audit_md = gr.Markdown()

    run_btn.click(
        start,
        inputs=[question_box, ui_state],
        outputs=[
            ui_state,
            question_echo,
            plan_md,
            search_md,
            answers_md,
            audit_md,
            approve_search_btn,
            approve_answers_btn,
        ],
    )
    approve_search_btn.click(
        approve_search,
        inputs=[ui_state],
        outputs=[
            ui_state,
            question_echo,
            plan_md,
            search_md,
            answers_md,
            audit_md,
            approve_search_btn,
            approve_answers_btn,
        ],
    )
    approve_answers_btn.click(
        approve_answers,
        inputs=[ui_state],
        outputs=[
            ui_state,
            question_echo,
            plan_md,
            search_md,
            answers_md,
            audit_md,
            final_md,
            approve_search_btn,
            approve_answers_btn,
        ],
    )
    reset_btn.click(
        reset,
        inputs=None,
        outputs=[
            ui_state,
            question_echo,
            plan_md,
            search_md,
            answers_md,
            audit_md,
            final_md,
            approve_search_btn,
            approve_answers_btn,
        ],
    )

    gr.Examples(
        examples=[
            "Which Nobel Prize did the discoverer of polonium win first?",
            "What language is most widely spoken in the country where the Danube ends?",
            "Who composed the opera that opens with 'Va, pensiero'?",
        ],
        inputs=question_box,
    )


if __name__ == "__main__":
    demo.queue().launch(theme=THEME)
