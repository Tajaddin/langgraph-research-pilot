"""CLI entrypoint.

Usage:
    research-pilot "Who was the first person to summit K2?"
    research-pilot --no-hitl "Capital of Burkina Faso?"
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid

from langgraph_research_pilot.graph import build_graph


def _print_state_block(label: str, state: dict) -> None:
    print(f"\n=== {label} ===")
    plan = state.get("plan") or []
    if plan:
        print("Plan:")
        for sub in plan:
            print(f"  [{sub['id']}] {sub['text']}")
    results = state.get("search_results") or {}
    if results:
        print("\nSearch hits:")
        for sub_id, hits in results.items():
            print(f"  sub {sub_id}: {len(hits)} hits")
            for hit in hits[:2]:
                print(f"    - {hit['title']} ({hit['url']})")
    answers = state.get("answers") or {}
    if answers:
        print("\nSub-answers:")
        for sub_id, ans in answers.items():
            print(f"  [{sub_id}] {ans}")
    final = state.get("final_answer")
    if final:
        print(f"\nFinal answer:\n{final}")


def _prompt_continue(label: str) -> bool:
    try:
        choice = input(f"\n[HITL] Approve {label}? [y/n/feedback...]: ").strip()
    except (EOFError, KeyboardInterrupt):
        return False
    if not choice or choice.lower().startswith("y"):
        return True
    return not choice.lower().startswith("n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="research-pilot", description=__doc__)
    parser.add_argument("question", type=str, help="Research question")
    parser.add_argument("--llm", type=str, default=None, help="LLM backend (mock, hf, anthropic)")
    parser.add_argument("--search", type=str, default=None, help="Search backend (mock, wikipedia)")
    parser.add_argument("--no-hitl", action="store_true", help="Skip human approval gates")
    parser.add_argument("--thread", type=str, default=None, help="Resume from a specific thread id")
    parser.add_argument("--json", dest="as_json", action="store_true", help="Emit final state as JSON")
    args = parser.parse_args(argv)

    from langgraph_research_pilot.llm import get_llm
    from langgraph_research_pilot.search import get_search

    llm = get_llm(args.llm)
    search = get_search(args.search)
    graph = build_graph(
        llm=llm,
        search=search,
        interrupt_search=not args.no_hitl,
        interrupt_answer=not args.no_hitl,
    )

    thread_id = args.thread or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    state = graph.invoke({"question": args.question}, config=config)
    _print_state_block("After search (awaiting review)", state)

    if not args.no_hitl:
        if not _prompt_continue("search results"):
            print("Aborted at search-review gate.")
            return 2
        state = graph.invoke(None, config=config)
        _print_state_block("After sub-answers (awaiting review)", state)
        if not _prompt_continue("sub-question answers"):
            print("Aborted at answer-review gate.")
            return 2
        state = graph.invoke(None, config=config)

    if args.as_json:
        out = {
            "question": state.get("question"),
            "plan": list(state.get("plan") or []),
            "answers": state.get("answers", {}),
            "final_answer": state.get("final_answer"),
            "thread_id": thread_id,
        }
        print(json.dumps(out, indent=2))
    else:
        _print_state_block("Final", state)
    return 0


if __name__ == "__main__":
    sys.exit(main())
