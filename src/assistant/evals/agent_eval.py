"""Agent eval (lesson 6.4): drive the agent over curated tasks, score 4 metrics.

    uv run evals --agent        # run all tasks, print the agent scoreboard, save a run

Four numbers (the dossier's rows):
  task success rate   : did the final answer / final DB state pass the task's check?
  valid tool-call rate: of every tool the agent ran, how many were well-formed (real
                        tool, no "Error:" result)?
  average steps       : model turns per task  (tool calls + 1 answering turn)
  median latency      : wall-clock seconds per task

Two things make this different from the golden eval:
  1. There is no human at an eval, so writes are AUTO-APPROVED: when the loop yields
     approval_request, we call resume_agent(approved=True) ourselves and continue.
  2. Each task runs and is scored against its own in-memory database, seeded with
     the persona calendar and one reminder.
"""

import json
import statistics
import time
from datetime import datetime
from pathlib import Path

from .. import agent
from ..agent import tools
from ..mcp import client as mcp_client
from ..trace import capture
from .agent_checks import CHECKS
from .stats import wilson

TASKS = Path("data/evals/agent_tasks.jsonl")
OUT_DIR = Path("data/evals")
SEED_REMINDER = ("Submit expense report", "2024-05-16")  # baseline row OWN1 reads


def load_tasks() -> list[dict]:
    with TASKS.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def drive(task_text: str) -> list:
    """Run one task to completion, auto-approving every write. Returns all events."""
    events, gen = [], agent.run_agent(task_text, [], mcp_client.SCHEMAS)
    while True:
        pending = None
        for e in gen:
            events.append(e)
            if e.kind == "approval_request":
                pending = e  # loop stopped here; we approve on the agent's behalf
                break
        if pending is None:
            return events  # generator ran out -> a `done` was the last event
        gen = agent.resume_agent(
            pending.data["state"],
            pending.text,
            pending.data["args"],
            True,
            mcp_client.SCHEMAS,
            tainted=pending.data.get("tainted", False),
        )


def _tool_stats(events: list) -> tuple[int, int]:
    """(valid calls, total calls). A call is valid when it ran a real tool and the
    result wasn't an error string (errors-as-results, from agent._execute)."""
    names = [e.text for e in events if e.kind == "tool_call"]
    results = [e.data.get("result", "") for e in events if e.kind == "tool_result"]
    valid = sum(
        1
        for name, res in zip(names, results)
        if name in tools._BODIES and not res.startswith("Error:")
    )
    return valid, len(names)


def run_task(task: dict, driver=drive) -> dict:
    """Run and score one task against a disposable database. `driver` is the agent
    under test — the hand-written loop (default) or agent_pydantic.drive (6.6)."""
    if not tools.CALENDAR_SEED.is_file():
        raise FileNotFoundError(tools.CALENDAR_SEED)
    with tools.temporary_database() as conn:
        with conn:
            conn.execute(
                "INSERT INTO reminders (text,date) VALUES (?,?)", SEED_REMINDER
            )
        t0 = time.perf_counter()
        events = driver(task["task"])
        latency_s = time.perf_counter() - t0

        answer = next((e.text for e in reversed(events) if e.kind == "done"), "")
        try:
            passed = bool(CHECKS[task["check"]](answer, conn))
        except Exception as e:  # noqa: BLE001 — a broken check is a fail, not a crash
            passed, answer = False, f"[check error: {e}] {answer}"

    n_calls = sum(1 for e in events if e.kind == "tool_call")
    valid_calls, total_calls = _tool_stats(events)

    capture(task["task"], events)  # persist a trace per task, like every real turn
    return {
        "id": task["id"],
        "task": task["task"],
        "kind": task["kind"],
        "passed": passed,
        "answer": answer,
        "steps": n_calls + 1,  # tool turns + the final answering turn
        "valid_calls": valid_calls,
        "total_calls": total_calls,
        "latency_s": round(latency_s, 2),
    }


def score(results: list[dict]) -> dict:
    passed = sum(r["passed"] for r in results)
    valid = sum(r["valid_calls"] for r in results)
    total = sum(r["total_calls"] for r in results)
    return {
        "task_success": [passed, len(results)],
        "valid_tool_calls": [valid, total],
        "avg_steps": round(statistics.mean(r["steps"] for r in results), 1),
        "median_lat_s": round(statistics.median(r["latency_s"] for r in results), 1),
    }


def print_scoreboard(s: dict, results: list[dict]) -> None:
    def rate(label: str, pair: list[int]) -> None:
        k, n = pair
        p, lo, hi = wilson(k, n)
        print(f"  {label:<22} {p:6.1%}  [{lo:4.0%}, {hi:4.0%}]   ({k}/{n})")

    print("\n" + "=" * 60)
    print(f"AGENT SCOREBOARD   n={len(results)} tasks")
    print("=" * 60)
    rate("task success", s["task_success"])
    rate("valid tool-call rate", s["valid_tool_calls"])
    print(f"  {'average steps':<22} {s['avg_steps']:>6}")
    print(f"  {'median latency':<22} {s['median_lat_s']:>5}s")
    print("=" * 60)
    print("  per task:")
    for r in results:
        mark = "ok  " if r["passed"] else "FAIL"
        print(
            f"    {mark} {r['id']:5} steps={r['steps']} "
            f"calls={r['valid_calls']}/{r['total_calls']} "
            f"{r['latency_s']:4.1f}s  {r['task'][:40]}"
        )
    print("=" * 60)


def run(limit: int | None = None, framework: bool = False) -> None:
    tasks = load_tasks()[:limit] if limit else load_tasks()
    driver, label, tag = drive, "hand-written loop (agent.py)", "ours"
    if framework:
        from ..agent import pydantic_loop as agent_pydantic

        driver, label, tag = (
            agent_pydantic.drive,
            "Pydantic AI (agent_pydantic.py)",
            "pai",
        )
    print(f"Running {len(tasks)} agent tasks via {label}…")
    results = []
    for i, task in enumerate(tasks, 1):
        r = run_task(task, driver)
        results.append(r)
        mark = "ok  " if r["passed"] else "FAIL"
        print(f"  [{i:2}/{len(tasks)}] {mark} {r['id']:5} {r['task'][:45]}", flush=True)

    s = score(results)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = OUT_DIR / f"agent-run-{tag}-{stamp}.json"
    out.write_text(json.dumps({"ts": stamp, "score": s, "results": results}, indent=2))
    print_scoreboard(s, results)
    print(f"\nSaved {out}")
