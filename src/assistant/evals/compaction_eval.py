"""Compare the history policies on the twelve prepared long-chat probes."""

import json
from datetime import datetime
from pathlib import Path

from ..context import (
    LOCAL_HISTORY_BUDGET,
    compact,
    estimate_tokens,
    messages_to_text,
    trim_history,
)
from ..context.summary import summarize_history
from ..models.ollama import LOCAL_CONTEXT_WINDOW, ask_without_stream
from ..respond import PRIMARY_LOCAL_MODEL
from .judge import LOCAL_JUDGE, judge_one

PROBES = Path("data/assets/memory/longmemeval_32.json")
GROUPS = {"recalls-fact", "combines-sessions"}
ANSWER_SYSTEM = (
    "Answer the final question using the conversation history. Put the requested "
    "answer first, in one short sentence. For a count, state the total first. "
    "If the conversation lacks the evidence, say so."
)


def _answer(history: list[dict], question: str, expected: str) -> dict:
    answer = ask_without_stream(
        [
            {"role": "system", "content": ANSWER_SYSTEM},
            *history,
            {"role": "user", "content": question},
        ],
        PRIMARY_LOCAL_MODEL,
        options={"temperature": 0, "num_predict": 192},
    )
    verdict, reason = judge_one(question, expected, answer, model=LOCAL_JUDGE)
    return {
        "answer": answer,
        "verdict": verdict,
        "reason": reason,
        "history_tokens_est": round(estimate_tokens(messages_to_text(history))),
    }


def run() -> None:
    rows = json.loads(PROBES.read_text())
    results = []
    for row in rows:
        if row["probe_group"] not in GROUPS:
            continue
        history = [
            {"role": message["role"], "content": message["content"]}
            for session in row["haystack_sessions"]
            for message in session
        ]
        before = estimate_tokens(messages_to_text(history))
        trimmed, _ = trim_history(history, before, LOCAL_HISTORY_BUDGET)
        compacted = compact(
            history, lambda text: summarize_history(text, PRIMARY_LOCAL_MODEL)
        )
        question, expected = row["question"], str(row["answer"])
        trim = _answer(trimmed, question, expected)
        # The same input gets one answer across both policies on under-budget rows.
        changed = compacted != trimmed
        result = {
            "id": row["question_id"],
            "group": row["probe_group"],
            "question": question,
            "expected": expected,
            "compacted": changed,
            "trim": trim,
            "compact": _answer(compacted, question, expected) if changed else trim,
        }
        if changed:
            result["summary"] = compacted[0]["content"]
        results.append(result)
        print(result["id"], trim["verdict"], result["compact"]["verdict"], flush=True)

    out = Path("data/evals") / f"compaction-{datetime.now():%Y%m%dT%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "model": PRIMARY_LOCAL_MODEL,
                "judge": LOCAL_JUDGE,
                "answer_system": ANSWER_SYSTEM,
                "context_window": LOCAL_CONTEXT_WINDOW,
                "history_budget": LOCAL_HISTORY_BUDGET,
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )
    for label, group in (
        ("All probes", results),
        ("Compacted probes", [row for row in results if row["compacted"]]),
    ):
        for policy in ("trim", "compact"):
            passed = sum(row[policy]["verdict"] == "PASS" for row in group)
            print(f"{label} · {policy}: {passed}/{len(group)}")
    print(out)
