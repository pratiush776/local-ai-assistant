"""CLI surface over the shared brain (`respond.py`).

A thin dev REPL: it prints the trace (route, status, timings) dimmed and streams
the answer. All the logic lives in `respond()`; this file only does I/O. The demo
GUI (`gui.py`) consumes the same `respond()`.
"""

from datetime import UTC, datetime

import anthropic
import httpx

from .models.ollama import ask_with_stream
from .respond import PRIMARY_LOCAL_MODEL, respond, respond_resume
from .trace import capture

DIM, RESET = "\033[2m", "\033[0m"


def _trace(text: str) -> None:
    print(f"{DIM}· {text}{RESET}")


def _consume(stream, events: list, state: dict):
    """Print an event stream. Returns the approval_request Event if the run paused
    for a write, else None. Accumulates the final answer into state['answer']."""
    for ev in stream:
        events.append(ev)
        if ev.kind in ("status", "route", "memory_recall"):
            _trace(ev.text)
        elif ev.kind == "compaction":
            _trace(ev.text)
            if "history" in ev.data:
                state["history"] = ev.data["history"]
        elif ev.kind == "tool_call":
            _trace(f"→ {ev.text}({ev.data.get('args', {})})")
        elif ev.kind == "tool_result":
            _trace(f"  ↳ {str(ev.data.get('result', '')).splitlines()[0][:70]}")
        elif ev.kind == "delta":
            if not state["started"]:
                print("assistant › ", end="", flush=True)
                state["started"] = True
            print(ev.text, end="", flush=True)
            state["answer"] += ev.text
        elif ev.kind == "citations":
            print(f"\n{DIM}{ev.text}{RESET}", end="")
        elif ev.kind == "timing":
            _trace(" ".join(f"{k}={v}" for k, v in ev.data.items()))
        elif ev.kind == "approval_request":
            return ev
        elif ev.kind == "done":
            state["answer"] = ev.text or state["answer"]
    return None


def prewarm(model: str = PRIMARY_LOCAL_MODEL) -> None:
    ask_with_stream([], model, warmup=True)


def chat() -> None:
    prewarm()
    messages: list[dict] = []
    escalate_next = False
    print("Type /escalate to send the next message to Claude, /quit to exit.")
    while True:
        user_msg = input("\nyou › ").strip()
        if user_msg in ("/quit", "quit", "exit"):
            break
        if user_msg == "/escalate":
            escalate_next = True
            _trace("next message will go to Claude")
            continue
        if not user_msg:
            continue

        state, events = {"answer": "", "started": False}, []
        observed_at = datetime.now(UTC)
        try:
            pause = _consume(
                respond(
                    user_msg, messages, force_escalate=escalate_next, use_memory=True
                ),
                events,
                state,
            )
            while pause is not None:  # a write is waiting — ask, then resume
                ok = input(
                    f"\n{DIM}approve {pause.text}({pause.data['args']})? [y/N] › "
                    f"{RESET}"
                ).strip().lower() in ("y", "yes")
                pause = _consume(
                    respond_resume(
                        pause.data["state"],
                        pause.text,
                        pause.data["args"],
                        ok,
                        tainted=pause.data.get("tainted", False),
                    ),
                    events,
                    state,
                )
            capture(
                user_msg,
                events,
                memory_source_time=observed_at,
                memory_model=PRIMARY_LOCAL_MODEL,
            )
        except (httpx.HTTPError, anthropic.APIError):
            print("\n[network error — nothing stored]")
            escalate_next = False
            continue
        answer = state["answer"]
        if answer and not state["started"]:  # action turns aren't streamed — print once
            print(f"assistant › {answer}")
        print()

        escalate_next = False
        messages = state.get("history", messages) + [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": answer},
        ]


def main() -> None:
    chat()
