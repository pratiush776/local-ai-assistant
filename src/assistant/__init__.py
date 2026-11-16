"""CLI surface over the shared brain (`respond.py`).

A thin dev REPL: it prints the trace (route, status, timings) dimmed and streams
the answer. All the logic lives in `respond()`; this file only does I/O. The demo
GUI (`gui.py`) consumes the same `respond()`.
"""

import anthropic
import httpx

from .context import estimate_tokens, messages_to_text, trim_history
from .ollama import ask_with_stream
from .respond import PRIMARY_LOCAL_MODEL, respond

CONTEXT_BUDGET = 4096
DIM, RESET = "\033[2m", "\033[0m"


def _trace(text: str) -> None:
    print(f"{DIM}· {text}{RESET}")


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

        answer, started = "", False
        try:
            for ev in respond(user_msg, messages, force_escalate=escalate_next):
                if ev.kind in ("status", "route"):
                    _trace(ev.text)
                elif ev.kind == "delta":
                    if not started:
                        print("assistant › ", end="", flush=True)
                        started = True
                    print(ev.text, end="", flush=True)
                    answer += ev.text
                elif ev.kind == "citations":
                    print(f"\n{DIM}{ev.text}{RESET}", end="")
                elif ev.kind == "timing":
                    _trace(" ".join(f"{k}={v}" for k, v in ev.data.items()))
                elif ev.kind == "done":
                    answer = ev.text or answer
        except (httpx.HTTPError, anthropic.APIError):
            print("\n[network error — nothing stored]")
            escalate_next = False
            continue
        print()

        escalate_next = False
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": answer})
        ctx = estimate_tokens(messages_to_text(messages))
        messages, _ = trim_history(messages, ctx, CONTEXT_BUDGET)


def main() -> None:
    chat()
