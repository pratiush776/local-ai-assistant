"""The agent loop: let a text model DO things by calling tools, step by step.

run_agent drives one request to completion: call the model, run any tools it
asks for, feed results back, repeat — until it answers or a stop condition trips.
"""

import json
import time
from collections.abc import Iterator

import httpx

from assistant import tools
from assistant.ollama import OLLAMA_URL
from assistant.respond import PRIMARY_LOCAL_MODEL, Event

MAX_STEPS = 8  # hard ceiling on model calls (cost/time)
MAX_SECONDS = 120  # wall-clock budget

# Without this, the model defaults to "I'm an LLM, I can't see your files" and
# refuses to call tools. It must know it HAS tools and should use them.
AGENT_SYSTEM = (
    f"You are Emma's personal assistant. Today is {tools.PERSONA_TODAY}. "
    "You have tools to read her files and manage her calendar and reminders. "
    "When a request needs her files, calendar or reminders, CALL the right tool — "
    "do not say you lack access. Use the tool results to answer. "
    f"For dates, treat 'today' as {tools.PERSONA_TODAY} and compute relative dates from it."
)


def _chat(messages: list[dict], tool_schemas: list[dict]) -> dict:
    """One non-streaming model call. Returns the assistant `message` dict."""
    payload = {
        "model": PRIMARY_LOCAL_MODEL,
        "messages": messages,
        "tools": tool_schemas,
        "think": False,
        "stream": False,
    }
    resp = httpx.post(OLLAMA_URL, json=payload, timeout=MAX_SECONDS)
    resp.raise_for_status()
    return resp.json()["message"]


def _execute(name: str, args: dict) -> str:
    """Run one tool, turning any failure into a result string (errors-as-results)."""
    try:
        return tools.run_tool(name, args)
    except Exception as e:  # noqa: BLE001 — ANY failure goes back to the model
        return f"Error: {e}"


def _record_call(
    messages: list[dict], name: str, args: dict, result: str
) -> Iterator[Event]:
    """Emit the trace events for one tool call and append its result message.
    Shared by the normal loop and the post-approval resume."""
    yield Event("tool_call", text=name, data={"args": args})
    yield Event("tool_result", text=name, data={"result": result})
    messages.append({"role": "tool", "tool_name": name, "content": result})


def run_agent(
    user_msg: str, history: list[dict], tool_schemas: list[dict]
) -> Iterator[Event]:
    # Don't mutate the caller's history; build our own working copy.
    # Lead with the system prompt so the model knows it HAS tools (fixes refusals).
    messages = [
        {"role": "system", "content": AGENT_SYSTEM},
        *history,
        {"role": "user", "content": user_msg},
    ]
    yield from _loop(messages, tool_schemas)


def _loop(messages: list[dict], tool_schemas: list[dict]) -> Iterator[Event]:
    """The shared loop. Both run_agent and resume_agent drive this."""
    seen: set[tuple[str, str]] = set()  # (name, args) signatures already run
    deadline = time.monotonic() + MAX_SECONDS

    for step in range(MAX_STEPS):
        msg = _chat(messages, tool_schemas)
        messages.append(msg)  # keep the assistant turn WHOLE (incl. tool_calls)

        calls = msg.get("tool_calls")
        if not calls:  # no tools asked for -> this is the answer
            yield Event("done", text=msg.get("content", ""), data={"steps": step + 1})
            return

        for call in calls:
            name = call["function"]["name"]
            args = call["function"]["arguments"]
            signature = (name, json.dumps(args, sort_keys=True))

            # Stop condition: the exact same call already ran -> it's stuck, not progressing.
            if signature in seen:
                yield Event(
                    "done",
                    text="I kept repeating the same step without progress, "
                    "so I stopped. I couldn't complete that.",
                    data={"steps": step + 1, "stopped": "repeat"},
                )
                return
            seen.add(signature)

            if needs_approval(name):
                yield Event(
                    "approval_request",
                    text=name,
                    data={"args": args, "state": messages},
                )
                return

            yield from _record_call(messages, name, args, _execute(name, args))

        if time.monotonic() > deadline:  # wall-clock stop
            yield Event(
                "done",
                text="This took too long, so I stopped.",
                data={"steps": step + 1, "stopped": "time"},
            )
            return

    # Fell out of the loop -> hit MAX_STEPS without a final answer.
    yield Event(
        "done",
        text="I reached my step limit before finishing.",
        data={"steps": MAX_STEPS, "stopped": "max_steps"},
    )


def needs_approval(name: str) -> bool:
    return name in tools.WRITE_TOOLS


def resume_agent(
    state: list[dict], name: str, args: dict, approved: bool, tool_schemas: list[dict]
) -> Iterator[Event]:
    """Continue a run that paused for approval. `state` is the saved messages list;
    `name`/`args` are the write that was waiting (read off the approval_request event)."""
    # Approve and deny differ only in the result string; both flow through the
    # same record-then-continue path.
    result = _execute(name, args) if approved else f"The user declined to run '{name}'."
    yield from _record_call(state, name, args, result)
    yield from _loop(state, tool_schemas)
