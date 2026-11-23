"""The agent loop: let a text model DO things by calling tools, step by step.

run_agent drives one request to completion: call the model, run any tools it
asks for, feed results back, repeat — until it answers or a stop condition trips.
"""

import json
import textwrap
import time
from collections.abc import Iterator

import httpx

from assistant.agent import security, tools
from assistant.mcp import client as mcp_client
from assistant.models.ollama import OLLAMA_URL
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
    """Run one tool, turning any failure into a result string (errors-as-results).
    MCP-backed tools go through the MCP client; the rest call tools.py directly."""
    try:
        if name in mcp_client.MCP_TOOL_NAMES:
            return mcp_client.run(name, args)
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
    # The trace keeps the RAW result; the model sees it wrapped as untrusted (layer b).
    messages.append(
        {
            "role": "tool",
            "tool_name": name,
            "content": security.wrap_untrusted(name, result),
        }
    )


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


def _loop(
    messages: list[dict], tool_schemas: list[dict], *, tainted: bool = False
) -> Iterator[Event]:
    """The shared loop. Both run_agent and resume_agent drive this. `tainted` is
    seeded by the caller so a run RESUMED after approval stays tainted (it is carried
    in the approval state); a fresh run starts untainted."""
    seen: set[tuple[str, str]] = set()  # (name, args) signatures already run
    # has this run read untrusted (stranger-written) content yet?  (may arrive True)
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

            # Host-side policy layer: the single interception point. Checked BEFORE
            # approval so a doomed call is never shown for authorization, and content-
            # blind so it covers ANY outbound tool, including foreign MCP ones.
            blocked = security.check_call(name, args, tainted=tainted)
            if blocked is not None:
                yield from _record_call(messages, name, args, blocked)
                continue

            # Contact-exfil guard: a tainted outbound send is NOT auto-run even if a
            # policy would allow it (a contact recipient). It is forced to approval
            # and the request carries a loud banner so the human decides, warned.
            warning = security.exfil_warning(name, tainted=tainted)
            if needs_approval(name) or warning is not None:
                yield Event(
                    "approval_request",
                    text=name,
                    data={
                        "args": args,
                        "state": messages,
                        "tainted": tainted,  # carried so a resumed run stays tainted
                        "warnings": [warning] if warning else [],
                    },
                )
                return

            result = _execute(name, args)
            if name in security.UNTRUSTED_READERS:
                tainted = True  # a stranger's words are now in context
            yield from _record_call(messages, name, args, result)

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
    # Risk rides on each tool (its read_only_hint annotation); the client reads it.
    return mcp_client.needs_approval(name)


def format_approval(
    name: str,
    args: dict,
    description: str | None = None,
    warnings: list[str] | None = None,
) -> str:
    """Render a pending write as the exact payload the user approves (WYSIWYS):
    the REAL argument values, never a paraphrase — this view is the authorization.

    Generic on purpose: it walks whatever args the tool carries, so a tool we never
    wrote (a foreign MCP server's) renders just as faithfully as our own. Long or
    multi-line values (an email body, a file's content) are shown in full on their
    own indented lines so nothing important is hidden; short values sit inline.

    `description` is the tool's own menu description — helpful context, shown as a
    subordinate line. It is tool/server-provided text (untrusted for a foreign
    server), so it never replaces the args: the args remain what is authorized.

    `warnings` are host advisories (e.g. the contact-exfil taint banner). They ride
    at the TOP, above the action, so the human sees the risk before the payload."""
    lines: list[str] = list(warnings or [])
    if lines:
        lines.append("")  # blank line separating the banner from the payload
    lines.append(f"Action: {name}")
    if description:
        lines.append(f"  ({description.strip()})")
    for key, value in args.items():
        value = str(value)
        if "\n" in value or len(value) > 60:
            lines.append(f"  {key}:\n{textwrap.indent(value, '    ')}")
        else:
            lines.append(f"  {key}: {value}")
    return "\n".join(lines)


def resume_agent(
    state: list[dict],
    name: str,
    args: dict,
    approved: bool,
    tool_schemas: list[dict],
    *,
    tainted: bool = False,
) -> Iterator[Event]:
    """Continue a run that paused for approval. `state` is the saved messages list;
    `name`/`args` are the write that was waiting (read off the approval_request event).
    `tainted` comes from that same approval event's data, so a run that read untrusted
    content before pausing stays tainted after it resumes (a reset would forget it)."""
    # Approve and deny differ only in the result string; both flow through the
    # same record-then-continue path.
    result = _execute(name, args) if approved else f"The user declined to run '{name}'."
    yield from _record_call(state, name, args, result)
    yield from _loop(state, tool_schemas, tainted=tainted)
