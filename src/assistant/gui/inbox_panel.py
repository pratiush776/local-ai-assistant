"""Inbox controls for the background-jobs queue (Lesson 9.1 glue).

A headless job's proposed write lands in `pending_actions` (see jobs/pending.py).
This panel lists those items, shows WHAT each job wanted to do (its title, the task
prompt, and the exact rendered action), and lets the human approve or deny. Approve
runs the write for real via `resume_agent`; deny drops it. Either way the queue row
is resolved so it leaves the list.
"""

from html import escape

import gradio as gr

from ..agent import format_approval, resume_agent
from ..agent import tools as agent_tools
from ..jobs import (
    get_pending,
    list_deliveries,
    list_pending,
    queue_pending,
    resolve_pending,
)
from ..mcp import client as mcp_client
from ..respond import Event
from ..trace.recorder import record_turn, write_trace


def _choices() -> list[tuple[str, int]]:
    """Radio options: a one-line label per pending item, carrying its row id."""
    return [(f"{p.title} · {p.name}", p.id) for p in list_pending()]


def _detail(pending_id: int | None) -> gr.update:
    p = get_pending(pending_id) if pending_id else None
    if p is None:
        return gr.update(value="", visible=False)
    warnings = (
        ["⚠ this job read untrusted content before proposing this"]
        if p.tainted
        else None
    )
    action = format_approval(p.name, p.args, warnings=warnings)
    html = (
        f"<div class='inbox-detail'>"
        f"<div class='inbox-task'>Task: {escape(p.prompt)}</div>"
        f"<pre class='inbox-action'>{escape(action)}</pre>"
        f"<div class='inbox-trace'>job <code>{escape(p.job_id)}</code> · "
        f"trace <code>{escape(p.trace_id)}</code></div>"
        f"</div>"
    )
    return gr.update(value=html, visible=True)


def _status(message: str) -> gr.update:
    return gr.update(
        value=f"<p class='inbox-status'>{escape(message)}</p>", visible=bool(message)
    )


def load_inbox(selected_id: int | None = None, message: str = ""):
    """Refresh the list. Keeps the current selection if it is still pending.
    Returns updates for (radio, detail, actions group, status)."""
    choices = _choices()
    keep = selected_id if any(cid == selected_id for _, cid in choices) else None
    return (
        gr.update(choices=choices, value=keep),
        _detail(keep),
        gr.update(visible=keep is not None),
        _status(message),
    )


def approve(selected_id: int | None):
    p = get_pending(selected_id) if selected_id else None
    if p is None:
        return load_inbox(None, "Select a pending action first.")
    if not p.state:
        # A suggestion (9.3): no paused agent behind it, so run the tool directly.
        result = agent_tools.run_tool(p.name, p.args)
        record = record_turn(
            f"[approved suggestion] {p.title}",
            [
                Event(kind="tool_call", text=p.name, data={"args": p.args}),
                Event(kind="tool_result", text=p.name, data={"result": result}),
            ],
        )
        write_trace(record)
        resolve_pending(p.id, True)
        return load_inbox(None, f"Approved — ran {p.name}.")
    # Run the approved write for real, then continue the job. The loop stops again
    # at the NEXT write (if any), which we re-queue so nothing runs unapproved.
    events, follow = [], None
    for event in resume_agent(
        p.state, p.name, p.args, True, mcp_client.SCHEMAS, tainted=p.tainted
    ):
        events.append(event)
        if event.kind == "approval_request":
            follow = event
            break
    record = record_turn(f"[approved] {p.title}", events)
    write_trace(record)
    resolve_pending(p.id, True)
    if follow is not None:
        queue_pending(
            p.job_id,
            p.title,
            p.prompt,
            follow.text,
            follow.data["args"],
            follow.data["state"],
            follow.data["tainted"],
            record["turn_id"],
        )
    return load_inbox(None, f"Approved — ran {p.name}.")


def deny(selected_id: int | None):
    p = get_pending(selected_id) if selected_id else None
    if p is None:
        return load_inbox(None, "Select a pending action first.")
    resolve_pending(p.id, False)  # drop it; nothing runs
    return load_inbox(None, f"Denied — {p.name} was not run.")


def load_briefs(selected_id: int | None = None):
    """Refresh the briefs/notifications list. Returns updates for (radio, body)."""
    items = list_deliveries()
    choices = [
        (f"{d.title} · {d.created_at[5:16].replace('T', ' ')}", d.id) for d in items
    ]
    keep = selected_id if any(cid == selected_id for _, cid in choices) else None
    if keep is None and items:  # default to the most recent
        keep = items[0].id
    body = next((d.body for d in items if d.id == keep), "")
    if not items:
        body = "*No briefs yet. Run the morning brief to see one here.*"
    return gr.update(choices=choices, value=keep), gr.update(value=body)


CSS = """
#inbox-list .wrap, #briefs-list .wrap { display: grid; gap: 8px; max-height: 200px; overflow-y: auto; }
.inbox-detail { font-size: 13px; }
.inbox-task { color: #60584c; margin-bottom: 6px; }
.inbox-action { background: #fffdf9; border: 1px solid #e2dbcf; border-radius: 8px;
    padding: 8px; white-space: pre-wrap; font-size: 12px; }
.inbox-trace { color: #60584c; font-size: 11px; margin-top: 6px; }
.inbox-status { color: #60584c; font-size: 12px; }
"""
