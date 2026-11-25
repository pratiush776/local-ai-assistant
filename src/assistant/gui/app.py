"""Gradio demo UI over the shared brain (`respond.py`).

Run:  uv run assistant-gui     (or: uv run python -m assistant.gui)

Left: an ink-on-off-white chat (☰ clears it). Right: a tabbed dev rail —
Memory (your files, with citation tracing), Flow (the routing/retrieval steps of
the last turn), and Settings (every live knob). Click an answer bubble to open
its cited sources. Same `respond()` as the CLI; this is presentation only.
"""

import logging
import shutil
import sys
from datetime import UTC, datetime, timedelta
from html import escape
from pathlib import Path

import gradio as gr
import httpx

from ..agent import format_approval
from ..agent.tools import PERSONA_TODAY
from ..jobs import Job
from ..jobs.briefs import morning_brief_job
from ..jobs.control import jobs_enabled, set_jobs_enabled
from ..jobs.scheduler import build_scheduler, schedule_job
from ..jobs.suggest import scan_inbox
from ..knowledge import rag
from ..mcp import client as mcp_client
from ..respond import respond, respond_resume
from ..trace import capture
from ..trace.recorder import read_memory_result
from . import inbox_panel, memory_panel

# The real modules for live config writes. We fetch them from sys.modules (not via
# `import assistant.respond`) because the package __init__ rebinds `respond`/`decide`
# to FUNCTIONS, so a plain import would hand us the function, not the module.
_RESP = sys.modules["assistant.respond"]
_DEC = sys.modules["assistant.router.decide"]

# One in-process scheduler for background jobs (Unit 9). Started in main().
_scheduler = build_scheduler()


def _schedule_demo_job():
    """Lesson 9.1 Done-when: schedule a job to run itself ~1 min from now, with
    nobody watching. Its proposed write should appear in the Inbox on its own."""
    when = datetime.now(UTC) + timedelta(seconds=60)
    job = Job(
        id="demo-inbox",
        title="Demo: expense-report reminder",
        prompt="Add a reminder to submit the expense report on 2024-05-16.",
    )
    schedule_job(_scheduler, job, "date", run_date=when)
    return gr.update(
        value=(
            f"<p class='inbox-status'>Scheduled “{job.title}” for "
            f"{when:%H:%M:%S} UTC (~1 min). It runs headless — watch it appear here.</p>"
        ),
        visible=True,
    )


def _run_brief_soon():
    """Schedule the morning brief to run ~1 min from now (so the Done-when is
    testable in-session); the real daily cron is registered in main()."""
    when = datetime.now(UTC) + timedelta(seconds=60)
    schedule_job(_scheduler, morning_brief_job(), "date", run_date=when)
    return gr.update(
        value=(
            f"<p class='inbox-status'>Morning brief scheduled for {when:%H:%M:%S} UTC "
            f"(~1 min). It runs headless — watch Briefs below.</p>"
        ),
        visible=True,
    )


def _toggle_jobs(on: bool):
    """Settings kill switch: pause/resume all background jobs (Lesson 9.4)."""
    set_jobs_enabled(on)
    state = "running" if on else "PAUSED"
    return gr.update(
        value=f"<p class='inbox-status'>Background jobs are {state}.</p>", visible=True
    )


def _run_scan():
    """Scan unread email for dated commitments and queue proposals (Lesson 9.3)."""
    if not jobs_enabled():
        logging.getLogger(__name__).info("scan skipped — background jobs paused")
        return
    try:
        n = scan_inbox(model=_RESP.PRIMARY_LOCAL_MODEL, today=PERSONA_TODAY)
        logging.getLogger(__name__).info("suggestion scan queued %d proposal(s)", n)
    except Exception:
        logging.getLogger(__name__).exception("suggestion scan failed")


def _scan_now():
    """Run a suggestion scan on a background thread; hits appear in the Inbox to
    approve. Runs headless like the other jobs so the UI never blocks on the model."""
    when = datetime.now(UTC) + timedelta(seconds=2)
    _scheduler.add_job(
        _run_scan, "date", run_date=when, id="suggest-scan-now", replace_existing=True
    )
    return gr.update(
        value=(
            "<p class='inbox-status'>Scanning unread email for commitments — "
            "proposals will appear above in a few seconds.</p>"
        ),
        visible=True,
    )


MEMORY_DIR = Path("data/docs")

CSS = """
footer { display: none !important; }
.gradio-container { height: 100vh !important; background: #faf8f4 !important;
    color: #15130f !important;
    font-family: ui-sans-serif, -apple-system, "Segoe UI", Roboto, sans-serif; }
.block, .form, .wrap { background: transparent !important; border: none !important;
    box-shadow: none !important; }
/* left column flexes to fill the width the fixed-width rail leaves */
#maincol { flex: 1 1 0 !important; min-width: 0 !important; }
/* chat: fixed viewport-based height that leaves room for the composer; scrolls
   INTERNALLY so content never pushes the input off-screen */
#chat { background: #fdfcfa !important; border: 1px solid #e7e2d9 !important;
    border-radius: 14px !important; height: calc(100vh - 214px) !important; }
#chat .icon-button-wrapper, #chat .message-buttons, #chat button[aria-label],
#chat .icon-buttons { display: none !important; }
/* fixed-height bottom slot: holds the input OR the Approve/Deny row, never reflows */
#composer { height: 88px; margin-bottom: 16px; }
#composer #msgbox { margin-bottom: 0; }
#approvalbar { gap: 10px; }
#approvalbar button { min-height: 46px; }
#newchat { max-width: 110px; }
#newchat button { background: transparent !important; border: 1px solid #ddd6c9 !important;
    border-radius: 10px !important; color: #15130f !important; font-size: 12px !important;
    padding: 3px 12px !important; min-width: 0 !important; }
/* right rail: fixed width, pinned to the right boundary (does not grow) */
#rail { border-left: 1px solid #e7e2d9; padding-left: 18px; flex: 0 0 360px !important; }
/* each rail tab panel scrolls internally when its content exceeds the viewport */
#rail .tabitem { max-height: calc(100vh - 96px); overflow-y: auto; padding-right: 6px; }
#content, #flow { font-size: 13.5px; line-height: 1.5; padding-right: 8px; }
button.primary { background: #15130f !important; color: #faf8f4 !important; }
@media (max-width: 720px) {
    #assistant-layout { flex-direction: column; }
    #rail { flex: 1 1 auto !important; min-width: 0 !important; width: 100%;
        border-left: 0; border-top: 1px solid #e7e2d9; padding: 12px 0 0; }
    #chat { height: 52vh !important; }
}
"""


# ---------------- memory data ----------------
def _choices() -> list[tuple[str, str]]:
    try:
        return [(f"{Path(s).name}  ·  {n}", s) for s, n in rag.list_sources()]
    except Exception:  # noqa: BLE001 — panel fallback: any backend error -> empty list
        return []


def _ollama_models() -> list[str]:
    try:
        r = httpx.get("http://localhost:11434/api/tags", timeout=5)
        return [m["name"] for m in r.json().get("models", [])]
    except (httpx.HTTPError, ValueError, KeyError):
        return []


def _remember(files: list[str]) -> list[str]:
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    notes = []
    for f in files:
        dest = MEMORY_DIR / Path(f).name
        if Path(f).resolve() != dest.resolve():
            shutil.copy(f, dest)
        counts = rag.ingest(str(dest))
        notes.append(
            f"📎 Added **{dest.name}** ({counts.get('inserted', 0)} new chunks)."
        )
    return notes


# ---------------- the navigable Memory rail ----------------
def _render(nav: list, last_cite: dict):
    """Returns updates for (title, back, selector, content)."""
    view, arg = nav[-1]
    back = gr.update(visible=len(nav) > 1)
    if view == "files":
        return (
            "",
            back,
            gr.update(choices=_choices(), value=None, label="Your files", visible=True),
            "*Select a file to read its contents.*",
        )
    if view == "cites":
        texts = last_cite.get("texts", [])
        md = (
            "\n\n".join(f"**[{i}]**\n\n{t}" for i, t in enumerate(texts, 1))
            or "*No sources.*"
        )
        ids = last_cite.get("ids", [])
        srcmap = rag.sources_for_chunks(ids)
        srcs = list(dict.fromkeys(s for s in (srcmap.get(c) for c in ids) if s))
        choices = [(f"Open: {Path(s).name}", s) for s in srcs]
        return (
            "### Cited sources",
            back,
            gr.update(
                choices=choices,
                value=None,
                label="Open full document",
                visible=bool(choices),
            ),
            md,
        )
    rows = rag.chunks_for_source(arg)  # doc view
    md = (
        "\n\n".join(f"**{i}.**\n\n{t}" for i, (_, t) in enumerate(rows, 1))
        or "*Empty.*"
    )
    return (f"### {Path(arg).name}", back, gr.update(visible=False), md)


def go_pick(source, nav, last_cite):
    if source:
        nav = nav + [("doc", source)]
    return (nav, *_render(nav, last_cite))


def go_back(nav, last_cite):
    nav = nav[:-1] if len(nav) > 1 else nav
    return (nav, *_render(nav, last_cite))


def open_sources(turn_cites, cur_cite, nav, evt: gr.SelectData):
    """Click an answer bubble → show that turn's cited sources in the Memory tab."""
    idx = evt.index[0] if isinstance(evt.index, list) else evt.index
    cite = turn_cites.get(idx) or turn_cites.get(str(idx))
    if not cite or not cite.get("ids"):
        return (gr.skip(), gr.skip(), cur_cite, nav, *_render(nav, cur_cite))
    nav2 = [("files", None), ("cites", None)]
    return (
        gr.Tabs(selected="memory"),
        gr.Tabs(selected="memory-files"),
        cite,
        nav2,
        *_render(nav2, cite),
    )


# ---------------- the turn ----------------
def _fmt_time(ms: float) -> str:
    s = ms / 1000
    return f"{s:.1f}s" if s < 60 else f"{s / 60:.1f}min"


def _meta(route: str, timings: dict, has_cites: bool) -> str:
    t = _fmt_time(timings["total_ms"]) if timings.get("total_ms") else ""
    line = " · ".join(b for b in (route, t) if b)
    out = f"\n\n<sub>{line}</sub>" if line else ""
    if has_cites:
        out += "\n\n<sub>📎 click this message to view its sources</sub>"
    return out


def _flow_md(lines: list[str], timings: dict) -> str:
    md = "\n".join(lines)
    if timings:
        steps = [
            f"- {k[:-3].replace('_', ' ')}: {v} ms"
            for k, v in timings.items()
            if k.endswith("_ms")
        ]
        md += "\n\n**Timings**\n\n" + "\n".join(steps)
    return md


def _memory_flow(trace: dict, result: dict | None = None) -> str:
    turn_id = escape(trace["turn_id"])
    model = escape(trace["memory"]["model"])
    trigger = (
        "explicit remember request"
        if trace["memory"]["trigger"] == "remember_request"
        else "chat turn finished"
    )
    lines = [
        "<h4>Memory · last completed chat turn</h4>",
        f"<p>Trigger: {trigger} · Trace <code>{turn_id}</code> · {model}</p>",
    ]
    if result is None:
        return "".join([*lines, "<p>Queued for background extraction.</p>"])
    if result.get("error"):
        lines.append(f"<p>Memory job failed: {escape(result['error'])}</p>")
    elif not result["writes"]:
        lines.append("<p>No confirmed facts to write.</p>")
    for write in result["writes"]:
        row = f" · row {write['id']}" if write["id"] is not None else ""
        lines.append(
            f"<p><strong>{escape(write['action'])}{row}</strong> "
            f"{escape(write['candidate'])}<br>"
            f"<small>Source quote: {escape(write['evidence'])}</small></p>"
        )
        if write["previous"]:
            lines.append(f"<p><small>Previous: {escape(write['previous'])}</small></p>")
        if write["reason"]:
            lines.append(f"<p><small>{escape(write['reason'])}</small></p>")
    return "".join(lines)


def _refresh_memory_flow(trace: dict | None):
    if trace is None or (result := read_memory_result(trace)) is None:
        return gr.skip(), gr.skip()
    return _memory_flow(trace, result), gr.update(active=False)


_BLANK = {"text": "", "files": []}


def _apply_event(ev, ctx: dict) -> None:
    """Fold one event into the render context (mutates ctx)."""
    if ev.kind == "status":
        ctx["flow"].append(f"- {ev.text}")
        if not ctx["answer"]:
            ctx["display"][-1]["content"] = f"<sub>· {ev.text}</sub>"
    elif ev.kind == "compaction":
        ctx["flow"].append(f"- {ev.text}")
        if "history" in ev.data:
            ctx["history"] = ev.data["history"]
    elif ev.kind == "route":
        ctx["route_txt"] = ev.data["label"] or ev.data["route"]
        ctx["flow"].append(
            f"**Route:** `{ctx['route_txt']}` · conf {ev.data['confidence']:.2f}"
        )
    elif ev.kind == "memory_recall":
        ctx["flow"].append(f"- {ev.text}")
        for fact in ev.data["facts"]:
            safe = (
                escape(" ".join(fact.split()))
                .replace("[", "&#91;")
                .replace("`", "&#96;")
            )
            ctx["flow"].append(f"  - {safe}")
    elif ev.kind == "delta":
        ctx["answer"] += ev.text
        ctx["final"] = ctx["answer"]
        ctx["display"][-1]["content"] = ctx["answer"]
    elif ev.kind == "tool_call":
        ctx["flow"].append(f"**→ {ev.text}**(`{ev.data.get('args', {})}`)")
    elif ev.kind == "tool_result":
        res = str(ev.data.get("result", "")).replace("\n", " ")[:80]
        ctx["flow"].append(f"&nbsp;&nbsp;↳ {res}")
    elif ev.kind == "citations":
        ctx["cids"], ctx["ctexts"] = ev.data["ids"], ev.data.get("texts", [])
    elif ev.kind == "timing":
        ctx["timings"] = ev.data
    elif ev.kind == "done" and not ctx["answer"]:
        ctx["final"] = ev.text
        ctx["display"][-1]["content"] = ev.text


def _frame(ctx: dict, clean, turn_cites, *, ask=False, pending=None):
    """One GUI output tuple. `ask` toggles the Approve/Deny buttons."""
    return (
        _BLANK,
        ctx["display"],
        clean,
        ctx["last_cite"],
        _flow_md(ctx["flow"], ctx["timings"]),
        turn_cites,
        gr.update(visible=ask),
        gr.update(visible=ask),
        pending,
        gr.skip(),
        gr.skip(),
        gr.skip(),
    )


def _stream_frames(stream, ctx, clean, turn_cites):
    """Yield a frame per event. Returns the approval_request Event if paused, else None."""
    for ev in stream:
        ctx["events"].append(ev)
        if ev.kind == "approval_request":
            return ev
        _apply_event(ev, ctx)
        yield _frame(ctx, clean, turn_cites)
    return None


def _ask_frame(ctx, pause, clean, turn_cites):
    """Pause the turn: show the exact pending write and reveal the buttons."""
    # The faithful payload the user authorizes (WYSIWYS): real args + the tool's own
    # description as subordinate context. A monospace block wraps/scrolls inside the
    # chat; the Approve/Deny buttons sit in a separate fixed row, so a long body
    # can't push them off-screen.
    args = pause.data.get("args", {})
    payload = format_approval(
        pause.text,
        args,
        mcp_client.describe(pause.text),
        warnings=pause.data.get("warnings"),
    )
    ctx["flow"].append(f"**⏸ approve `{pause.text}`?**")
    # Signal the pause in the chat itself, not just the Flow tab: the bubble
    # otherwise stays on "Working on it…" and reads as stuck.
    ctx["display"][-1]["content"] = (
        "⏸ **Approval needed.** I'd like to run this:\n\n"
        f"```\n{payload}\n```\n\n"
        "Click **✓ Approve** or **✗ Deny** below."
    )
    pending = {
        "state": pause.data["state"],
        "name": pause.text,
        "args": pause.data["args"],
        "tainted": pause.data.get("tainted", False),  # carried so resume stays tainted
        "ctx": ctx,
        "clean": clean,
        "turn_cites": turn_cites,
    }
    return _frame(ctx, clean, turn_cites, ask=True, pending=pending)


def _finish(ctx, clean, turn_cites):
    """Persist the trace and emit the final, buttons-hidden frame."""
    trace = capture(
        ctx["text"],
        ctx["events"],
        memory_source_time=ctx["observed_at"],
        memory_model=_RESP.PRIMARY_LOCAL_MODEL,
    )
    display = ctx["display"]
    display[-1]["content"] = ctx["final"] + _meta(
        ctx["route_txt"], ctx["timings"], bool(ctx["cids"])
    )
    new_clean = ctx.get("history", clean) + [
        {"role": "user", "content": ctx["text"]},
        {"role": "assistant", "content": ctx["final"]},
    ]
    new_cite = {"ids": ctx["cids"], "texts": ctx["ctexts"]}
    new_turn_cites = {**turn_cites, len(display) - 1: new_cite}
    yield (
        _BLANK,
        display,
        new_clean,
        new_cite,
        _flow_md(ctx["flow"], ctx["timings"]),
        new_turn_cites,
        gr.update(visible=False),
        gr.update(visible=False),
        None,
        {"turn_id": trace["turn_id"], "memory": trace["memory"]},
        _memory_flow(trace),
        gr.update(active=True),
    )


def chat_fn(payload, display, clean, last_cite, turn_cites, pending):
    text = ((payload or {}).get("text") or "").strip()
    files = (payload or {}).get("files") or []
    notes = _remember(files) if files else []
    hide = gr.update(visible=False)

    low = text.lower()
    force = False
    if low.startswith("/claude"):
        force, text = True, text[7:].strip()
    elif low.startswith("/local"):
        text = text[6:].strip()

    if not text:  # files only
        if notes:
            display = display + [{"role": "assistant", "content": "\n".join(notes)}]
        yield (
            _BLANK,
            display,
            clean,
            {"ids": [], "texts": []},
            gr.update(),
            turn_cites,
            hide,
            hide,
            None,
            gr.skip(),
            gr.skip(),
            gr.skip(),
        )
        return

    if notes:
        display = display + [{"role": "assistant", "content": "\n".join(notes)}]
    display = display + [
        {"role": "user", "content": text},
        {"role": "assistant", "content": "<sub>· thinking…</sub>"},
    ]

    ctx = {
        "observed_at": datetime.now(UTC),
        "display": display,
        "flow": ["**Flow**"],
        "answer": "",
        "final": "",
        "route_txt": "",
        "timings": {},
        "cids": [],
        "ctexts": [],
        "events": [],
        "last_cite": last_cite,
        "text": text,
    }

    pause = yield from _stream_frames(
        respond(text, clean, force_escalate=force, use_memory=True),
        ctx,
        clean,
        turn_cites,
    )
    if pause is not None:  # a write is waiting — reveal Approve/Deny and stop
        yield _ask_frame(ctx, pause, clean, turn_cites)
        return
    yield from _finish(ctx, clean, turn_cites)


def resume_fn(approved: bool, pending):
    """Continue a paused action turn after the human clicked Approve/Deny."""
    if not pending:
        return
    ctx = pending["ctx"]
    stream = respond_resume(
        pending["state"],
        pending["name"],
        pending["args"],
        approved,
        tainted=pending.get("tainted", False),
    )
    pause = yield from _stream_frames(
        stream, ctx, pending["clean"], pending["turn_cites"]
    )
    if pause is not None:  # another write — ask again
        yield _ask_frame(ctx, pause, pending["clean"], pending["turn_cites"])
        return
    yield from _finish(ctx, pending["clean"], pending["turn_cites"])


def approve_click(pending):
    yield from resume_fn(True, pending)


def deny_click(pending):
    yield from resume_fn(False, pending)


APPROVAL_MARK = "Approval needed"


def _msg_text(msg) -> str:
    """Extract the text of a chatbot message, tolerating Gradio's content shapes
    (a plain str, or a list of {'text': ...} parts)."""
    if not isinstance(msg, dict):
        return ""
    c = msg.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return " ".join(
            str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in c
        )
    return str(c or "")


def _reveal_approval(chat_history):
    """While a write is pending, swap the message box for the Approve/Deny buttons
    — they are mutually exclusive (you are either typing or deciding), and hiding
    the input also stops a stray message from dropping the pending approval. Runs
    as a discrete `.then()`: a visibility change does NOT render when it rides on a
    streaming generator's final frame (Gradio 6), but a plain event returning
    component instances does. Reads the chatbot (whose updates DO land) to decide."""
    last = _msg_text(chat_history[-1]) if chat_history else ""
    pending = APPROVAL_MARK in last
    return (
        gr.update(visible=not pending),  # message box: hidden while deciding
        gr.Button(visible=pending),  # ✓ Approve
        gr.Button(visible=pending),  # ✗ Deny
    )


def clear_all():
    nav = [("files", None)]
    empty = {"ids": [], "texts": []}
    return (
        {"text": "", "files": []},
        [],
        [],
        empty,
        nav,
        "*Send a message to see the routing flow.*",
        {},
        gr.update(visible=False),  # approve_btn
        gr.update(visible=False),  # deny_btn
        None,  # pending_state
        None,  # memory_trace_state
        "",  # memory_flow
        gr.update(active=False),
        *_render(nav, empty),
    )


# ---------------- settings ----------------
def apply_settings(
    primary,
    router,
    escalate,
    embed,
    reranker,
    top_k,
    rrf_k,
    thr,
    winner,
    live,
    sysp,
    groundp,
):
    _RESP.PRIMARY_LOCAL_MODEL = primary.strip()
    _RESP.ESCALATE_MODEL = escalate.strip()
    _RESP.SYSTEM_PROMPT = sysp
    _DEC.MODEL = router.strip()
    _DEC.WINNER_THRESHOLD = float(winner)
    _DEC.LIVE_CUTOFF = float(live)
    rag.EMBED_MODEL = embed.strip()
    rag.TOP_K = int(top_k)
    rag.RRF_K = int(rrf_k)
    rag.RERANK_THRESHOLD = float(thr)
    rag.RAG_SYSTEM_PROMPT = groundp
    if reranker.strip() != rag.RERANKER_NAME:
        rag.RERANKER_NAME = reranker.strip()
        rag._RERANKER = None
    return "✅ Applied — effective on the next message."


# ---------------- layout ----------------
with gr.Blocks(title="assistant", fill_height=True) as demo:
    clean_state = gr.State([])
    cite_state = gr.State({"ids": [], "texts": []})
    nav_state = gr.State([("files", None)])
    turn_cites_state = gr.State({})
    pending_state = gr.State(None)  # a paused action turn awaiting Approve/Deny
    memory_trace_state = gr.State(None)
    selected_memory_state = gr.State(None)
    memory_confirmation_state = gr.State(None)

    with gr.Row(equal_height=False, elem_id="assistant-layout"):
        with gr.Column(scale=1, elem_id="maincol"):
            with gr.Row():
                clear_btn = gr.Button("New chat", scale=0, size="sm", elem_id="newchat")
            chatbot = gr.Chatbot(elem_id="chat", show_label=False, autoscroll=True)
            # One fixed-height bottom slot: the full-width input OR the Approve/Deny
            # row occupies it, swapped by _reveal_approval — same spot, no reflow.
            with gr.Column(elem_id="composer"):
                msg = gr.MultimodalTextbox(
                    show_label=False,
                    elem_id="msgbox",
                    placeholder="Message   ·   /claude to ask Claude   ·   drop a file to remember it",
                    file_types=[".pdf", ".md", ".txt"],
                    submit_btn=True,
                )
                with gr.Row(elem_id="approvalbar"):
                    approve_btn = gr.Button(
                        "✓ Approve", variant="primary", visible=False, scale=1
                    )
                    deny_btn = gr.Button(
                        "✗ Deny", variant="stop", visible=False, scale=1
                    )
        with gr.Column(scale=0, min_width=360, elem_id="rail"):
            with gr.Tabs() as rail_tabs:
                with gr.Tab("Memory", id="memory") as memory_tab:
                    with gr.Tabs(elem_id="memory-views") as memory_views:
                        with gr.Tab("Saved facts", id="saved-facts") as saved_tab:
                            memory_select = gr.Radio(
                                [],
                                value=None,
                                label="Saved facts",
                                show_label=False,
                                interactive=True,
                                elem_id="saved-facts-list",
                            )
                            with gr.Group(
                                visible=False, elem_id="saved-fact-detail"
                            ) as memory_detail:
                                memory_summary = gr.HTML("")
                                with gr.Row(elem_id="saved-fact-actions"):
                                    memory_edit = gr.Button("Edit", size="sm")
                                    memory_forget = gr.Button("Forget…", size="sm")
                            with gr.Group(visible=False) as memory_editor:
                                memory_text = gr.Textbox(
                                    label="Edit saved fact", lines=3
                                )
                                with gr.Row():
                                    memory_cancel_edit = gr.Button(
                                        "Cancel edit", size="sm"
                                    )
                                    memory_save = gr.Button(
                                        "Save edit", size="sm", variant="primary"
                                    )
                            with gr.Group(
                                visible=False, elem_id="saved-fact-confirm"
                            ) as memory_confirmation:
                                gr.Markdown("**Forget this saved fact?**")
                                with gr.Row():
                                    memory_cancel_forget = gr.Button(
                                        "Keep fact", size="sm"
                                    )
                                    memory_confirm = gr.Button(
                                        "Confirm forget", variant="stop", size="sm"
                                    )
                            memory_message = gr.HTML(
                                visible=False, elem_id="saved-fact-status"
                            )
                            memory_reload = gr.Button(
                                "Refresh facts",
                                size="sm",
                                scale=0,
                                elem_id="saved-fact-refresh",
                            )
                        with gr.Tab("Files", id="memory-files"):
                            with gr.Row():
                                back_btn = gr.Button("← Back", scale=0, visible=False)
                            rail_title = gr.Markdown("")
                            selector = gr.Dropdown(
                                choices=[],
                                label="Your files",
                                interactive=True,
                                value=None,
                            )
                            content = gr.Markdown(
                                "*Select a file to read its contents.*",
                                elem_id="content",
                            )
                with gr.Tab("Flow", id="flow"):
                    flow_md = gr.Markdown(
                        "*Send a message to see the routing flow.*", elem_id="flow"
                    )
                    memory_flow = gr.HTML("")
                    memory_timer = gr.Timer(1.0, active=False)
                with gr.Tab("Inbox", id="inbox") as inbox_tab:
                    inbox_select = gr.Radio(
                        [],
                        value=None,
                        label="Pending from background jobs",
                        interactive=True,
                        elem_id="inbox-list",
                    )
                    inbox_detail = gr.HTML("", visible=False)
                    with gr.Row(visible=False) as inbox_actions:
                        inbox_approve = gr.Button(
                            "✓ Approve", variant="primary", size="sm"
                        )
                        inbox_deny = gr.Button("✗ Deny", variant="stop", size="sm")
                    inbox_status = gr.HTML("", visible=False)
                    with gr.Row():
                        inbox_refresh = gr.Button("Refresh", size="sm", scale=0)
                        inbox_scan = gr.Button(
                            "🔎 Scan email for commitments", size="sm", scale=0
                        )
                        inbox_demo = gr.Button(
                            "⏱ Schedule demo job (+1 min)", size="sm", scale=0
                        )
                    gr.Markdown("---\n#### Briefs & notifications")
                    inbox_briefs = gr.Radio(
                        [],
                        value=None,
                        label="",
                        interactive=True,
                        elem_id="briefs-list",
                    )
                    inbox_brief_body = gr.Markdown("")
                    inbox_brief_now = gr.Button(
                        "⏱ Run morning brief in 1 min", size="sm", scale=0
                    )
                    inbox_timer = gr.Timer(3.0, active=True)
                with gr.Tab("Settings", id="settings"):
                    ollama = _ollama_models()
                    with gr.Accordion("Models", open=True):
                        s_primary = gr.Dropdown(
                            ollama,
                            value=_RESP.PRIMARY_LOCAL_MODEL,
                            allow_custom_value=True,
                            label="Primary local model",
                        )
                        s_router = gr.Dropdown(
                            ollama,
                            value=_DEC.MODEL,
                            allow_custom_value=True,
                            label="Router model",
                        )
                        s_escalate = gr.Textbox(
                            _RESP.ESCALATE_MODEL, label="Escalate model (Claude)"
                        )
                        s_embed = gr.Textbox(rag.EMBED_MODEL, label="Embedding model")
                        s_reranker = gr.Textbox(
                            rag.RERANKER_NAME, label="Reranker model"
                        )
                    with gr.Accordion("Background jobs", open=True):
                        s_jobs_on = gr.Checkbox(
                            value=jobs_enabled(),
                            label="Background jobs enabled (kill switch)",
                        )
                        s_jobs_status = gr.HTML("", visible=False)
                    with gr.Accordion("Retrieval", open=True):
                        s_topk = gr.Slider(
                            1, 50, value=rag.TOP_K, step=1, label="Top-K"
                        )
                        s_rrfk = gr.Slider(
                            1, 200, value=rag.RRF_K, step=1, label="RRF k"
                        )
                        s_thr = gr.Number(
                            value=rag.RERANK_THRESHOLD,
                            label="Rerank keep-threshold",
                            minimum=0,
                            maximum=1,
                        )
                    with gr.Accordion("Routing", open=True):
                        s_winner = gr.Slider(
                            0,
                            1,
                            value=_DEC.WINNER_THRESHOLD,
                            step=0.01,
                            label="Winner threshold",
                        )
                        s_live = gr.Slider(
                            0, 1, value=_DEC.LIVE_CUTOFF, step=0.01, label="Live cutoff"
                        )
                    with gr.Accordion("Prompts", open=False):
                        s_sys = gr.Textbox(
                            _RESP.SYSTEM_PROMPT, label="System prompt", lines=3
                        )
                        s_ground = gr.Textbox(
                            rag.RAG_SYSTEM_PROMPT, label="Grounding prompt", lines=6
                        )
                    apply_btn = gr.Button("Apply", variant="primary")
                    apply_status = gr.Markdown("")

    render_out = [rail_title, back_btn, selector, content]
    saved_out = [
        memory_select,
        selected_memory_state,
        memory_detail,
        memory_summary,
        memory_editor,
        memory_text,
        memory_confirmation,
        memory_confirmation_state,
        memory_message,
    ]
    demo.load(memory_panel.load_memories, outputs=saved_out)
    for change in (
        memory_tab.select,
        saved_tab.select,
        memory_select.input,
        memory_reload.click,
    ):
        change(memory_panel.load_memories, [memory_select], saved_out)
    memory_edit.click(
        memory_panel.open_edit,
        [selected_memory_state],
        [memory_editor, memory_text, memory_confirmation, memory_confirmation_state],
    )
    memory_forget.click(
        memory_panel.open_confirmation,
        [selected_memory_state],
        [memory_editor, memory_confirmation, memory_confirmation_state],
    )
    memory_cancel_edit.click(lambda: gr.update(visible=False), outputs=memory_editor)
    memory_cancel_forget.click(
        lambda: (gr.update(visible=False), None),
        outputs=[memory_confirmation, memory_confirmation_state],
    )
    memory_save.click(
        memory_panel.save_edit, [selected_memory_state, memory_text], saved_out
    )
    memory_confirm.click(
        memory_panel.forget,
        [memory_confirmation_state],
        saved_out,
        api_visibility="private",
    )
    inbox_out = [inbox_select, inbox_detail, inbox_actions, inbox_status]
    demo.load(inbox_panel.load_inbox, outputs=inbox_out)
    for ev in (inbox_tab.select, inbox_select.input, inbox_refresh.click):
        ev(inbox_panel.load_inbox, [inbox_select], inbox_out)
    inbox_timer.tick(
        inbox_panel.load_inbox,
        [inbox_select],
        inbox_out,
        queue=False,
        api_visibility="private",
        show_progress="hidden",
    )
    inbox_approve.click(inbox_panel.approve, [inbox_select], inbox_out)
    inbox_deny.click(inbox_panel.deny, [inbox_select], inbox_out)
    inbox_demo.click(_schedule_demo_job, None, inbox_status)
    inbox_scan.click(_scan_now, None, inbox_status)
    s_jobs_on.change(_toggle_jobs, [s_jobs_on], s_jobs_status)

    briefs_out = [inbox_briefs, inbox_brief_body]
    demo.load(inbox_panel.load_briefs, outputs=briefs_out)
    inbox_briefs.input(inbox_panel.load_briefs, [inbox_briefs], briefs_out)
    inbox_timer.tick(
        inbox_panel.load_briefs,
        [inbox_briefs],
        briefs_out,
        queue=False,
        api_visibility="private",
        show_progress="hidden",
    )
    inbox_brief_now.click(_run_brief_soon, None, inbox_status)

    chat_out = [
        msg,
        chatbot,
        clean_state,
        cite_state,
        flow_md,
        turn_cites_state,
        approve_btn,
        deny_btn,
        pending_state,
        memory_trace_state,
        memory_flow,
        memory_timer,
    ]
    memory_timer.tick(
        _refresh_memory_flow,
        [memory_trace_state],
        [memory_flow, memory_timer],
        queue=False,
        api_visibility="private",
        show_progress="hidden",
    )
    sub = msg.submit(
        chat_fn,
        [msg, chatbot, clean_state, cite_state, turn_cites_state, pending_state],
        chat_out,
    )
    # Reveal/hide the Approve/Deny buttons as a discrete event (see _reveal_approval):
    # a Button visibility change does not render from a streaming generator's final
    # frame in Gradio 6, so a plain .then returning gr.Button instances does it,
    # reading the chatbot (which DOES update from the generator) to decide.
    reveal = (_reveal_approval, [chatbot], [msg, approve_btn, deny_btn])
    sub.then(_render, [nav_state, cite_state], render_out).then(*reveal)
    approve_btn.click(approve_click, [pending_state], chat_out).then(*reveal)
    deny_btn.click(deny_click, [pending_state], chat_out).then(*reveal)

    chatbot.select(
        open_sources,
        [turn_cites_state, cite_state, nav_state],
        [rail_tabs, memory_views, cite_state, nav_state, *render_out],
    )
    selector.input(go_pick, [selector, nav_state, cite_state], [nav_state, *render_out])
    back_btn.click(go_back, [nav_state, cite_state], [nav_state, *render_out])
    clear_btn.click(
        clear_all,
        None,
        [
            msg,
            chatbot,
            clean_state,
            cite_state,
            nav_state,
            flow_md,
            turn_cites_state,
            approve_btn,
            deny_btn,
            pending_state,
            memory_trace_state,
            memory_flow,
            memory_timer,
            *render_out,
        ],
    )
    apply_btn.click(
        apply_settings,
        [
            s_primary,
            s_router,
            s_escalate,
            s_embed,
            s_reranker,
            s_topk,
            s_rrfk,
            s_thr,
            s_winner,
            s_live,
            s_sys,
            s_ground,
        ],
        apply_status,
    )
    demo.load(_render, [nav_state, cite_state], render_out)


def main() -> None:
    # Register the real recurring jobs, then start the scheduler. (Jobs live in the
    # in-memory store, so they are re-registered on each launch — see 9.1 parking note.)
    schedule_job(_scheduler, morning_brief_job(), "cron", hour=7, minute=0)
    # Daily suggestion scan: read new mail, queue any dated commitments for approval.
    _scheduler.add_job(
        _run_scan, "cron", hour=7, minute=5, id="suggest-scan", replace_existing=True
    )
    _scheduler.start()  # background jobs run while the app is up
    demo.launch(
        theme=gr.themes.Monochrome(),
        css=CSS + memory_panel.CSS + inbox_panel.CSS,
    )


if __name__ == "__main__":
    main()
