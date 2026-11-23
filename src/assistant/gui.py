"""Gradio demo UI over the shared brain (`respond.py`).

Run:  uv run assistant-gui     (or: uv run python -m assistant.gui)

Left: an ink-on-off-white chat (☰ clears it). Right: a tabbed dev rail —
Memory (your files, with citation tracing), Flow (the routing/retrieval steps of
the last turn), and Settings (every live knob). Click an answer bubble to open
its cited sources. Same `respond()` as the CLI; this is presentation only.
"""

import shutil
import sys
from pathlib import Path

import gradio as gr
import httpx

from . import mcp_client, rag
from .agent import format_approval
from .respond import respond, respond_resume
from .trace import capture

# real modules (not the shadowed `respond`/`decide` names) for live config writes
_RESP = sys.modules[f"{__package__}.respond"]
_DEC = sys.modules[f"{__package__}.decide"]

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
            "### Memory",
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
        return (gr.update(), cur_cite, nav, *_render(nav, cur_cite))
    nav2 = [("files", None), ("cites", None)]
    return (gr.Tabs(selected="memory"), cite, nav2, *_render(nav2, cite))


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


_BLANK = {"text": "", "files": []}


def _apply_event(ev, ctx: dict) -> None:
    """Fold one event into the render context (mutates ctx)."""
    if ev.kind == "status":
        ctx["flow"].append(f"- {ev.text}")
        if not ctx["answer"]:
            ctx["display"][-1]["content"] = f"<sub>· {ev.text}</sub>"
    elif ev.kind == "route":
        ctx["route_txt"] = ev.data["label"] or ev.data["route"]
        ctx["flow"].append(
            f"**Route:** `{ctx['route_txt']}` · conf {ev.data['confidence']:.2f}"
        )
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
    payload = format_approval(pause.text, args, mcp_client.describe(pause.text))
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
        "ctx": ctx,
        "clean": clean,
        "turn_cites": turn_cites,
    }
    return _frame(ctx, clean, turn_cites, ask=True, pending=pending)


def _finish(ctx, clean, turn_cites):
    """Persist the trace and emit the final, buttons-hidden frame."""
    capture(ctx["text"], ctx["events"])
    display = ctx["display"]
    display[-1]["content"] = ctx["final"] + _meta(
        ctx["route_txt"], ctx["timings"], bool(ctx["cids"])
    )
    new_clean = clean + [
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
        )
        return

    if notes:
        display = display + [{"role": "assistant", "content": "\n".join(notes)}]
    display = display + [
        {"role": "user", "content": text},
        {"role": "assistant", "content": "<sub>· thinking…</sub>"},
    ]

    ctx = {
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
        respond(text, clean, force_escalate=force), ctx, clean, turn_cites
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
        pending["state"], pending["name"], pending["args"], approved
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

    with gr.Row(equal_height=False):
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
                with gr.Tab("Memory", id="memory"):
                    with gr.Row():
                        back_btn = gr.Button("← Back", scale=0, visible=False)
                    rail_title = gr.Markdown("### Memory")
                    selector = gr.Radio(
                        choices=[], label="Your files", interactive=True
                    )
                    content = gr.Markdown(
                        "*Select a file to read its contents.*", elem_id="content"
                    )
                with gr.Tab("Flow", id="flow"):
                    flow_md = gr.Markdown(
                        "*Send a message to see the routing flow.*", elem_id="flow"
                    )
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
    ]
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
        [rail_tabs, cite_state, nav_state, *render_out],
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
    demo.launch(theme=gr.themes.Monochrome(), css=CSS)


if __name__ == "__main__":
    main()
