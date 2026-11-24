"""Saved fact controls for the Memory tab."""

import sqlite3
from html import escape

import gradio as gr
import httpx

from ..memory.background import after_pending_writes
from ..memory.store import MemoryRecord, delete_memory, edit_memory, list_memories
from ..trace.recorder import find_turn

CSS = """
#memory-views { --memory-ink: #242019; --memory-line: #e2dbcf; --memory-muted: #60584c;
    --memory-surface: #fffdf9; --memory-selected: #eee9df; }
#memory-views .tabitem { padding: 8px 1px 20px; max-height: none; overflow: visible; }
#memory-views .tabitem > .column { gap: 12px !important; }
#saved-facts-list { padding: 0 !important; margin: 0 !important; }
#saved-facts-list .wrap { display: grid; gap: 8px;
    max-height: 256px; overflow-y: auto; padding: 2px; }
#saved-facts-list label:has(input[type=radio]) { display: flex; align-items: center;
    min-height: 44px; width: 100%; box-sizing: border-box; margin: 0; padding: 9px 12px;
    border: 1px solid var(--memory-line); border-radius: 10px;
    background: var(--memory-surface); cursor: pointer; overflow-wrap: anywhere; }
#saved-facts-list label:has(input:checked) { background: var(--memory-selected);
    border-color: #877864; }
#saved-facts-list label:has(input[type=radio]) span { color: var(--memory-ink) !important; }
#memory-views :is(button, input, textarea, summary):focus-visible {
    outline: 2px solid #76614a; outline-offset: 2px; }
#memory-views button { min-height: 44px; }
#saved-fact-detail, #saved-fact-confirm { padding: 14px; border: 1px solid
    var(--memory-line) !important; border-radius: 12px;
    background: var(--memory-surface) !important; }
#saved-fact-confirm { background: #fff2ee !important; border-color: #dec2b8 !important; }
#saved-fact-detail #saved-fact-detail, #saved-fact-confirm #saved-fact-confirm {
    padding: 0; border: 0 !important; background: transparent !important; }
#saved-fact-detail .styler, #saved-fact-confirm .styler { background: transparent !important; }
#saved-fact-detail .html-container { padding: 0 !important; }
#saved-fact-detail :is(.html-container, .gradio-style) { background: transparent !important; }
#saved-fact-detail p { margin: 0 !important; overflow-wrap: anywhere; }
#saved-fact-detail .fact-statement { font-weight: 650; line-height: 1.45; }
#saved-fact-detail details { margin-top: 8px; color: var(--memory-muted); }
#saved-fact-detail summary { display: list-item; cursor: pointer; min-height: 44px;
    padding-block: 11px; box-sizing: border-box; }
#saved-fact-detail blockquote { margin: 0 0 8px; padding-left: 10px;
    border-left: 2px solid var(--memory-line); white-space: pre-wrap; overflow-wrap: anywhere; }
#saved-fact-detail .fact-trace { font-size: 12px; }
#saved-fact-actions { display: grid !important; grid-template-columns: 1fr 1fr;
    gap: 8px !important; margin-top: 8px; }
#saved-fact-actions button { border: 1px solid var(--memory-line) !important;
    border-radius: 8px !important; }
#saved-fact-refresh { border: 0 !important; background: transparent !important;
    border-radius: 8px; color: var(--memory-muted) !important; padding-inline: 10px; }
#saved-fact-refresh:hover { background: var(--memory-selected) !important; }
#saved-fact-status { font-size: 13px; line-height: 1.5; }
"""


def _status(message: str):
    return gr.update(
        value=f'<p role="status">{escape(message)}</p>', visible=bool(message)
    )


def _detail(current: MemoryRecord) -> str:
    trace = find_turn(current.source_trace_id)
    if trace is None:
        provenance = f"<p>Trace <code>{escape(current.source_trace_id)}</code> is unavailable.</p>"
    else:
        provenance = (
            f"<blockquote>{escape(trace['query'])}</blockquote>"
            f'<p class="fact-trace">Trace <code>{escape(trace["turn_id"])}</code></p>'
        )
    return (
        f'<p class="fact-statement">{escape(current.text)}</p>'
        f"<details><summary>Original chat message</summary>{provenance}</details>"
    )


def load_memories(selected_id=None, message=""):
    rows = list_memories()
    current = next((row for row in rows if row.id == selected_id), None)
    choices = [(row.text, row.id) for row in rows]
    return (
        gr.update(
            choices=choices, value=current.id if current else None, visible=bool(rows)
        ),
        current,
        gr.update(visible=current is not None),
        _detail(current) if current else "",
        gr.update(visible=False),
        current.text if current else "",
        gr.update(visible=False),
        None,
        _status(message or ("" if rows else "No saved facts yet.")),
    )


def open_edit(current: MemoryRecord | None):
    return (
        gr.update(visible=current is not None),
        current.text if current else "",
        gr.update(visible=False),
        None,
    )


def open_confirmation(current: MemoryRecord | None):
    return gr.update(visible=False), gr.update(visible=current is not None), current


def _error(message: str):
    return (*(gr.skip() for _ in range(8)), _status(message))


def save_edit(current: MemoryRecord | None, text: str):
    if current is None:
        return load_memories(message="Select a saved fact.")
    try:
        saved = after_pending_writes(edit_memory, current, text)
    except (ValueError, sqlite3.Error, httpx.HTTPError) as exc:
        return _error(f"Could not save the edit. {exc}")
    message = (
        "Saved." if saved else "The fact changed or the edited value already exists."
    )
    return load_memories(current.id, message)


def forget(current: MemoryRecord | None):
    if current is None:
        return _error("Select a saved fact.")
    try:
        removed = after_pending_writes(delete_memory, current.id, expected=current)
    except sqlite3.Error as exc:
        return _error(f"Could not forget the fact. {exc}")
    if removed:
        return load_memories(message="Forgot the saved fact.")
    return load_memories(
        current.id, "The fact changed. Review the current value and try again."
    )
