"""Agent tools: schemas (what the model sees) + bodies (what actually runs).

Six tools over Emma's world:
  read   : search_files, read_file, list_events, list_reminders
  write  : add_event, add_reminder   (risk level set here in code, used in 6.3)

State lives in SQLite (data/assistant.db), seeded once from the persona calendar.
Every tool reads "now" from PERSONA_TODAY — Emma's world is frozen at 2024-05-15.

Contract agent.py depends on:
  TOOL_SCHEMAS : list[dict]        # the tool menu sent to the model
  run_tool(name, args) -> str      # execute one tool, return text; may raise
  WRITE_TOOLS  : set[str]          # which tools change state (approval gate, 6.3)
"""

import datetime as dt
import json
import sqlite3
from functools import lru_cache
from pathlib import Path

from assistant import rag

PERSONA_TODAY = "2024-05-15"
DB_PATH = Path("data/assistant.db")
CALENDAR_SEED = Path("data/assets/persona/calendar.json")
FILES_INDEX = Path("data/assets/persona/files_index.json")

WRITE_TOOLS: set[str] = {"add_event", "add_reminder"}


# --- storage -----------------------------------------------------------------
def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables and seed events from the persona calendar (idempotent)."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY, title TEXT, start TEXT, end TEXT, location TEXT);
            CREATE TABLE IF NOT EXISTS reminders (
                id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT, date TEXT,
                done INTEGER DEFAULT 0);
        """)
        already = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        if not already:
            seed = json.loads(CALENDAR_SEED.read_text())
            conn.executemany(
                "INSERT INTO events (id,title,start,end,location) VALUES (?,?,?,?,?)",
                [
                    (e["id"], e["title"], e["start"], e["end"], e.get("location", ""))
                    for e in seed["events"]
                ],
            )


# --- bodies (my glue) --------------------------------------------------------
def _search_files(query: str) -> str:
    hits = rag.rag_search(query)  # list[(cid, text)]
    if not hits:
        return "No matching passages found."
    return "\n\n".join(f"[{cid}] {text}" for cid, text in hits)


@lru_cache(maxsize=1)
def _files() -> list[dict]:
    return json.loads(FILES_INDEX.read_text())


def _read_file(filename: str) -> str:
    for f in _files():
        if f["filename"].lower() == filename.lower():
            return f.get("raw_text", "")[:2000]
    names = ", ".join(sorted(f["filename"] for f in _files())[:10])
    return f"No file named '{filename}'. Some files: {names}"


def _list_events(date: str) -> str:
    with _db() as conn:
        rows = conn.execute(
            "SELECT title,start,end,location FROM events "
            "WHERE start LIKE ? ORDER BY start",
            (f"{date}%",),
        ).fetchall()
    if not rows:
        return f"No events on {date}."
    return "\n".join(
        f"{r['start'][11:]}–{r['end'][11:]}  {r['title']}"
        f"{' @ ' + r['location'] if r['location'] else ''}"
        for r in rows
    )


def _add_event(
    title: str, date: str, time: str = "09:00", duration_min: int = 60
) -> str:
    start = dt.datetime.fromisoformat(f"{date}T{time}")
    end = start + dt.timedelta(minutes=int(duration_min))
    new_id = f"local-{int(dt.datetime.now().timestamp())}"
    with _db() as conn:
        conn.execute(
            "INSERT INTO events (id,title,start,end,location) VALUES (?,?,?,?,?)",
            (
                new_id,
                title,
                start.isoformat(timespec="minutes"),
                end.isoformat(timespec="minutes"),
                "",
            ),
        )
    return f"Added '{title}' on {date} {time} ({duration_min} min)."


def _list_reminders() -> str:
    with _db() as conn:
        rows = conn.execute("SELECT text,date FROM reminders ORDER BY date").fetchall()
    if not rows:
        return "No reminders set."
    return "\n".join(f"{r['date']}: {r['text']}" for r in rows)


def _add_reminder(text: str, date: str) -> str:
    with _db() as conn:
        conn.execute("INSERT INTO reminders (text,date) VALUES (?,?)", (text, date))
    return f"Reminder set for {date}: {text}"


_BODIES = {
    "search_files": lambda a: _search_files(a["query"]),
    "read_file": lambda a: _read_file(a["filename"]),
    "list_events": lambda a: _list_events(a["date"]),
    "add_event": lambda a: _add_event(
        a["title"], a["date"], a.get("time", "09:00"), a.get("duration_min", 60)
    ),
    "list_reminders": lambda a: _list_reminders(),
    "add_reminder": lambda a: _add_reminder(a["text"], a["date"]),
}


def run_tool(name: str, args: dict) -> str:
    if name not in _BODIES:
        raise ValueError(f"No such tool: {name}")
    return _BODIES[name](args)


def _s(desc: str) -> dict:
    return {"type": "string", "description": desc}


# --- SCHEMAS -----------------------------------------------------------------
# search_files is written for you as the pattern. Write the other five below so
# their property NAMES match what each body reads from `args` (see _BODIES):
#   read_file     -> filename (str, required)
#   list_events   -> date (str "YYYY-MM-DD", required)
#   add_event     -> title (str, req), date (str, req), time (str "HH:MM", opt),
#                    duration_min (integer, opt)
#   list_reminders-> (no arguments)
#   add_reminder  -> text (str, req), date (str "YYYY-MM-DD", req)
TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search the user's personal files and return matching passages. "
            "Use this to answer questions about the user's documents.",
            "parameters": {
                "type": "object",
                "properties": {"query": _s("What to search the user's files for.")},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "given one known filename, returns that file's full text",
            "parameters": {
                "type": "object",
                "properties": {"filename": _s("Name of file of whose text you want.")},
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_events",
            "description": "List the user's calendar events on a specific date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": _s("The date in 'YYYY-MM-DD' to list events for.")
                },
                "required": ["date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_event",
            "description": "Add an event to the user's calendar. Use when the user asks to "
            "schedule or book something.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": _s("Short title of the event."),
                    "date": _s("Event date in 'YYYY-MM-DD'."),
                    "time": _s(
                        "Start time in 'HH:MM' (24h). Defaults to 09:00 if omitted."
                    ),
                    "duration_min": {
                        "type": "integer",
                        "description": "Length in minutes. Defaults to 60.",
                    },
                },
                "required": ["title", "date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_reminders",
            "description": "List all of the user's reminders.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_reminder",
            "description": "Add a reminder for the user on a given date. Use when the user "
            "asks to be reminded of something.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": _s("What to remind the user about."),
                    "date": _s("Reminder date in 'YYYY-MM-DD'."),
                },
                "required": ["text", "date"],
            },
        },
    },
]


init_db()
