"""Agent tools: schemas (what the model sees) + bodies (what actually runs).

Seven tools over Emma's world:
  read   : search_files, read_file, list_events, find_event, list_reminders
  write  : add_event, add_reminder   (risk level set here in code, used in 6.3)

App state lives in SQLite (data/assistant.db), seeded from the persona calendar.
Agent evals use a separate in-memory database for each task.
Every tool reads "now" from PERSONA_TODAY — Emma's world is frozen at 2024-05-15.

Contract agent.py depends on:
  TOOL_SCHEMAS : list[dict]        # the tool menu sent to the model
  run_tool(name, args) -> str      # execute one tool, return text; may raise
  WRITE_TOOLS  : set[str]          # which tools change state (approval gate, 6.3)
"""

import datetime as dt
import json
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from contextvars import ContextVar
from functools import lru_cache
from pathlib import Path

from assistant import rag

PERSONA_TODAY = "2024-05-15"
DB_PATH = Path("data/assistant.db")
CALENDAR_SEED = Path("data/assets/persona/calendar.json")
FILES_INDEX = Path("data/assets/persona/files_index.json")

WRITE_TOOLS: set[str] = {"add_event", "add_reminder"}
_temporary_db: ContextVar[sqlite3.Connection | None] = ContextVar(
    "temporary_db", default=None
)


# --- storage -----------------------------------------------------------------
@contextmanager
def _db() -> Iterator[sqlite3.Connection]:
    temporary = _temporary_db.get()
    if temporary is not None:
        with temporary:
            yield temporary
        return
    with closing(sqlite3.connect(DB_PATH)) as conn, conn:
        conn.row_factory = sqlite3.Row
        yield conn


@contextmanager
def temporary_database() -> Iterator[sqlite3.Connection]:
    """Give tool calls in this context a fresh, seeded SQLite database.

    check_same_thread=False: a framework agent (agent_pydantic.py) runs sync tools
    in a worker thread, so this one connection is touched from more than one thread.
    Access stays sequential (one tool call at a time), so sharing it is safe here."""
    with closing(sqlite3.connect(":memory:", check_same_thread=False)) as conn:
        conn.row_factory = sqlite3.Row
        token = _temporary_db.set(conn)
        try:
            init_db()
            yield conn
        finally:
            _temporary_db.reset(token)


EVENT_COLS = "id,title,start,end,location,participants,description"


def _event_rows(seed: dict) -> list[tuple]:
    """Map persona calendar JSON -> event table rows. One source of the column order,
    shared by init-seed and the eval's reset_db (participants list -> comma string)."""
    return [
        (
            e["id"],
            e["title"],
            e["start"],
            e["end"],
            e.get("location", ""),
            ", ".join(e.get("participants", [])),
            e.get("description", ""),
        )
        for e in seed["events"]
    ]


def init_db() -> None:
    """Create tables and seed events from the persona calendar (idempotent)."""
    if _temporary_db.get() is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY, title TEXT, start TEXT, end TEXT, location TEXT,
                participants TEXT, description TEXT);
            CREATE TABLE IF NOT EXISTS reminders (
                id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT, date TEXT,
                done INTEGER DEFAULT 0);
        """)
        # migrate an events table created before participants/description existed
        have = {r[1] for r in conn.execute("PRAGMA table_info(events)")}
        for col in ("participants", "description"):
            if col not in have:
                conn.execute(f"ALTER TABLE events ADD COLUMN {col} TEXT")
        already = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        if (
            not already and CALENDAR_SEED.exists()
        ):  # skip seeding if assets absent (e.g. CI)
            seed = json.loads(CALENDAR_SEED.read_text())
            conn.executemany(
                f"INSERT INTO events ({EVENT_COLS}) VALUES (?,?,?,?,?,?,?)",
                _event_rows(seed),
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
            "SELECT title,start,end,location,participants,description FROM events "
            "WHERE start LIKE ? ORDER BY start",
            (f"{date}%",),
        ).fetchall()
    if not rows:
        return f"No events on {date}."
    out = []
    for r in rows:
        line = f"{r['start'][11:]}–{r['end'][11:]}  {r['title']}"
        if r["location"]:
            line += f" @ {r['location']}"
        if r["participants"]:
            line += f"\n    with: {r['participants']}"
        if r["description"]:
            line += f"\n    note: {r['description']}"
        out.append(line)
    return "\n".join(out)


def _find_event(title: str) -> str:
    """Search events by title/keyword across ALL dates (list_events only sees one day).
    Lets the agent locate an event it was given by name, e.g. 'the dentist appointment',
    without first knowing its date — then it has the date for a follow-up action."""
    with _db() as conn:
        rows = conn.execute(
            "SELECT title,start,end,location FROM events "
            "WHERE title LIKE ? ORDER BY start",
            (f"%{title}%",),
        ).fetchall()
    if not rows:
        return f"No event matching '{title}'."
    # spans dates, so each line shows the DATE (unlike list_events' time-only lines)
    return "\n".join(
        f"{r['start'][:10]} {r['start'][11:]}–{r['end'][11:]}  {r['title']}"
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
    "find_event": lambda a: _find_event(a["title"]),
    "add_event": lambda a: _add_event(
        a["title"], a["date"], a.get("time", "09:00"), a.get("duration_min", 60)
    ),
    "list_reminders": lambda a: _list_reminders(),
    "add_reminder": lambda a: _add_reminder(a["text"], a["date"]),
}


_db_ready = False


def _ensure_db() -> None:
    """Set up + seed the DB on first tool use (not at import, so importing this
    module stays side-effect-free — e.g. under pytest with no persona assets)."""
    global _db_ready
    if _temporary_db.get() is None and not _db_ready:
        init_db()
        _db_ready = True


def run_tool(name: str, args: dict) -> str:
    if name not in _BODIES:
        raise ValueError(f"No such tool: {name}")
    _ensure_db()
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
            "name": "find_event",
            "description": "Find calendar events by title or keyword across ALL dates. "
            "Use when the user names an event without giving its date (e.g. 'my dentist "
            "appointment', 'the networking event') and you need to know when it is.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": _s(
                        "A word or phrase from the event's title to search for."
                    )
                },
                "required": ["title"],
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
# NB: no init_db() at import — it runs lazily via _ensure_db() on first tool use.
