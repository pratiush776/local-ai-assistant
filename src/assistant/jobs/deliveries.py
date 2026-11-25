"""Deliveries: read-only things a background job produced for me to SEE (Lesson 9.2).

Unlike pending_actions (writes awaiting approve/deny), a delivery is just information
— a morning brief, a notification. It is stored so the Inbox can show it, and it also
fires a desktop notification so I notice it without the app in focus.
"""

import logging
import sqlite3
import subprocess
import sys
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ..storage import APP_DB_PATH

log = logging.getLogger(__name__)


@dataclass
class Delivery:
    id: int
    job_id: str
    title: str
    body: str  # the full text (e.g. the brief) to read in the Inbox
    trace_id: str
    created_at: str


@contextmanager
def _connect(path: Path) -> Iterator[sqlite3.Connection]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as db, db:
        db.row_factory = sqlite3.Row
        db.execute("""
            CREATE TABLE IF NOT EXISTS deliveries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                title TEXT NOT NULL,
                body TEXT NOT NULL,
                trace_id TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        yield db


def _notify(title: str, body: str) -> None:
    """Fire a desktop notification. macOS via osascript; a no-op (logged) elsewhere."""
    if sys.platform != "darwin":
        log.info("desktop notification (no-op on %s): %s", sys.platform, title)
        return
    preview = (body.splitlines()[0] if body.strip() else "")[:200]

    def esc(s: str) -> str:
        return s.replace("\\", "\\\\").replace('"', '\\"')

    script = f'display notification "{esc(preview)}" with title "{esc(title)}"'
    try:
        subprocess.run(["osascript", "-e", script], check=False, timeout=5)
    except Exception:
        log.exception("desktop notification failed")


def deliver(
    job_id: str, title: str, body: str, trace_id: str, *, path: Path = APP_DB_PATH
) -> int:
    """Store a delivery AND fire a desktop notification. Returns the row id."""
    with _connect(path) as db:
        cur = db.execute(
            "INSERT INTO deliveries (job_id, title, body, trace_id, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (job_id, title, body, trace_id, datetime.now(UTC).isoformat()),
        )
        row_id = int(cur.lastrowid)
    _notify(title, body)
    return row_id


def list_deliveries(*, limit: int = 20, path: Path = APP_DB_PATH) -> list[Delivery]:
    """Most recent deliveries first."""
    with _connect(path) as db:
        rows = db.execute(
            "SELECT * FROM deliveries ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [
            Delivery(
                id=r["id"],
                job_id=r["job_id"],
                title=r["title"],
                body=r["body"],
                trace_id=r["trace_id"],
                created_at=r["created_at"],
            )
            for r in rows
        ]
