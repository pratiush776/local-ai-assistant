"""The approval queue: writes a headless job proposed but could not run.

A background job runs with nobody to approve, so when its agent proposes a write
the loop stops and hands back the pending action plus the saved message `state`.
We persist that here as a row; the GUI Inbox lists these rows and, on approval,
feeds `state`/`name`/`args` back into `resume_agent` to finish the run.

Glue (tutor-written), mirroring the memory store's SQLite pattern. The one row
that matters conceptually is `state`: it is the whole resumable run, not a summary.
"""

import json
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ..storage import APP_DB_PATH


@dataclass
class PendingAction:
    id: int
    job_id: str
    title: str  # one-line label for the Inbox list ("Morning brief")
    prompt: str  # the full task we gave the job (detail / expand)
    name: str  # the write tool that is waiting
    args: dict  # its arguments (what will actually run on approval)
    state: list[dict]  # saved messages -> resume_agent continues from here
    tainted: bool  # did the run read untrusted content before pausing?
    trace_id: str  # the job run's trace, so the Inbox can link to it
    status: str  # pending | approved | denied
    created_at: str


@contextmanager
def _connect(path: Path) -> Iterator[sqlite3.Connection]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as db, db:
        db.row_factory = sqlite3.Row
        db.execute("""
            CREATE TABLE IF NOT EXISTS pending_actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                title TEXT NOT NULL,
                prompt TEXT NOT NULL,
                name TEXT NOT NULL,
                args TEXT NOT NULL,
                state TEXT NOT NULL,
                tainted INTEGER NOT NULL,
                trace_id TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                created_at TEXT NOT NULL,
                resolved_at TEXT
            )
        """)
        yield db


def _row_to_action(row: sqlite3.Row) -> PendingAction:
    return PendingAction(
        id=row["id"],
        job_id=row["job_id"],
        title=row["title"],
        prompt=row["prompt"],
        name=row["name"],
        args=json.loads(row["args"]),
        state=json.loads(row["state"]),
        tainted=bool(row["tainted"]),
        trace_id=row["trace_id"],
        status=row["status"],
        created_at=row["created_at"],
    )


def queue_pending(
    job_id: str,
    title: str,
    prompt: str,
    name: str,
    args: dict,
    state: list[dict],
    tainted: bool,
    trace_id: str,
    *,
    path: Path = APP_DB_PATH,
) -> int:
    """Persist one proposed write as a pending item. Returns its row id.

    `title`/`prompt` describe the job that proposed the write, so the Inbox can
    show WHAT it was trying to do, not just the bare tool call."""
    with _connect(path) as db:
        cur = db.execute(
            """
            INSERT INTO pending_actions
                (job_id, title, prompt, name, args, state, tainted, trace_id,
                 status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
            """,
            (
                job_id,
                title,
                prompt,
                name,
                json.dumps(args),
                json.dumps(state),
                int(tainted),
                trace_id,
                datetime.now(UTC).isoformat(),
            ),
        )
        return int(cur.lastrowid)


def list_pending(*, path: Path = APP_DB_PATH) -> list[PendingAction]:
    """Every item still waiting for a decision, oldest first."""
    with _connect(path) as db:
        rows = db.execute(
            "SELECT * FROM pending_actions WHERE status = 'pending' ORDER BY id"
        )
        return [_row_to_action(r) for r in rows]


def get_pending(pending_id: int, *, path: Path = APP_DB_PATH) -> PendingAction | None:
    with _connect(path) as db:
        row = db.execute(
            "SELECT * FROM pending_actions WHERE id = ?", (pending_id,)
        ).fetchone()
        return _row_to_action(row) if row else None


def resolve_pending(
    pending_id: int, approved: bool, *, path: Path = APP_DB_PATH
) -> bool:
    """Mark an item approved/denied. Only acts on a still-pending row (so a double
    click can't resolve twice). Returns True if this call changed it. Executing the
    approved write itself is the Inbox's job via resume_agent, not this function."""
    with _connect(path) as db:
        cur = db.execute(
            """
            UPDATE pending_actions
               SET status = ?, resolved_at = ?
             WHERE id = ? AND status = 'pending'
            """,
            (
                "approved" if approved else "denied",
                datetime.now(UTC).isoformat(),
                pending_id,
            ),
        )
        return cur.rowcount > 0
