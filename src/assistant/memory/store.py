"""Store current facts and shortlist existing rows for consolidation."""

import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from pathlib import Path

import sqlite_vec

from ..knowledge.rag import embed
from ..storage import APP_DB_PATH
from .memory import StoredFact, decide_write, extract_facts

MATCH_LIMIT = 5
RECALL_LIMIT = 3
# Chosen in Lesson 8.4: 17/32 vs 15/32 at 0.42 on the 32 prepared probes. No global
# cosine cutoff separates true matches (0.44-0.465) from hard-negative distractors
# (0.418-0.432); recall is a weak gate, so extraction quality carries the load (8.5).
RECALL_MAX_DISTANCE = 0.50


class MemoryRecord(StoredFact):
    source_trace_id: str


class MemoryMatch(MemoryRecord):
    distance: float


@contextmanager
def _connect(path: Path) -> Iterator[sqlite3.Connection]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as db, db:
        db.row_factory = sqlite3.Row
        db.enable_load_extension(True)
        sqlite_vec.load(db)
        db.enable_load_extension(False)
        db.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL UNIQUE,
                embedding BLOB NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                observed_at TEXT NOT NULL,
                source_trace_id TEXT NOT NULL
            )
        """)
        yield db


def _shortlist(path: Path, embedding: bytes) -> list[StoredFact]:
    with _connect(path) as db:
        rows = db.execute(
            """
            SELECT id, text, observed_at FROM memories
            ORDER BY vec_distance_cosine(embedding, ?), id LIMIT ?
            """,
            (embedding, MATCH_LIMIT),
        )
        return [StoredFact.model_validate(dict(row)) for row in rows]


def rank_memories(user_msg: str, *, path: Path = APP_DB_PATH) -> list[MemoryMatch]:
    """Return the three nearest rows, including their distances and provenance."""
    if not user_msg.strip() or not path.exists():
        return []

    with _connect(path) as db:
        if db.execute("SELECT 1 FROM memories LIMIT 1").fetchone() is None:
            return []

        vector = embed([user_msg], "search_query")[0]
        query_embedding = sqlite_vec.serialize_float32(vector)
        rows = db.execute(
            """
            SELECT id, text, observed_at, source_trace_id,
                   vec_distance_cosine(embedding, ?) AS distance
            FROM memories
            ORDER BY distance, id LIMIT ?
            """,
            (query_embedding, RECALL_LIMIT),
        ).fetchall()
    return [MemoryMatch.model_validate(dict(row)) for row in rows]


def recall(user_msg: str, *, path: Path = APP_DB_PATH) -> list[str]:
    """Return up to three current facts within the recall distance cutoff."""
    return [
        match.text
        for match in rank_memories(user_msg, path=path)
        if match.distance <= RECALL_MAX_DISTANCE
    ]


def list_memories(*, path: Path = APP_DB_PATH) -> list[MemoryRecord]:
    if not path.exists():
        return []
    with _connect(path) as db:
        rows = db.execute(
            "SELECT id, text, observed_at, source_trace_id FROM memories ORDER BY updated_at DESC, id"
        )
        return [MemoryRecord.model_validate(dict(row)) for row in rows]


def _save(
    path: Path,
    text: str,
    embedding: bytes,
    observed_at: datetime,
    source_trace_id: str,
    target: StoredFact | None,
) -> int | None:
    now = datetime.now(UTC).isoformat(timespec="microseconds")
    source_time = observed_at.astimezone(UTC).isoformat(timespec="microseconds")
    with _connect(path) as db:
        if target is None:
            cursor = db.execute(
                """
                INSERT INTO memories
                    (text, embedding, created_at, updated_at, observed_at, source_trace_id)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(text) DO NOTHING
                """,
                (text, embedding, now, now, source_time, source_trace_id),
            )
            return cursor.lastrowid if cursor.rowcount else None

        expected = target.observed_at.astimezone(UTC).isoformat(timespec="microseconds")
        cursor = db.execute(
            """
            UPDATE OR IGNORE memories
            SET text = ?, embedding = ?, updated_at = ?, observed_at = ?, source_trace_id = ?
            WHERE id = ? AND observed_at = ? AND observed_at < ?
            """,
            (
                text,
                embedding,
                now,
                source_time,
                source_trace_id,
                target.id,
                expected,
                source_time,
            ),
        )
        return target.id if cursor.rowcount else None


def store_turn(
    turn: str,
    *,
    observed_at: datetime,
    source_trace_id: str,
    model: str,
    path: Path = APP_DB_PATH,
) -> Iterator[dict]:
    """Yield each committed write (or NOOP) as the turn is consolidated."""
    if observed_at.utcoffset() is None:
        raise ValueError("The source observation time must include a timezone.")
    facts = extract_facts(turn, model=model)
    if not facts:
        return
    vectors = embed([fact.text for fact in facts], "search_document")
    for fact, vector in zip(facts, vectors, strict=True):
        embedding = sqlite_vec.serialize_float32(vector)
        similar = _shortlist(path, embedding)
        decision = decide_write(fact, similar, observed_at=observed_at, model=model)
        row_id = None
        target = None
        if decision.action in ("ADD", "UPDATE"):
            target = next(
                (row for row in similar if row.id == decision.target_id), None
            )
            if decision.action == "ADD" or target is not None:
                row_id = _save(
                    path, fact.text, embedding, observed_at, source_trace_id, target
                )
        yield {
            "action": decision.action if row_id is not None else "NOOP",
            "id": row_id,
            "candidate": fact.text,
            "evidence": fact.evidence,
            "previous": target.text
            if row_id is not None and target is not None
            else None,
            "reason": (
                "The row changed or the value already exists."
                if decision.action in ("ADD", "UPDATE") and row_id is None
                else None
            ),
        }


def edit_memory(record: MemoryRecord, text: str, *, path: Path = APP_DB_PATH) -> bool:
    text = text.strip()
    if not text:
        raise ValueError("Memory text cannot be blank.")
    if text == record.text:
        return True
    observed_at = datetime.now(UTC)
    vector = sqlite_vec.serialize_float32(embed([text], "search_document")[0])
    return (
        _save(path, text, vector, observed_at, record.source_trace_id, record)
        is not None
    )


def delete_memory(
    memory_id: int, *, expected: StoredFact | None = None, path: Path = APP_DB_PATH
) -> bool:
    """Forget a specific current memory when explicitly requested by the app."""
    with _connect(path) as db:
        if expected is not None:
            version = expected.observed_at.astimezone(UTC).isoformat(
                timespec="microseconds"
            )
            return bool(
                db.execute(
                    "DELETE FROM memories WHERE id = ? AND observed_at = ?",
                    (memory_id, version),
                ).rowcount
            )
        return bool(
            db.execute("DELETE FROM memories WHERE id = ?", (memory_id,)).rowcount
        )
