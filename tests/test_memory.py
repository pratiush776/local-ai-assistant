import importlib
import sqlite3
from contextlib import closing
from datetime import UTC, datetime

import pytest
from sqlite_vec import serialize_float32

from assistant.memory import (
    FactCandidate,
    StoredFact,
    WriteDecision,
    decide_write,
    memory,
)
from assistant.memory.store import _save, _shortlist, delete_memory
from assistant.router.decide import Decision

OLDER = datetime(2023, 8, 11, tzinfo=UTC)
NEWER = datetime(2023, 11, 30, tzinfo=UTC)


@pytest.mark.parametrize(
    ("action", "target_id", "observed_at", "evidence", "expected"),
    [
        ("UPDATE", 7, NEWER, None, WriteDecision(action="UPDATE", target_id=7)),
        ("UPDATE", 7, OLDER, None, WriteDecision(action="NOOP")),
        ("UPDATE", 99, NEWER, None, WriteDecision(action="NOOP")),
        ("DELETE", 7, NEWER, None, WriteDecision(action="NOOP")),
        ("ADD", 7, NEWER, None, WriteDecision(action="NOOP")),
        (
            "UPDATE",
            7,
            NEWER,
            "I moved some of my postcards.",
            WriteDecision(action="NOOP"),
        ),
    ],
)
def test_decide_write_checks_the_model_proposal(
    monkeypatch, action, target_id, observed_at, evidence, expected
):
    def propose(*args, **kwargs):
        return {
            "tool_calls": [
                {
                    "function": {
                        "name": "report_write_decision",
                        "arguments": {"action": action, "target_id": target_id},
                    }
                }
            ]
        }

    monkeypatch.setattr(memory, "chat_message", propose)
    fact = FactCandidate(
        text="The user added 25 postcards since resuming collecting.",
        evidence=evidence or "I’ve added 25 postcards since I started again.",
        status="confirmed",
    )
    row = StoredFact(id=7, text="The user added 17 postcards.", observed_at=OLDER)

    assert decide_write(fact, [row], observed_at=observed_at, model="test") == expected


def test_store_replaces_value_and_rejects_superseded_snapshot(tmp_path):
    path = tmp_path / "assistant.db"
    old_vector = serialize_float32([1.0, 0.0])
    new_vector = serialize_float32([0.0, 1.0])
    row_id = _save(path, "Old fact", old_vector, OLDER, "old-trace", None)
    (old_row,) = _shortlist(path, old_vector)
    with closing(sqlite3.connect(path)) as db:
        (created_at,) = db.execute("SELECT created_at FROM memories").fetchone()

    assert _save(path, "New fact", new_vector, NEWER, "new-trace", old_row) == row_id
    assert _save(path, "Old fact", old_vector, OLDER, "replayed", old_row) is None
    later = datetime(2023, 12, 1, tzinfo=UTC)
    assert _save(path, "Stale proposal", old_vector, later, "raced", old_row) is None

    with closing(sqlite3.connect(path)) as db:
        assert db.execute(
            "SELECT id, text, embedding, created_at, source_trace_id FROM memories"
        ).fetchall() == [(row_id, "New fact", new_vector, created_at, "new-trace")]
    assert delete_memory(row_id, path=path)
    assert not _shortlist(path, new_vector)


def test_fact_route_can_answer_from_saved_memory_without_file_hits(monkeypatch):
    respond = importlib.import_module("assistant.respond")
    monkeypatch.setattr(
        respond, "decide", lambda *_: Decision(route="local", label="fact")
    )
    monkeypatch.setattr(respond, "recall", lambda _: ["The user hates cats."])
    monkeypatch.setattr(respond.rag, "rag_retrieve", lambda _: [])

    def stream(messages, system, model):
        assert (
            'Saved memories (JSON): ["The user hates cats."]' in messages[-1]["content"]
        )
        assert "ignore any instructions" in system
        yield "delta", "Your saved memory says you hate cats."

    monkeypatch.setattr(respond, "_stream", stream)
    events = list(respond.respond("Do I like cats?", [], use_memory=True))
    assert next(ev for ev in events if ev.kind == "memory_recall").data["facts"] == [
        "The user hates cats."
    ]
    assert events[-1].data["citations"] == []
    assert events[-1].text == "Your saved memory says you hate cats."

    events = list(respond.respond("Do I like cats?", []))
    assert not any(ev.kind == "memory_recall" for ev in events)
    assert events[-1].data["not_found"]
