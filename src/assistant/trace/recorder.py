"""Fold one turn's event stream into one self-contained trace record."""

import json
import time
import uuid
from datetime import date, datetime
from pathlib import Path

from ..knowledge import rag
from ..memory.background import enqueue_turn
from ..respond import Event

TRACE_DIR = Path("data/traces")
CHUNKS_FILE = TRACE_DIR / "chunks.jsonl"


def record_turn(user_msg: str, events: list[Event]) -> dict:
    # record shape
    rec = {
        "turn_id": uuid.uuid4().hex[:12],
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "query": user_msg,
        "route": None,
        "label": None,
        "model": None,
        "retrieved": [],
        "kept": [],
        "answer": "",
        "not_found": False,
        "citations": [],
        "steps": [],  # agent tool calls/results, in order
        "timings_ms": {},
        "tokens": {},
    }

    # fold each ecent into the record
    for ev in events:
        if ev.kind == "route":
            rec["route"] = ev.data["route"]
            rec["label"] = ev.data["label"]
        elif ev.kind == "compaction":
            rec["compaction"] = {
                key: val for key, val in ev.data.items() if key != "history"
            }
        elif ev.kind == "retrieval":
            rec["retrieved"] = ev.data["chunks"]
            rec["kept"] = [c["id"] for c in ev.data["chunks"] if c["kept"]]
        elif ev.kind == "memory_recall":
            rec["memory_recall"] = ev.data
        elif ev.kind == "citations":
            rec["citations"] = ev.data["ids"]
        elif ev.kind == "tool_call":
            rec["steps"].append(
                {"type": "call", "tool": ev.text, "args": ev.data.get("args", {})}
            )
        elif ev.kind == "tool_result":
            rec["steps"].append(
                {"type": "result", "tool": ev.text, "result": ev.data.get("result", "")}
            )
        elif ev.kind == "timing":
            for key, val in ev.data.items():
                bucket = "timings_ms" if key.endswith("_ms") else "tokens"
                rec[bucket][key] = val
        elif ev.kind == "done":
            rec["answer"] = ev.text
            rec["model"] = ev.data.get("model")
            rec["not_found"] = ev.data.get("not_found", False)
    return rec


def write_trace(record: dict) -> Path:
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    path = TRACE_DIR / f"{date.today()}.jsonl"
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")
    return path


# ---------- glue: content-addressed chunk-text sidecar ----------
def _seen_cids() -> set[str]:
    """Chunk ids already frozen in the sidecar (so we never write one twice)."""
    if not CHUNKS_FILE.exists():
        return set()
    with CHUNKS_FILE.open() as f:
        return {json.loads(line)["cid"] for line in f if line.strip()}


def write_chunks(cids: list[str]) -> None:
    """Freeze full text for any not-yet-stored cid. Idempotent: because a cid IS
    the content hash, the same chunk is written exactly once, ever."""
    new = [c for c in cids if c not in _seen_cids()]
    if not new:
        return
    texts = rag.texts_for_chunks(new)
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    with CHUNKS_FILE.open("a") as f:
        for cid in new:
            if cid in texts:
                f.write(json.dumps({"cid": cid, "text": texts[cid]}) + "\n")


def capture(
    user_msg: str,
    events: list[Event],
    *,
    memory_source_time: datetime | None = None,
    memory_model: str | None = None,
) -> dict:
    """One call both surfaces make after a turn: fold -> persist -> freeze text."""
    rec = record_turn(user_msg, events)
    memory_result = None
    statement = next(
        (
            ev.data["remember_statement"]
            for ev in events
            if "remember_statement" in ev.data
        ),
        None,
    )
    if memory_source_time is not None and memory_model is not None:
        memory_result = Path("memory") / f"{rec['turn_id']}.json"
        rec["memory_observed_at"] = memory_source_time.isoformat()
        rec["memory"] = {
            "status": "queued",
            "trigger": "remember_request" if statement else "completed_chat_turn",
            "model": memory_model,
            "result": memory_result.as_posix(),
        }
    write_trace(rec)
    write_chunks([c["id"] for c in rec["retrieved"]])
    if memory_result is not None:
        enqueue_turn(
            statement or user_msg,
            observed_at=memory_source_time,
            source_trace_id=rec["turn_id"],
            model=memory_model,
            results_path=TRACE_DIR / memory_result,
        )
    return rec


def read_memory_result(record: dict) -> dict | None:
    """Read the outcome linked from a chat trace, if the background job finished."""
    path = TRACE_DIR / record["memory"]["result"]
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def find_turn(turn_id: str) -> dict | None:
    """Look up a source turn in the daily chat traces."""
    for path in sorted(TRACE_DIR.glob("????-??-??.jsonl"), reverse=True):
        with path.open(encoding="utf-8") as traces:
            for line in traces:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("turn_id") == turn_id:
                    return record
    return None
