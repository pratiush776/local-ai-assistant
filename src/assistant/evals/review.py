"""Walk real traces one at a time: question, chunk TEXT + scores, answer, my note.

Run:  uv run assistant-review              # review traces that have no note yet
      uv run assistant-review --all        # re-walk every trace
      uv run assistant-review --notes       # dump all notes (for grouping)
"""
import json
import sys
import time
from pathlib import Path

TRACE_DIR = Path("data/traces")
CHUNKS_FILE = TRACE_DIR / "chunks.jsonl"
NOTES_FILE = Path("data/evals/notes.jsonl")

PREVIEW = 300  # chars of chunk text to show


# ---------- load ----------
def load_chunks() -> dict[str, str]:
    """cid -> full frozen text, from the trace sidecar."""
    if not CHUNKS_FILE.exists():
        return {}
    with CHUNKS_FILE.open() as f:
        return {r["cid"]: r["text"] for r in map(json.loads, filter(str.strip, f))}


def load_traces() -> list[dict]:
    """Every trace record across all date files, oldest first."""
    recs = []
    for path in sorted(TRACE_DIR.glob("*.jsonl")):
        if path.name == CHUNKS_FILE.name:
            continue
        with path.open() as f:
            recs += [json.loads(line) for line in f if line.strip()]
    return recs


def noted_ids() -> set[str]:
    if not NOTES_FILE.exists():
        return set()
    with NOTES_FILE.open() as f:
        return {json.loads(line)["turn_id"] for line in f if line.strip()}


def save_note(rec: dict, note: str) -> None:
    NOTES_FILE.parent.mkdir(parents=True, exist_ok=True)
    row = {"turn_id": rec["turn_id"], "query": rec["query"],
           "note": note, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with NOTES_FILE.open("a") as f:
        f.write(json.dumps(row) + "\n")


# ---------- render ----------
def show(rec: dict, chunks: dict[str, str], n: int, total: int) -> None:
    print("\n" + "=" * 72)
    print(f"[{n}/{total}]  {rec.get('ts')}   route={rec.get('route')} "
          f"label={rec.get('label')} model={rec.get('model')} "
          f"not_found={rec.get('not_found')}")
    print(f"\nQ: {rec['query']}\n")
    for i, c in enumerate(rec["retrieved"], 1):
        tag = "KEPT" if c["kept"] else "drop"
        text = chunks.get(c["id"], "<text not in sidecar>").replace("\n", " ")
        print(f"  {i:2}. [{tag}] {c['score']:.4f}  {text[:PREVIEW]}")
    cites = len(rec.get("citations") or [])
    print(f"\nA ({cites} citations): {rec['answer']}")
    print("=" * 72)


# ---------- modes ----------
def dump_notes() -> None:
    if not NOTES_FILE.exists():
        print("No notes yet.")
        return
    with NOTES_FILE.open() as f:
        rows = [json.loads(line) for line in f if line.strip()]
    print(f"{len(rows)} notes:\n")
    for r in rows:
        print(f"- {r['query']}\n    {r['note']}\n")


def walk(review_all: bool) -> None:
    chunks = load_chunks()
    traces = load_traces()
    done = set() if review_all else noted_ids()
    todo = [r for r in traces if r["turn_id"] not in done]
    if not todo:
        print("Nothing to review (all traces noted). Use --all to re-walk.")
        return
    print(f"{len(todo)} trace(s) to review. Enter = skip, note = save, q = quit.")
    for i, rec in enumerate(todo, 1):
        show(rec, chunks, i, len(todo))
        note = input("note> ").strip()
        if note == "q":
            break
        if note:
            save_note(rec, note)


def main() -> None:
    if "--notes" in sys.argv:
        dump_notes()
    else:
        walk(review_all="--all" in sys.argv)


if __name__ == "__main__":
    main()
