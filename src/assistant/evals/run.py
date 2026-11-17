"""Score the 40-row golden set — the knowledge-layer scoreboard in one command.

    uv run evals            # run all rows, print the scoreboard, save data/evals/run-*.json

Two measurements, kept apart on purpose (see run_row):
  - retrieval hit@5  : pure retrieval via rag.rag_retrieve(), answerable rows only,
                       so a router punt is never mislabelled as a retrieval miss.
  - answer behaviour : the full system via respond(), for refusal accuracy + (next
                       session) the judge's answer pass rate.
"""
import json
import time
from datetime import datetime
from pathlib import Path

from .. import rag
from ..respond import respond
from ..trace import record_turn
from . import checks
from .stats import wilson

GOLDEN = Path("data/assets/evals/golden.jsonl")
OUT_DIR = Path("data/evals")
K = 5


def load_golden(limit: int | None = None) -> list[dict]:
    with GOLDEN.open() as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return rows[:limit] if limit else rows


def run_row(row: dict) -> dict:
    """One golden row -> the flat facts the checks need, from two sources."""
    q = row["question"]

    # (1) pure retrieval, isolated from the router
    retrieved = rag.rag_retrieve(q)  # (cid, text, score), best-first
    cids = [cid for cid, _, _ in retrieved]
    id2src = rag.sources_for_chunks(cids)
    ret_sources = [id2src.get(cid, "") for cid in cids]

    # (2) the real system answer, through respond()
    rec = record_turn(q, list(respond(q, [])))

    return {
        "id": row["id"],
        "question": q,
        "answerable": row["answerable"],
        "expected_answer": row["expected_answer"],
        "expected_sources": row["source_files"],
        "retrieved_sources": ret_sources,
        "answer": rec["answer"],
        "route": rec["route"],
        "label": rec["label"],  # fact/chat/... — this, not route, says if it retrieved
        "not_found": rec["not_found"],
        "citations": rec["citations"],
        "retrieved_ids": [c["id"] for c in rec["retrieved"]],
    }


def score(results: list[dict]) -> dict:
    """Fold per-row checks into counts. hit@5 is answerable-only; refusal and
    citation run on every row."""
    answerable = [r for r in results if r["answerable"]]
    hits = sum(
        checks.hit_at_k(r["retrieved_sources"], r["expected_sources"], K)
        for r in answerable
    )
    refusal_ok = sum(
        checks.refusal_correct(r["answer"], r["not_found"], r["answerable"])
        for r in results
    )
    cite_ok = sum(
        checks.citation_valid(r["citations"], r["retrieved_ids"]) for r in results
    )
    return {
        "n": len(results),
        "n_answerable": len(answerable),
        "hit_at_k": [hits, len(answerable)],
        "refusal": [refusal_ok, len(results)],
        "citation": [cite_ok, len(results)],
    }


def print_scoreboard(s: dict) -> None:
    def line(label: str, pair: list[int]) -> None:
        k, n = pair
        p, lo, hi = wilson(k, n)
        print(f"  {label:<20} {p:6.1%}  [{lo:4.0%}, {hi:4.0%}]   ({k}/{n})")

    print("\n" + "=" * 58)
    print(f"GOLDEN SCOREBOARD   n={s['n']}   ({s['n_answerable']} answerable)")
    print("=" * 58)
    line("retrieval hit@5", s["hit_at_k"])
    line("refusal accuracy", s["refusal"])
    line("citation validity", s["citation"])
    print("  answer pass rate     <pending judge.py — next session>")
    print("=" * 58)


def main() -> None:
    import sys

    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])
    rows = load_golden(limit)

    print(f"Running {len(rows)} golden rows through the assistant…")
    results, t0 = [], time.perf_counter()
    for i, row in enumerate(rows, 1):
        r = run_row(row)
        results.append(r)
        print(
            f"  [{i:2}/{len(rows)}] {r['id']} label={r['label'] or '—':6} "
            f"{row['question'][:46]}",
            flush=True,
        )
    dur = time.perf_counter() - t0

    s = score(results)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = OUT_DIR / f"run-{stamp}.json"
    out.write_text(
        json.dumps(
            {"ts": stamp, "dur_s": round(dur, 1), "score": s, "results": results},
            indent=2,
        )
    )
    print_scoreboard(s)
    print(f"\nSaved {out}  ({dur:.0f}s)")


if __name__ == "__main__":
    main()
