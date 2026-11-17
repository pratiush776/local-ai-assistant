"""Ingest every file in one or more folders into the chunk store.

    uv run python -m assistant.ingest_folder data/assets/persona/files data/assets/persona/documents
    uv run python -m assistant.ingest_folder --fresh <folders...>   # wipe the store first

Each file goes through rag.ingest() (Docling -> HybridChunker -> content-hash ids ->
embed -> upsert). --fresh empties the chunks table first, for switching corpora cleanly
(e.g. dropping the old synthetic files when we load the persona).
"""

import sys
import time
from pathlib import Path

import psycopg

from . import rag

SKIP = {".DS_Store"}  # noise, not documents


def _wipe() -> int:
    with psycopg.connect(rag.DSN, autocommit=True) as conn:
        try:
            n = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
            conn.execute("TRUNCATE chunks")
            return n
        except psycopg.errors.UndefinedTable:
            return 0


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    fresh = "--fresh" in args
    folders = [a for a in args if a != "--fresh"]
    if not folders:
        print(__doc__)
        raise SystemExit(2)

    if fresh:
        removed = _wipe()
        print(f"[fresh] emptied store ({removed} chunks removed)\n")

    files = sorted(
        p
        for folder in folders
        for p in Path(folder).rglob("*")
        if p.is_file() and p.name not in SKIP
    )
    print(f"Ingesting {len(files)} file(s) from {', '.join(folders)}:\n")

    inserted = 0
    t0 = time.perf_counter()
    for i, path in enumerate(files, 1):
        ts = time.perf_counter()
        res = rag.ingest(str(path))
        inserted += res["inserted"]
        print(
            f"  [{i:2}/{len(files)}] {path.name:40} "
            f"+{res['inserted']:3} chunks  ({time.perf_counter() - ts:4.1f}s)"
        )

    total = rag.list_sources()
    chunks = sum(n for _, n in total)
    print(
        f"\nDone in {time.perf_counter() - t0:.0f}s: "
        f"{len(total)} sources, {chunks} chunks total (+{inserted} new)."
    )


if __name__ == "__main__":
    main()
