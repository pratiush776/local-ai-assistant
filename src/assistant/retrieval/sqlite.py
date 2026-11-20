"""SQLite backend: FTS5 `bm25()` (real Okapi BM25, the reason ParadeDB existed)
for the keyword lane + sqlite-vec brute-force cosine KNN for the vector lane,
fused app-side with RRF. In-process, no server, near-zero idle memory, permissive
licence. Brute-force is exact and correct at a personal-corpus scale (Unit 3).

Three tables share one rowid:
  chunks       canonical (chunk_id ↔ rowid ↔ source, text)
  chunks_fts   FTS5 over text (porter stemmer), keyed by the same rowid
  vec_chunks   sqlite-vec vec0 embeddings, keyed by the same rowid
so the two lanes return rowids and we fuse, then map rowid -> (chunk_id, text).

Connections are thread-local (see _get_conn): a SQLite connection is not safe to
share across threads, so each thread gets its own, opened once and reused. That
keeps reuse safe AND pays the ~14 ms connect+extension-load only once per thread.
WAL + busy_timeout ready the file for the concurrent reads/writes that background
jobs (Unit 9) and multi-client access (Unit 10) will bring.
"""

import hashlib
import re
import sqlite3
import threading
from pathlib import Path

import sqlite_vec
from docling.chunking import HybridChunker  # type: ignore[reportPrivateImportUsage]
from docling.document_converter import DocumentConverter
from sqlite_vec import serialize_float32

DB_PATH = Path("data/rag.db")
EMBED_DIM = 768  # nomic-embed-text; L2-normalized, so cosine == this metric

_local = threading.local()  # holds one connection per thread


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS chunks ("
        "chunk_id TEXT UNIQUE NOT NULL, source TEXT NOT NULL, text TEXT NOT NULL)"
    )
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5("
        "chunk_id UNINDEXED, text, tokenize='porter unicode61')"
    )
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0("
        f"embedding float[{EMBED_DIM}] distance_metric=cosine)"
    )


def _get_conn() -> sqlite3.Connection:
    """This thread's connection, opened once and cached. check_same_thread stays
    True (its default): it's the guard that catches a connection used off its own
    thread — thread-local makes sure that never happens while reusing the handle."""
    conn = getattr(_local, "conn", None)
    if conn is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.execute("PRAGMA journal_mode=WAL")  # readers don't block the writer
        conn.execute("PRAGMA busy_timeout=5000")  # a late writer waits, not errors
        _ensure_schema(conn)
        _local.conn = conn
    return conn


def _fts_query(question: str) -> str:
    """Turn a natural question into a safe FTS5 MATCH expression: quote each word
    token (so punctuation/apostrophes can't break the syntax) and OR them, so any
    term can match. Porter stemming still applies inside the quoted terms."""
    terms = re.findall(r"\w+", question.lower())
    return " OR ".join(f'"{t}"' for t in terms)


class SQLiteBackend:
    def hybrid_retrieve(self, question: str, k: int) -> list[tuple[str, str, float]]:
        from assistant import rag  # lazy: avoids a circular import at load

        rrf_k = rag.RRF_K
        qvec = serialize_float32(rag.embed([question], "search_query")[0])
        conn = _get_conn()
        # vector lane: brute-force cosine KNN -> rowid ranked by distance
        vec = conn.execute(
            "SELECT rowid, distance FROM vec_chunks "
            "WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
            (qvec, k),
        ).fetchall()
        # keyword lane: FTS5 real BM25 -> rowid ranked (bm25 is negative, asc = best)
        kw_rows = []
        match = _fts_query(question)
        if match:
            kw_rows = conn.execute(
                "SELECT rowid, bm25(chunks_fts) AS s FROM chunks_fts "
                "WHERE chunks_fts MATCH ? ORDER BY s LIMIT ?",
                (match, k),
            ).fetchall()

        # --- app-side RRF: 1/(rrf_k + rank) summed across the lanes it appears in ---
        rrf: dict[int, float] = {}
        for rank, (rid, _) in enumerate(vec, start=1):
            rrf[rid] = rrf.get(rid, 0.0) + 1.0 / (rrf_k + rank)
        for rank, (rid, _) in enumerate(kw_rows, start=1):
            rrf[rid] = rrf.get(rid, 0.0) + 1.0 / (rrf_k + rank)

        top = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:k]
        if not top:
            return []
        rowids = [rid for rid, _ in top]
        qmarks = ",".join("?" * len(rowids))
        meta = {
            rid: (cid, text)
            for rid, cid, text in conn.execute(
                f"SELECT rowid, chunk_id, text FROM chunks WHERE rowid IN ({qmarks})",
                rowids,
            )
        }
        return [(meta[rid][0], meta[rid][1], score) for rid, score in top]

    def ingest(self, path: str) -> dict:
        from assistant import rag

        doc = DocumentConverter().convert(path).document
        chunker = HybridChunker()
        rows = []
        for ch in chunker.chunk(doc):
            text = chunker.contextualize(chunk=ch)
            cid = hashlib.sha256(text.encode()).hexdigest()[:16]
            rows.append((cid, path, text))
        vecs = rag.embed([t for _, _, t in rows], "search_document")

        conn = _get_conn()
        before = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
        for (cid, src, text), v in zip(rows, vecs):
            cur = conn.execute(
                "INSERT OR IGNORE INTO chunks(chunk_id, source, text) VALUES (?,?,?)",
                (cid, src, text),
            )
            if cur.rowcount == 1:  # a genuinely new chunk
                rid = cur.lastrowid
                conn.execute(
                    "INSERT INTO chunks_fts(rowid, chunk_id, text) VALUES (?,?,?)",
                    (rid, cid, text),
                )
                conn.execute(
                    "INSERT INTO vec_chunks(rowid, embedding) VALUES (?,?)",
                    (rid, serialize_float32(v)),
                )
        after = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]

        # stale sweep: drop this file's chunks that are no longer present
        current = {cid for cid, _, _ in rows}
        stale = [
            rid
            for rid, cid in conn.execute(
                "SELECT rowid, chunk_id FROM chunks WHERE source=?", (path,)
            )
            if cid not in current
        ]
        for rid in stale:
            conn.execute("DELETE FROM chunks WHERE rowid=?", (rid,))
            conn.execute("DELETE FROM chunks_fts WHERE rowid=?", (rid,))
            conn.execute("DELETE FROM vec_chunks WHERE rowid=?", (rid,))
        conn.commit()
        return {
            "inserted": after - before,
            "deleted": len(stale),
            "total": after - len(stale),
        }

    def wipe(self) -> int:
        conn = _get_conn()
        n = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
        conn.execute("DELETE FROM chunks")
        conn.execute("DELETE FROM chunks_fts")
        conn.execute("DELETE FROM vec_chunks")
        conn.commit()
        return n

    def list_sources(self) -> list[tuple[str, int]]:
        conn = _get_conn()
        return conn.execute(
            "SELECT source, count(*) FROM chunks GROUP BY source ORDER BY source"
        ).fetchall()

    def chunks_for_source(self, source: str) -> list[tuple[str, str]]:
        conn = _get_conn()
        return conn.execute(
            "SELECT chunk_id, text FROM chunks WHERE source=? ORDER BY chunk_id",
            (source,),
        ).fetchall()

    def sources_for_chunks(self, cids: list[str]) -> dict[str, str]:
        if not cids:
            return {}
        conn = _get_conn()
        qmarks = ",".join("?" * len(cids))
        rows = conn.execute(
            f"SELECT chunk_id, source FROM chunks WHERE chunk_id IN ({qmarks})", cids
        ).fetchall()
        return {cid: src for cid, src in rows}

    def texts_for_chunks(self, cids: list[str]) -> dict[str, str]:
        if not cids:
            return {}
        conn = _get_conn()
        qmarks = ",".join("?" * len(cids))
        rows = conn.execute(
            f"SELECT chunk_id, text FROM chunks WHERE chunk_id IN ({qmarks})", cids
        ).fetchall()
        return {cid: text for cid, text in rows}
