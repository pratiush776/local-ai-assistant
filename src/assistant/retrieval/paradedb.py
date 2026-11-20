"""ParadeDB backend: Postgres with pgvector (cosine KNN) + pg_search (real BM25),
fused with RRF inside one SQL round-trip. This is the original rag.py retrieval
code, moved behind the RetrievalBackend seam unchanged. Best for a hosted/large
corpus where DB-side fusion and an HNSW index earn their keep.

Needs the ParadeDB container on :5432. `pg_search` is AGPL (see memory).
"""

import hashlib

import psycopg
from docling.chunking import HybridChunker  # type: ignore[reportPrivateImportUsage]
from docling.document_converter import DocumentConverter
from pgvector.psycopg import register_vector

DSN = "postgresql://postgres:rag@localhost:5432/postgres"


class ParadeDBBackend:
    def hybrid_retrieve(self, question: str, k: int) -> list[tuple[str, str, float]]:
        from assistant import rag  # lazy: avoids a circular import at load

        SQL = """
WITH vec AS (
    SELECT chunk_id, ROW_NUMBER() OVER (ORDER BY embedding <=> %(q)s) rank
    FROM chunks ORDER BY embedding <=> %(q)s LIMIT %(k)s
),kw AS (
    SELECT chunk_id, ROW_NUMBER() OVER (ORDER BY paradedb.score(chunk_id) DESC) rank
    FROM chunks WHERE chunk_id @@@ paradedb.match('text', %(t)s) LIMIT %(k)s
)
SELECT c.chunk_id, c.text,
    COALESCE(1.0/(%(rrf_k)s+vec.rank),0) + COALESCE(1.0/(%(rrf_k)s+kw.rank),0) AS rrf
FROM vec FULL OUTER JOIN kw USING (chunk_id)
JOIN chunks c ON c.chunk_id = COALESCE(vec.chunk_id, kw.chunk_id)
ORDER BY rrf DESC LIMIT %(k)s;
"""
        query_vector = rag.embed([question], "search_query")[0]
        with psycopg.connect(DSN) as conn:
            register_vector(conn)
            rows = conn.execute(
                SQL, {"q": query_vector, "t": question, "rrf_k": rag.RRF_K, "k": k}
            ).fetchall()
        return [(cid, text, float(rrf)) for cid, text, rrf in rows]

    def ingest(self, path: str) -> dict:
        from assistant import rag

        doc = DocumentConverter().convert(path).document
        chunker = HybridChunker()
        chunks = list(chunker.chunk(doc))

        rows = []
        for ch in chunks:
            text = chunker.contextualize(chunk=ch)
            cid = hashlib.sha256(text.encode()).hexdigest()[:16]
            rows.append((cid, path, text))

        with psycopg.connect(DSN, autocommit=True) as conn:
            register_vector(conn)
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute("CREATE EXTENSION IF NOT EXISTS pg_search")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                chunk_id text PRIMARY KEY,
                source text NOT NULL,
                text text NOT NULL,
                embedding vector(768)
                )
        """)
            vecs = rag.embed([t for _, _, t in rows], "search_document")
            before = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]  # type: ignore[reportPrivateImportUsage]
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO chunks (chunk_id, source, text, embedding) VALUES (%s,%s,%s,%s)"
                    "ON CONFLICT (chunk_id) DO NOTHING",
                    [(cid, src, text, v) for (cid, src, text), v in zip(rows, vecs)],
                )
            after = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]  # type: ignore[reportPrivateImportUsage]

            conn.execute(
                "CREATE INDEX IF NOT EXISTS chunks_hnsw ON chunks USING hnsw (embedding vector_cosine_ops)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS chunks_bm25 ON chunks USING bm25 (chunk_id, text) WITH (key_field='chunk_id')"
            )

            current_ids = [cid for cid, _, _ in rows]
            deleted = conn.execute(
                "DELETE FROM chunks WHERE source = %s AND chunk_id <> ALL(%s)",
                (path, current_ids),
            ).rowcount
            return {
                "inserted": after - before,
                "deleted": deleted,
                "total": after - deleted,
            }

    def wipe(self) -> int:
        with psycopg.connect(DSN, autocommit=True) as conn:
            try:
                n = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
                conn.execute("TRUNCATE chunks")
                return n
            except psycopg.errors.UndefinedTable:
                return 0

    def list_sources(self) -> list[tuple[str, int]]:
        with psycopg.connect(DSN) as conn:
            return conn.execute(
                "SELECT source, count(*) FROM chunks GROUP BY source ORDER BY source"
            ).fetchall()

    def chunks_for_source(self, source: str) -> list[tuple[str, str]]:
        with psycopg.connect(DSN) as conn:
            return conn.execute(
                "SELECT chunk_id, text FROM chunks WHERE source = %s ORDER BY chunk_id",
                (source,),
            ).fetchall()

    def sources_for_chunks(self, cids: list[str]) -> dict[str, str]:
        if not cids:
            return {}
        with psycopg.connect(DSN) as conn:
            rows = conn.execute(
                "SELECT chunk_id, source FROM chunks WHERE chunk_id = ANY(%s)", (cids,)
            ).fetchall()
        return {cid: src for cid, src in rows}

    def texts_for_chunks(self, cids: list[str]) -> dict[str, str]:
        if not cids:
            return {}
        with psycopg.connect(DSN) as conn:
            rows = conn.execute(
                "SELECT chunk_id, text FROM chunks WHERE chunk_id = ANY(%s)", (cids,)
            ).fetchall()
        return {cid: text for cid, text in rows}
