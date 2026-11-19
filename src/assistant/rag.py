import hashlib
from typing import Literal

import httpx
import numpy as np
import psycopg
from docling.chunking import HybridChunker  # type: ignore[reportPrivateImportUsage]
from docling.document_converter import DocumentConverter
from pgvector.psycopg import register_vector
from sentence_transformers import CrossEncoder

# --- config (single source of truth) ---
DSN = "postgresql://postgres:rag@localhost:5432/postgres"
OLLAMA_URL = "http://localhost:11434/api/"
EMBED_MODEL = "nomic-embed-text"
GEN_MODEL = "qwen3.5:9b"
RERANKER_NAME = "BAAI/bge-reranker-v2-m3"
RRF_K = 60
# recall-favouring pre-filter, not the honesty gate: no single threshold separates
# relevant from noise (France/Lisbon noise scored 0.31 > real answers at 0.24), so we
# keep this low and let grounded generation refuse near-miss noise. Provisional until
# the personal-files eval (#5) calibrates it. See ROADMAP "To review".
RERANK_THRESHOLD = 0.01
# KEEP_N: when set, keep the top-N chunks by rank and ignore RERANK_THRESHOLD.
# None -> the old score-threshold behaviour. A fixed cut is kept-set-size-invariant,
# so it isolates the reranker (5.5) and sidesteps the uncalibrated threshold.
KEEP_N: int | None = None
TOP_K = 10
# The cross-encoder is OFF by default: it costs ~2.3 GB RAM + a cold ~7 s load on an
# 18 GB laptop that already swaps. Off -> the hybrid RRF order goes straight to
# generation. Lesson 5.5 measures whether it earns its memory and may flip this.
RERANK_ENABLED = False
PERSONA_TODAY = "2024-05-15"  # the assistant's clock while it runs as the persona
_RERANKER = None  # module-level cache, starts empty
RAG_SYSTEM_PROMPT = """Answer using ONLY the numbered context below. Cite the source number [n] with exact text used from the context to create the response
for reference so that the user can trace it back for every fact respectively.
If the context does not contain the answer, say you don't have it truthfully."""


def embed(
    texts: list[str], task_prefix: Literal["search_query", "search_document"]
) -> np.ndarray:
    payload = {
        "model": EMBED_MODEL,
        "input": [f"{task_prefix}: {text}" for text in texts],
    }
    r = httpx.post(OLLAMA_URL + "embed", json=payload, timeout=120)
    r.raise_for_status()
    return np.array(r.json()["embeddings"])


def hybrid_retrieve(
    question: str, k=TOP_K
) -> list[tuple[str, str, float]]:  # (chunk_id, text, rrf_score), #3
    # DB search
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
    query_vector = embed([question], "search_query")[0]
    with psycopg.connect(DSN) as conn:
        register_vector(conn)
        rows = conn.execute(
            SQL, {"q": query_vector, "t": question, "rrf_k": RRF_K, "k": k}
        ).fetchall()
    return [(cid, text, float(rrf)) for cid, text, rrf in rows]


def _reranker() -> CrossEncoder:
    global _RERANKER
    if _RERANKER is None:
        _RERANKER = CrossEncoder(RERANKER_NAME)
    return _RERANKER


def warmup_reranker() -> None:
    _reranker()


def rag_retrieve(question: str) -> list[tuple[str, str, float]]:
    """Every candidate as (chunk_id, text, score), best first.
    Reranker ON  -> score is the cross-encoder relevance score.
    Reranker OFF -> score is the hybrid RRF score (order unchanged)."""
    hits = hybrid_retrieve(question, TOP_K)  # (cid, text, rrf)
    if not RERANK_ENABLED:
        return hits
    scores = _reranker().predict([[question, text] for _, text, _ in hits])
    reranked = sorted(zip(hits, scores), key=lambda x: x[1], reverse=True)
    return [(cid, text, float(score)) for (cid, text, _), score in reranked]


def rag_search(question: str) -> list[tuple[str, str]]:
    chunks = rag_retrieve(question)  # best-first
    if KEEP_N is not None:
        return [(cid, text) for cid, text, _ in chunks[:KEEP_N]]
    return [(cid, text) for cid, text, score in chunks if score >= RERANK_THRESHOLD]


def format_context(hits: list[tuple[str, str]]) -> str:
    return "".join(
        [f"[{i + 1}] (source={cid}) {text}\n" for i, (cid, text) in enumerate(hits)]
    )


# ---- read-only views of what's in memory (for the GUI's Memory panel) ----
def list_sources() -> list[tuple[str, int]]:
    """Every ingested file and how many chunks it holds."""
    with psycopg.connect(DSN) as conn:
        return conn.execute(
            "SELECT source, count(*) FROM chunks GROUP BY source ORDER BY source"
        ).fetchall()


def chunks_for_source(source: str) -> list[tuple[str, str]]:
    """The indexed chunks (id, text) of one file — the actual retrieval units."""
    with psycopg.connect(DSN) as conn:
        return conn.execute(
            "SELECT chunk_id, text FROM chunks WHERE source = %s ORDER BY chunk_id",
            (source,),
        ).fetchall()


def sources_for_chunks(cids: list[str]) -> dict[str, str]:
    """Map cited chunk ids back to their source file, for citation tracing."""
    if not cids:
        return {}
    with psycopg.connect(DSN) as conn:
        rows = conn.execute(
            "SELECT chunk_id, source FROM chunks WHERE chunk_id = ANY(%s)", (cids,)
        ).fetchall()
    return {cid: src for cid, src in rows}


def texts_for_chunks(cids: list[str]) -> dict[str, str]:
    """Map chunk ids to their full text — the content the trace sidecar freezes."""
    if not cids:
        return {}
    with psycopg.connect(DSN) as conn:
        rows = conn.execute(
            "SELECT chunk_id, text FROM chunks WHERE chunk_id = ANY(%s)", (cids,)
        ).fetchall()
    return {cid: text for cid, text in rows}


def ingest(path: str) -> dict:
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
        vecs = embed([t for _, _, t in rows], "search_document")
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
