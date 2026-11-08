import httpx, numpy as np, psycopg
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker  # type: ignore[reportPrivateImportUsage]
import hashlib
from pgvector.psycopg import register_vector


converter = DocumentConverter()

result = converter.convert("data/docs/notes.md")
doc = result.document

# print(doc.export_to_markdown())

chunker = HybridChunker()
chunks = list(chunker.chunk(doc))

for i, ch in enumerate(chunks):
    text = chunker.contextualize(chunk=ch)
    cid = hashlib.sha256(text.encode()).hexdigest()[:16]
    print(f"[{i}] id={cid} ({len(text)} chars)")
    print("---")

# ----------- database ingest
DSN = "postgresql://postgres:rag@localhost:5432/postgres"


def embed(texts: list[str]) -> list[np.ndarray]:
    r = httpx.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "nomic-embed-text",
            "input": [f"search_document: {t}" for t in texts],
        },
        timeout=600,
    )
    r.raise_for_status()
    return [np.array(v) for v in r.json()["embeddings"]]


rows = []
for ch in chunks:
    text = chunker.contextualize(chunk=ch)
    cid = hashlib.sha256(text.encode()).hexdigest()[:16]
    rows.append((cid, "data/docs/notes.md", text))

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
    vecs = embed([t for _, _, t in rows])
    before = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]  # type: ignore[reportPrivateImportUsage]
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO chunks (chunk_id, source, text, embedding) VALUES (%s,%s,%s,%s)"
            "ON CONFLICT (chunk_id) DO NOTHING",
            [(cid, src, text, v) for (cid, src, text), v in zip(rows, vecs)],
        )
    after = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]  # type: ignore[reportPrivateImportUsage]
    print(f"inserted {after - before} new chunks (table now has {after})")

    conn.execute(
        "CREATE INDEX IF NOT EXISTS chunks_hnsw ON chunks USING hnsw (embedding vector_cosine_ops)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS chunks_bm25 ON chunks USING bm25 (chunk_id, text) WITH (key_field='chunk_id')"
    )

    current_ids = [cid for cid, _, _ in rows]
    deleted = conn.execute(
        "DELETE FROM chunks WHERE source = %s AND chunk_id <> ALL(%s)",
        ("data/docs/notes.md", current_ids),
    ).rowcount
    print(f"delted {deleted} stale chunks")
