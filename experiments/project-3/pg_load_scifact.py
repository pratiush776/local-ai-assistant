import httpx, numpy as np, psycopg
from pgvector.psycopg import register_vector
from datasets import load_dataset

DSN = "postgresql://postgres:rag@localhost:5432/postgres"
URL = "http://localhost:11434/api/embed"


def embed_batch(texts: list[str]) -> list:
    r = httpx.post(URL, json={"model": "nomic-embed-text", "input": texts}, timeout=600)
    r.raise_for_status()
    return r.json()["embeddings"]


corpus = load_dataset("BeIR/scifact", "corpus")["corpus"]
# BEIR convention: embed title + text together; str() the id
docs = [(str(r["_id"]), f'{r["title"]} {r["text"]}'.strip()) for r in corpus]

with psycopg.connect(DSN, autocommit=True) as conn:
    register_vector(conn)
    #add extensions for vector & pg_search (for keyword) field types
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.execute("CREATE EXTENSION IF NOT EXISTS pg_search")

    conn.execute("DROP TABLE IF EXISTS scifact")
    conn.execute("""
        CREATE TABLE scifact (
            doc_id    text PRIMARY KEY,
            text      text NOT NULL,
            embedding vector(768),
            fts       tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
        )
    """)
    BATCH = 128
    with conn.cursor() as cur:
        for i in range(0, len(docs), BATCH):
            chunk = docs[i:i + BATCH]
            vecs = embed_batch([f"search_document: {t}" for _, t in chunk])
            cur.executemany(
                "INSERT INTO scifact (doc_id, text, embedding) VALUES (%s, %s, %s)",
                [(did, txt, np.array(v)) for (did, txt), v in zip(chunk, vecs)],
            )
            print(f"  embedded {min(i + BATCH, len(docs))}/{len(docs)}", flush=True)
    conn.execute("CREATE INDEX ON scifact USING hnsw (embedding vector_cosine_ops)")
    conn.execute("CREATE INDEX scifact_bm25 ON scifact USING bm25 (doc_id, text) WITH (key_field='doc_id')")
    print("done, rows:", conn.execute("SELECT count(*) FROM scifact").fetchone()[0])

