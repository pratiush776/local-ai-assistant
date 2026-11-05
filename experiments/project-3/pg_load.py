'''building docs table, load 100 sentences-embeddings, index both lanes'''

import httpx
import numpy as np
import psycopg
from pgvector.psycopg import register_vector

DSN = "postgresql://postgres:rag@localhost:5432/postgres"

def embed(texts: list[str]) -> list:
    r = httpx.post("http://localhost:11434/api/embed", json={"model":"nomic-embed-text","input":texts}, timeout=300)
    r.raise_for_status()
    return r.json()["embeddings"]

sentences = open("data/hundred_sentences.txt").read().splitlines()
vectors = [np.array(v) for v in embed([f"search_document: {s}" for s in sentences])]

with psycopg.connect(DSN, autocommit=True) as conn:
    register_vector(conn)

    conn.execute("DROP TABLE IF EXISTS docs")
    conn.execute("""
    CREATE TABLE docs(
    id bigserial PRIMARY KEY,
    sentence text NOT NULL,
    embedding vector(768),
    fts tsvector GENERATED ALWAYS AS (to_tsvector('english',sentence)) STORED
    )
    """)

    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO docs (sentence, embedding) VALUES (%s, %s)",
            list(zip(sentences, vectors))
        )

    conn.execute("CREATE INDEX ON docs USING hnsw (embedding vector_cosine_ops)")
    conn.execute("CREATE INDEX ON docs USING gin (fts)")

    count = conn.execute("SELECT count(*) FROM docs").fetchone()
    count = count[0] if count else count
    print("rows loaded:", count)