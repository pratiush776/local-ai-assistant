"""Uses GIN index and @@ operation on fts field(deprecated) to conceptuazlize native Postgres behavior for keyword search
Upgraded to ParadeDB with actual BM25 indexing corpus rather than keyword matching (@@)"""

import httpx
import numpy as np
import psycopg
from pgvector.psycopg import register_vector

DSN = "postgresql://postgres:rag@localhost:5432/postgres"


def embed_query(text: str) -> np.ndarray:
    r = httpx.post(
        "http://localhost:11434/api/embed",
        json={"model": "nomic-embed-text", "input": [f"search_query: {text}"]},
        timeout=120,
    )
    r.raise_for_status()
    return np.array(r.json()["embeddings"][0])


query = "how to take care of my pet"
q = embed_query(query)

with psycopg.connect(DSN) as conn:
    register_vector(conn)

    print("---- VECTOR (cosine distance, vai HNSW) ---")
    for s, dist in conn.execute(
        """SELECT sentence, embedding <=> %s AS distance
        FROM docs ORDER BY embedding <=> %s LIMIT 5""",
        (q, q),
    ).fetchall():
        print(f" {dist:.3f} {s}")

    print("--- KEYWORD (full-text, via GIN) ---")
    rows = conn.execute(
        """SELECT sentence ,ts_rank(fts, websearch_to_tsquery('english', %s)) AS rank
        FROM docs 
        WHERE fts @@ websearch_to_tsquery('english', %s)
        ORDER BY rank DESC 
        LIMIT 5""",
        (query, query),
    ).fetchall()
    for s, rnk in rows:
        print(f" {rnk:.3f} {s}")
    if not rows:
        print(" (no keyword matches)")
