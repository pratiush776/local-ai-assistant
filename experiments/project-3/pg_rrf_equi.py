
import httpx, numpy as np, psycopg
from pgvector.psycopg import register_vector

DSN = "postgresql://postgres:rag@localhost:5432/postgres"

def embed_query(text:str) -> np.ndarray:
    r = httpx.post("http://localhost:11434/api/embed",
                   json={
                       "model":"nomic-embed-text",
                       "input":[f"query_document: {text}"]
                   },
                   timeout=300)
    r.raise_for_status()
    return np.array(r.json()["embeddings"][0])

query = "beagle" 
q = embed_query(query)

SQL ="""
WITH vector AS (                                        -- lane 1 ranking
    SELECT id, ROW_NUMBER() OVER (ORDER BY embedding <=> %(q)s) AS rank
    FROM docs ORDER BY embedding <=> %(q)s LIMIT 20
),
keyword AS (                                            -- lane 2 ranking (zero-filtered by WHERE)
    SELECT id, ROW_NUMBER() OVER (
             ORDER BY ts_rank(fts, websearch_to_tsquery('english', %(text)s)) DESC) AS rank
    FROM docs WHERE fts @@ websearch_to_tsquery('english', %(text)s) ORDER BY rank LIMIT 20
)
SELECT d.sentence,
       COALESCE(1.0/(60 + v.rank), 0.0)                 -- this doc's vector contribution
     + COALESCE(1.0/(60 + k.rank), 0.0) AS rrf          -- + its keyword contribution
FROM vector v
FULL OUTER JOIN keyword k USING (id)                    -- keep docs from EITHER lane
JOIN docs d ON d.id = COALESCE(v.id, k.id)
ORDER BY rrf DESC
LIMIT 5;
"""

with psycopg.connect(DSN) as conn:
    register_vector(conn)
    for sentence, rrf in conn.execute(SQL, {"q":q, "text":query}).fetchall():
        print(f" {rrf:.4f} {sentence}")
