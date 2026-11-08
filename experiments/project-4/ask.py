import httpx, numpy as np, psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import CrossEncoder

DSN = "postgres://postgres:rag@localhost:5432/postgres"


def embed_query(q: str) -> np.ndarray:
    r = httpx.post(
        "http://localhost:11434/api/embed",
        json={"model": "nomic-embed-text", "input": [f"search_query: {q}"]},
        timeout=600,
    )
    r.raise_for_status()
    return np.array(r.json()["embeddings"][0])


question = "what is my bank balance?"

q = embed_query(question)

SQL = """
WITH vec AS (
    SELECT chunk_id, ROW_NUMBER() OVER (ORDER BY embedding <=> %(q)s) rank
    FROM chunks ORDER BY embedding <=> %(q)s LIMIT 10
),
kw AS (
    SELECT chunk_id, ROW_NUMBER() OVER (ORDER BY paradedb.score(chunk_id) DESC) rank
    FROM chunks WHERE chunk_id @@@ paradedb.match('text',%(t)s) LIMIT 10
)
SELECT c.chunk_id, c.text,
    COALESCE(1.0/(60+vec.rank),0) + COALESCE(1.0 / (60 + kw.rank),0) AS rrf
FROM vec FULL OUTER JOIN kw USING (chunk_id)
JOIN chunks c ON c.chunk_id = COALESCE(vec.chunk_id,kw.chunk_id)
ORDER BY rrf DESC LIMIT 10;
"""

with psycopg.connect(DSN) as conn:
    register_vector(conn)
    retrieved = [
        (cid, text)
        for cid, text, _ in conn.execute(SQL, {"q": q, "t": question}).fetchall()
    ]  # SQL now LIMIT 10

# --- rerank: cross-encoder scores each (query, chunk) pair jointly ---

reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")  # downloads once

pairs = [[question, text] for _, text in retrieved]  # the (query, chunk) pairs
scores = reranker.predict(pairs)  # one relevance score per pair

reranked = sorted(zip(retrieved, scores), key=lambda r: r[1], reverse=True)
# print(f"\nRERANKED  (query: {question!r})")
# for (cid, text), s in reranked:
#     print(f"  {s:+.3f}  [{cid[:8]}]  {text[:80]}")

# ---- generate: threshold shortlist, then pass to local LLM for answer with retrieval + chat context
THRESHOLD = 0.3
kept = [(cid, text) for (cid, text), s in reranked if s >= THRESHOLD]
if not kept:
    print("\nnot found - no relevant chunk found")
else:
    context = "\n\n".join(
        f"[{i + 1}] (source={cid}) {text}" for i, (cid, text) in enumerate(kept)
    )
    prompt = (
        "Answer the question using ONLY the context below."
        "Cite the source number [n] for each fact you use."
        'If the context does not contain the answer, reply exactly: "I don\'t know".\n\n'
        f"Context: \n{context}\n\nQuestion: {question}\nAnswer:"
    )
    r = httpx.post(
        "http://localhost:11434/api/chat",
        json={
            "model": "qwen3.5:9b",
            "messages": [{"role": "user", "content": prompt}],
            "think": False,
            "stream": False,
        },
        timeout=120,
    )
    r.raise_for_status()
    print("\nQuery:", question)
    print("\nAnswer:", r.json()["message"]["content"])
