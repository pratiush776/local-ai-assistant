"""Compare rerankers on a SciFact slice: recall@3/@5/MRR + ms/query.
Retrieve the hybrid top-20 once per query, then rerank with each model on the
same inputs. A 'no rerank' baseline (hybrid order) sits on top for reference."""
import gc
import time
import httpx
import numpy as np
import psycopg
from collections import defaultdict
from datasets import load_dataset
from pgvector.psycopg import register_vector
from sentence_transformers import CrossEncoder  # type: ignore[reportPrivateImportUsage]

DSN = "postgresql://postgres:rag@localhost:5432/postgres"
N = 100
MODELS = [
    "BAAI/bge-reranker-v2-m3",              # current (~2.3 GB)
    "BAAI/bge-reranker-base",               # smaller (~0.5 GB)
    "cross-encoder/ms-marco-MiniLM-L-6-v2", # tiny (~0.08 GB)
]

qrels = load_dataset("BeIR/scifact-qrels")["test"]
queries = load_dataset("BeIR/scifact", "queries")["queries"]
relevant = defaultdict(set)
for r in qrels:
    relevant[str(r["query-id"])].add(str(r["corpus-id"]))
qtext = {str(r["_id"]): r["text"] for r in queries}
test_ids = list(relevant)[:N]


def embed(batch):
    r = httpx.post("http://localhost:11434/api/embed",
                   json={"model": "nomic-embed-text", "input": [f"search_query: {t}" for t in batch]}, timeout=600)
    r.raise_for_status()
    return [np.array(v) for v in r.json()["embeddings"]]


qvecs = {}
for i in range(0, len(test_ids), 128):
    c = test_ids[i:i + 128]
    for q, v in zip(c, embed([qtext[x] for x in c])):
        qvecs[q] = v

HYBRID_SQL = """
WITH vec AS (
  SELECT doc_id, ROW_NUMBER() OVER (ORDER BY embedding <=> %(q)s) rank
  FROM scifact ORDER BY embedding <=> %(q)s LIMIT 20),
kw AS (
  SELECT doc_id, ROW_NUMBER() OVER (ORDER BY paradedb.score(doc_id) DESC) rank
  FROM scifact WHERE doc_id @@@ paradedb.match('text', %(t)s) LIMIT 20)
SELECT c.doc_id, c.text,
       COALESCE(1.0/(60+vec.rank),0) + COALESCE(1.0/(60+kw.rank),0) AS rrf
FROM vec FULL OUTER JOIN kw USING (doc_id)
JOIN scifact c ON c.doc_id = COALESCE(vec.doc_id, kw.doc_id)
ORDER BY rrf DESC LIMIT 20;
"""


def recall_at_k(ranked, rel, k):
    return len(rel.intersection(ranked[:k])) / len(rel)


def rr(ranked, rel):
    for i, d in enumerate(ranked, 1):
        if d in rel:
            return 1.0 / i
    return 0.0


def metrics(order_by_qid):
    agg = defaultdict(float)
    for qid in test_ids:
        ids, rel = order_by_qid[qid], relevant[qid]
        agg["r@3"] += recall_at_k(ids, rel, 3)
        agg["r@5"] += recall_at_k(ids, rel, 5)
        agg["mrr"] += rr(ids, rel)
    return {k: v / len(test_ids) for k, v in agg.items()}


# retrieve every shortlist ONCE (same inputs for all rerankers)
shortlists = {}
with psycopg.connect(DSN) as conn:
    register_vector(conn)
    for qid in test_ids:
        rows = conn.execute(HYBRID_SQL, {"q": qvecs[qid], "t": qtext[qid]}).fetchall()
        shortlists[qid] = [(doc_id, text) for doc_id, text, _ in rows]

print(f"{'model':40}{'r@3':>8}{'r@5':>8}{'MRR':>8}{'ms/q':>9}")
base = metrics({qid: [d for d, _ in shortlists[qid]] for qid in test_ids})
print(f"{'hybrid (no rerank)':40}{base['r@3']:>8.3f}{base['r@5']:>8.3f}{base['mrr']:>8.3f}{'-':>9}")

for m in MODELS:
    model = CrossEncoder(m)
    t0 = time.perf_counter()
    reranked = {}
    for qid in test_ids:
        sl = shortlists[qid]
        scores = model.predict([[qtext[qid], t] for _, t in sl])
        order = sorted(range(len(sl)), key=lambda i: scores[i], reverse=True)
        reranked[qid] = [sl[i][0] for i in order]
    ms = (time.perf_counter() - t0) * 1000 / len(test_ids)
    mm = metrics(reranked)
    print(f"{m:40}{mm['r@3']:>8.3f}{mm['r@5']:>8.3f}{mm['mrr']:>8.3f}{ms:>9.1f}")
    del model
    gc.collect()
