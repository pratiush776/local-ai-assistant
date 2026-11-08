"""Measure the reranker's impact on SciFact: hybrid top-20, metrics before vs
after a cross-encoder rerank. recall@3, recall@5, MRR."""

import httpx
import numpy as np
import psycopg
from collections import defaultdict
from datasets import load_dataset
from pgvector.psycopg import register_vector
from sentence_transformers import CrossEncoder  # type: ignore[reportPrivateImportUsage]

DSN = "postgresql://postgres:rag@localhost:5432/postgres"
SHORTLIST = 20

qrels = load_dataset("BeIR/scifact-qrels")["test"]
queries = load_dataset("BeIR/scifact", "queries")["queries"]
relevant = defaultdict(set)
for r in qrels:
    relevant[str(r["query-id"])].add(str(r["corpus-id"]))  # type: ignore[dataset specific]
qtext = {str(r["_id"]): r["text"] for r in queries}  # type: ignore[dataset specific]
test_ids = list(relevant)


def embed(batch):
    r = httpx.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "nomic-embed-text",
            "input": [f"search_query: {t}" for t in batch],
        },
        timeout=600,
    )
    r.raise_for_status()
    return [np.array(v) for v in r.json()["embeddings"]]


qvecs = {}
for i in range(0, len(test_ids), 128):
    chunk = test_ids[i : i + 128]
    for qid, v in zip(chunk, embed([qtext[q] for q in chunk])):
        qvecs[qid] = v

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


reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
agg = {m: defaultdict(float) for m in ("before", "after")}

with psycopg.connect(DSN) as conn:
    register_vector(conn)
    for j, qid in enumerate(test_ids, 1):
        rows = conn.execute(HYBRID_SQL, {"q": qvecs[qid], "t": qtext[qid]}).fetchall()
        ids_before = [doc_id for doc_id, _, _ in rows]
        scores = reranker.predict([[qtext[qid], text] for _, text, _ in rows])
        order = sorted(range(len(rows)), key=lambda i: scores[i], reverse=True)
        ids_after = [rows[i][0] for i in order]
        rel = relevant[qid]
        for name, ids in (("before", ids_before), ("after", ids_after)):
            agg[name]["r@3"] += recall_at_k(ids, rel, 3)
            agg[name]["r@5"] += recall_at_k(ids, rel, 5)
            agg[name]["mrr"] += rr(ids, rel)
        if j % 100 == 0:
            print(f"  {j}/{len(test_ids)}", flush=True)

n = len(test_ids)
print(f"\n{'':8}{'recall@3':>10}{'recall@5':>10}{'MRR':>10}")
for name in ("before", "after"):
    a = agg[name]
    print(f"{name:8}{a['r@3'] / n:>10.4f}{a['r@5'] / n:>10.4f}{a['mrr'] / n:>10.4f}")
