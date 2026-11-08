"""Grounded three-way diagnosis: context limit + how often each reranker demotes
the relevant doc below hybrid's position, on a SciFact slice."""
import httpx
import numpy as np
import psycopg
from collections import defaultdict
from datasets import load_dataset
from pgvector.psycopg import register_vector
from sentence_transformers import CrossEncoder  # type: ignore[reportPrivateImportUsage]

DSN = "postgresql://postgres:rag@localhost:5432/postgres"
N = 40
MODELS = [
    "BAAI/bge-reranker-v2-m3",
    "BAAI/bge-reranker-base",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
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


qvecs = {q: v for q, v in zip(test_ids, embed([qtext[q] for q in test_ids]))}

HYBRID_SQL = """
WITH vec AS (SELECT doc_id, ROW_NUMBER() OVER (ORDER BY embedding <=> %(q)s) rank
             FROM scifact ORDER BY embedding <=> %(q)s LIMIT 20),
kw AS (SELECT doc_id, ROW_NUMBER() OVER (ORDER BY paradedb.score(doc_id) DESC) rank
       FROM scifact WHERE doc_id @@@ paradedb.match('text', %(t)s) LIMIT 20)
SELECT c.doc_id, c.text, COALESCE(1.0/(60+vec.rank),0)+COALESCE(1.0/(60+kw.rank),0) rrf
FROM vec FULL OUTER JOIN kw USING (doc_id)
JOIN scifact c ON c.doc_id = COALESCE(vec.doc_id, kw.doc_id)
ORDER BY rrf DESC LIMIT 20;
"""
shortlists = {}
with psycopg.connect(DSN) as conn:
    register_vector(conn)
    for qid in test_ids:
        shortlists[qid] = [(d, t) for d, t, _ in conn.execute(HYBRID_SQL, {"q": qvecs[qid], "t": qtext[qid]}).fetchall()]


def rank_of_rel(ids, rel):
    for i, d in enumerate(ids, 1):
        if d in rel:
            return i
    return 99


hyb_rank = {qid: rank_of_rel([d for d, _ in shortlists[qid]], relevant[qid]) for qid in test_ids}

print(f"{'model':40}{'ctx_limit':>10}{'truncated%':>12}{'demotions':>11}")
for m in MODELS:
    model = CrossEncoder(m)
    limit = model.model.config.max_position_embeddings
    over = sum(len(model.tokenizer.encode(qtext[qid], t, truncation=False)) > limit
               for qid in test_ids for _, t in shortlists[qid])
    total = sum(len(shortlists[qid]) for qid in test_ids)
    demoted = 0
    for qid in test_ids:
        sl = shortlists[qid]
        s = model.predict([[qtext[qid], t] for _, t in sl])
        ids = [sl[i][0] for i in sorted(range(len(sl)), key=lambda i: s[i], reverse=True)]
        demoted += rank_of_rel(ids, relevant[qid]) > hyb_rank[qid]
    print(f"{m:40}{limit:>10}{over / total:>11.0%}{demoted:>8}/{N}")
    del model
