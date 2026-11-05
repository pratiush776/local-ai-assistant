import httpx, numpy as np, psycopg
from collections import defaultdict
from pgvector.psycopg import register_vector
from datasets import load_dataset

DSN = "postgresql://postgres:rag@localhost:5432/postgres"
URL = "http://localhost:11434/api/embed"
K = 10

# --- load the answer key (qrels) and the query texts ---
qrels   = load_dataset("BeIR/scifact-qrels")["test"]
queries = load_dataset("BeIR/scifact", "queries")["queries"]

relevant = defaultdict(set)                     # query_id -> {relevant doc_ids}
for r in qrels:
    relevant[str(r["query-id"])].add(str(r["corpus-id"]))   # str() both — the type trap
qtext   = {str(r["_id"]): r["text"] for r in queries}       # query_id -> claim text
test_ids = list(relevant)                       # the 300 queries that actually have labels

# --- embed all 300 query texts once, in batches of 128 ---
def embed(texts: list[str]) -> list:
    r = httpx.post(URL, json={"model": "nomic-embed-text", "input": texts}, timeout=600)
    r.raise_for_status()
    return [np.array(v) for v in r.json()["embeddings"]]

qvecs = {}
for i in range(0, len(test_ids), 128):
    chunk = test_ids[i:i + 128]
    for qid, v in zip(chunk, embed([f"search_query: {qtext[q]}" for q in chunk])):
        qvecs[qid] = v

# --- the three lanes, each returns a list of top-K doc_ids for a query id ---
def vector_lane(conn, qid):
    return [row[0] for row in conn.execute(
        "SELECT doc_id FROM scifact ORDER BY embedding <=> %s LIMIT %s", (qvecs[qid], K))]

def keyword_lane(conn, qid):
    return [row[0] for row in conn.execute(
        "SELECT doc_id FROM scifact WHERE doc_id @@@ paradedb.match('text', %s) "
        "ORDER BY paradedb.score(doc_id) DESC LIMIT %s",
        (qtext[qid], K))]

def hybrid_lane(conn, qid):
    sql = """
    WITH vector AS (
      SELECT doc_id, ROW_NUMBER() OVER (ORDER BY embedding <=> %(q)s) AS rank
      FROM scifact ORDER BY embedding <=> %(q)s LIMIT 20),
    keyword AS (
      SELECT doc_id, ROW_NUMBER() OVER (
        ORDER BY paradedb.score(doc_id) DESC) AS rank
      FROM scifact WHERE doc_id @@@ paradedb.match('text', %(t)s) LIMIT 20)
    SELECT COALESCE(v.doc_id, k.doc_id) AS doc_id,
           COALESCE(1.0/(60+v.rank),0) + COALESCE(1.0/(60+k.rank),0) AS rrf
    FROM vector v FULL OUTER JOIN keyword k USING (doc_id)
    ORDER BY rrf DESC LIMIT %(k)s"""
    return [row[0] for row in conn.execute(sql, {"q": qvecs[qid], "t": qtext[qid], "k": K})]

def recall_at_k(retrieved: list[str], relevant_ids: set[str]) -> float:
    n = len(relevant_ids.intersection(retrieved))
    return n/len(relevant_ids)

def main() -> None:
    lanes = {"vector": vector_lane, "keyword": keyword_lane, "hybrid": hybrid_lane}
    lane_total_recall = defaultdict(float)
    with psycopg.connect(DSN) as conn:
        register_vector(conn)
        for qid in test_ids:
            for lane in lanes:
                lane_recall = recall_at_k(lanes[lane](conn,qid),relevant[qid])
                lane_total_recall[lane]+=lane_recall
    lane_avg_recall = {lane:total_recall/len(test_ids) for lane,total_recall in lane_total_recall.items()}
    print("Lane\tAvg Recall")
    for lane,avg_recall in lane_avg_recall.items():
        print(f"{lane}\t{avg_recall}")
if __name__ == "__main__":
    main()


