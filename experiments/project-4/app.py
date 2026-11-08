from contextlib import asynccontextmanager
import time
import httpx, numpy as np, psycopg
from fastapi import FastAPI
from pydantic import BaseModel
from pgvector.psycopg import register_vector
from sentence_transformers import CrossEncoder

DSN = "postgresql://postgres:rag@localhost:5432/postgres"
THRESHOLD = 0.3
STATE = {}

HYBRID_SQL = """
WITH vec AS (
  SELECT chunk_id, ROW_NUMBER() OVER (ORDER BY embedding <=> %(q)s) rank
  FROM chunks ORDER BY embedding <=> %(q)s LIMIT 10),
kw AS (
  SELECT chunk_id, ROW_NUMBER() OVER (ORDER BY paradedb.score(chunk_id) DESC) rank
  FROM chunks WHERE chunk_id @@@ paradedb.match('text', %(t)s) LIMIT 10)
SELECT c.chunk_id, c.text,
       COALESCE(1.0/(60+vec.rank),0) + COALESCE(1.0/(60+kw.rank),0) AS rrf
FROM vec FULL OUTER JOIN kw USING (chunk_id)
JOIN chunks c ON c.chunk_id = COALESCE(vec.chunk_id, kw.chunk_id)
ORDER BY rrf DESC LIMIT 10;
"""


# ----pipeline
def embed_query(q: str):
    r = httpx.post(
        "http://localhost:11434/api/embed",
        json={"model": "nomic-embed-text", "input": [f"search_query: {q}"]},
        timeout=300,
    )
    r.raise_for_status()
    d = r.json()
    return np.array(d["embeddings"][0]), d  # vector + raw response (has load_duration)


def generate(question: str, kept: list[tuple[str, str]]) -> str:
    context = "\n\n".join(
        f"[{i + 1}] (source={cid}) {text}" for i, (cid, text) in enumerate(kept)
    )
    prompt = (
        "Answer the question using ONLY the context below. Cite the source number [n] "
        'for each fact. If the context does not contain the answer, reply exactly: "I don\'t know".\n\n'
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    r = httpx.post(
        "http://localhost:11434/api/chat",
        json={
            "model": "qwen3.5:9b",
            "messages": [{"role": "user", "content": prompt}],
            "think": False,
            "stream": False,
            "keep_alive": "30m",
        },
        timeout=120,
    )
    r.raise_for_status()
    d = r.json()
    return d["message"]["content"], d  # answer + raw response (has load_duration, eval_count)


# --- load model once at startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE["reranker"] = CrossEncoder("BAAI/bge-reranker-v2-m3")
    httpx.post(
        "http://localhost:11434/api/embed",
        json={"model": "nomic-embed-text", "input": ["warmup"]},
        timeout=120,
    )
    httpx.post(
        "http://localhost:11434/api/chat",
        json={
            "model": "qwen3.5:9b",
            "messages": [{"role": "user", "content": "hi"}],
            "think": False,
            "stream": False,
            "keep_alive": "30m",
        },
        timeout=120,
    )
    yield
    STATE.clear()


app = FastAPI(lifespan=lifespan)


# ---- request/response shapes with pydantic
class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str | None = None
    citations: list[str] = []
    not_found: bool = False
    timings: dict = {}


# ---endpoint
@app.post("/ask")
def ask(req: AskRequest) -> AskResponse:
    T = {}
    t0 = time.perf_counter()
    q_vec, emeta = embed_query(req.question)
    T["1_embed_ms"] = round((time.perf_counter() - t0) * 1000)

    t1 = time.perf_counter()
    with psycopg.connect(DSN) as conn:
        register_vector(conn)
        shortlist = [
            (cid, text)
            for cid, text, _ in conn.execute(
                HYBRID_SQL, {"q": q_vec, "t": req.question}
            ).fetchall()
        ]
    T["2_retrieve_ms"] = round((time.perf_counter() - t1) * 1000)  # both lanes, one SQL query

    t2 = time.perf_counter()
    scores = STATE["reranker"].predict([[req.question, t] for _, t in shortlist])
    T["3_rerank_ms"] = round((time.perf_counter() - t2) * 1000)

    kept = [(cid, t) for (cid, t), s in zip(shortlist, scores) if s >= THRESHOLD]
    if not kept:
        T["total_ms"] = round((time.perf_counter() - t0) * 1000)
        print("TIMINGS (not_found):", T)
        return AskResponse(not_found=True, timings=T)

    t3 = time.perf_counter()
    answer, gmeta = generate(req.question, kept)
    T["4_generate_ms"] = round((time.perf_counter() - t3) * 1000)

    # Ollama's own load_duration: ~0 = model was warm; big = it cold-loaded this request
    T["embed_model_load_ms"] = round(emeta.get("load_duration", 0) / 1e6)
    T["gen_model_load_ms"] = round(gmeta.get("load_duration", 0) / 1e6)
    T["gen_output_tokens"] = gmeta.get("eval_count")
    T["total_ms"] = round((time.perf_counter() - t0) * 1000)
    print("TIMINGS:", T)
    return AskResponse(
        answer=answer, citations=[cid for cid, _ in kept], timings=T
    )


# --- webpage endpoint
from fastapi.responses import HTMLResponse
from pathlib import Path


@app.get("/", response_class=HTMLResponse)
def home():
    # serve the page sitting next to app.py (works regardless of where uvicorn runs)
    return Path(__file__).parent.joinpath("index.html").read_text()
