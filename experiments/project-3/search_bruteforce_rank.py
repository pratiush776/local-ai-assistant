"""bruteforce vector search over 100 sentences cropus using nomic-embed-text and numpy"""

import httpx
import numpy as np

URL = "http://localhost:11434/api/embed"
MODEL = "nomic-embed-text"

# embed list of txt into maxtrix of vectors
def embed(texts: list[str]) -> np.ndarray:
    r = httpx.post(URL, json={"model":MODEL, "input":texts}, timeout=300)
    r.raise_for_status()
    data = r.json()
    return np.array(data["embeddings"])

sentences = open("data/hundred_sentences.txt").read().splitlines()
D = embed([f"search_document: {sentence}" for sentence in sentences])

query = "animals that are not pets"
q = embed([f"search_query: {query}"])[0]

scores = D @ q

top = np.argsort(scores)[::-1][:5]
for i in top:
    print(f'{scores[i]:.3f} {sentences[i]}')



