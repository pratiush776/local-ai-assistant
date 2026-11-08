# ollama embeddings call
import httpx

URL = "http://localhost:11434/api/embed"

payload = {
    "model": "nomic-embed-text",
    "input": [
        "search_document: The cat sat on the mat.",
        "search_document: A dog barked in the yard.",
    ],
}

r = httpx.post(URL, json=payload, timeout=300)
r.raise_for_status()
data = r.json()

print(data)
vecs = data["embeddings"]
print("number of vectors:", len(vecs))
print("length of first vector:", len(vecs[0]))
print("first 5 numbers: ", vecs[0][:5])
