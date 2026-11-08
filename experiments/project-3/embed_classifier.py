import json
import numpy as np
import httpx
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# --- block 1: load the 800 labeled rows ---
rows = [json.loads(l) for l in open("data/testset.jsonl")]
texts = [r["text"] for r in rows]
labels = [r["label"] for r in rows]


# --- block 2: embed every text = the FEATURES. note the 'classification:' prefix ---
# nomic has a task prefix made for this (not search_query) — it tunes the vector for classifying
def embed(batch: list[str]) -> list:
    r = httpx.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "nomic-embed-text",
            "input": [f"classification: {t}" for t in batch],
        },
        timeout=600,
    )
    r.raise_for_status()
    return r.json()["embeddings"]


X = []
for i in range(0, len(texts), 128):
    X.extend(embed(texts[i : i + 128]))
X = np.array(X)  # shape (800, 768) — 800 rows, 768 features each
y = np.array(labels)  # shape (800,)     — the 4-label answers

# --- block 3: split into train vs held-out test (stratify keeps all 4 labels balanced in both) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# --- block 4: train the classifier (learns the 768 weights per label) ---
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# --- block 5: predict on the UNSEEN test rows and score ---
pred = clf.predict(X_test)
print("accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))
