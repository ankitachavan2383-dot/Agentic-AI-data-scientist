"""Lightweight RAG over the curated statistics/ML guidance in /knowledge.

TF-IDF keeps the system dependency-free and deterministic. To scale up, swap `retrieve`
for pgvector + embeddings; the interface (list of {id, source, title, text}) stays the same.
"""
from functools import lru_cache
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KB_DIR = Path(__file__).resolve().parents[2] / "knowledge"


@lru_cache(maxsize=1)
def _index():
    chunks = []
    for f in sorted(KB_DIR.glob("*.md")):
        for sec in f.read_text().split("\n## ")[1:]:
            title, _, body = sec.partition("\n")
            chunks.append({"source": f.name, "title": title.strip(), "text": body.strip()})
    if not chunks:
        return chunks, None, None
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    mat = vec.fit_transform([c["title"] + " " + c["text"] for c in chunks])
    return chunks, vec, mat


def retrieve(query: str, k: int = 3, min_score: float = 0.05) -> list[dict]:
    chunks, vec, mat = _index()
    if not chunks:
        return []
    scores = cosine_similarity(vec.transform([query]), mat)[0]
    top = scores.argsort()[::-1][:k]
    return [{"id": f"K{i + 1}", **chunks[j], "score": round(float(scores[j]), 3)} for i, j in enumerate(top) if scores[j] >= min_score]
