import math
import re
from collections import Counter

from src.core.models import Chunk, RetrievalResult


def _tokenize(text: str) -> list[str]:
    return re.findall(r'\b[a-z]{2,}\b', text.lower())


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0.0


class VectorIndex:
    """In-memory TF-IDF index with cosine similarity search."""

    def __init__(self):
        self._chunks: list[Chunk] = []
        self._vocab: list[str] = []
        self._doc_freq: Counter = Counter()

    def add(self, chunks: list[Chunk]) -> None:
        self._chunks.extend(chunks)
        self._rebuild()

    def _rebuild(self) -> None:
        corpus = [_tokenize(c.content) for c in self._chunks]

        self._doc_freq = Counter()
        for tokens in corpus:
            self._doc_freq.update(set(tokens))
        self._vocab = sorted(self._doc_freq.keys())

        n = len(self._chunks)
        for chunk, tokens in zip(self._chunks, corpus):
            tf = Counter(tokens)
            total = sum(tf.values()) or 1
            chunk.embedding = [
                (tf[w] / total) * math.log((n + 1) / (self._doc_freq[w] + 1))
                if tf[w] > 0 else 0.0
                for w in self._vocab
            ]

    def _embed(self, text: str) -> list[float]:
        tokens = _tokenize(text)
        tf = Counter(tokens)
        total = sum(tf.values()) or 1
        n = len(self._chunks)
        return [
            (tf.get(w, 0) / total) * math.log((n + 1) / (self._doc_freq[w] + 1))
            for w in self._vocab
        ]

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if not self._chunks:
            return []
        query_vec = self._embed(query)
        scored = [
            RetrievalResult(chunk=c, score=_cosine(query_vec, c.embedding))
            for c in self._chunks
            if c.embedding
        ]
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]

    def __len__(self) -> int:
        return len(self._chunks)
