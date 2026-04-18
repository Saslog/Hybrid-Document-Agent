"""
retrieval/vector_store.py
=========================
Dense Vector Retrieval using simulated or real sentence embeddings.

In simulation mode: uses a deterministic hash-based pseudo-embedding
that preserves some semantic properties for demo purposes.

In production mode: uses sentence-transformers (all-MiniLM-L6-v2)
for real 384-dim dense embeddings with cosine similarity ANN search.

Architecture:
    ┌──────────────────┐
    │  Query Text      │
    └────────┬─────────┘
             │ encode()
    ┌────────▼─────────┐
    │  Query Vector    │  384-dim float32
    └────────┬─────────┘
             │ cosine_similarity(query_vec, chunk_vecs)
    ┌────────▼─────────┐
    │  Ranked Chunks   │  top-k by similarity score
    └──────────────────┘
"""

from __future__ import annotations
import math
import hashlib
import re
from typing import List, Tuple, Dict, Optional

from core.models import DocumentChunk
from core.config import RetrievalConfig


class VectorStore:
    """
    Dense vector store for DocumentChunks.

    Supports:
      - Simulated embeddings (no external dependencies)
      - Real embeddings via sentence-transformers (if installed)
    """

    def __init__(self, config: RetrievalConfig, use_real_embeddings: bool = False):
        self.cfg = config
        self.use_real = use_real_embeddings
        self._chunks: List[DocumentChunk] = []
        self._embeddings: List[List[float]] = []
        self._encoder = None

        if use_real_embeddings:
            self._load_encoder()

    def _load_encoder(self):
        """Lazy-load sentence-transformers encoder."""
        try:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.cfg.embedding_model)
        except ImportError:
            print("Warning: sentence-transformers not installed. Using simulated embeddings.")
            self.use_real = False

    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Encode and store chunks."""
        for chunk in chunks:
            embedding = self._encode(chunk.content)
            chunk.embedding = embedding
            self._chunks.append(chunk)
            self._embeddings.append(embedding)

    def search(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """
        Search for top-k chunks by cosine similarity.

        Returns list of (chunk_index, cosine_similarity_score).
        """
        if not self._chunks:
            return []

        query_vec = self._encode(query)
        scores: List[Tuple[int, float]] = []

        for idx, chunk_vec in enumerate(self._embeddings):
            sim = self._cosine_similarity(query_vec, chunk_vec)
            scores.append((idx, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def get_chunk(self, idx: int) -> DocumentChunk:
        return self._chunks[idx]

    def size(self) -> int:
        return len(self._chunks)

    # ─────────────────────────────────────────
    # Embedding functions
    # ─────────────────────────────────────────

    def _encode(self, text: str) -> List[float]:
        """Encode text to a vector."""
        if self.use_real and self._encoder:
            return self._encoder.encode(text).tolist()
        return self._simulated_embedding(text)

    def _simulated_embedding(self, text: str, dim: int = 384) -> List[float]:
        """
        Deterministic pseudo-embedding for simulation.

        Strategy: Hash overlapping n-grams and project onto unit sphere.
        This preserves partial semantic similarity for demo purposes
        (e.g., "refund policy" and "return policy" share n-grams
        and will have higher similarity than random texts).
        """
        tokens = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
        vec = [0.0] * dim

        # Unigram contributions
        for token in tokens:
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            for d in range(min(8, dim)):
                component = ((h >> (d * 8)) & 0xFF) / 255.0 - 0.5
                vec[h % dim] += component * 0.5
                vec[(h // 2) % dim] += component * 0.3

        # Bigram contributions (captures phrase semantics)
        for i in range(len(tokens) - 1):
            bigram = f"{tokens[i]}_{tokens[i+1]}"
            h = int(hashlib.md5(bigram.encode()).hexdigest(), 16)
            for d in range(min(4, dim)):
                component = ((h >> (d * 8)) & 0xFF) / 255.0 - 0.5
                vec[(h * 3) % dim] += component * 0.7

        # L2 normalize to unit sphere
        magnitude = math.sqrt(sum(x * x for x in vec))
        if magnitude > 0:
            vec = [x / magnitude for x in vec]

        return vec

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """
        Cosine similarity between two L2-normalized vectors.

        For unit vectors: cos_sim(a, b) = dot_product(a, b)

        Since both vectors are L2-normalized in _simulated_embedding,
        we just compute the dot product.
        """
        if len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        # If not pre-normalized, compute full formula:
        # norm_a = math.sqrt(sum(x*x for x in a))
        # norm_b = math.sqrt(sum(x*x for x in b))
        # return dot / (norm_a * norm_b + 1e-9)
        return max(-1.0, min(1.0, dot))
