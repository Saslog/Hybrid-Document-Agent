"""
cache/semantic_cache.py
=======================
Semantic Cache Layer using cosine similarity.

Instead of exact-match caching (which misses paraphrases), this cache
stores query embeddings and retrieves semantically similar past answers.

Example:
    "What is the refund policy?" and
    "Tell me about returns and refunds"
    → will both hit the same cache entry if cosine_similarity > 0.92

Cache backends:
    - InMemoryCache: Dict-based, fast, no persistence (default)
    - RedisCache:    Redis-backed with TTL, survives restarts

TTL Strategy:
    - Static documents (policies, contracts): 24h–7d TTL
    - Dynamic data (financial reports): 1h–6h TTL
    - Real-time data: no caching
"""

from __future__ import annotations
import time
import hashlib
import math
import re
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass, field


@dataclass
class CacheEntry:
    """A single cached query-answer pair."""
    key: str
    query_text: str
    query_embedding: List[float]
    response: str
    route: str
    model_used: str
    cost_saved: float
    timestamp: float = field(default_factory=time.time)
    ttl_seconds: int = 3600
    hit_count: int = 0

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.timestamp) > self.ttl_seconds

    def touch(self):
        """Record a cache hit."""
        self.hit_count += 1


class InMemorySemanticCache:
    """
    In-memory semantic cache using cosine similarity for lookup.

    Lookup complexity: O(n) where n = number of cached entries.
    For larger deployments, replace with FAISS ANN index for O(log n).
    """

    def __init__(
        self,
        similarity_threshold: float = 0.92,
        max_size: int = 1000,
        default_ttl: int = 3600,
    ):
        self.similarity_threshold = similarity_threshold
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._store: Dict[str, CacheEntry] = {}
        self._hit_count = 0
        self._miss_count = 0

    def lookup(self, query_text: str) -> Optional[CacheEntry]:
        """
        Look up a semantically similar cached query.

        Algorithm:
        1. Encode the query to a vector
        2. Compare against all cached query vectors using cosine similarity
        3. Return the best match if similarity > threshold
        4. Evict expired entries during lookup (lazy eviction)
        """
        query_vec = self._encode(query_text)
        best_entry: Optional[CacheEntry] = None
        best_similarity = 0.0

        expired_keys = []
        for key, entry in self._store.items():
            if entry.is_expired:
                expired_keys.append(key)
                continue

            sim = self._cosine_similarity(query_vec, entry.query_embedding)
            if sim > best_similarity:
                best_similarity = sim
                best_entry = entry

        # Lazy eviction
        for key in expired_keys:
            del self._store[key]

        if best_entry and best_similarity >= self.similarity_threshold:
            best_entry.touch()
            self._hit_count += 1
            return best_entry

        self._miss_count += 1
        return None

    def store(
        self,
        query_text: str,
        response: str,
        route: str,
        model_used: str,
        cost_saved: float,
        ttl_seconds: Optional[int] = None,
    ) -> str:
        """Store a query-response pair in the cache."""
        # Evict oldest entries if at capacity
        if len(self._store) >= self.max_size:
            self._evict_lru()

        query_vec = self._encode(query_text)
        key = self._make_key(query_text)
        ttl = ttl_seconds or self.default_ttl

        entry = CacheEntry(
            key=key,
            query_text=query_text,
            query_embedding=query_vec,
            response=response,
            route=route,
            model_used=model_used,
            cost_saved=cost_saved,
            ttl_seconds=ttl,
        )
        self._store[key] = entry
        return key

    def invalidate(self, key: str) -> bool:
        """Remove a specific cache entry."""
        if key in self._store:
            del self._store[key]
            return True
        return False

    def clear(self) -> None:
        """Clear all cache entries."""
        self._store.clear()
        self._hit_count = 0
        self._miss_count = 0

    def stats(self) -> Dict[str, Any]:
        total = self._hit_count + self._miss_count
        return {
            "size": len(self._store),
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "hit_rate": round(self._hit_count / total, 3) if total > 0 else 0.0,
            "max_size": self.max_size,
        }

    # ─────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────

    def _encode(self, text: str, dim: int = 128) -> List[float]:
        """Lightweight simulated embedding (same logic as VectorStore)."""
        tokens = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
        vec = [0.0] * dim

        for token in tokens:
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            for d in range(min(8, dim)):
                component = ((h >> (d * 8)) & 0xFF) / 255.0 - 0.5
                vec[h % dim] += component * 0.5

        for i in range(len(tokens) - 1):
            bigram = f"{tokens[i]}_{tokens[i+1]}"
            h = int(hashlib.md5(bigram.encode()).hexdigest(), 16)
            vec[(h * 3) % dim] += 0.4

        magnitude = math.sqrt(sum(x * x for x in vec))
        if magnitude > 0:
            vec = [x / magnitude for x in vec]
        return vec

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        if len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        return max(-1.0, min(1.0, dot))

    @staticmethod
    def _make_key(text: str) -> str:
        return hashlib.sha256(text.strip().lower().encode()).hexdigest()[:16]

    def _evict_lru(self) -> None:
        """Evict the oldest 10% of entries."""
        sorted_entries = sorted(
            self._store.items(),
            key=lambda x: x[1].timestamp
        )
        evict_count = max(1, len(self._store) // 10)
        for key, _ in sorted_entries[:evict_count]:
            del self._store[key]
