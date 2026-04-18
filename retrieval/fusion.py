"""
retrieval/fusion.py
===================
Reciprocal Rank Fusion (RRF) and Cross-Encoder Reranking.

──────────────────────────────────────────────────────────────────
RECIPROCAL RANK FUSION (RRF)
──────────────────────────────────────────────────────────────────
RRF merges multiple ranked lists into a single unified ranking
without requiring score normalization.

Formula (Cormack et al., 2009):
    RRFScore(d) = Σ_r 1 / (k + rank_r(d))

Where:
    k         = smoothing constant (default 60) — prevents top-ranked
                docs from dominating; reduces sensitivity to outliers
    rank_r(d) = position of document d in ranking r (1-indexed)

Why RRF over score fusion?
    - BM25 scores (unbounded positive floats) and cosine similarity
      scores ([-1, 1]) live on completely different scales.
    - Normalizing them is tricky and dataset-dependent.
    - RRF uses only rank positions, making it scale-invariant.
    - Empirically matches or beats weighted score fusion in most benchmarks.

──────────────────────────────────────────────────────────────────
CROSS-ENCODER RERANKING
──────────────────────────────────────────────────────────────────
After RRF narrows candidates to top-20, a cross-encoder reranker
scores each (query, passage) pair jointly using full attention.

Difference from bi-encoder:
    Bi-encoder: encode(query) + encode(passage) → separate vectors → dot product
    Cross-encoder: encode(query + passage) → single relevance score

Cross-encoders are 10–50× more accurate but too slow for full corpus
(O(n) inference passes). We use them only on top-20 RRF candidates.
"""

from __future__ import annotations
import math
import re
import hashlib
from typing import List, Tuple, Dict

from core.models import DocumentChunk, RetrievedChunk


def reciprocal_rank_fusion(
    bm25_results: List[Tuple[int, float]],
    vector_results: List[Tuple[int, float]],
    k: int = 60,
) -> List[Tuple[int, float]]:
    """
    Merge BM25 and vector ranked lists using Reciprocal Rank Fusion.

    Args:
        bm25_results:   List of (chunk_idx, bm25_score) sorted by score desc
        vector_results: List of (chunk_idx, cosine_score) sorted by score desc
        k:              RRF smoothing constant (default 60)

    Returns:
        List of (chunk_idx, rrf_score) sorted by rrf_score desc
    """
    rrf_scores: Dict[int, float] = {}

    # BM25 ranked list
    for rank, (idx, _score) in enumerate(bm25_results, start=1):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k + rank)

    # Vector ranked list
    for rank, (idx, _score) in enumerate(vector_results, start=1):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k + rank)

    # Sort by combined RRF score
    merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return merged


class CrossEncoderReranker:
    """
    Cross-encoder reranker for final chunk selection.

    In simulation: uses a deterministic overlap-based scoring function
    that approximates cross-encoder behavior.

    In production: uses sentence-transformers cross-encoder
    (cross-encoder/ms-marco-MiniLM-L-6-v2).
    """

    def __init__(self, use_real_model: bool = False, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.use_real = use_real_model
        self.model_name = model_name
        self._model = None

        if use_real_model:
            self._load_model()

    def _load_model(self):
        try:
            from sentence_transformers.cross_encoder import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        except ImportError:
            print("Warning: sentence-transformers not installed. Using simulated reranker.")
            self.use_real = False

    def rerank(
        self,
        query: str,
        candidates: List[RetrievedChunk],
        top_k: int = 5,
    ) -> List[RetrievedChunk]:
        """
        Score each (query, chunk) pair and return top_k by rerank score.
        """
        if not candidates:
            return []

        if self.use_real and self._model:
            pairs = [(query, rc.chunk.content) for rc in candidates]
            scores = self._model.predict(pairs)
            for rc, score in zip(candidates, scores):
                rc.rerank_score = float(score)
        else:
            # Simulated cross-encoder: term overlap + position bias
            for rc in candidates:
                rc.rerank_score = self._simulated_rerank_score(query, rc.chunk.content)

        # Set final score = rerank score (overrides RRF)
        for rc in candidates:
            rc.final_score = rc.rerank_score

        reranked = sorted(candidates, key=lambda x: x.final_score, reverse=True)
        return reranked[:top_k]

    def _simulated_rerank_score(self, query: str, passage: str) -> float:
        """
        Simulated cross-encoder score based on:
        1. Token overlap (Jaccard-like)
        2. Bigram overlap (phrase matching)
        3. Length normalization penalty
        4. Position of first match (earlier = better)

        This is NOT a real cross-encoder but approximates ranking quality
        well enough for demo/testing purposes.
        """
        q_tokens = set(re.findall(r'\b[a-zA-Z0-9]+\b', query.lower()))
        p_tokens = re.findall(r'\b[a-zA-Z0-9]+\b', passage.lower())
        p_set = set(p_tokens)

        # Token overlap
        overlap = len(q_tokens & p_set)
        token_score = overlap / (len(q_tokens) + 1)

        # Bigram overlap
        q_bigrams = set()
        q_list = list(q_tokens)
        for i in range(len(q_list) - 1):
            q_bigrams.add(f"{q_list[i]}_{q_list[i+1]}")

        p_bigrams = set()
        for i in range(len(p_tokens) - 1):
            p_bigrams.add(f"{p_tokens[i]}_{p_tokens[i+1]}")

        bigram_overlap = len(q_bigrams & p_bigrams)
        bigram_score = bigram_overlap / (len(q_bigrams) + 1)

        # Length penalty (very long passages dilute signal)
        length_penalty = min(1.0, 200 / max(len(p_tokens), 1))

        # Position score: reward passages where query terms appear early
        first_match_pos = len(p_tokens)
        for term in q_tokens:
            for i, t in enumerate(p_tokens):
                if t == term:
                    first_match_pos = min(first_match_pos, i)
                    break
        position_score = 1.0 - (first_match_pos / max(len(p_tokens), 1))

        # Weighted combination
        score = (
            token_score  * 0.40 +
            bigram_score * 0.30 +
            position_score * 0.20 +
            length_penalty * 0.10
        )

        return round(score, 4)
