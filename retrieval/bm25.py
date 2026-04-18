"""
retrieval/bm25.py
=================
BM25 (Best Match 25) Implementation from scratch.

BM25 is a probabilistic ranking function used in sparse retrieval.
It improves on TF-IDF by:
  1. Applying a saturation function to term frequency (so repeating a word
     many times doesn't linearly inflate the score)
  2. Normalizing by document length

Formula:
    score(D, Q) = Σ IDF(qi) × (f(qi,D) × (k1+1)) / (f(qi,D) + k1×(1-b+b×|D|/avgdl))

Where:
    f(qi, D)  = term frequency of query term qi in document D
    |D|       = length of document D (in tokens)
    avgdl     = average document length in corpus
    k1        = term frequency saturation parameter (typically 1.2–2.0)
    b         = length normalization parameter (0–1, typically 0.75)
    IDF(qi)   = log((N - n(qi) + 0.5) / (n(qi) + 0.5) + 1)
    N         = total number of documents
    n(qi)     = number of documents containing qi
"""

from __future__ import annotations
import math
import re
from collections import defaultdict, Counter
from typing import List, Tuple, Dict

from core.models import DocumentChunk


class BM25Index:
    """
    BM25 index over a collection of DocumentChunks.

    Usage:
        index = BM25Index(k1=1.5, b=0.75)
        index.build(chunks)
        results = index.search(query_text, top_k=20)
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b  = b

        self._chunks: List[DocumentChunk] = []
        self._doc_lengths: List[int] = []
        self._avgdl: float = 0.0
        self._N: int = 0

        # Inverted index: term → {doc_idx: term_freq}
        self._inverted: Dict[str, Dict[int, int]] = defaultdict(dict)

        # Document frequency: term → number of docs containing term
        self._df: Dict[str, int] = defaultdict(int)

        self._built = False

    def build(self, chunks: List[DocumentChunk]) -> None:
        """Build the BM25 index from a list of DocumentChunks."""
        self._chunks = chunks
        self._N = len(chunks)

        total_tokens = 0
        for idx, chunk in enumerate(chunks):
            tokens = self._tokenize(chunk.content)
            chunk.bm25_tokens = tokens
            self._doc_lengths.append(len(tokens))
            total_tokens += len(tokens)

            # Count term frequencies
            tf = Counter(tokens)
            for term, freq in tf.items():
                self._inverted[term][idx] = freq
                self._df[term] += 1

        self._avgdl = total_tokens / self._N if self._N > 0 else 1.0
        self._built = True

    def search(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """
        Search the index.

        Returns list of (chunk_index, bm25_score) sorted by score descending.
        """
        if not self._built:
            raise RuntimeError("BM25Index must be built before searching.")

        query_terms = self._tokenize(query)
        scores: Dict[int, float] = defaultdict(float)

        for term in set(query_terms):  # deduplicate query terms
            if term not in self._inverted:
                continue

            idf = self._compute_idf(term)
            postings = self._inverted[term]

            for doc_idx, tf in postings.items():
                dl   = self._doc_lengths[doc_idx]
                norm = self.k1 * (1 - self.b + self.b * (dl / self._avgdl))
                tf_norm = (tf * (self.k1 + 1)) / (tf + norm)
                scores[doc_idx] += idf * tf_norm

        # Sort by score descending
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def get_chunk(self, idx: int) -> DocumentChunk:
        return self._chunks[idx]

    # ─────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────

    def _compute_idf(self, term: str) -> float:
        """
        Robertson-Sparck Jones IDF with smoothing to avoid negative values:
            IDF = log((N - n(q) + 0.5) / (n(q) + 0.5) + 1)
        """
        n_q = self._df.get(term, 0)
        return math.log((self._N - n_q + 0.5) / (n_q + 0.5) + 1)

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text: lowercase, remove stopwords, keep alphanumeric.
        Minimal stopword list for efficiency.
        """
        STOPWORDS = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
            "for", "of", "with", "by", "from", "is", "are", "was", "were",
            "be", "been", "being", "have", "has", "had", "do", "does", "did",
            "will", "would", "could", "should", "may", "might", "shall",
            "this", "that", "these", "those", "it", "its", "as", "not",
            "all", "any", "both", "each", "few", "more", "most", "other",
            "some", "such", "than", "then", "so", "if", "can",
        }
        tokens = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
        return [t for t in tokens if t not in STOPWORDS and len(t) > 1]
