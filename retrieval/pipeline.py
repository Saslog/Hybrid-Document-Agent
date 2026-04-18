"""
retrieval/pipeline.py
=====================
RAG Retrieval Pipeline Orchestrator.

Combines BM25 + Dense Vector retrieval via RRF, then reranks with
a cross-encoder to produce the final top-k context chunks.

Full pipeline:
    Query
      │
      ├─► BM25 Search ──────────────────────────────────┐
      │   (sparse, exact keyword match)                  │
      │                                                  ├─► RRF Merge ─► Cross-Encoder Rerank ─► Top-K Chunks
      └─► Vector Search ─────────────────────────────────┘
          (dense, semantic similarity)
"""

from __future__ import annotations
import time
import re
from typing import List, Optional

from core.models import (
    Query, Document, DocumentChunk, RetrievedChunk, RetrievalResult
)
from core.config import RetrievalConfig
from retrieval.bm25 import BM25Index
from retrieval.vector_store import VectorStore
from retrieval.fusion import reciprocal_rank_fusion, CrossEncoderReranker


class DocumentChunker:
    """Splits documents into overlapping chunks for retrieval."""

    def __init__(self, chunk_size: int = 300, overlap: int = 50):
        self.chunk_size = chunk_size  # words
        self.overlap = overlap        # words

    def chunk_document(self, doc: Document) -> List[DocumentChunk]:
        """
        Split document content into overlapping word-based chunks.

        Overlap ensures sentences near chunk boundaries are covered
        by at least one complete chunk.
        """
        words = doc.content.split()
        chunks: List[DocumentChunk] = []
        idx = 0
        chunk_index = 0

        while idx < len(words):
            chunk_words = words[idx: idx + self.chunk_size]
            chunk_text = " ".join(chunk_words).strip()

            if len(chunk_text) > 20:  # skip near-empty trailing chunks
                chunk = DocumentChunk.from_document(doc, chunk_text, chunk_index)
                chunks.append(chunk)
                chunk_index += 1

            idx += self.chunk_size - self.overlap

        doc.chunks = chunks
        return chunks


class RetrievalPipeline:
    """
    Full RAG retrieval pipeline.

    Builds indexes over a document corpus and answers retrieval
    queries using hybrid BM25 + dense vector search with RRF fusion
    and cross-encoder reranking.
    """

    def __init__(self, config: RetrievalConfig):
        self.cfg = config
        self.chunker  = DocumentChunker(config.chunk_size, config.chunk_overlap)
        self.bm25     = BM25Index(k1=config.bm25_k1, b=config.bm25_b)
        self.vectors  = VectorStore(config, use_real_embeddings=False)
        self.reranker = CrossEncoderReranker(use_real_model=False)
        self._all_chunks: List[DocumentChunk] = []
        self._indexed = False

    def index(self, documents: List[Document]) -> None:
        """
        Index all documents: chunk → encode → build BM25 + vector indexes.
        """
        all_chunks: List[DocumentChunk] = []

        for doc in documents:
            chunks = self.chunker.chunk_document(doc)
            all_chunks.extend(chunks)

        self._all_chunks = all_chunks

        # Build BM25 index (inverted index over all chunks)
        self.bm25.build(all_chunks)

        # Build vector store (encode all chunks)
        self.vectors.add_chunks(all_chunks)

        self._indexed = True

    def retrieve(self, query: Query, top_k: Optional[int] = None) -> RetrievalResult:
        """
        Execute the full retrieval pipeline for a query.

        Returns a RetrievalResult with the top-k most relevant chunks.
        """
        if not self._indexed:
            raise RuntimeError("Pipeline must be indexed before retrieving.")

        top_k = top_k or self.cfg.reranker_top_k
        start = time.time()

        # ── Stage 1: BM25 sparse retrieval ────────────────────────────
        bm25_results = self.bm25.search(
            query.text,
            top_k=self.cfg.bm25_top_k
        )
        # bm25_results: [(chunk_idx, bm25_score), ...]

        # ── Stage 2: Dense vector retrieval ───────────────────────────
        vector_results = self.vectors.search(
            query.text,
            top_k=self.cfg.vector_top_k
        )
        # vector_results: [(chunk_idx, cosine_score), ...]

        # ── Stage 3: RRF Fusion ────────────────────────────────────────
        # Merge both ranked lists into a single unified ranking
        rrf_results = reciprocal_rank_fusion(
            bm25_results,
            vector_results,
            k=self.cfg.rrf_k,
        )
        # rrf_results: [(chunk_idx, rrf_score), ...]

        # ── Stage 4: Build RetrievedChunk objects ──────────────────────
        # Map scores back to chunk objects
        bm25_score_map  = dict(bm25_results)
        vector_score_map = dict(vector_results)
        rrf_score_map   = dict(rrf_results)

        # Take top-20 from RRF for reranking
        rrf_top = rrf_results[:min(20, len(rrf_results))]

        retrieved_chunks: List[RetrievedChunk] = []
        for chunk_idx, rrf_score in rrf_top:
            chunk = self._all_chunks[chunk_idx]
            rc = RetrievedChunk(
                chunk=chunk,
                bm25_score=bm25_score_map.get(chunk_idx, 0.0),
                vector_score=vector_score_map.get(chunk_idx, 0.0),
                rrf_score=rrf_score,
                rerank_score=0.0,
                final_score=rrf_score,  # will be overwritten by reranker
            )
            retrieved_chunks.append(rc)

        # ── Stage 5: Cross-Encoder Reranking ──────────────────────────
        if self.cfg.use_reranker and retrieved_chunks:
            final_chunks = self.reranker.rerank(
                query=query.text,
                candidates=retrieved_chunks,
                top_k=top_k,
            )
        else:
            # No reranker: use RRF score as final
            for rc in retrieved_chunks:
                rc.final_score = rc.rrf_score
            final_chunks = sorted(
                retrieved_chunks,
                key=lambda x: x.final_score,
                reverse=True
            )[:top_k]

        latency_ms = (time.time() - start) * 1000

        return RetrievalResult(
            query_id=query.query_id,
            chunks=final_chunks,
            retrieval_latency_ms=round(latency_ms, 2),
            bm25_candidates=len(bm25_results),
            vector_candidates=len(vector_results),
            after_rrf=len(rrf_top),
            after_rerank=len(final_chunks),
        )

    def corpus_size(self) -> int:
        return len(self._all_chunks)
