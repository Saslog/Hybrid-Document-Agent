"""
core/agent.py
=============
HybridDocumentAgent — Top-level orchestrator.

Wires together all components:
    Classifier → Router → Cache → Retrieval → Inference → Response
"""

from __future__ import annotations
import asyncio
import logging
import time
from typing import List, Dict, Any, Optional

from core.models import (
    Query, Document, AgentResult, SessionStats, RouteDecision
)
from core.config import AgentConfig
from core.classifier import QueryClassifier
from core.router import QueryRouter
from retrieval.pipeline import RetrievalPipeline
from cache.semantic_cache import InMemorySemanticCache
from models.inference import EdgeModelAdapter, CloudModelAdapter, HybridAdapter

logger = logging.getLogger(__name__)


class HybridDocumentAgent:
    """
    Hybrid Edge-Cloud Document Agent.

    Full query processing pipeline:
    ┌─────────┐    ┌────────────┐    ┌────────┐    ┌───────────┐
    │  Query  │──► │ Classifier │──► │ Router │──► │   Cache   │
    └─────────┘    └────────────┘    └────────┘    └─────┬─────┘
                                                         │ miss
                                                    ┌────▼──────┐
                                                    │ Retrieval │
                                                    └─────┬─────┘
                                                          │
                                          ┌───────────────┼───────────────┐
                                          ▼               ▼               ▼
                                     Edge Model    Cloud Model      Hybrid
                                          └───────────────┼───────────────┘
                                                          │
                                                    ┌─────▼──────┐
                                                    │  Response  │
                                                    └────────────┘
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config    = config or AgentConfig()
        self.classifier = QueryClassifier(self.config.routing)
        self.router    = QueryRouter(self.config)
        self.retrieval  = RetrievalPipeline(self.config.retrieval)
        self.cache     = InMemorySemanticCache(
            similarity_threshold=self.config.cache.similarity_threshold,
            max_size=self.config.cache.max_size,
            default_ttl=self.config.cache.ttl_seconds,
        )
        self.edge_model  = EdgeModelAdapter(self.config)
        self.cloud_model = CloudModelAdapter(self.config)
        self.hybrid_model = HybridAdapter(self.config)

        self._stats = SessionStats()
        self._indexed = False

        logging.basicConfig(
            level=getattr(logging, self.config.log_level, logging.INFO),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )

    async def index_documents(self, documents: List[Document]) -> None:
        """Index documents into the retrieval pipeline."""
        logger.info(f"Indexing {len(documents)} documents...")
        await asyncio.get_event_loop().run_in_executor(
            None, self.retrieval.index, documents
        )
        self._indexed = True
        logger.info(
            f"Indexed {self.retrieval.corpus_size()} chunks "
            f"from {len(documents)} documents."
        )

    async def process(self, query: Query) -> AgentResult:
        """
        Process a single query through the full pipeline.

        Returns an AgentResult with the response + all metadata.
        """
        total_start = time.time()

        # ── 1. Classify ────────────────────────────────────────────────
        classification = self.classifier.classify(query)
        logger.debug(f"Classification: {classification}")

        # ── 2. Cache check ─────────────────────────────────────────────
        cache_entry = None
        cache_hit   = False
        if self.config.cache.enabled:
            cache_entry = self.cache.lookup(query.text)
            cache_hit   = cache_entry is not None

        # ── 3. Route ───────────────────────────────────────────────────
        routing = self.router.route(
            classification,
            cache_hit=cache_hit,
            cache_key=cache_entry.key if cache_entry else None,
        )
        if self.config.log_routing_decisions:
            logger.info(f"Query: '{query.text[:60]}...' → {routing}")

        # ── 4. Serve from cache if hit ────────────────────────────────
        if cache_hit and cache_entry:
            total_latency = (time.time() - total_start) * 1000
            result = AgentResult(
                query_id=query.query_id,
                session_id=query.session_id,
                query_text=query.text,
                response=cache_entry.response,
                route=RouteDecision.CACHE.value,
                model_used="SemanticCache",
                complexity_score=classification.complexity_score,
                latency_ms=round(total_latency, 1),
                cost=0.0,
                cloud_cost_equivalent=routing.estimated_cloud_cost,
                cost_savings=routing.estimated_cloud_cost,
                cache_hit=True,
                sources=[],
                routing_reasoning=routing.reasoning,
                signals=classification.signals,
            )
            self._update_stats(result, routing)
            return result

        # ── 5. Retrieve context ────────────────────────────────────────
        if not self._indexed:
            raise RuntimeError(
                "No documents indexed. Call index_documents() first."
            )

        retrieval_result = await asyncio.get_event_loop().run_in_executor(
            None, self.retrieval.retrieve, query
        )

        # ── 6. Inference ───────────────────────────────────────────────
        response_text, model_latency = await self._run_inference(
            routing.route, query.text, retrieval_result
        )

        # ── 7. Assemble result ─────────────────────────────────────────
        total_latency = (time.time() - total_start) * 1000

        sources = [
            {
                "doc_id":    rc.chunk.doc_id,
                "doc_title": rc.chunk.doc_title,
                "score":     round(rc.final_score, 4),
                "snippet":   rc.chunk.content[:100] + "...",
            }
            for rc in retrieval_result.top_chunks[:3]
        ]

        result = AgentResult(
            query_id=query.query_id,
            session_id=query.session_id,
            query_text=query.text,
            response=response_text,
            route=routing.route.value,
            model_used=routing.model_name,
            complexity_score=classification.complexity_score,
            latency_ms=round(total_latency, 1),
            cost=routing.estimated_cost,
            cloud_cost_equivalent=routing.estimated_cloud_cost,
            cost_savings=routing.savings,
            cache_hit=False,
            sources=sources,
            routing_reasoning=routing.reasoning,
            signals=classification.signals,
        )

        # ── 8. Cache the result ────────────────────────────────────────
        if self.config.cache.enabled:
            self.cache.store(
                query_text=query.text,
                response=response_text,
                route=routing.route.value,
                model_used=routing.model_name,
                cost_saved=routing.savings,
            )

        self._update_stats(result, routing)
        return result

    async def _run_inference(self, route, query_text, retrieval_result):
        """Dispatch to the correct model adapter."""
        if route == RouteDecision.EDGE:
            return await self.edge_model.generate(query_text, retrieval_result)
        elif route == RouteDecision.CLOUD:
            return await self.cloud_model.generate(query_text, retrieval_result)
        elif route == RouteDecision.HYBRID:
            return await self.hybrid_model.generate(query_text, retrieval_result)
        else:
            # Fallback
            return await self.edge_model.generate(query_text, retrieval_result)

    def _update_stats(self, result: AgentResult, routing) -> None:
        """Update session statistics."""
        self._stats.total_queries += 1
        self._stats.latencies.append(result.latency_ms)
        self._stats.total_cost_actual    += result.cost
        self._stats.total_cost_cloud_only += result.cloud_cost_equivalent

        route = result.route
        if route == "edge":
            self._stats.edge_count += 1
        elif route == "cloud":
            self._stats.cloud_count += 1
        elif route == "hybrid":
            self._stats.hybrid_count += 1
        elif route == "cache":
            self._stats.cache_hits += 1
            self._stats.edge_count += 1  # cache counts as edge-side

    def get_session_stats(self) -> Dict[str, Any]:
        """Return session statistics as a dict."""
        stats = self._stats.to_dict()
        stats["cache_stats"] = self.cache.stats()
        stats["corpus_size"] = self.retrieval.corpus_size()
        return stats
