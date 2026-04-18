"""
api/server.py
=============
FastAPI REST API server for the Hybrid Edge-Cloud Document Agent.

Endpoints:
    POST /query          — Process a single query
    POST /documents      — Add documents to the index
    GET  /stats          — Session statistics
    GET  /health         — Health check
    POST /cache/clear    — Clear semantic cache
    GET  /docs           — Auto-generated OpenAPI docs (FastAPI built-in)

Usage:
    pip install fastapi uvicorn
    python -m api.server
    # → http://localhost:8000/docs
"""

from __future__ import annotations
import asyncio
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from core.agent import HybridDocumentAgent
from core.config import AgentConfig
from core.models import Query, Document


# ─────────────────────────────────────────────
# Pydantic request/response schemas
# ─────────────────────────────────────────────

class QueryRequest(BaseModel):
    text: str = Field(..., description="The query text", example="What is the refund policy?")
    session_id: Optional[str] = Field(None, description="Optional session ID for tracking")

class DocumentRequest(BaseModel):
    id: str
    title: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class QueryResponse(BaseModel):
    query_id: str
    response: str
    route: str
    model_used: str
    complexity_score: float
    latency_ms: float
    cost: float
    cost_savings: float
    cache_hit: bool
    sources: List[Dict[str, Any]]
    routing_reasoning: str

class StatsResponse(BaseModel):
    total_queries: int
    edge_count: int
    cloud_count: int
    hybrid_count: int
    cache_hits: int
    total_cost_actual: float
    total_cost_cloud_only: float
    cost_saved: float
    savings_pct: float
    avg_latency_ms: float
    cache_stats: Dict[str, Any]
    corpus_size: int


# ─────────────────────────────────────────────
# App factory
# ─────────────────────────────────────────────

def create_app(config: Optional[AgentConfig] = None) -> "FastAPI":
    if not HAS_FASTAPI:
        raise ImportError("FastAPI and uvicorn are required. Run: pip install fastapi uvicorn")

    app = FastAPI(
        title="Hybrid Edge-Cloud Document Agent",
        description="Intelligent query routing for cost-optimized RAG systems",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    agent = HybridDocumentAgent(config=config or AgentConfig())

    @app.get("/health")
    async def health():
        return {"status": "healthy", "indexed": agent._indexed}

    @app.post("/documents")
    async def add_documents(docs: List[DocumentRequest]):
        """Index documents into the retrieval corpus."""
        documents = [
            Document(id=d.id, title=d.title, content=d.content, metadata=d.metadata)
            for d in docs
        ]
        await agent.index_documents(documents)
        return {
            "indexed": len(documents),
            "corpus_size": agent.retrieval.corpus_size()
        }

    @app.post("/query", response_model=QueryResponse)
    async def process_query(req: QueryRequest):
        """Process a query through the hybrid routing pipeline."""
        if not agent._indexed:
            raise HTTPException(
                status_code=400,
                detail="No documents indexed. POST to /documents first."
            )
        query = Query(
            text=req.text,
            session_id=req.session_id or "api_session"
        )
        result = await agent.process(query)
        return QueryResponse(
            query_id=result.query_id,
            response=result.response,
            route=result.route,
            model_used=result.model_used,
            complexity_score=result.complexity_score,
            latency_ms=result.latency_ms,
            cost=result.cost,
            cost_savings=result.cost_savings,
            cache_hit=result.cache_hit,
            sources=result.sources,
            routing_reasoning=result.routing_reasoning,
        )

    @app.get("/stats", response_model=StatsResponse)
    async def get_stats():
        """Get session statistics."""
        return StatsResponse(**agent.get_session_stats())

    @app.post("/cache/clear")
    async def clear_cache():
        """Clear the semantic cache."""
        agent.cache.clear()
        return {"status": "cleared"}

    return app


if __name__ == "__main__":
    if not HAS_FASTAPI:
        print("Install FastAPI: pip install fastapi uvicorn")
        sys.exit(1)

    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
