"""
core/models.py
==============
All data models used throughout the Hybrid Edge-Cloud Agent.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from enum import Enum
import time
import uuid


# ─────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────

class RouteDecision(str, Enum):
    EDGE    = "edge"
    CLOUD   = "cloud"
    HYBRID  = "hybrid"
    CACHE   = "cache"


class QueryIntent(str, Enum):
    FACTUAL_LOOKUP   = "factual_lookup"    # "What is X?"
    DOCUMENT_SEARCH  = "document_search"   # "Find clauses about Y"
    MULTI_DOC_ANALYSIS = "multi_doc_analysis"  # "Compare all reports"
    REASONING        = "reasoning"         # "Analyze trends and forecast"
    DEFINITION       = "definition"        # "Define force majeure"
    CROSS_REFERENCE  = "cross_reference"   # "Cross-ref policy A with policy B"
    SUMMARIZATION    = "summarization"     # "Summarize the document"


class DataClassification(str, Enum):
    PUBLIC       = "public"
    INTERNAL     = "internal"
    CONFIDENTIAL = "confidential"
    SECRET       = "secret"


# ─────────────────────────────────────────────
# Document model
# ─────────────────────────────────────────────

@dataclass
class Document:
    """A document in the corpus."""
    id: str
    title: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunks: List["DocumentChunk"] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def get_classification(self) -> DataClassification:
        cls = self.metadata.get("classification", "internal")
        return DataClassification(cls) if cls in DataClassification._value2member_map_ else DataClassification.INTERNAL

    def word_count(self) -> int:
        return len(self.content.split())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "word_count": self.word_count(),
            "metadata": self.metadata,
        }


@dataclass
class DocumentChunk:
    """A chunk/passage extracted from a Document for retrieval."""
    id: str
    doc_id: str
    doc_title: str
    content: str
    chunk_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = field(default=None, repr=False)
    bm25_tokens: List[str] = field(default_factory=list)

    @classmethod
    def from_document(cls, doc: Document, content: str, chunk_index: int) -> "DocumentChunk":
        return cls(
            id=f"{doc.id}_chunk_{chunk_index:03d}",
            doc_id=doc.id,
            doc_title=doc.title,
            content=content,
            chunk_index=chunk_index,
            metadata=doc.metadata.copy(),
        )


# ─────────────────────────────────────────────
# Query model
# ─────────────────────────────────────────────

@dataclass
class Query:
    """An incoming user query."""
    text: str
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    query_id: str   = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def token_count(self) -> int:
        """Approximate token count (words × 1.3)."""
        return int(len(self.text.split()) * 1.3)

    def word_count(self) -> int:
        return len(self.text.split())


# ─────────────────────────────────────────────
# Classification result
# ─────────────────────────────────────────────

@dataclass
class ClassificationResult:
    """Output of the QueryClassifier."""
    query_id: str
    complexity_score: float        # 0.0–1.0
    sensitivity_score: float       # 0.0–1.0
    intent: QueryIntent
    token_count: int
    is_multi_doc: bool
    requires_reasoning: bool
    is_simple_intent: bool
    detected_entities: List[str]
    detected_doc_refs: List[str]
    confidence: float
    signals: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"ClassificationResult("
            f"complexity={self.complexity_score:.2f}, "
            f"intent={self.intent.value}, "
            f"multi_doc={self.is_multi_doc}, "
            f"reasoning={self.requires_reasoning})"
        )


# ─────────────────────────────────────────────
# Retrieval models
# ─────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    """A chunk returned by the retrieval pipeline with scores."""
    chunk: DocumentChunk
    bm25_score: float = 0.0
    vector_score: float = 0.0
    rrf_score: float = 0.0          # Reciprocal Rank Fusion score
    rerank_score: float = 0.0       # Cross-encoder reranker score
    final_score: float = 0.0

    def __lt__(self, other: "RetrievedChunk") -> bool:
        return self.final_score < other.final_score


@dataclass
class RetrievalResult:
    """Output of the RAG retrieval pipeline."""
    query_id: str
    chunks: List[RetrievedChunk]
    retrieval_latency_ms: float
    bm25_candidates: int
    vector_candidates: int
    after_rrf: int
    after_rerank: int
    cache_hit: bool = False

    @property
    def top_chunks(self) -> List[RetrievedChunk]:
        return sorted(self.chunks, key=lambda x: x.final_score, reverse=True)

    def to_context_string(self, max_chunks: int = 5) -> str:
        parts = []
        for i, rc in enumerate(self.top_chunks[:max_chunks], 1):
            parts.append(
                f"[Source {i}: {rc.chunk.doc_title}]\n{rc.chunk.content}"
            )
        return "\n\n---\n\n".join(parts)


# ─────────────────────────────────────────────
# Routing decision
# ─────────────────────────────────────────────

@dataclass
class RoutingDecision:
    """The router's output — where to send the query."""
    query_id: str
    route: RouteDecision
    model_name: str
    classification: ClassificationResult
    reasoning: str                  # Human-readable explanation
    estimated_cost: float
    estimated_cloud_cost: float     # What it would have cost cloud-only
    cache_hit: bool = False
    cache_key: Optional[str] = None

    @property
    def savings(self) -> float:
        return max(0.0, self.estimated_cloud_cost - self.estimated_cost)

    def __str__(self) -> str:
        return (
            f"Route: {self.route.value.upper()} → {self.model_name} | "
            f"Est. cost: ${self.estimated_cost:.5f} | "
            f"Savings: ${self.savings:.5f}"
        )


# ─────────────────────────────────────────────
# Final agent result
# ─────────────────────────────────────────────

@dataclass
class AgentResult:
    """The complete result returned to the caller."""
    query_id: str
    session_id: str
    query_text: str
    response: str
    route: str
    model_used: str
    complexity_score: float
    latency_ms: float
    cost: float
    cloud_cost_equivalent: float
    cost_savings: float
    cache_hit: bool
    sources: List[Dict[str, Any]]
    routing_reasoning: str
    signals: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def savings_pct(self) -> float:
        if self.cloud_cost_equivalent == 0:
            return 0.0
        return (self.cost_savings / self.cloud_cost_equivalent) * 100


# ─────────────────────────────────────────────
# Session statistics
# ─────────────────────────────────────────────

@dataclass
class SessionStats:
    """Accumulated statistics for a session."""
    total_queries: int = 0
    edge_count: int = 0
    cloud_count: int = 0
    hybrid_count: int = 0
    cache_hits: int = 0
    total_cost_actual: float = 0.0
    total_cost_cloud_only: float = 0.0
    latencies: List[float] = field(default_factory=list)

    @property
    def cost_saved(self) -> float:
        return max(0.0, self.total_cost_cloud_only - self.total_cost_actual)

    @property
    def savings_pct(self) -> float:
        if self.total_cost_cloud_only == 0:
            return 0.0
        return (self.cost_saved / self.total_cost_cloud_only) * 100

    @property
    def avg_latency_ms(self) -> float:
        return sum(self.latencies) / len(self.latencies) if self.latencies else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_queries": self.total_queries,
            "edge_count": self.edge_count,
            "cloud_count": self.cloud_count,
            "hybrid_count": self.hybrid_count,
            "cache_hits": self.cache_hits,
            "total_cost_actual": round(self.total_cost_actual, 6),
            "total_cost_cloud_only": round(self.total_cost_cloud_only, 6),
            "cost_saved": round(self.cost_saved, 6),
            "savings_pct": round(self.savings_pct, 2),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
        }
