"""
core/config.py
==============
Configuration dataclasses for all agent components.
Supports defaults + YAML override.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import os


@dataclass
class RoutingConfig:
    """Controls the routing decision thresholds."""
    # Complexity score thresholds
    edge_simple_threshold: float = 0.30       # < this → edge simple
    edge_medium_threshold: float = 0.55       # < this (no multi-doc) → edge medium
    cloud_threshold: float = 0.65             # >= this → cloud

    # Cost weights
    cost_weight: float = 0.4                  # how much cost matters in routing
    accuracy_weight: float = 0.6             # how much accuracy matters

    # Force override (None | "edge" | "cloud" | "hybrid")
    force_route: Optional[str] = None

    # Cache similarity threshold (cosine)
    cache_similarity_threshold: float = 0.92

    # Complexity scoring weights (must sum to ~1.0)
    token_weight: float = 0.40
    multi_doc_weight: float = 0.30
    reasoning_weight: float = 0.25
    simple_intent_penalty: float = 0.20

    # Token normalization denominator
    token_norm_denom: int = 80


@dataclass
class EdgeModelConfig:
    """Configuration for the local (edge) inference model."""
    # Model identifiers (used in simulation; swap for real model paths)
    simple_model: str = "Mistral-7B-Instruct-v0.2-Q4_K_M"
    medium_model: str = "Phi-3-Mini-4k-Instruct-Q4"

    # Simulated latency ranges (ms)
    simple_latency_min: int = 80
    simple_latency_max: int = 200
    medium_latency_min: int = 150
    medium_latency_max: int = 320

    # Cost per token (USD) - quantized local model (electricity + hardware amortization)
    cost_per_input_token: float = 0.000_000_5
    cost_per_output_token: float = 0.000_001_0

    # Max context length (tokens)
    max_context_tokens: int = 4096

    # Real model path (for llama.cpp / ollama integration)
    model_path: Optional[str] = None
    ollama_base_url: str = "http://localhost:11434"
    use_ollama: bool = False


@dataclass
class CloudModelConfig:
    """Configuration for cloud LLM APIs."""
    provider: str = "openai"                  # "openai" | "anthropic" | "simulated"
    model_name: str = "gpt-4o"

    # Fallback model for hybrid route cloud portion
    fallback_model: str = "claude-3-5-sonnet-20241022"

    # Simulated latency ranges (ms)
    latency_min: int = 900
    latency_max: int = 2800

    # Cost per 1K tokens (USD) — approximate 2024 pricing
    input_cost_per_1k: float = 0.005          # GPT-4o input
    output_cost_per_1k: float = 0.015         # GPT-4o output

    # API keys (read from environment if not set)
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None

    # Max output tokens
    max_output_tokens: int = 1024

    def get_openai_key(self) -> Optional[str]:
        return self.openai_api_key or os.getenv("OPENAI_API_KEY")

    def get_anthropic_key(self) -> Optional[str]:
        return self.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")


@dataclass
class HybridModelConfig:
    """Configuration for the hybrid routing mode."""
    # Edge handles retrieval; cloud handles final synthesis
    edge_model: str = "Mistral-7B-Instruct-v0.2-Q4_K_M"
    cloud_model: str = "gpt-4o"

    # Simulated latency
    latency_min: int = 500
    latency_max: int = 1400

    # Cost splits
    edge_retrieval_cost_per_token: float = 0.000_000_5
    cloud_synthesis_cost_per_1k: float = 0.005


@dataclass
class RetrievalConfig:
    """Configuration for the RAG retrieval pipeline."""
    # Chunking
    chunk_size: int = 300             # words per chunk
    chunk_overlap: int = 50           # overlap between chunks

    # BM25
    bm25_top_k: int = 20              # candidates from BM25
    bm25_k1: float = 1.5              # BM25 k1 parameter
    bm25_b: float = 0.75              # BM25 b parameter

    # Dense vector retrieval
    vector_top_k: int = 20            # candidates from vector search
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # RRF (Reciprocal Rank Fusion)
    rrf_k: int = 60                   # RRF smoothing constant

    # Reranker
    reranker_top_k: int = 5           # final chunks after reranking
    use_reranker: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class CacheConfig:
    """Configuration for the semantic cache layer."""
    enabled: bool = True
    backend: str = "memory"            # "memory" | "redis"
    redis_url: str = "redis://localhost:6379"
    ttl_seconds: int = 3600            # default 1 hour
    max_size: int = 1000               # max cached entries (memory backend)
    similarity_threshold: float = 0.92


@dataclass
class AgentConfig:
    """Top-level agent configuration."""
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    edge: EdgeModelConfig = field(default_factory=EdgeModelConfig)
    cloud: CloudModelConfig = field(default_factory=CloudModelConfig)
    hybrid: HybridModelConfig = field(default_factory=HybridModelConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)

    # Simulation mode (no real API calls)
    simulation_mode: bool = True

    # Logging
    log_level: str = "INFO"
    log_routing_decisions: bool = True

    @classmethod
    def from_yaml(cls, path: str) -> "AgentConfig":
        """Load config from YAML file."""
        try:
            import yaml
            with open(path) as f:
                data = yaml.safe_load(f)
            cfg = cls()
            # Apply overrides from YAML
            for section, values in data.items():
                if hasattr(cfg, section) and isinstance(values, dict):
                    section_cfg = getattr(cfg, section)
                    for k, v in values.items():
                        if hasattr(section_cfg, k):
                            setattr(section_cfg, k, v)
            return cfg
        except ImportError:
            print("Warning: pyyaml not installed; using default config.")
            return cls()
        except FileNotFoundError:
            print(f"Warning: Config file {path} not found; using defaults.")
            return cls()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "routing": self.routing.__dict__,
            "edge": self.edge.__dict__,
            "cloud": self.cloud.__dict__,
            "hybrid": self.hybrid.__dict__,
            "retrieval": self.retrieval.__dict__,
            "cache": self.cache.__dict__,
            "simulation_mode": self.simulation_mode,
        }
