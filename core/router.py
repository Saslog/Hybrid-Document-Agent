"""
core/router.py
==============
Cost-Aware Query Router

Takes a ClassificationResult and produces a RoutingDecision.

Decision tree (in priority order):
  1. Cache hit → serve from cache (zero cost)
  2. Force override → honour config.routing.force_route
  3. complexity < SIMPLE_THRESH AND NOT multi_doc → EDGE (simple model)
  4. complexity < MEDIUM_THRESH AND NOT multi_doc → EDGE (medium model)
  5. multi_doc AND complexity > 0.4 → HYBRID (edge retrieval + cloud synthesis)
  6. sensitivity == HIGH → CLOUD (data must not leave org edge or requires audit)
  7. complexity >= CLOUD_THRESH OR requires_reasoning → CLOUD
  8. Default fallback → EDGE medium

Cost estimation formulas are documented inline.
"""

from __future__ import annotations
import logging
from typing import Optional

from core.models import (
    ClassificationResult, RoutingDecision, RouteDecision, QueryIntent
)
from core.config import AgentConfig

logger = logging.getLogger(__name__)


class QueryRouter:
    """
    Deterministic, rule-based router with cost-awareness.

    The routing logic is intentionally transparent (no ML model)
    so operators can audit and tune thresholds directly.
    """

    def __init__(self, config: AgentConfig):
        self.cfg = config
        self.rcfg = config.routing
        self.ecfg = config.edge
        self.ccfg = config.cloud
        self.hcfg = config.hybrid

    def route(
        self,
        classification: ClassificationResult,
        cache_hit: bool = False,
        cache_key: Optional[str] = None,
    ) -> RoutingDecision:
        """
        Produce a routing decision for the given classification.
        """

        # ── 1. Force override ──────────────────────────────────────────
        if self.rcfg.force_route:
            return self._forced_route(classification, self.rcfg.force_route)

        # ── 2. Cache hit ───────────────────────────────────────────────
        if cache_hit:
            return RoutingDecision(
                query_id=classification.query_id,
                route=RouteDecision.CACHE,
                model_name="SemanticCache",
                classification=classification,
                reasoning="Cache hit: semantically similar query found in cache",
                estimated_cost=0.0,
                estimated_cloud_cost=self._estimate_cloud_cost(
                    classification.token_count, output_tokens=150
                ),
                cache_hit=True,
                cache_key=cache_key,
            )

        c  = classification.complexity_score
        md = classification.is_multi_doc
        rr = classification.requires_reasoning
        hi_sensitivity = classification.sensitivity_score > 0.6
        tokens = classification.token_count

        # ── 3. Edge simple ─────────────────────────────────────────────
        if (c < self.rcfg.edge_simple_threshold
                and not md
                and not rr
                and not hi_sensitivity):
            model = self.ecfg.simple_model
            cost  = self._estimate_edge_cost(tokens, output_tokens=100)
            cloud = self._estimate_cloud_cost(tokens, output_tokens=100)
            return RoutingDecision(
                query_id=classification.query_id,
                route=RouteDecision.EDGE,
                model_name=model,
                classification=classification,
                reasoning=(
                    f"Simple factual lookup (complexity={c:.2f} < "
                    f"{self.rcfg.edge_simple_threshold}). "
                    "Edge model fully capable, minimal cost."
                ),
                estimated_cost=cost,
                estimated_cloud_cost=cloud,
            )

        # ── 4. Edge medium ─────────────────────────────────────────────
        if (c < self.rcfg.edge_medium_threshold
                and not md
                and not hi_sensitivity):
            model = self.ecfg.medium_model
            cost  = self._estimate_edge_cost(tokens, output_tokens=200)
            cloud = self._estimate_cloud_cost(tokens, output_tokens=200)
            return RoutingDecision(
                query_id=classification.query_id,
                route=RouteDecision.EDGE,
                model_name=model,
                classification=classification,
                reasoning=(
                    f"Medium complexity (complexity={c:.2f} < "
                    f"{self.rcfg.edge_medium_threshold}), single document. "
                    "Edge medium model handles this efficiently."
                ),
                estimated_cost=cost,
                estimated_cloud_cost=cloud,
            )

        # ── 5. Hybrid (multi-doc + moderate complexity) ────────────────
        if md and c > 0.40:
            model = f"{self.hcfg.edge_model} + {self.hcfg.cloud_model}"
            cost  = self._estimate_hybrid_cost(tokens, output_tokens=400)
            cloud = self._estimate_cloud_cost(tokens * 3, output_tokens=400)  # multi-doc cloud would be ~3× tokens
            return RoutingDecision(
                query_id=classification.query_id,
                route=RouteDecision.HYBRID,
                model_name=model,
                classification=classification,
                reasoning=(
                    "Multi-document query detected. "
                    f"Edge model ({self.hcfg.edge_model}) handles retrieval/chunking; "
                    f"Cloud model ({self.hcfg.cloud_model}) performs final synthesis. "
                    "60-80% token reduction vs sending raw docs to cloud."
                ),
                estimated_cost=cost,
                estimated_cloud_cost=cloud,
            )

        # ── 6. High sensitivity → Cloud (audit trail + data governance) ──
        if hi_sensitivity and c > 0.30:
            model = self.ccfg.model_name
            cost  = self._estimate_cloud_cost(tokens, output_tokens=250)
            return RoutingDecision(
                query_id=classification.query_id,
                route=RouteDecision.CLOUD,
                model_name=model,
                classification=classification,
                reasoning=(
                    f"High sensitivity detected (score={classification.sensitivity_score:.2f}). "
                    "Cloud route enforced for audit compliance and data governance."
                ),
                estimated_cost=cost,
                estimated_cloud_cost=cost,
            )

        # ── 7. Cloud (complex reasoning / high complexity) ─────────────
        if c >= self.rcfg.cloud_threshold or rr:
            model = self.ccfg.model_name
            out_tokens = 600 if rr else 300
            cost  = self._estimate_cloud_cost(tokens, output_tokens=out_tokens)
            return RoutingDecision(
                query_id=classification.query_id,
                route=RouteDecision.CLOUD,
                model_name=model,
                classification=classification,
                reasoning=(
                    f"High complexity (score={c:.2f}) or reasoning required. "
                    "Cloud LLM needed for accurate multi-step reasoning."
                ),
                estimated_cost=cost,
                estimated_cloud_cost=cost,
            )

        # ── 8. Default: edge medium ────────────────────────────────────
        model = self.ecfg.medium_model
        cost  = self._estimate_edge_cost(tokens, output_tokens=200)
        cloud = self._estimate_cloud_cost(tokens, output_tokens=200)
        return RoutingDecision(
            query_id=classification.query_id,
            route=RouteDecision.EDGE,
            model_name=model,
            classification=classification,
            reasoning="Default: medium complexity, single doc. Edge medium model.",
            estimated_cost=cost,
            estimated_cloud_cost=cloud,
        )

    # ─────────────────────────────────────────
    # Cost estimation helpers
    # ─────────────────────────────────────────

    def _estimate_edge_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Edge cost = electricity + hardware amortization approximated
        as a tiny per-token rate for the quantized local model.

        Formula:
            cost = (input_tokens × input_rate) + (output_tokens × output_rate)
        """
        return (
            input_tokens  * self.ecfg.cost_per_input_token +
            output_tokens * self.ecfg.cost_per_output_token
        )

    def _estimate_cloud_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Cloud cost = API pricing (per 1K tokens).

        Formula:
            cost = (input_tokens / 1000 × input_rate_1k)
                 + (output_tokens / 1000 × output_rate_1k)
        """
        return (
            (input_tokens  / 1000) * self.ccfg.input_cost_per_1k +
            (output_tokens / 1000) * self.ccfg.output_cost_per_1k
        )

    def _estimate_hybrid_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Hybrid cost = edge retrieval portion + reduced cloud synthesis.

        The edge model pre-processes and distills context,
        so cloud only sees ~30% of the raw token volume.
        """
        edge_portion  = input_tokens * self.hcfg.edge_retrieval_cost_per_token
        cloud_portion = (
            (int(input_tokens * 0.30) / 1000) * self.hcfg.cloud_synthesis_cost_per_1k +
            (output_tokens / 1000) * self.ccfg.output_cost_per_1k
        )
        return edge_portion + cloud_portion

    def _forced_route(
        self, classification: ClassificationResult, route_str: str
    ) -> RoutingDecision:
        """Override routing with a forced destination."""
        route_map = {
            "edge":  RouteDecision.EDGE,
            "cloud": RouteDecision.CLOUD,
            "hybrid": RouteDecision.HYBRID,
        }
        route = route_map.get(route_str.lower(), RouteDecision.EDGE)
        model_map = {
            RouteDecision.EDGE:   self.ecfg.medium_model,
            RouteDecision.CLOUD:  self.ccfg.model_name,
            RouteDecision.HYBRID: f"{self.hcfg.edge_model} + {self.hcfg.cloud_model}",
        }
        return RoutingDecision(
            query_id=classification.query_id,
            route=route,
            model_name=model_map[route],
            classification=classification,
            reasoning=f"Forced route override: {route_str}",
            estimated_cost=self._estimate_edge_cost(classification.token_count, 200),
            estimated_cloud_cost=self._estimate_cloud_cost(classification.token_count, 200),
        )
