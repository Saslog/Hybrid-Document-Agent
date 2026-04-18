"""
core/classifier.py
==================
Query Classifier — the NLP pipeline that scores every incoming query
across multiple dimensions before the router makes a routing decision.

Pipeline:
  1. Tokenization + token count
  2. Intent detection (regex pattern matching)
  3. Named entity recognition (lightweight rule-based)
  4. Multi-document reference detection
  5. Complexity scoring (weighted formula)
  6. Sensitivity scoring
"""

from __future__ import annotations
import re
import math
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass

from core.models import Query, ClassificationResult, QueryIntent
from core.config import RoutingConfig


# ─────────────────────────────────────────────
# Intent Pattern Library
# ─────────────────────────────────────────────

# Each pattern: (regex, intent, complexity_adjustment)
INTENT_PATTERNS: List[Tuple[re.Pattern, QueryIntent, float]] = [
    # Factual lookups — simplest
    (re.compile(r'\b(what is|what are|what\'s|where is|who is|when is|how much|how many|is there)\b', re.I),
     QueryIntent.FACTUAL_LOOKUP, -0.20),

    # Definitions
    (re.compile(r'\b(define|definition of|what does .+ mean|explain .+ term|meaning of)\b', re.I),
     QueryIntent.DEFINITION, -0.15),

    # Document search — medium
    (re.compile(r'\b(find|search|locate|show|list|get|retrieve|look for)\b.{0,30}\b(clause|section|policy|term|provision|paragraph)\b', re.I),
     QueryIntent.DOCUMENT_SEARCH, 0.0),

    # Summarization
    (re.compile(r'\b(summarize|summary of|give me an overview|brief description|tldr|key points)\b', re.I),
     QueryIntent.SUMMARIZATION, 0.10),

    # Cross-reference — complex
    (re.compile(r'\b(cross.?reference|compare .+ with|reconcile|align|contrast|check .+ against)\b', re.I),
     QueryIntent.CROSS_REFERENCE, 0.25),

    # Multi-doc analysis — most complex
    (re.compile(r'\b(analyze all|across all|all .+ reports|every document|entire corpus)\b', re.I),
     QueryIntent.MULTI_DOC_ANALYSIS, 0.35),

    # Reasoning/forecasting — cloud territory
    (re.compile(r'\b(analyze|analyse|forecast|predict|project|trend|strategy|recommend|insight|implication|assess|evaluate|synthesize)\b', re.I),
     QueryIntent.REASONING, 0.30),
]

# Multi-doc signals
MULTI_DOC_PATTERNS = re.compile(
    r'('
    # Pattern A: quantifier + document noun
    r'\b(all|across|between|compar|multiple|each|every|both|several|various|different)\b'
    r'.{0,40}'
    r'\b(document|report|file|contract|handbook|policy|agreement|guideline|section)\b'
    r'|'
    # Pattern B: explicit cross-reference between named sources
    r'\b(cross.?reference|cross.?ref|compare.+with|align.+with|reconcile)\b'
    r'|'
    # Pattern C: "X policy with Y guidelines" style (two named policy docs)
    r'\b\w+\s+(policy|handbook|guidelines?|terms)\b.{0,40}\b\w+\s+(guidelines?|policy|handbook|terms)\b'
    r')',
    re.I
)

# Sensitivity signals
SENSITIVITY_PATTERNS = re.compile(
    r'\b(confidential|private|pii|personal data|secret|classified|restricted|sensitive|credential|password|ssn|salary|compensation)\b',
    re.I
)

# Simple intent strong signals (reduces complexity)
SIMPLE_INTENT_PATTERNS = re.compile(
    r'^(what is|what are|what\'s|where is|who is|when is|define|list the|show me the|what does)\b',
    re.I
)

# Named entity patterns (doc references, dates, monetary amounts)
ENTITY_PATTERNS: Dict[str, re.Pattern] = {
    "document_ref": re.compile(
        r'\b([A-Z][a-z]+ (?:Handbook|Agreement|Contract|Policy|Report|Guidelines?|Manual)(?:\s+v?\d+\.?\d*)?|'
        r'Q[1-4]\s+\d{4}|Section \d+(?:\.\d+)*|Appendix [A-Z])\b'
    ),
    "date_range": re.compile(r'\b(20\d{2}[-–]20\d{2}|Q[1-4]\s*(?:to|through|–|-)\s*Q[1-4]|\d{4}\s+to\s+\d{4})\b', re.I),
    "financial":  re.compile(r'\$[\d,]+(?:\.\d+)?[KMB]?|\b\d+%\s+(?:growth|increase|revenue)\b', re.I),
    "org_ref":    re.compile(r'\b(?:enterprise|SMB|consumer|department|division|team|segment)\b', re.I),
}


class QueryClassifier:
    """
    Lightweight NLP classifier that scores query complexity
    without invoking any heavyweight models.

    Scoring formula:
        complexity = min(1.0,
            (token_count / TOKEN_NORM  × token_weight) +
            (multi_doc_flag            × multi_doc_weight) +
            (reasoning_flag            × reasoning_weight) -
            (simple_intent_flag        × simple_intent_penalty) +
            intent_complexity_adj
        )
    """

    def __init__(self, config: RoutingConfig):
        self.cfg = config

    def classify(self, query: Query) -> ClassificationResult:
        text = query.text
        tokens = self._tokenize(text)
        token_count = int(len(tokens) * 1.3)  # approx BPE tokens

        # ── Intent detection ──
        intent, intent_complexity_adj = self._detect_intent(text)

        # ── Multi-doc detection ──
        is_multi_doc = bool(MULTI_DOC_PATTERNS.search(text))

        # ── Simple intent check ──
        is_simple_intent = bool(SIMPLE_INTENT_PATTERNS.match(text.strip()))

        # ── Reasoning flag ──
        requires_reasoning = intent in (
            QueryIntent.REASONING,
            QueryIntent.MULTI_DOC_ANALYSIS,
            QueryIntent.CROSS_REFERENCE,
        )

        # ── Named entity extraction ──
        entities, doc_refs = self._extract_entities(text)

        # Boost multi-doc if multiple doc refs found
        if len(doc_refs) >= 2:
            is_multi_doc = True

        # ── Complexity scoring ──
        token_component     = min(1.0, token_count / self.cfg.token_norm_denom) * self.cfg.token_weight
        multi_doc_component = self.cfg.multi_doc_weight  if is_multi_doc       else 0.0
        reasoning_component = self.cfg.reasoning_weight  if requires_reasoning  else 0.0
        simple_penalty      = self.cfg.simple_intent_penalty if is_simple_intent else 0.0

        complexity_score = max(0.0, min(1.0,
            token_component +
            multi_doc_component +
            reasoning_component -
            simple_penalty +
            intent_complexity_adj
        ))

        # ── Sensitivity scoring ──
        sensitivity_matches = SENSITIVITY_PATTERNS.findall(text)
        sensitivity_score = min(1.0, 0.15 + len(sensitivity_matches) * 0.25)

        # ── Confidence ──
        # Higher confidence when intent pattern match is strong
        confidence = 0.75 + (0.15 if is_simple_intent or requires_reasoning else 0.0)
        confidence = min(1.0, confidence)

        # ── Signals dict for debugging ──
        signals = {
            "token_count":           token_count,
            "token_component":       round(token_component, 3),
            "multi_doc_component":   round(multi_doc_component, 3),
            "reasoning_component":   round(reasoning_component, 3),
            "simple_penalty":        round(simple_penalty, 3),
            "intent_adj":            round(intent_complexity_adj, 3),
            "entities_found":        len(entities),
            "doc_refs_found":        doc_refs,
            "sensitivity_matches":   sensitivity_matches,
        }

        return ClassificationResult(
            query_id=query.query_id,
            complexity_score=round(complexity_score, 3),
            sensitivity_score=round(sensitivity_score, 3),
            intent=intent,
            token_count=token_count,
            is_multi_doc=is_multi_doc,
            requires_reasoning=requires_reasoning,
            is_simple_intent=is_simple_intent,
            detected_entities=entities,
            detected_doc_refs=doc_refs,
            confidence=round(confidence, 3),
            signals=signals,
        )

    # ─────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace + punctuation tokenizer."""
        return re.findall(r'\b\w+\b', text.lower())

    def _detect_intent(self, text: str) -> Tuple[QueryIntent, float]:
        """
        Match text against intent patterns.
        Returns the highest-priority (lowest threshold) match
        and its complexity adjustment.
        Last match wins for adjustment (most specific).
        """
        matched_intent = QueryIntent.DOCUMENT_SEARCH  # default
        total_adj = 0.0

        for pattern, intent, adj in INTENT_PATTERNS:
            if pattern.search(text):
                matched_intent = intent
                total_adj += adj

        # Clamp adjustment
        total_adj = max(-0.25, min(0.40, total_adj))
        return matched_intent, total_adj

    def _extract_entities(self, text: str) -> Tuple[List[str], List[str]]:
        """Extract named entities and document references."""
        all_entities: List[str] = []
        doc_refs: List[str] = []

        for entity_type, pattern in ENTITY_PATTERNS.items():
            matches = pattern.findall(text)
            all_entities.extend(matches)
            if entity_type == "document_ref":
                doc_refs.extend(matches)

        return list(set(all_entities)), list(set(doc_refs))
