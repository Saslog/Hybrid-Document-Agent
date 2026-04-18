"""
models/inference.py
===================
Model inference adapters for Edge and Cloud backends.

Each adapter implements the same async interface:
    async def generate(query_text, context, max_tokens) -> (response_text, latency_ms)

Adapters:
    EdgeModelAdapter   — local quantized LLM (simulated or real via Ollama/llama.cpp)
    CloudModelAdapter  — cloud API (simulated or real via OpenAI/Anthropic)
    HybridAdapter      — chains Edge retrieval + Cloud synthesis
"""

from __future__ import annotations
import asyncio
import random
import time
import re
from typing import Tuple, Optional

from core.config import AgentConfig
from core.models import RetrievalResult


# ─────────────────────────────────────────────
# Response templates for simulation
# ─────────────────────────────────────────────

EDGE_SIMPLE_TEMPLATES = [
    "Based on {source}, {answer}",
    "According to the {source}, {answer}",
    "The {source} states: {answer}",
]

CLOUD_ANALYSIS_TEMPLATES = [
    "**Comprehensive Analysis:**\n\n{answer}\n\n**Key Takeaways:**\n{takeaways}",
    "**Strategic Assessment:**\n\n{answer}\n\n**Recommendations:**\n{takeaways}",
]


def _extract_answer_from_context(query: str, context: str) -> str:
    """
    Extract the most relevant sentence(s) from context for the query.
    Used by simulated models to produce plausible answers.
    """
    q_terms = set(re.findall(r'\b[a-zA-Z0-9]{3,}\b', query.lower()))
    sentences = re.split(r'(?<=[.!?])\s+', context)

    scored: list = []
    for sent in sentences:
        if len(sent.strip()) < 20:
            continue
        s_terms = set(re.findall(r'\b[a-zA-Z0-9]{3,}\b', sent.lower()))
        overlap = len(q_terms & s_terms)
        scored.append((overlap, sent.strip()))

    scored.sort(reverse=True)
    top = [s for _, s in scored[:3] if s]
    return " ".join(top) if top else context[:300]


# ─────────────────────────────────────────────
# Edge Model Adapter
# ─────────────────────────────────────────────

class EdgeModelAdapter:
    """
    Local edge model inference adapter.

    Production options:
        - Ollama (easiest): set use_ollama=True in config
        - llama.cpp Python bindings: set model_path in config
        - HuggingFace Transformers: use AutoModelForCausalLM with BitsAndBytes quantization

    Simulation: generates plausible responses by extracting key sentences
    from retrieved context using a simple overlap heuristic.
    """

    def __init__(self, config: AgentConfig):
        self.cfg   = config.edge
        self.rcfg  = config.routing
        self._llm  = None

        if self.cfg.use_ollama and not config.simulation_mode:
            self._init_ollama()

    def _init_ollama(self):
        """Initialize Ollama client for local LLM inference."""
        try:
            import httpx
            self._ollama_url = f"{self.cfg.ollama_base_url}/api/generate"
            print(f"Edge model: Ollama at {self._ollama_url}")
        except ImportError:
            print("Warning: httpx not installed. Falling back to simulation.")

    async def generate(
        self,
        query_text: str,
        retrieval: RetrievalResult,
        max_tokens: int = 256,
    ) -> Tuple[str, float]:
        """
        Generate a response given query + retrieved context.

        Returns:
            (response_text, latency_ms)
        """
        context = retrieval.to_context_string(max_chunks=3)

        if self.cfg.use_ollama and self._llm:
            return await self._ollama_generate(query_text, context, max_tokens)
        else:
            return await self._simulated_generate(query_text, context, retrieval)

    async def _simulated_generate(
        self,
        query: str,
        context: str,
        retrieval: RetrievalResult,
    ) -> Tuple[str, float]:
        """Simulate edge model response."""
        latency_min = self.cfg.simple_latency_min
        latency_max = self.cfg.medium_latency_max
        latency_ms  = random.randint(latency_min, latency_max)

        await asyncio.sleep(latency_ms / 1000)

        answer = _extract_answer_from_context(query, context)
        sources = list({rc.chunk.doc_title for rc in retrieval.top_chunks[:2]})
        source_str = sources[0] if sources else "the document"

        response = (
            f"Based on {source_str}: {answer}"
            if answer
            else "I could not find a direct answer in the available documents."
        )

        return response, float(latency_ms)

    async def _ollama_generate(
        self,
        query: str,
        context: str,
        max_tokens: int,
    ) -> Tuple[str, float]:
        """Real Ollama API call."""
        import httpx
        prompt = f"""You are a helpful document assistant. Answer using only the provided context.

Context:
{context}

Question: {query}

Answer concisely based on the context above:"""

        start = time.time()
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(self._ollama_url, json={
                    "model": self.cfg.simple_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": max_tokens, "temperature": 0.1},
                })
                data = resp.json()
                response = data.get("response", "No response generated.")
                latency_ms = (time.time() - start) * 1000
                return response, latency_ms
        except Exception as e:
            return f"Edge model error: {e}", 0.0


# ─────────────────────────────────────────────
# Cloud Model Adapter
# ─────────────────────────────────────────────

class CloudModelAdapter:
    """
    Cloud LLM adapter (OpenAI / Anthropic / simulated).

    In simulation mode: produces richer, more detailed responses
    than the edge adapter by synthesizing across more context chunks.
    """

    def __init__(self, config: AgentConfig):
        self.cfg  = config.cloud
        self.sim  = config.simulation_mode

    async def generate(
        self,
        query_text: str,
        retrieval: RetrievalResult,
        max_tokens: int = 600,
    ) -> Tuple[str, float]:
        """Generate response using cloud LLM."""
        context = retrieval.to_context_string(max_chunks=5)

        if not self.sim and self.cfg.get_openai_key():
            return await self._openai_generate(query_text, context, max_tokens)
        elif not self.sim and self.cfg.get_anthropic_key():
            return await self._anthropic_generate(query_text, context, max_tokens)
        else:
            return await self._simulated_generate(query_text, context, retrieval)

    async def _simulated_generate(
        self,
        query: str,
        context: str,
        retrieval: RetrievalResult,
    ) -> Tuple[str, float]:
        """Simulate cloud model — richer, multi-step reasoning response."""
        latency_ms = random.randint(self.cfg.latency_min, self.cfg.latency_max)
        await asyncio.sleep(latency_ms / 1000)

        sources = list({rc.chunk.doc_title for rc in retrieval.top_chunks[:4]})
        answer  = _extract_answer_from_context(query, context)

        # Detect query type for response format
        is_analysis = any(w in query.lower() for w in ["analyze", "compare", "trend", "forecast", "assess"])
        is_crossref  = any(w in query.lower() for w in ["cross-reference", "cross reference", "compare with", "align"])

        if is_analysis:
            response = (
                f"**Analysis based on {len(sources)} source(s):**\n\n"
                f"{answer}\n\n"
                f"**Key Findings:** The documents indicate multiple relevant dimensions "
                f"to consider. Cross-document synthesis reveals consistent themes aligned "
                f"with the query intent.\n\n"
                f"*Sources reviewed: {', '.join(sources)}*"
            )
        elif is_crossref:
            response = (
                f"**Cross-Reference Analysis:**\n\n"
                f"Reviewing {', '.join(sources)}:\n\n"
                f"{answer}\n\n"
                f"**Alignment Assessment:** The referenced documents show both "
                f"consistent requirements and complementary provisions that together "
                f"define the complete policy framework."
            )
        else:
            response = (
                f"**Comprehensive Response:**\n\n"
                f"{answer}\n\n"
                f"*Retrieved from: {', '.join(sources) if sources else 'corpus'}*"
            )

        return response, float(latency_ms)

    async def _openai_generate(
        self, query: str, context: str, max_tokens: int
    ) -> Tuple[str, float]:
        """Real OpenAI API call."""
        try:
            import openai
            client = openai.AsyncOpenAI(api_key=self.cfg.get_openai_key())
            start = time.time()
            resp = await client.chat.completions.create(
                model=self.cfg.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful document assistant. Answer questions based strictly on the provided context."},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
                ],
                max_tokens=max_tokens,
                temperature=0.1,
            )
            latency_ms = (time.time() - start) * 1000
            return resp.choices[0].message.content, latency_ms
        except Exception as e:
            return f"OpenAI error: {e}", 0.0

    async def _anthropic_generate(
        self, query: str, context: str, max_tokens: int
    ) -> Tuple[str, float]:
        """Real Anthropic API call."""
        try:
            import anthropic
            client = anthropic.AsyncAnthropic(api_key=self.cfg.get_anthropic_key())
            start = time.time()
            resp = await client.messages.create(
                model=self.cfg.model_name,
                max_tokens=max_tokens,
                messages=[{
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer based only on the context provided."
                }],
            )
            latency_ms = (time.time() - start) * 1000
            return resp.content[0].text, latency_ms
        except Exception as e:
            return f"Anthropic error: {e}", 0.0


# ─────────────────────────────────────────────
# Hybrid Adapter
# ─────────────────────────────────────────────

class HybridAdapter:
    """
    Hybrid inference: Edge model handles retrieval/filtering,
    Cloud model handles final synthesis.

    Token reduction strategy:
        1. Edge model selects and summarizes top chunks (~30% of raw tokens)
        2. Compressed context sent to cloud model
        3. Cloud model synthesizes final answer

    This reduces cloud token consumption by 60-80% vs sending raw docs.
    """

    def __init__(self, config: AgentConfig):
        self.edge_adapter  = EdgeModelAdapter(config)
        self.cloud_adapter = CloudModelAdapter(config)
        self.cfg = config.hybrid

    async def generate(
        self,
        query_text: str,
        retrieval: RetrievalResult,
        max_tokens: int = 500,
    ) -> Tuple[str, float]:
        """
        Two-stage hybrid generation:
        Stage 1 (Edge): Extract and summarize relevant passages
        Stage 2 (Cloud): Synthesize final answer from compressed context
        """
        start = time.time()

        # Stage 1: Edge extraction (simulated as context compression)
        raw_context    = retrieval.to_context_string(max_chunks=5)
        compressed_ctx = self._compress_context(query_text, raw_context)

        # Create a synthetic retrieval with compressed context
        compressed_retrieval = _CompressedRetrieval(
            query_id=retrieval.query_id,
            compressed_context=compressed_ctx,
            original_chunks=retrieval.chunks,
        )

        # Stage 2: Cloud synthesis on compressed context
        response, cloud_latency = await self.cloud_adapter.generate(
            query_text, compressed_retrieval, max_tokens
        )

        total_latency = (time.time() - start) * 1000
        return response, total_latency

    def _compress_context(self, query: str, context: str) -> str:
        """
        Simulate edge model context compression.
        Extracts the most query-relevant sentences to reduce cloud token count.
        """
        q_terms = set(re.findall(r'\b[a-zA-Z0-9]{3,}\b', query.lower()))
        sentences = re.split(r'(?<=[.!?])\s+|\n+', context)

        scored = []
        for sent in sentences:
            if len(sent.strip()) < 15:
                continue
            s_terms = set(re.findall(r'\b[a-zA-Z0-9]{3,}\b', sent.lower()))
            score = len(q_terms & s_terms)
            scored.append((score, sent.strip()))

        scored.sort(reverse=True)
        # Take top ~30% of sentences by relevance
        n_keep = max(3, len(scored) // 3)
        kept = [s for _, s in scored[:n_keep]]
        return "\n".join(kept)


class _CompressedRetrieval:
    """Wrapper to make compressed context work with CloudModelAdapter interface."""

    def __init__(self, query_id, compressed_context, original_chunks):
        self.query_id = query_id
        self._context = compressed_context
        self.chunks = original_chunks

    def to_context_string(self, max_chunks: int = 5) -> str:
        return self._context

    @property
    def top_chunks(self):
        return sorted(self.chunks, key=lambda x: x.final_score, reverse=True)
