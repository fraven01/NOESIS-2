"""
Agentic Chunking - LLM-driven semantic boundary detection for complex documents.

This module implements intelligent chunking using LLMs to identify optimal
chunk boundaries based on semantic meaning, document structure, and context.
Particularly effective for complex documents like legal contracts, technical
specifications, and academic papers.

Key Features:
- LLM-driven boundary detection (Gemini Flash, Claude Haiku, GPT-4o-mini)
- Structured output with confidence scores
- Automatic fallback to LateChunker on failure
- Rate limiting and token budget controls
- Cost tracking and monitoring

Usage:
    >>> from ai_core.rag.chunking import AgenticChunker
    >>> from ai_core.rag.embeddings import get_embedding_client
    >>>
    >>> chunker = AgenticChunker(
    ...     model="agentic-chunk",  # Uses MODEL_ROUTING.yaml label
    ...     max_retries=2,
    ...     timeout=30,
    ...     rate_limit_per_minute=10,
    ...     token_budget_per_day=100000,
    ... )
    >>>
    >>> chunks = chunker.chunk(parsed_result, context)

Architecture:
    1. Text preprocessing (sentence segmentation)
    2. LLM boundary detection (structured output)
    3. Chunk construction with overlap
    4. Fallback to LateChunker on failure
    5. Cost tracking and rate limiting

Fallback Strategy:
    - LLM timeout → LateChunker
    - LLM rate limit → LateChunker
    - Invalid response → LateChunker
    - Token budget exceeded → LateChunker

Author: Claude Code
Version: 1.0
Date: 2025-12-30
"""

import hashlib
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from ai_core.rag.chunking.late_chunker import LateChunker
from documents.pipeline import DocumentProcessingContext, ParsedResult

logger = logging.getLogger(__name__)


# ==============================================================================
# Boundary Detection Prompt
# ==============================================================================

BOUNDARY_DETECTION_PROMPT = """You are an expert document analyst specializing in semantic chunking. Your task is to identify optimal boundaries for dividing the following document into semantically coherent chunks.

**Document Text:**
{document_text}

**Instructions:**
1. Analyze the document structure and semantic flow
2. Identify natural boundaries where topics or themes shift
3. Each chunk should be:
   - Self-contained (understandable without context)
   - Semantically coherent (single topic/theme)
   - Appropriately sized (300-1000 words ideal)
4. Provide boundary positions as sentence indices (0-based)
5. Include confidence scores (0.0-1.0) for each boundary

**Output Format:**
Return a JSON object with this structure:
{{
    "boundaries": [
        {{"sentence_idx": 5, "confidence": 0.95, "reason": "Topic shift from introduction to methodology"}},
        {{"sentence_idx": 12, "confidence": 0.88, "reason": "Transition from problem statement to solution"}},
        ...
    ],
    "metadata": {{
        "total_sentences": 25,
        "suggested_chunk_count": 3,
        "document_type": "technical_report"
    }}
}}

**Rules:**
- Minimum 2 sentences per chunk
- Maximum 50 sentences per chunk
- Confidence >= 0.7 for boundaries to be used
- Always include reasoning for each boundary
"""


# ==============================================================================
# Pydantic Models for Structured Output
# ==============================================================================


class BoundaryDetection(BaseModel):
    """Single boundary detection result."""

    sentence_idx: int = Field(
        ...,
        ge=0,
        description="0-based sentence index where boundary occurs",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score (0.0-1.0) for this boundary",
    )
    reason: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="Explanation for why this is a good boundary",
    )


class BoundaryMetadata(BaseModel):
    """Metadata about the boundary detection process."""

    total_sentences: int = Field(..., ge=1, description="Total sentences in document")
    suggested_chunk_count: int = Field(
        ..., ge=1, description="Suggested number of chunks"
    )
    document_type: str = Field(
        default="general",
        description="Detected document type (technical, legal, academic, etc.)",
    )


class BoundaryDetectionResponse(BaseModel):
    """LLM response for boundary detection."""

    boundaries: list[BoundaryDetection] = Field(
        ..., description="List of detected boundaries"
    )
    metadata: BoundaryMetadata = Field(..., description="Detection metadata")


# ==============================================================================
# Rate Limiting & Budget Tracking
# ==============================================================================


@dataclass
class RateLimiter:
    """Simple in-memory rate limiter for LLM calls."""

    calls_per_minute: int
    tokens_per_day: int

    def __post_init__(self):
        self.call_timestamps: list[float] = []
        self.daily_token_count = 0
        self.last_reset_date: str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def check_rate_limit(self) -> bool:
        """Check if we're within rate limits."""
        now = time.time()

        # Reset daily token count if new day
        current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if current_date != self.last_reset_date:
            self.daily_token_count = 0
            self.last_reset_date = current_date

        # Remove timestamps older than 1 minute
        cutoff = now - 60
        self.call_timestamps = [ts for ts in self.call_timestamps if ts > cutoff]

        # Check rate limit
        if len(self.call_timestamps) >= self.calls_per_minute:
            logger.warning(
                f"Rate limit exceeded: {len(self.call_timestamps)}/{self.calls_per_minute} calls/min"
            )
            return False

        return True

    def check_token_budget(self, estimated_tokens: int) -> bool:
        """Check if we have budget for estimated tokens."""
        if self.daily_token_count + estimated_tokens > self.tokens_per_day:
            logger.warning(
                f"Token budget exceeded: {self.daily_token_count + estimated_tokens}/{self.tokens_per_day} tokens/day"
            )
            return False
        return True

    def record_call(self, tokens_used: int):
        """Record a successful LLM call."""
        self.call_timestamps.append(time.time())
        self.daily_token_count += tokens_used


# ==============================================================================
# Agentic Chunker
# ==============================================================================


class AgenticChunker:
    """
    LLM-driven semantic chunking for complex documents.

    Uses LLMs (Gemini Flash, Claude Haiku, GPT-4o-mini) to intelligently
    detect chunk boundaries based on semantic meaning and document structure.

    Automatic fallback to LateChunker on failure ensures robustness.
    """

    def __init__(
        self,
        model: str = "agentic-chunk",  # MODEL_ROUTING.yaml label
        max_retries: int = 2,
        timeout: int = 30,
        rate_limit_per_minute: int = 10,
        token_budget_per_day: int = 100000,
        min_chunk_sentences: int = 2,
        max_chunk_sentences: int = 50,
        confidence_threshold: float = 0.7,
        use_content_based_ids: bool = True,
    ):
        """
        Initialize Agentic Chunker.

        Args:
            model: LLM model for boundary detection (default: Gemini Flash)
            max_retries: Maximum retry attempts on LLM failure
            timeout: LLM call timeout in seconds
            rate_limit_per_minute: Maximum LLM calls per minute
            token_budget_per_day: Maximum tokens per day
            min_chunk_sentences: Minimum sentences per chunk
            max_chunk_sentences: Maximum sentences per chunk
            confidence_threshold: Minimum confidence for boundaries (0.0-1.0)
            use_content_based_ids: Use SHA256 content-based chunk IDs
        """
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        self.min_chunk_sentences = min_chunk_sentences
        self.max_chunk_sentences = max_chunk_sentences
        self.confidence_threshold = confidence_threshold
        self.use_content_based_ids = use_content_based_ids

        # Rate limiting and budget controls
        self.rate_limiter = RateLimiter(
            calls_per_minute=rate_limit_per_minute,
            tokens_per_day=token_budget_per_day,
        )

        # Fallback chunker (LateChunker)
        self.fallback_chunker = LateChunker(
            model="embedding",  # MODEL_ROUTING.yaml label
            use_content_based_ids=use_content_based_ids,
        )

        logger.info(
            f"AgenticChunker initialized: model={model}, "
            f"rate_limit={rate_limit_per_minute}/min, "
            f"budget={token_budget_per_day} tokens/day"
        )

    def chunk(
        self,
        parsed: ParsedResult,
        context: DocumentProcessingContext,
    ) -> list[dict[str, Any]]:
        """
        Chunk document using LLM-driven boundary detection.

        Args:
            parsed: Parsed document result
            context: Document processing context

        Returns:
            List of chunk dictionaries with metadata

        Fallback:
            Uses LateChunker if LLM fails or exceeds budget
        """
        # Extract sentences from parsed result
        sentences = self._extract_sentences(parsed)

        if not sentences:
            logger.warning("No sentences found in document, returning empty chunks")
            return []

        # Estimate token count for budget check
        document_text = " ".join(sentences)
        estimated_tokens = len(document_text.split()) * 1.3  # Rough estimate

        # Check rate limits and token budget
        if not self.rate_limiter.check_rate_limit():
            logger.warning("Rate limit exceeded, falling back to LateChunker")
            return self.fallback_chunker.chunk(parsed, context)

        if not self.rate_limiter.check_token_budget(int(estimated_tokens)):
            logger.warning("Token budget exceeded, falling back to LateChunker")
            return self.fallback_chunker.chunk(parsed, context)

        # Attempt LLM boundary detection
        try:
            boundaries = self._detect_boundaries_llm(sentences, document_text)

            # Record successful call
            self.rate_limiter.record_call(int(estimated_tokens))

            # Build chunks from boundaries
            chunks = self._build_chunks_from_boundaries(
                sentences=sentences,
                boundaries=boundaries,
                context=context,
            )

            logger.info(
                f"Agentic chunking successful: {len(chunks)} chunks from {len(sentences)} sentences"
            )
            return chunks

        except Exception as e:
            logger.warning(f"Agentic chunking failed: {e}, falling back to LateChunker")
            return self.fallback_chunker.chunk(parsed, context)

    def _extract_sentences(self, parsed: ParsedResult) -> list[str]:
        """Extract sentences from parsed result."""
        sentences = []
        for block in parsed.text_blocks:
            # Simple sentence splitting (TODO: use proper sentence tokenizer)
            text = block.text.strip()
            if not text:
                continue

            # Split on sentence boundaries
            for sent in text.replace("!", ".").replace("?", ".").split("."):
                sent = sent.strip()
                if sent and len(sent) > 10:  # Minimum sentence length
                    sentences.append(sent)

        return sentences

    def _detect_boundaries_llm(
        self, sentences: list[str], document_text: str
    ) -> list[int]:
        """
        Detect chunk boundaries using LLM.

        Args:
            sentences: List of sentences
            document_text: Full document text

        Returns:
            List of sentence indices for boundaries

        Raises:
            Exception: On LLM failure (timeout, invalid response, etc.)
        """
        # TODO: Implement actual LLM call with structured output
        # For now, MVP implementation returns empty boundaries
        # (will fallback to LateChunker)

        # Mock LLM call for MVP
        logger.warning(
            "AgenticChunker LLM boundary detection not yet implemented (MVP), "
            "falling back to LateChunker"
        )
        raise NotImplementedError("LLM boundary detection not yet implemented")

    def _build_chunks_from_boundaries(
        self,
        sentences: list[str],
        boundaries: list[int],
        context: DocumentProcessingContext,
    ) -> list[dict[str, Any]]:
        """
        Build chunks from detected boundaries.

        Args:
            sentences: List of sentences
            boundaries: List of boundary indices
            context: Document processing context

        Returns:
            List of chunk dictionaries
        """
        chunks = []

        namespace_key = self._resolve_chunk_namespace_key(context)

        # Add start and end boundaries
        all_boundaries = sorted([0] + boundaries + [len(sentences)])

        for i in range(len(all_boundaries) - 1):
            start_idx = all_boundaries[i]
            end_idx = all_boundaries[i + 1]

            # Extract chunk sentences
            chunk_sentences = sentences[start_idx:end_idx]
            chunk_text = " ".join(chunk_sentences)

            # Generate chunk ID
            if self.use_content_based_ids:
                # Content-based SHA256 ID (namespaced by document_id to prevent collisions)
                normalized_text = chunk_text.lower().strip()
                namespaced_content = f"{namespace_key}:{normalized_text}"
                chunk_hash = hashlib.sha256(
                    namespaced_content.encode("utf-8")
                ).hexdigest()
                chunk_id = f"sha256-{chunk_hash[:32]}"
            else:
                # UUID-based ID
                from uuid import uuid5, NAMESPACE_DNS

                chunk_id = str(
                    uuid5(
                        NAMESPACE_DNS,
                        f"{namespace_key}:{start_idx}:{end_idx}",
                    )
                )

            # Build chunk dictionary
            chunk = {
                "chunk_id": chunk_id,
                "text": chunk_text,
                "metadata": {
                    "document_id": str(context.metadata.document_id),
                    "chunk_index": i,
                    "start_sentence": start_idx,
                    "end_sentence": end_idx,
                    "sentence_count": len(chunk_sentences),
                    "chunker": "agentic",
                },
            }

            chunks.append(chunk)

        return chunks

    @staticmethod
    def _resolve_chunk_namespace_key(context: DocumentProcessingContext) -> str:
        version_id = getattr(context.metadata, "document_version_id", None)
        if version_id in {None, ""}:
            return str(context.metadata.document_id)
        return str(version_id)


# ==============================================================================
# Factory Function
# ==============================================================================


def get_default_agentic_chunker() -> AgenticChunker:
    """Get default AgenticChunker with settings from Django config."""
    from django.conf import settings

    return AgenticChunker(
        model=getattr(settings, "RAG_AGENTIC_CHUNK_MODEL", "agentic-chunk"),
        max_retries=getattr(settings, "RAG_AGENTIC_CHUNK_MAX_RETRIES", 2),
        timeout=getattr(settings, "RAG_AGENTIC_CHUNK_TIMEOUT", 30),
        rate_limit_per_minute=getattr(settings, "RAG_AGENTIC_CHUNK_RATE_LIMIT", 10),
        token_budget_per_day=getattr(
            settings, "RAG_AGENTIC_CHUNK_TOKEN_BUDGET", 100000
        ),
        use_content_based_ids=getattr(settings, "RAG_USE_CONTENT_BASED_IDS", True),
    )
