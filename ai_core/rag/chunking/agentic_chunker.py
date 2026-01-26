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
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from pydantic import BaseModel, Field, ValidationError

from ai_core.rag.chunking.late_chunker import LateChunker
from ai_core.rag.chunking.utils import (
    build_chunk_prefix,
    extract_numbered_list_index,
    find_numbered_list_runs,
    is_numbered_list_item,
    split_sentences,
)
from ai_core.rag.contextual_enrichment import (
    generate_contextual_prefixes,
    resolve_contextual_enrichment_config,
)
from ai_core.llm import client as llm_client
from ai_core.llm.client import LlmClientError, LlmTimeoutError, RateLimitError
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
        enable_contextual_enrichment: bool = False,
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
        self.enable_contextual_enrichment = enable_contextual_enrichment

        # Rate limiting and budget controls
        self.rate_limiter = RateLimiter(
            calls_per_minute=rate_limit_per_minute,
            tokens_per_day=token_budget_per_day,
        )

        # Fallback chunker (LateChunker)
        self.fallback_chunker = LateChunker(
            model="embedding",  # MODEL_ROUTING.yaml label
            use_content_based_ids=use_content_based_ids,
            enable_contextual_enrichment=enable_contextual_enrichment,
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
        document: Any | None = None,
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
            return self._apply_chunk_counts(
                self.fallback_chunker.chunk(parsed, context, document=document)
            )

        if not self.rate_limiter.check_token_budget(int(estimated_tokens)):
            logger.warning("Token budget exceeded, falling back to LateChunker")
            return self._apply_chunk_counts(
                self.fallback_chunker.chunk(parsed, context, document=document)
            )

        # Attempt LLM boundary detection
        document_title = self._resolve_document_title(document)
        document_ref = self._resolve_document_ref(document) or document_title
        doc_type = self._resolve_doc_type(document)

        try:
            boundaries = self._detect_boundaries_llm(
                sentences,
                document_text,
                context,
            )

            # Record successful call
            self.rate_limiter.record_call(int(estimated_tokens))

            # Build chunks from boundaries
            chunks = self._build_chunks_from_boundaries(
                sentences=sentences,
                boundaries=boundaries,
                context=context,
                document_title=document_title,
                document_ref=document_ref,
                doc_type=doc_type,
            )
            if self.enable_contextual_enrichment:
                chunks = self._apply_contextual_enrichment(
                    chunks,
                    document_text,
                    context,
                )

            logger.info(
                "agentic_chunker_completed",
                extra={
                    "chunk_count": len(chunks),
                    "sentence_count": len(sentences),
                    "document_id": str(context.metadata.document_id),
                    "document_ref": document_ref,
                    "doc_type": doc_type,
                },
            )
            return self._apply_chunk_counts(chunks)

        except Exception as e:
            logger.warning(
                "agentic_chunker_failed",
                extra={
                    "error": str(e),
                    "document_id": str(context.metadata.document_id),
                    "document_ref": document_ref,
                    "doc_type": doc_type,
                },
            )
            return self._apply_chunk_counts(
                self.fallback_chunker.chunk(parsed, context, document=document)
            )

    def _extract_sentences(self, parsed: ParsedResult) -> list[str]:
        """Extract sentences from parsed result."""
        sentences = []
        for block in parsed.text_blocks:
            # Simple sentence splitting (TODO: use proper sentence tokenizer)
            text = block.text.strip()
            if not text:
                continue

            # Split on sentence boundaries
            for sent in split_sentences(text):
                if sent and len(sent) > 10:  # Minimum sentence length
                    sentences.append(sent)

        return sentences

    def _detect_boundaries_llm(
        self,
        sentences: list[str],
        document_text: str,
        context: DocumentProcessingContext,
    ) -> tuple[list[int], dict[int, int]]:
        """
        Detect chunk boundaries using LLM.

        Args:
            sentences: List of sentences
            document_text: Full document text

        Returns:
            Tuple of (boundary indices, overlap map)

        Raises:
            Exception: On LLM failure (timeout, invalid response, etc.)
        """
        if not sentences:
            return []

        _ = document_text

        indexed_text = "\n".join(
            f"{idx}: {sentence}" for idx, sentence in enumerate(sentences)
        )
        prompt = BOUNDARY_DETECTION_PROMPT.format(document_text=indexed_text)

        metadata = {
            "tenant_id": getattr(context.metadata, "tenant_id", None),
            "case_id": getattr(context.metadata, "case_id", None),
            "trace_id": context.trace_id or getattr(context.metadata, "trace_id", None),
            "prompt_version": "agentic_chunk_boundaries_v1",
        }

        def _extract_payload(text: str) -> dict[str, Any]:
            cleaned = (text or "").strip()
            if cleaned.startswith("```") and cleaned.endswith("```"):
                lines = cleaned.splitlines()
                if len(lines) >= 3:
                    cleaned = "\n".join(lines[1:-1]).strip()
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise ValueError("llm payload missing json object")
            fragment = cleaned[start : end + 1]
            data = json.loads(fragment)
            if not isinstance(data, dict):
                raise ValueError("llm payload must be a json object")
            return data

        def _normalize_boundaries(
            response: BoundaryDetectionResponse,
        ) -> tuple[list[int], dict[int, int]]:
            total_sentences = len(sentences)
            if response.metadata.total_sentences != total_sentences:
                logger.warning(
                    "agentic_chunker_sentence_count_mismatch",
                    extra={
                        "expected": total_sentences,
                        "reported": response.metadata.total_sentences,
                    },
                )

            candidates: list[int] = []
            for boundary in response.boundaries:
                if boundary.confidence < self.confidence_threshold:
                    continue
                idx = boundary.sentence_idx
                if idx <= 0 or idx >= total_sentences:
                    continue
                candidates.append(idx)

            candidates = sorted(set(candidates))

            run_map, run_lengths = self._build_list_run_map(sentences)
            overlap_by_boundary: dict[int, int] = {}

            filtered: list[int] = []
            last_idx = 0
            for idx in candidates:
                if idx - last_idx < self.min_chunk_sentences:
                    continue
                skip_candidate = False
                run = run_map.get(idx - 1)
                if run is not None and idx < run[1]:
                    run_len = run_lengths.get(run, 0)
                    if run_len <= self.max_chunk_sentences:
                        logger.debug(
                            "agentic_chunker_list_boundary_skipped",
                            extra={
                                "boundary_idx": idx,
                                "run_start": run[0],
                                "run_end": run[1],
                                "run_len": run_len,
                            },
                        )
                        continue
                    overlap_by_boundary[idx] = 1
                while idx - last_idx > self.max_chunk_sentences:
                    insert_idx = last_idx + self.max_chunk_sentences
                    run = run_map.get(insert_idx - 1)
                    if run is not None and insert_idx < run[1]:
                        run_len = run_lengths.get(run, 0)
                        if run_len <= self.max_chunk_sentences:
                            insert_idx = run[1]
                        else:
                            overlap_by_boundary[insert_idx] = 1
                    if insert_idx <= last_idx:
                        break
                    filtered.append(insert_idx)
                    last_idx = insert_idx
                    if insert_idx >= idx:
                        skip_candidate = True
                        break
                if skip_candidate:
                    continue
                filtered.append(idx)
                last_idx = idx

            while total_sentences - last_idx > self.max_chunk_sentences:
                insert_idx = last_idx + self.max_chunk_sentences
                run = run_map.get(insert_idx - 1)
                if run is not None and insert_idx < run[1]:
                    run_len = run_lengths.get(run, 0)
                    if run_len <= self.max_chunk_sentences:
                        insert_idx = run[1]
                    else:
                        overlap_by_boundary[insert_idx] = 1
                if insert_idx <= last_idx:
                    break
                filtered.append(insert_idx)
                last_idx = insert_idx

            if filtered and total_sentences - filtered[-1] < self.min_chunk_sentences:
                filtered.pop()

            return sorted(set(filtered)), overlap_by_boundary

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = llm_client.call(
                    self.model,
                    prompt,
                    metadata,
                    response_format={"type": "json_object"},
                    timeout_s=self.timeout,
                )
                payload = _extract_payload(str(response.get("text") or ""))
                parsed = BoundaryDetectionResponse.model_validate(payload)
                boundaries = _normalize_boundaries(parsed)
                boundary_list, _overlap_by_boundary = boundaries
                logger.info(
                    "agentic_chunker_boundaries_detected",
                    extra={
                        "boundary_count": len(boundary_list),
                        "sentence_count": len(sentences),
                        "document_id": str(context.metadata.document_id),
                    },
                )
                return boundaries
            except RateLimitError as exc:
                logger.warning(
                    "agentic_chunker_llm_rate_limited",
                    extra={
                        "document_id": str(context.metadata.document_id),
                        "error": str(exc),
                    },
                )
                raise
            except (
                LlmClientError,
                LlmTimeoutError,
                ValueError,
                ValidationError,
            ) as exc:
                last_error = exc
                logger.warning(
                    "agentic_chunker_llm_attempt_failed",
                    extra={
                        "attempt": attempt + 1,
                        "max_retries": self.max_retries,
                        "document_id": str(context.metadata.document_id),
                        "error": str(exc),
                    },
                )
                if attempt < self.max_retries:
                    time.sleep(1)
                    continue
                break

        if last_error is not None:
            raise last_error
        raise RuntimeError("agentic chunker failed without exception")

    def _build_chunks_from_boundaries(
        self,
        sentences: list[str],
        boundaries: list[int] | tuple[list[int], dict[int, int]],
        context: DocumentProcessingContext,
        document_title: str | None = None,
        document_ref: str | None = None,
        doc_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Build chunks from detected boundaries.

        Args:
            sentences: List of sentences
            boundaries: (boundary indices, overlap map)
            context: Document processing context

        Returns:
            List of chunk dictionaries
        """
        chunks = []

        namespace_key = self._resolve_chunk_namespace_key(context)

        # Add start and end boundaries
        if isinstance(boundaries, tuple):
            boundary_list, overlap_by_boundary = boundaries
        else:
            boundary_list = boundaries
            overlap_by_boundary = {}
        all_boundaries = sorted([0] + boundary_list + [len(sentences)])
        merged_boundaries = [all_boundaries[0]]
        idx = 1
        while idx < len(all_boundaries):
            start_idx = merged_boundaries[-1]
            end_idx = all_boundaries[idx]
            if idx < len(all_boundaries) - 1:
                chunk_text = " ".join(sentences[start_idx:end_idx]).strip()
                if self._is_heading_only_chunk(chunk_text):
                    idx += 1
                    merged_boundaries.append(all_boundaries[idx])
                    idx += 1
                    continue
            merged_boundaries.append(end_idx)
            idx += 1
        all_boundaries = merged_boundaries
        run_map, _run_lengths = self._build_list_run_map(sentences)

        total_chunks = len(all_boundaries) - 1
        for i in range(total_chunks):
            start_idx = all_boundaries[i]
            end_idx = all_boundaries[i + 1]

            # Extract chunk sentences with optional overlap for list splits
            overlap = overlap_by_boundary.get(start_idx, 0)
            if overlap:
                start_idx = max(0, start_idx - overlap)
            chunk_sentences = sentences[start_idx:end_idx]
            chunk_text = " ".join(chunk_sentences)

            list_header = self._resolve_list_header(sentences, run_map, start_idx)
            chunk_position = self._format_chunk_position(i, total_chunks)
            prefix = build_chunk_prefix(
                document_ref=document_ref or document_title,
                doc_type=doc_type,
                section_path=None,
                chunk_position=chunk_position,
                list_header=list_header,
            )
            if i == 0 and prefix:
                self._log_prefix_sample(
                    document_id=str(context.metadata.document_id),
                    document_ref=document_ref or document_title,
                    doc_type=doc_type,
                    prefix=prefix,
                    section_path=(),
                    chunk_position=chunk_position,
                )
            chunk_text = f"{prefix}{chunk_text}" if prefix else chunk_text

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

    @staticmethod
    def _resolve_document_title(document: Any | None) -> str | None:
        if document is None:
            return None
        if isinstance(document, dict):
            meta = document.get("meta") or {}
            title = meta.get("title") or document.get("title")
            return str(title).strip() if title else None
        meta = getattr(document, "meta", None)
        title = getattr(meta, "title", None) if meta is not None else None
        if title:
            return str(title).strip()
        return None

    @staticmethod
    def _resolve_document_ref(document: Any | None) -> str | None:
        if document is None:
            return None
        if isinstance(document, dict):
            meta = document.get("meta") or {}
            for key in ("document_ref", "doc_ref", "external_id", "ref", "title"):
                value = meta.get(key) or document.get(key)
                if value:
                    return str(value).strip()
            return None
        meta = getattr(document, "meta", None)
        for key in ("document_ref", "doc_ref", "external_id", "ref", "title"):
            value = getattr(meta, key, None) if meta is not None else None
            if value:
                return str(value).strip()
        return None

    @staticmethod
    def _resolve_doc_type(document: Any | None) -> str | None:
        if document is None:
            return None
        if isinstance(document, dict):
            meta = document.get("meta") or {}
            for key in ("doc_type", "doc_class", "document_type", "type"):
                value = meta.get(key) or document.get(key)
                if value:
                    return str(value).strip()
            return None
        meta = getattr(document, "meta", None)
        for key in ("doc_type", "doc_class", "document_type", "type"):
            value = getattr(meta, key, None) if meta is not None else None
            if value:
                return str(value).strip()
        return None

    @staticmethod
    def _format_chunk_position(
        index: int, total: int, *, label: str = "Chunk"
    ) -> str | None:
        if total <= 0:
            return None
        return f"{label} {index + 1} von {total}"

    @staticmethod
    def _log_prefix_sample(
        *,
        document_id: str,
        document_ref: str | None,
        doc_type: str | None,
        prefix: str,
        section_path: Sequence[str],
        chunk_position: str | None,
    ) -> None:
        if not prefix.strip():
            return
        try:
            logger.info(
                "ingestion.chunk.prefix_sample",
                extra={
                    "document_id": document_id,
                    "document_ref": document_ref,
                    "doc_type": doc_type,
                    "section_path": list(section_path),
                    "chunk_position": chunk_position,
                    "prefix": prefix.strip(),
                },
            )
        except Exception:
            pass

    @staticmethod
    def _apply_chunk_counts(
        chunks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        total = len(chunks)
        for idx, chunk in enumerate(chunks):
            meta = chunk.get("metadata")
            if not isinstance(meta, dict):
                meta = dict(meta) if isinstance(meta, Mapping) else {}
                chunk["metadata"] = meta
            meta["chunk_index"] = idx
            meta["chunk_count"] = total
        return chunks

    def _apply_contextual_enrichment(
        self,
        chunks: list[dict[str, Any]],
        document_text: str,
        context: DocumentProcessingContext,
    ) -> list[dict[str, Any]]:
        if not chunks:
            return chunks
        if self._has_contextual_prefix(chunks):
            return chunks
        config = resolve_contextual_enrichment_config(True)
        if not config.enabled:
            return chunks
        started_at = time.perf_counter()
        entries = [
            {"text": str(chunk.get("text") or ""), "metadata": chunk.get("metadata")}
            for chunk in chunks
        ]
        prefixes = generate_contextual_prefixes(
            document_text,
            entries,
            context,
            config,
        )
        duration_ms = int(round((time.perf_counter() - started_at) * 1000))
        logger.info(
            "agentic_chunker_contextual_enrichment",
            extra={
                "document_id": str(context.metadata.document_id),
                "chunk_count": len(chunks),
                "duration_ms": duration_ms,
            },
        )
        for chunk, prefix in zip(chunks, prefixes):
            if not prefix:
                continue
            meta = chunk.get("metadata")
            if not isinstance(meta, dict):
                meta = dict(meta) if isinstance(meta, Mapping) else {}
                chunk["metadata"] = meta
            meta["contextual_prefix"] = prefix
        return chunks

    @staticmethod
    def _has_contextual_prefix(chunks: Sequence[Mapping[str, Any]]) -> bool:
        for chunk in chunks:
            meta = chunk.get("metadata")
            if not isinstance(meta, Mapping):
                continue
            value = meta.get("contextual_prefix")
            if isinstance(value, str) and value.strip():
                return True
        return False

    @staticmethod
    def _build_list_run_map(
        sentences: list[str],
    ) -> tuple[dict[int, tuple[int, int]], dict[tuple[int, int], int]]:
        runs = find_numbered_list_runs(sentences)
        run_map: dict[int, tuple[int, int]] = {}
        run_lengths: dict[tuple[int, int], int] = {}
        for start, end in runs:
            run_lengths[(start, end)] = end - start
            for idx in range(start, end):
                run_map[idx] = (start, end)
        return run_map, run_lengths

    @staticmethod
    def _resolve_list_header(
        sentences: list[str],
        run_map: dict[int, tuple[int, int]],
        start_idx: int,
    ) -> str | None:
        run = run_map.get(start_idx)
        if not run:
            return None
        run_start, _run_end = run
        if start_idx <= run_start:
            return None
        header_idx = run_start - 1
        if header_idx < 0:
            return None
        candidate = sentences[header_idx].strip()
        if candidate and not is_numbered_list_item(candidate):
            return candidate
        item_number = extract_numbered_list_index(sentences[start_idx])
        if item_number is None:
            return "Fortsetzung"
        return f"Fortsetzung (Punkt {item_number})"

    @staticmethod
    def _is_heading_only_chunk(text: str) -> bool:
        if not text:
            return False
        if len(text) > 140:
            return False
        if "\n" in text:
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            if len(lines) > 3:
                return False
        if any(punct in text for punct in ".!?"):
            return False
        return True


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
        enable_contextual_enrichment=getattr(
            settings, "RAG_CONTEXTUAL_ENRICHMENT", False
        ),
    )
