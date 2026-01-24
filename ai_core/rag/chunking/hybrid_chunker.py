"""Hybrid Chunker combining Late and Agentic chunking strategies.

Main implementation of the SOTA chunking system for NOESIS 2 RAG.

Modes:
- LATE (default): Embed full document first, then chunk with contextual embeddings
- AGENTIC (Phase 2): LLM-driven boundary detection with automatic fallback to Late
- HYBRID: Auto-select based on document characteristics (future)

Integrates with:
- RAG Routing Rules for mode selection
- Quality Metrics (LLM-as-Judge) for chunk evaluation
- DocumentChunker protocol for universal_ingestion_graph
- Automatic fallback: AgenticChunker → LateChunker on failure/rate limits/budget exceeded
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence, Tuple

from ai_core.rag.chunking.utils import split_sentences
from common.logging import get_logger

logger = get_logger(__name__)


class ChunkerMode(str, Enum):
    """Chunker mode selection."""

    LATE = "late"  # Late Chunking (default)
    AGENTIC = "agentic"  # Agentic Chunking (Phase 2)
    HYBRID = "hybrid"  # Auto-select (future)


@dataclass(frozen=True)
class ChunkerConfig:
    """Configuration for HybridChunker (uses MODEL_ROUTING.yaml labels)."""

    mode: ChunkerMode = ChunkerMode.LATE
    late_chunk_model: str = "embedding"  # MODEL_ROUTING.yaml label
    late_chunk_max_tokens: int = 8000
    agentic_chunk_model: str = "agentic-chunk"  # MODEL_ROUTING.yaml label
    enable_quality_metrics: bool = True
    enable_contextual_enrichment: bool = False
    quality_model: str = "quality-eval"  # MODEL_ROUTING.yaml label
    quality_max_workers: int = 8  # Parallel workers for quality evaluation
    quality_sample_rate: float = 1.0  # Evaluate all chunks (0.0-1.0)
    quality_timeout: int = 60  # Timeout per chunk evaluation (seconds)
    max_chunk_tokens: int = 450
    overlap_tokens: int = 80
    similarity_threshold: float = 0.7

    # Embedding-based similarity (default on)
    use_embedding_similarity: bool = True
    allow_jaccard_fallback: bool = False
    window_size: int = 3  # Sentences per window for embedding
    batch_size: int = 16  # Windows to embed in parallel
    use_content_based_ids: bool = True  # SHA256-hash IDs (deterministic)
    adaptive_chunking_enabled: bool = True  # Adaptive, structure-first late chunking
    asset_chunks_enabled: bool = True  # Emit asset-derived chunks


def get_default_chunker_config() -> ChunkerConfig:
    """Get default chunker configuration from settings."""
    from django.conf import settings

    return ChunkerConfig(
        mode=ChunkerMode(getattr(settings, "RAG_CHUNKER_MODE", "late")),
        late_chunk_model=getattr(settings, "RAG_LATE_CHUNK_MODEL", "embedding"),
        late_chunk_max_tokens=getattr(settings, "RAG_LATE_CHUNK_MAX_TOKENS", 8000),
        agentic_chunk_model=getattr(
            settings, "RAG_AGENTIC_CHUNK_MODEL", "agentic-chunk"
        ),
        enable_quality_metrics=getattr(settings, "RAG_ENABLE_QUALITY_METRICS", True),
        enable_contextual_enrichment=getattr(
            settings, "RAG_CONTEXTUAL_ENRICHMENT", False
        ),
        quality_model=getattr(settings, "RAG_QUALITY_EVAL_MODEL", "quality-eval"),
        quality_max_workers=getattr(settings, "RAG_QUALITY_MAX_WORKERS", 8),
        quality_sample_rate=getattr(settings, "RAG_QUALITY_SAMPLE_RATE", 1.0),
        quality_timeout=getattr(settings, "RAG_QUALITY_TIMEOUT", 60),
        max_chunk_tokens=getattr(settings, "RAG_CHUNK_TARGET_TOKENS", 450),
        overlap_tokens=getattr(settings, "RAG_CHUNK_OVERLAP_TOKENS", 80),
        # Embedding-based similarity
        use_embedding_similarity=getattr(
            settings, "RAG_USE_EMBEDDING_SIMILARITY", True
        ),
        allow_jaccard_fallback=getattr(settings, "RAG_ALLOW_JACCARD_FALLBACK", False),
        window_size=getattr(settings, "RAG_CHUNKING_WINDOW_SIZE", 3),
        batch_size=getattr(settings, "RAG_CHUNKING_BATCH_SIZE", 16),
        use_content_based_ids=getattr(settings, "RAG_USE_CONTENT_BASED_IDS", True),
        adaptive_chunking_enabled=getattr(
            settings, "RAG_ADAPTIVE_CHUNKING_ENABLED", True
        ),
        asset_chunks_enabled=getattr(settings, "RAG_ASSET_CHUNKS_ENABLED", True),
    )


def get_chunker_config_from_routing(
    *,
    tenant_id: str,
    collection_id: str | None = None,
    doc_class: str | None = None,
    workflow_id: str | None = None,
    process: str | None = None,
) -> ChunkerConfig:
    """
    Get chunker configuration with routing-based mode selection.

    Resolves chunker mode from RAG routing rules (config/rag_routing_rules.yaml)
    based on tenant, collection, doc_class, workflow, and process context.

    All other configuration values are loaded from Django settings (same as
    get_default_chunker_config).

    Args:
        tenant_id: Tenant identifier (required)
        collection_id: Optional collection identifier
        doc_class: Optional document class
        workflow_id: Optional workflow identifier
        process: Optional process name

    Returns:
        ChunkerConfig with routing-resolved mode

    Example:
        >>> from ai_core.rag.chunking import get_chunker_config_from_routing
        >>> config = get_chunker_config_from_routing(
        ...     tenant_id="enterprise",
        ...     collection_id="legal-contracts",
        ...     doc_class="legal_contract"
        ... )
        >>> config.mode  # ChunkerMode.AGENTIC (from routing rules)
    """
    from django.conf import settings

    from ai_core.rag.routing_rules import (
        resolve_chunker_mode,
        resolve_contextual_enrichment,
    )

    # Resolve mode from routing rules
    mode_str = resolve_chunker_mode(
        tenant=tenant_id,
        collection_id=collection_id,
        doc_class=doc_class,
        workflow_id=workflow_id,
        process=process,
    )
    enrichment_override = resolve_contextual_enrichment(
        tenant=tenant_id,
        collection_id=collection_id,
        doc_class=doc_class,
        workflow_id=workflow_id,
        process=process,
    )

    # Load all other config from Django settings
    return ChunkerConfig(
        mode=ChunkerMode(mode_str),
        late_chunk_model=getattr(settings, "RAG_LATE_CHUNK_MODEL", "embedding"),
        late_chunk_max_tokens=getattr(settings, "RAG_LATE_CHUNK_MAX_TOKENS", 8000),
        agentic_chunk_model=getattr(
            settings, "RAG_AGENTIC_CHUNK_MODEL", "agentic-chunk"
        ),
        enable_quality_metrics=getattr(settings, "RAG_ENABLE_QUALITY_METRICS", True),
        enable_contextual_enrichment=(
            enrichment_override
            if enrichment_override is not None
            else getattr(settings, "RAG_CONTEXTUAL_ENRICHMENT", False)
        ),
        quality_model=getattr(settings, "RAG_QUALITY_EVAL_MODEL", "quality-eval"),
        max_chunk_tokens=getattr(settings, "RAG_CHUNK_TARGET_TOKENS", 450),
        overlap_tokens=getattr(settings, "RAG_CHUNK_OVERLAP_TOKENS", 80),
        # Embedding-based similarity
        use_embedding_similarity=getattr(
            settings, "RAG_USE_EMBEDDING_SIMILARITY", True
        ),
        allow_jaccard_fallback=getattr(settings, "RAG_ALLOW_JACCARD_FALLBACK", False),
        window_size=getattr(settings, "RAG_CHUNKING_WINDOW_SIZE", 3),
        batch_size=getattr(settings, "RAG_CHUNKING_BATCH_SIZE", 16),
        use_content_based_ids=getattr(settings, "RAG_USE_CONTENT_BASED_IDS", True),
        adaptive_chunking_enabled=getattr(
            settings, "RAG_ADAPTIVE_CHUNKING_ENABLED", True
        ),
        asset_chunks_enabled=getattr(settings, "RAG_ASSET_CHUNKS_ENABLED", True),
    )


class HybridChunker:
    """
    Hybrid Chunker implementing DocumentChunker protocol.

    Orchestrates Late and Agentic chunking modes with quality evaluation.

    Example:
        >>> from ai_core.rag.chunking import HybridChunker, get_default_chunker_config
        >>> chunker = HybridChunker(get_default_chunker_config())
        >>> chunks, stats = chunker.chunk(document, parsed, context=context, config=config)
    """

    def __init__(self, config: ChunkerConfig):
        """Initialize HybridChunker with configuration."""
        self.config = config

        # Initialize Late Chunker
        from .late_chunker import LateChunker

        self.late_chunker = LateChunker(
            model=config.late_chunk_model,
            max_tokens=config.late_chunk_max_tokens,
            target_tokens=config.max_chunk_tokens,
            overlap_tokens=config.overlap_tokens,
            similarity_threshold=config.similarity_threshold,
            # Embedding-based similarity
            use_embedding_similarity=config.use_embedding_similarity,
            allow_jaccard_fallback=config.allow_jaccard_fallback,
            window_size=config.window_size,
            batch_size=config.batch_size,
            use_content_based_ids=config.use_content_based_ids,
            adaptive_enabled=config.adaptive_chunking_enabled,
            asset_chunks_enabled=config.asset_chunks_enabled,
            enable_contextual_enrichment=config.enable_contextual_enrichment,
        )

        # Initialize Agentic Chunker (Phase 2)
        from .agentic_chunker import AgenticChunker

        self.agentic_chunker = AgenticChunker(
            model=config.agentic_chunk_model,
            use_content_based_ids=config.use_content_based_ids,
            enable_contextual_enrichment=config.enable_contextual_enrichment,
        )

        # Initialize Quality Evaluator (if enabled)
        if config.enable_quality_metrics:
            from ai_core.rag.quality.llm_judge import ChunkQualityEvaluator

            self.quality_evaluator = ChunkQualityEvaluator(
                model=config.quality_model,
                max_workers=config.quality_max_workers,
                sample_rate=config.quality_sample_rate,
                timeout=config.quality_timeout,
            )
        else:
            self.quality_evaluator = None

        logger.info(
            "hybrid_chunker_initialized",
            extra={
                "mode": config.mode.value,
                "enable_quality_metrics": config.enable_quality_metrics,
            },
        )

    def chunk(
        self,
        document: Any,  # NormalizedDocument (unused but part of protocol)
        parsed: Any,  # ParsedResult
        *,
        context: Any,  # DocumentProcessingContext
        config: Any,  # DocumentPipelineConfig
    ) -> Tuple[Sequence[Mapping[str, Any]], Mapping[str, Any]]:
        """
        Chunk document using selected strategy.

        Implements DocumentChunker protocol from documents/pipeline.py:481-492.

        Args:
            document: NormalizedDocument (unused)
            parsed: ParsedResult with text_blocks
            context: DocumentProcessingContext with metadata
            config: DocumentPipelineConfig (unused for now)

        Returns:
            (chunks, statistics) where:
            - chunks: List of chunk dicts with chunk_id, text, parent_ref, metadata
            - statistics: Dict with chunk.count, chunker.mode, quality.scores, etc.
        """
        _ = (document, config)  # Unused but part of protocol

        # 1. Resolve chunker mode
        mode = self._resolve_mode(context, config)

        # 2. Execute chunking
        if mode == ChunkerMode.AGENTIC:
            # Phase 2: Agentic Chunking with automatic fallback to Late
            chunks = self._chunk_agentic(parsed, context, document)
            actual_mode = "agentic"
        else:
            chunks = self._chunk_late(parsed, context, document)
            actual_mode = "late"

        # 3. Evaluate quality (Phase 1)
        quality_scores = []
        quality_feedback: list[dict[str, object]] = []
        if self.quality_evaluator:
            try:
                quality_scores = self.quality_evaluator.evaluate(chunks, context)
                quality_feedback = [score.to_dict() for score in quality_scores]

                # Add quality metadata to chunks
                chunks = self.quality_evaluator.add_quality_to_chunks(
                    chunks, quality_scores
                )

                logger.info(
                    "hybrid_chunker_quality_evaluated",
                    extra={
                        "document_id": str(context.metadata.document_id),
                        "chunk_count": len(chunks),
                        "mean_overall": (
                            sum(s.overall for s in quality_scores) / len(quality_scores)
                            if quality_scores
                            else 0
                        ),
                    },
                )

            except Exception as exc:
                logger.error(
                    "hybrid_chunker_quality_failed",
                    extra={
                        "document_id": str(context.metadata.document_id),
                        "error": str(exc),
                    },
                    exc_info=exc,
                )

        # 4. Build statistics
        stats = self._build_statistics(chunks, quality_scores, actual_mode)

        sentence_count = self._count_sentences(parsed)
        chunk_summaries = self._summarize_chunks(chunks)
        document_ref = self._resolve_document_ref(document)
        doc_type = self._resolve_doc_type(document)

        logger.info(
            "hybrid_chunker_completed",
            document_id=str(context.metadata.document_id),
            chunk_count=len(chunks),
            chunker_mode=actual_mode,
            sentence_count=sentence_count,
            chunks=chunk_summaries,
            model_feedback=quality_feedback,
            scores=quality_feedback,
            statistics=stats,
            document_ref=document_ref,
            doc_type=doc_type,
        )

        return chunks, stats

    def _resolve_mode(
        self,
        context: Any,
        config: Any,
    ) -> ChunkerMode:
        """
        Resolve chunker mode from routing rules.

        For now, use configured mode. In Phase 3, this will query RAG Routing Rules.
        """
        # TODO Phase 3: Query routing rules
        # from ai_core.rag.routing_rules import resolve_chunker_mode
        # mode = resolve_chunker_mode(
        #     tenant=context.metadata.tenant_id,
        #     doc_class=getattr(context.metadata, "doc_class", None),
        #     collection_id=getattr(context.metadata, "collection_id", None),
        # )

        return self.config.mode

    def _chunk_late(
        self,
        parsed: Any,
        context: Any,
        document: Any,
    ) -> list[Mapping[str, Any]]:
        """Execute Late Chunking."""
        return self.late_chunker.chunk(parsed, context, document=document)

    def _chunk_agentic(
        self,
        parsed: Any,
        context: Any,
        document: Any,
    ) -> list[Mapping[str, Any]]:
        """
        Execute Agentic Chunking with automatic fallback to Late.

        AgenticChunker handles fallback internally (rate limits, budget, LLM failures).
        """
        return self.agentic_chunker.chunk(parsed, context, document=document)

    def _build_statistics(
        self,
        chunks: list,
        quality_scores: list,
        mode: str,
    ) -> dict:
        """Build statistics dict for return value."""
        stats = {
            "chunk.count": len(chunks),
            "chunker.mode": mode,
            "chunker.version": "hybrid-v1",
        }

        # Add quality statistics
        if quality_scores:
            from ai_core.rag.quality.llm_judge import compute_quality_statistics

            quality_stats = compute_quality_statistics(quality_scores)

            stats["quality.mean_coherence"] = quality_stats["mean_coherence"]
            stats["quality.mean_completeness"] = quality_stats["mean_completeness"]
            stats["quality.mean_overall"] = quality_stats["mean_overall"]
            stats["quality.min_overall"] = quality_stats["min_overall"]
            stats["quality.max_overall"] = quality_stats["max_overall"]

        return stats

    @staticmethod
    def _count_sentences(parsed: Any) -> int:
        total = 0
        text_blocks = getattr(parsed, "text_blocks", None) or []
        for block in text_blocks:
            text = getattr(block, "text", "")
            if not text:
                continue
            total += len(split_sentences(str(text)))
        return total

    @staticmethod
    def _summarize_chunks(
        chunks: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, object]]:
        summaries: list[dict[str, object]] = []
        for chunk in chunks:
            chunk_id = chunk.get("chunk_id")
            text_value = chunk.get("text")
            metadata = chunk.get("metadata")
            entry = {
                "chunk_id": chunk_id,
                "text_length": len(text_value) if isinstance(text_value, str) else None,
            }
            if isinstance(metadata, Mapping):
                if "chunk_index" in metadata:
                    entry["chunk_index"] = metadata.get("chunk_index")
                if "sentence_count" in metadata:
                    entry["sentence_count"] = metadata.get("sentence_count")
            summaries.append(entry)
        return summaries

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


class RoutingAwareChunker:
    """
    Routing-aware chunker wrapper for tenant/collection/doc_class-specific configuration.

    This wrapper resolves chunker configuration from RAG routing rules based on
    the request context (tenant_id, collection_id, workflow_id), then delegates
    to HybridChunker with the appropriate config.

    Caches HybridChunker instances by config hash to avoid re-instantiation overhead.

    Example:
        >>> from ai_core.rag.chunking import RoutingAwareChunker
        >>> chunker = RoutingAwareChunker()
        >>> # chunker.chunk() will resolve config from context.metadata at call time
    """

    def __init__(self):
        """Initialize RoutingAwareChunker with empty cache."""
        self._chunker_cache: dict[str, HybridChunker] = {}

    def chunk(
        self,
        document: Any,  # NormalizedDocument
        parsed: Any,  # ParsedResult
        *,
        context: Any,  # DocumentProcessingContext
        config: Any,  # DocumentPipelineConfig
    ) -> Tuple[Sequence[Mapping[str, Any]], Mapping[str, Any]]:
        """
        Chunk document using routing-resolved configuration.

        Resolves chunker mode from RAG routing rules based on context.metadata,
        then delegates to HybridChunker with the appropriate config.

        Args:
            document: NormalizedDocument
            parsed: ParsedResult with text_blocks
            context: DocumentProcessingContext with metadata (tenant_id, collection_id, workflow_id)
            config: DocumentPipelineConfig

        Returns:
            (chunks, statistics) where:
            - chunks: List of chunk dicts with chunk_id, text, parent_ref, metadata
            - statistics: Dict with chunk.count, chunker.mode, quality.scores, routing.tenant_id, etc.
        """
        # Extract routing parameters from context.metadata
        metadata = context.metadata
        tenant_id = metadata.tenant_id
        collection_id = str(metadata.collection_id) if metadata.collection_id else None
        workflow_id = metadata.workflow_id

        # Resolve chunker config from routing rules
        chunker_config = get_chunker_config_from_routing(
            tenant_id=tenant_id,
            collection_id=collection_id,
            workflow_id=workflow_id,
            # doc_class and process not available in metadata (future enhancement)
        )

        # Create cache key from config (use mode as simple key for MVP)
        cache_key = f"{tenant_id}:{collection_id}:{chunker_config.mode.value}"

        # Get or create HybridChunker for this config
        if cache_key not in self._chunker_cache:
            logger.debug(
                "routing_aware_chunker_create",
                extra={
                    "tenant_id": tenant_id,
                    "collection_id": collection_id,
                    "mode": chunker_config.mode.value,
                    "cache_key": cache_key,
                },
            )
            self._chunker_cache[cache_key] = HybridChunker(chunker_config)

        # Delegate to cached HybridChunker
        chunker = self._chunker_cache[cache_key]
        chunks, stats = chunker.chunk(document, parsed, context=context, config=config)

        # Add routing metadata to stats
        stats["routing.tenant_id"] = tenant_id
        stats["routing.collection_id"] = collection_id
        stats["routing.workflow_id"] = workflow_id
        stats["routing.cache_key"] = cache_key

        return chunks, stats
