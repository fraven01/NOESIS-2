"""Late Chunking implementation for NOESIS 2 RAG system.

Late Chunking embeds full documents first, then chunks based on contextual embeddings.
This preserves semantic context lost in traditional "chunk-then-embed" approaches.

Key Features:
- Full document embedding with long-context models (8k tokens)
- Boundary detection using embedding similarity
- Contextual embedding preservation in chunk metadata
- Fallback splitting for documents exceeding token limits

Default Model: "embedding" label from MODEL_ROUTING.yaml
  Dev: oai-embed-small (OpenAI text-embedding-3-small, 8k context, 1536D)
  Prod: oai-embed-large (OpenAI text-embedding-3-large, 8k context, 3072D)
"""

from __future__ import annotations

import hashlib
import time
import logging
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Sequence, Tuple
from uuid import UUID, uuid5

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

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LateBoundary:
    """Represents a chunk boundary detected via embedding similarity."""

    start_idx: int  # Sentence index (start)
    end_idx: int  # Sentence index (end, exclusive)
    similarity_score: float  # Similarity with next segment (lower = better boundary)


@dataclass(frozen=True)
class SentenceWindow:
    """Represents a sliding window of sentences for embedding-based similarity.

    Phase 2: SOTA embedding-based boundary detection uses overlapping windows
    of sentences (e.g., window_size=3 means 3 sentences per window).

    Example:
        Sentences: ["A", "B", "C", "D", "E"]
        Window size: 3
        Windows:
            - SentenceWindow(start_idx=0, end_idx=3, text="A B C")
            - SentenceWindow(start_idx=1, end_idx=4, text="B C D")
            - SentenceWindow(start_idx=2, end_idx=5, text="C D E")
    """

    start_idx: int  # First sentence index in window
    end_idx: int  # Last sentence index + 1 (exclusive)
    text: str  # Combined text of all sentences in window
    embedding: List[float] | None = None  # Optional embedding vector


@dataclass(frozen=True)
class TextSegment:
    """Structured text segment derived from parsed blocks."""

    index: int
    text: str
    kind: str
    section_path: Tuple[str, ...]
    page_index: Optional[int] = None


@dataclass(frozen=True)
class TextRun:
    """Contiguous run of segments sharing the same section path."""

    index: int
    section_path: Tuple[str, ...]
    segments: Tuple[TextSegment, ...]
    text: str
    page_index: Optional[int] = None


class LateChunker:
    """
    Late Chunking: Embed full document first, then chunk with contextual embeddings.

    Algorithm:
      1. Embed full document with long-context model
      2. Split text into sentences
      3. Detect chunk boundaries using embedding similarity
      4. Preserve contextual embeddings in chunk metadata

    Example:
        >>> from ai_core.rag.chunking import LateChunker
        >>> chunker = LateChunker(
        ...     model="embedding",  # MODEL_ROUTING.yaml label
        ...     max_tokens=8000,
        ...     target_tokens=450,
        ... )
        >>> chunks = chunker.chunk(parsed_result, context)
    """

    def __init__(
        self,
        *,
        model: str = "embedding",  # MODEL_ROUTING.yaml label
        max_tokens: int = 8000,
        target_tokens: int = 450,
        overlap_tokens: int = 80,
        similarity_threshold: float = 0.7,
        dimension: int = 1536,
        # Embedding-based Similarity (default on)
        use_embedding_similarity: bool = True,
        allow_jaccard_fallback: bool = False,
        window_size: int = 3,
        batch_size: int = 16,
        use_content_based_ids: bool = True,
        adaptive_enabled: bool = True,
        asset_chunks_enabled: bool = True,
        enable_contextual_enrichment: bool = False,
    ):
        """
        Initialize Late Chunker.

        Args:
            model: Embedding model to use (must be in litellm-config.yaml)
            max_tokens: Maximum tokens for full document embedding
            target_tokens: Target chunk size in tokens
            overlap_tokens: Overlap between chunks in tokens
            similarity_threshold: Similarity threshold for boundary detection (lower = more boundaries)
            dimension: Embedding dimension (must match model)
            use_embedding_similarity: Enable embedding-based window similarity (default: True)
            allow_jaccard_fallback: Allow Jaccard fallback when embedding fails (default: False)
            window_size: Sentences per window for embedding (Phase 2, default: 3)
            batch_size: Windows to embed in parallel (Phase 2, default: 16)
            use_content_based_ids: Use SHA256-based chunk IDs for determinism (Phase 2, default: True)
            adaptive_enabled: Enable adaptive, structure-first chunking (default: True)
            asset_chunks_enabled: Emit asset-derived chunks when available (default: True)
        """
        self.model = model
        self.max_tokens = max_tokens
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        self.similarity_threshold = similarity_threshold
        self.dimension = dimension

        # Embedding-based similarity
        self.use_embedding_similarity = use_embedding_similarity
        self.allow_jaccard_fallback = allow_jaccard_fallback
        self.window_size = window_size
        self.batch_size = batch_size
        self.use_content_based_ids = use_content_based_ids
        self.adaptive_enabled = adaptive_enabled
        self.asset_chunks_enabled = asset_chunks_enabled
        self.enable_contextual_enrichment = enable_contextual_enrichment

    def chunk(
        self,
        parsed: Any,  # ParsedResult
        context: Any,  # DocumentProcessingContext
        *,
        document: Any | None = None,
    ) -> List[Mapping[str, Any]]:
        """
        Chunk document using Late Chunking strategy.

        Args:
            parsed: ParsedResult with text_blocks
            context: DocumentProcessingContext with metadata

        Returns:
            List of chunk dicts with chunk_id, text, parent_ref, metadata
        """
        document_title = self._resolve_document_title(document)
        document_ref = self._resolve_document_ref(document) or document_title
        doc_type = self._resolve_doc_type(document)
        if not self.adaptive_enabled:
            chunks = self._chunk_legacy(
                parsed,
                context,
                document_title=document_title,
                document_ref=document_ref,
                doc_type=doc_type,
            )
        else:
            chunks = self._chunk_adaptive(
                parsed,
                context,
                document_title=document_title,
                document_ref=document_ref,
                doc_type=doc_type,
            )
        if self.enable_contextual_enrichment:
            full_text, _ = self._build_full_text(parsed)
            chunks = self._apply_contextual_enrichment(
                chunks,
                full_text,
                context,
            )
        return self._apply_chunk_counts(chunks)

    def _chunk_legacy(
        self,
        parsed: Any,
        context: Any,
        *,
        document_title: str | None,
        document_ref: str | None,
        doc_type: str | None,
    ) -> List[Mapping[str, Any]]:
        """Legacy Late Chunking path (kept for backward compatibility)."""
        # 1. Build full text from parsed blocks
        full_text, block_map = self._build_full_text(parsed)

        # 2. Check token limit
        token_count = self._count_tokens(full_text)
        if token_count > self.max_tokens:
            logger.warning(
                "late_chunker_document_too_long",
                extra={
                    "token_count": token_count,
                    "max_tokens": self.max_tokens,
                    "document_id": str(context.metadata.document_id),
                },
            )
            # Split into sections, process each independently
            return self._chunk_by_sections(
                parsed,
                context,
                document_title=document_title,
                document_ref=document_ref,
                doc_type=doc_type,
            )

        # 3. Split into sentences
        sentences = self._split_sentences(full_text)

        # 4. Detect chunk boundaries (embedding-based by default)
        list_run_map, list_run_tokens = self._build_list_run_map(sentences)
        boundaries = self._detect_boundaries(
            sentences,
            context,
            list_run_map=list_run_map,
            list_run_tokens=list_run_tokens,
        )
        boundaries = self._merge_list_run_boundaries(
            boundaries,
            list_run_map=list_run_map,
            list_run_tokens=list_run_tokens,
        )
        boundaries = self._merge_heading_only_boundaries(sentences, boundaries)

        # 5. Build chunks with metadata
        namespace_id = self._resolve_chunk_namespace_id(context)
        chunks = self._build_chunks(
            sentences=sentences,
            boundaries=boundaries,
            block_map=block_map,
            document_id=namespace_id,
            parsed=parsed,
            document_title=document_title,
            document_ref=document_ref,
            doc_type=doc_type,
            list_run_map=list_run_map,
        )

        logger.info(
            "late_chunker_completed",
            extra={
                "chunk_count": len(chunks),
                "document_id": str(context.metadata.document_id),
                "token_count": token_count,
                "document_ref": document_ref,
                "doc_type": doc_type,
            },
        )

        return chunks

    def _chunk_adaptive(
        self,
        parsed: Any,
        context: Any,
        *,
        document_title: str | None,
        document_ref: str | None,
        doc_type: str | None,
    ) -> List[Mapping[str, Any]]:
        """Adaptive, structure-first chunking with optional asset chunks."""
        full_text, _block_map = self._build_full_text(parsed)
        token_count = self._count_tokens(full_text)

        if token_count > self.max_tokens:
            logger.warning(
                "late_chunker_document_too_long",
                extra={
                    "token_count": token_count,
                    "max_tokens": self.max_tokens,
                    "document_id": str(context.metadata.document_id),
                    "chunker_mode": "adaptive",
                },
            )
            chunks = self._chunk_by_sections_adaptive(
                parsed,
                context,
                document_title=document_title,
                document_ref=document_ref,
                doc_type=doc_type,
            )
        else:
            runs = self._build_text_runs(parsed)
            chunks = []
            for run in runs:
                run_chunks = self._chunk_run(
                    run,
                    context,
                    document_title=document_title,
                    document_ref=document_ref,
                    doc_type=doc_type,
                )
                if run_chunks:
                    chunks.extend(run_chunks)

        if self.asset_chunks_enabled:
            asset_chunks = self._build_asset_chunks(
                parsed,
                context,
                document_title=document_title,
                document_ref=document_ref,
                doc_type=doc_type,
            )
            if asset_chunks:
                chunks.extend(asset_chunks)

        logger.info(
            "late_chunker_completed",
            extra={
                "chunk_count": len(chunks),
                "document_id": str(context.metadata.document_id),
                "token_count": token_count,
                "chunker_mode": "adaptive",
                "document_ref": document_ref,
                "doc_type": doc_type,
            },
        )

        return chunks

    def _build_full_text(self, parsed: Any) -> Tuple[str, dict]:
        """Build full text from parsed blocks, tracking block origins."""
        parts = []
        block_map = {}  # char_offset -> block_info

        current_offset = 0
        for idx, block in enumerate(parsed.text_blocks):
            text = block.text.strip()
            if not text:
                continue

            parts.append(text)
            block_map[current_offset] = {
                "index": idx,
                "kind": block.kind,
                "section_path": block.section_path or (),
                "page_index": block.page_index,
            }

            current_offset += len(text) + 2  # +2 for "\n\n" separator

        return "\n\n".join(parts), block_map

    def _build_text_runs(self, parsed: Any) -> List[TextRun]:
        """Group parsed blocks into structure-aware runs."""
        segments: list[TextSegment] = []
        for idx, block in enumerate(parsed.text_blocks):
            text = block.text.strip()
            if not text:
                continue
            section_path = tuple(block.section_path) if block.section_path else ()
            segments.append(
                TextSegment(
                    index=idx,
                    text=text,
                    kind=str(block.kind),
                    section_path=section_path,
                    page_index=block.page_index,
                )
            )

        if not segments:
            return []

        runs: list[TextRun] = []
        current_segments: list[TextSegment] = []
        current_section = segments[0].section_path
        run_index = 0

        for segment in segments:
            force_boundary = segment.kind == "heading" and current_segments
            if (
                segment.section_path != current_section and current_segments
            ) or force_boundary:
                runs.append(
                    self._finalize_text_run(
                        run_index,
                        current_section,
                        current_segments,
                    )
                )
                run_index += 1
                current_segments = []
                current_section = segment.section_path
            current_segments.append(segment)

        if current_segments:
            runs.append(
                self._finalize_text_run(run_index, current_section, current_segments)
            )

        return runs

    def _merge_short_runs(
        self,
        runs: List[TextRun],
        *,
        min_tokens: int,
    ) -> List[TextRun]:
        if not runs:
            return []
        if min_tokens <= 0:
            return runs

        merged: list[TextRun] = []
        carry_text = ""

        for run in runs:
            run_text = run.text or ""
            if carry_text:
                if run_text:
                    run_text = f"{carry_text}\n\n{run_text}"
                else:
                    run_text = carry_text
            token_count = self._count_tokens(run_text)

            if token_count < min_tokens:
                carry_text = run_text
                continue

            merged.append(
                TextRun(
                    index=run.index,
                    section_path=run.section_path,
                    segments=run.segments,
                    text=run_text,
                    page_index=run.page_index,
                )
            )
            carry_text = ""

        if carry_text:
            if merged:
                last = merged[-1]
                merged[-1] = TextRun(
                    index=last.index,
                    section_path=last.section_path,
                    segments=last.segments,
                    text=f"{last.text}\n\n{carry_text}",
                    page_index=last.page_index,
                )
            else:
                last_run = runs[-1]
                merged.append(
                    TextRun(
                        index=last_run.index,
                        section_path=last_run.section_path,
                        segments=last_run.segments,
                        text=carry_text,
                        page_index=last_run.page_index,
                    )
                )

        return merged

    def _finalize_text_run(
        self,
        index: int,
        section_path: Tuple[str, ...],
        segments: Sequence[TextSegment],
    ) -> TextRun:
        text_parts = [segment.text for segment in segments if segment.text]
        text = "\n\n".join(text_parts)
        page_index = next(
            (
                segment.page_index
                for segment in segments
                if segment.page_index is not None
            ),
            None,
        )
        return TextRun(
            index=index,
            section_path=section_path,
            segments=tuple(segments),
            text=text,
            page_index=page_index,
        )

    def _chunk_run(
        self,
        run: TextRun,
        context: Any,
        *,
        document_title: str | None,
        document_ref: str | None,
        doc_type: str | None,
    ) -> List[Mapping[str, Any]]:
        if not run.text:
            return []
        sentences = self._split_sentences(run.text)
        if not sentences:
            return []
        list_run_map, list_run_tokens = self._build_list_run_map(sentences)
        boundaries = self._detect_boundaries(
            sentences,
            context,
            list_run_map=list_run_map,
            list_run_tokens=list_run_tokens,
        )
        boundaries = self._merge_list_run_boundaries(
            boundaries,
            list_run_map=list_run_map,
            list_run_tokens=list_run_tokens,
        )
        boundaries = self._merge_heading_only_boundaries(sentences, boundaries)
        namespace_id = self._resolve_chunk_namespace_id(context)
        return self._build_chunks_adaptive(
            run=run,
            sentences=sentences,
            boundaries=boundaries,
            document_id=namespace_id,
            document_title=document_title,
            document_ref=document_ref,
            doc_type=doc_type,
            list_run_map=list_run_map,
        )

    def _count_tokens(self, text: str) -> int:
        """Estimate token count (rough heuristic: 1 token ≈ 4 chars)."""
        return len(text) // 4

    def _embed_full_document(self, text: str, context: Any) -> List[float]:
        """
        Embed full document with long-context model.

        TODO (Phase 2): Re-enable this for true embedding-based similarity.
        Currently unused to avoid unnecessary API calls during Phase 1.
        """
        from ai_core.rag.embeddings import get_embedding_client

        client = get_embedding_client()

        try:
            response = client.embed([text])
            embedding = response.vectors[0]  # Already a list

            logger.debug(
                "late_chunker_embed_success",
                extra={
                    "document_id": str(context.metadata.document_id),
                    "embedding_dimension": len(embedding),
                },
            )

            return embedding

        except Exception as exc:
            logger.error(
                "late_chunker_embed_failed",
                extra={
                    "document_id": str(context.metadata.document_id),
                    "error": str(exc),
                },
            )
            raise

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences (simple heuristic for now)."""
        return split_sentences(text)

    def _create_sentence_windows(
        self,
        sentences: List[str],
    ) -> List[SentenceWindow]:
        """
        Create sliding windows of sentences for embedding-based similarity.

        Phase 2: SOTA approach using overlapping windows for semantic boundary detection.

        Args:
            sentences: List of sentence strings

        Returns:
            List of SentenceWindow objects with overlapping sentence groups

        Example:
            Sentences: ["A", "B", "C", "D", "E"]
            Window size: 3
            Result:
                [
                    SentenceWindow(0, 3, "A B C"),
                    SentenceWindow(1, 4, "B C D"),
                    SentenceWindow(2, 5, "C D E"),
                ]
        """
        if len(sentences) == 0:
            return []

        windows = []

        # Create overlapping windows
        for i in range(len(sentences)):
            # Window end index (exclusive)
            end_idx = min(i + self.window_size, len(sentences))

            # Only create window if it has at least 1 sentence
            if end_idx > i:
                window_sentences = sentences[i:end_idx]
                window_text = " ".join(window_sentences)

                window = SentenceWindow(
                    start_idx=i,
                    end_idx=end_idx,
                    text=window_text,
                    embedding=None,  # Will be populated by _embed_windows_batch
                )
                windows.append(window)

        logger.debug(
            "late_chunker_windows_created",
            extra={
                "sentence_count": len(sentences),
                "window_count": len(windows),
                "window_size": self.window_size,
            },
        )

        return windows

    def _embed_windows_batch(
        self,
        windows: List[SentenceWindow],
        context: Any,
    ) -> List[SentenceWindow]:
        """
        Embed windows in batches for efficiency.

        Phase 2: Batch embedding (16 windows in parallel) for performance.

        Args:
            windows: List of SentenceWindow objects without embeddings
            context: DocumentProcessingContext for logging

        Returns:
            List of SentenceWindow objects with embeddings populated
        """
        from ai_core.rag.embeddings import get_embedding_client

        if not windows:
            return []

        client = get_embedding_client()
        embedded_windows = []

        # Process in batches
        for batch_start in range(0, len(windows), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(windows))
            batch = windows[batch_start:batch_end]

            # Extract texts for batch embedding
            texts = [w.text for w in batch]

            try:
                # Batch embed
                response = client.embed(texts)
                embeddings = response.vectors

                # Create new windows with embeddings
                for window, embedding in zip(batch, embeddings):
                    embedded_window = SentenceWindow(
                        start_idx=window.start_idx,
                        end_idx=window.end_idx,
                        text=window.text,
                        embedding=embedding,
                    )
                    embedded_windows.append(embedded_window)

                logger.debug(
                    "late_chunker_batch_embedded",
                    extra={
                        "document_id": str(context.metadata.document_id),
                        "batch_start": batch_start,
                        "batch_size": len(batch),
                        "embedding_dimension": len(embeddings[0]),
                    },
                )

            except Exception as exc:
                text_lengths = [len(text) for text in texts if isinstance(text, str)]
                logger.error(
                    (
                        "late_chunker_batch_embed_failed: %s (type=%s batch_start=%s "
                        "batch_size=%s text_count=%s max_chars=%s total_chars=%s model=%s)"
                    )
                    % (
                        str(exc),
                        type(exc).__name__,
                        batch_start,
                        len(batch),
                        len(texts),
                        max(text_lengths) if text_lengths else 0,
                        sum(text_lengths) if text_lengths else 0,
                        self.model,
                    ),
                    extra={
                        "document_id": str(context.metadata.document_id),
                        "batch_start": batch_start,
                        "batch_size": len(batch),
                        "text_count": len(texts),
                        "max_chars": max(text_lengths) if text_lengths else 0,
                        "total_chars": sum(text_lengths) if text_lengths else 0,
                        "model": self.model,
                        "exc_type": type(exc).__name__,
                        "error": str(exc),
                    },
                    exc_info=True,
                )
                # On failure, fallback to Jaccard similarity by raising exception
                raise

        return embedded_windows

    def _compute_cosine_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float],
    ) -> float:
        """
        Compute cosine similarity between two embedding vectors.

        Phase 2: SOTA embedding-based similarity for semantic boundary detection.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score between -1.0 and 1.0 (typically 0.0 to 1.0)

        Formula:
            cosine_similarity = (A · B) / (||A|| * ||B||)
            where · is dot product and ||A|| is L2 norm
        """
        import math

        # Validate dimensions
        if len(embedding1) != len(embedding2):
            raise ValueError(
                f"Embedding dimension mismatch: {len(embedding1)} vs {len(embedding2)}"
            )

        # Compute dot product
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))

        # Compute L2 norms
        norm1 = math.sqrt(sum(a * a for a in embedding1))
        norm2 = math.sqrt(sum(b * b for b in embedding2))

        # Handle zero vectors
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0

        # Compute cosine similarity
        similarity = dot_product / (norm1 * norm2)

        # Clamp to [-1, 1] to handle floating point errors
        return max(-1.0, min(1.0, similarity))

    def _detect_boundaries(
        self,
        sentences: List[str],
        context: Any = None,
        *,
        list_run_map: dict[int, tuple[int, int]] | None = None,
        list_run_tokens: dict[tuple[int, int], int] | None = None,
    ) -> List[LateBoundary]:
        """
        Detect chunk boundaries using similarity.

        Optional (use_embedding_similarity=False):
          - Uses Jaccard similarity (text-based, deterministic)
          - Only when embedding similarity is explicitly disabled

        Embedding similarity (use_embedding_similarity=True):
          - Uses embedding-based window similarity
          - Semantically aware, captures paraphrases/synonyms
          - Batch embedding for performance

        Strategy:
          - Accumulate sentences until target_tokens is reached
          - Check similarity between current segment and next window
          - Low similarity (<threshold) = good boundary
          - Create chunks with overlap

        Args:
            sentences: List of sentence strings
            context: DocumentProcessingContext (required for Phase 2)

        Returns:
            List of LateBoundary objects marking chunk boundaries
        """
        # Phase 2: Embedding-based similarity
        if self.use_embedding_similarity:
            if context is None:
                raise ValueError("Context required for embedding-based similarity")
            return self._detect_boundaries_embedding(
                sentences,
                context,
                list_run_map=list_run_map,
                list_run_tokens=list_run_tokens,
            )

        # Jaccard similarity (only when explicitly disabled)
        return self._detect_boundaries_jaccard(
            sentences,
            list_run_map=list_run_map,
            list_run_tokens=list_run_tokens,
        )

    def _detect_boundaries_jaccard(
        self,
        sentences: List[str],
        *,
        list_run_map: dict[int, tuple[int, int]] | None = None,
        list_run_tokens: dict[tuple[int, int], int] | None = None,
    ) -> List[LateBoundary]:
        """
        Detect chunk boundaries using Jaccard similarity (Phase 1).

        LIMITATION: Jaccard is NOT semantically aware - it misses:
        - Paraphrases ("mandatory arbitration" vs "binding dispute resolution")
        - Synonyms ("large" vs "big")
        - Legal/technical formulations
        """
        boundaries = []
        current_start = 0
        current_tokens = 0
        run_map = list_run_map or {}
        run_tokens = list_run_tokens or {}

        for i, sentence in enumerate(sentences):
            sentence_tokens = self._count_tokens(sentence)
            current_tokens += sentence_tokens

            # Check if we've reached target size
            if current_tokens >= self.target_tokens:
                run = run_map.get(i)
                in_list_run = run is not None and i + 1 < run[1]
                preserve_list = (
                    in_list_run and run_tokens.get(run, 0) <= self.target_tokens
                )
                if preserve_list:
                    continue
                # Check similarity with next sentence (if exists)
                if i + 1 < len(sentences):
                    similarity = self._compute_sentence_similarity(
                        sentences[current_start : i + 1],
                        [sentences[i + 1]],
                    )

                    # Low similarity = good boundary
                    if similarity < self.similarity_threshold:
                        boundaries.append(
                            LateBoundary(
                                start_idx=current_start,
                                end_idx=i + 1,
                                similarity_score=similarity,
                            )
                        )
                        # Start next chunk with overlap
                        overlap_sentences = self._get_overlap_sentences(
                            sentences[current_start : i + 1]
                        )
                        current_start = i + 1 - len(overlap_sentences)
                        current_tokens = sum(
                            self._count_tokens(s) for s in overlap_sentences
                        )
                        continue

                # Force boundary if no next sentence or high similarity
                boundaries.append(
                    LateBoundary(
                        start_idx=current_start,
                        end_idx=i + 1,
                        similarity_score=1.0,  # Forced boundary
                    )
                )
                if in_list_run and run_tokens.get(run, 0) > self.target_tokens:
                    overlap_sentences = self._get_overlap_sentences(
                        sentences[current_start : i + 1]
                    )
                    current_start = i + 1 - len(overlap_sentences)
                    current_tokens = sum(
                        self._count_tokens(s) for s in overlap_sentences
                    )
                else:
                    current_start = i + 1
                    current_tokens = 0

        # Add final chunk
        if current_start < len(sentences):
            boundaries.append(
                LateBoundary(
                    start_idx=current_start,
                    end_idx=len(sentences),
                    similarity_score=0.0,  # Document end
                )
            )

        return boundaries

    def _compute_sentence_similarity(
        self,
        segment1: List[str],
        segment2: List[str],
    ) -> float:
        """
        Compute similarity between two sentence segments using Jaccard similarity.

        Uses deterministic text-based similarity (Jaccard index) for boundary detection.
        Phase 2 will add embedding-based similarity for improved boundary detection.

        Args:
            segment1: List of sentences in current segment
            segment2: List of sentences in next segment

        Returns:
            Similarity score between 0.0 (completely different) and 1.0 (identical)
        """

        # Combine segments into text
        text1 = " ".join(segment1).lower()
        text2 = " ".join(segment2).lower()

        # Tokenize into words (simple split)
        words1 = set(text1.split())
        words2 = set(text2.split())

        # Compute Jaccard similarity: |intersection| / |union|
        if not words1 and not words2:
            return 1.0  # Both empty = identical

        if not words1 or not words2:
            return 0.0  # One empty = no overlap

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _detect_boundaries_embedding(
        self,
        sentences: List[str],
        context: Any,
        *,
        list_run_map: dict[int, tuple[int, int]] | None = None,
        list_run_tokens: dict[tuple[int, int], int] | None = None,
    ) -> List[LateBoundary]:
        """
        Detect chunk boundaries using embedding-based window similarity (Phase 2 - SOTA).

        Algorithm:
          1. Create overlapping windows of sentences (e.g., window_size=3)
          2. Batch embed all windows for efficiency
          3. Compute cosine similarity between adjacent windows
          4. Detect boundaries where similarity drops below threshold
          5. Accumulate sentences until target_tokens + low similarity

        This approach is semantically aware and captures:
          - Paraphrases ("mandatory arbitration" vs "binding dispute resolution")
          - Synonyms ("large" vs "big")
          - Legal/technical formulations
          - Topic transitions

        Args:
            sentences: List of sentence strings
            context: DocumentProcessingContext for logging

        Returns:
            List of LateBoundary objects marking semantic chunk boundaries
        """
        # 1. Create sliding windows
        windows = self._create_sentence_windows(sentences)

        if not windows:
            return []

        # 2. Batch embed all windows
        try:
            embedded_windows = self._embed_windows_batch(windows, context)
        except Exception as exc:
            log_extra = {
                "document_id": str(context.metadata.document_id),
                "error": str(exc),
                "exc_type": type(exc).__name__,
            }
            if self.allow_jaccard_fallback:
                logger.warning(
                    "late_chunker_embedding_failed_fallback_jaccard: %s (type=%s)",
                    str(exc),
                    type(exc).__name__,
                    extra=log_extra,
                )
                return self._detect_boundaries_jaccard(
                    sentences,
                    list_run_map=list_run_map,
                    list_run_tokens=list_run_tokens,
                )
            logger.error(
                "late_chunker_embedding_failed_no_fallback: %s (type=%s)",
                str(exc),
                type(exc).__name__,
                extra=log_extra,
            )
            raise

        # 3. Detect boundaries using cosine similarity between windows
        boundaries = []
        current_start = 0
        current_tokens = 0
        run_map = list_run_map or {}
        run_tokens = list_run_tokens or {}

        for i, sentence in enumerate(sentences):
            sentence_tokens = self._count_tokens(sentence)
            current_tokens += sentence_tokens

            # Check if we've reached target size
            if current_tokens >= self.target_tokens:
                run = run_map.get(i)
                in_list_run = run is not None and i + 1 < run[1]
                preserve_list = (
                    in_list_run and run_tokens.get(run, 0) <= self.target_tokens
                )
                if preserve_list:
                    continue
                # Check embedding similarity with next window (if exists)
                if i + 1 < len(sentences):
                    # Find windows that start at current position and next position
                    current_window = next(
                        (w for w in embedded_windows if w.start_idx == i), None
                    )
                    next_window = next(
                        (w for w in embedded_windows if w.start_idx == i + 1), None
                    )

                    if (
                        current_window
                        and next_window
                        and current_window.embedding
                        and next_window.embedding
                    ):
                        # Compute cosine similarity
                        similarity = self._compute_cosine_similarity(
                            current_window.embedding,
                            next_window.embedding,
                        )

                        logger.debug(
                            "late_chunker_window_similarity",
                            extra={
                                "document_id": str(context.metadata.document_id),
                                "window_idx": i,
                                "similarity": similarity,
                                "threshold": self.similarity_threshold,
                            },
                        )

                        # Low similarity = good boundary (semantic topic shift)
                        if similarity < self.similarity_threshold:
                            boundaries.append(
                                LateBoundary(
                                    start_idx=current_start,
                                    end_idx=i + 1,
                                    similarity_score=similarity,
                                )
                            )
                            # Start next chunk with overlap
                            overlap_sentences = self._get_overlap_sentences(
                                sentences[current_start : i + 1]
                            )
                            current_start = i + 1 - len(overlap_sentences)
                            current_tokens = sum(
                                self._count_tokens(s) for s in overlap_sentences
                            )
                            continue

                # Force boundary if no next sentence or high similarity
                boundaries.append(
                    LateBoundary(
                        start_idx=current_start,
                        end_idx=i + 1,
                        similarity_score=1.0,  # Forced boundary
                    )
                )
                if in_list_run and run_tokens.get(run, 0) > self.target_tokens:
                    overlap_sentences = self._get_overlap_sentences(
                        sentences[current_start : i + 1]
                    )
                    current_start = i + 1 - len(overlap_sentences)
                    current_tokens = sum(
                        self._count_tokens(s) for s in overlap_sentences
                    )
                else:
                    current_start = i + 1
                    current_tokens = 0

        # Add final chunk
        if current_start < len(sentences):
            boundaries.append(
                LateBoundary(
                    start_idx=current_start,
                    end_idx=len(sentences),
                    similarity_score=0.0,  # Document end
                )
            )

        logger.info(
            "late_chunker_embedding_boundaries_detected",
            extra={
                "document_id": str(context.metadata.document_id),
                "sentence_count": len(sentences),
                "boundary_count": len(boundaries),
                "window_count": len(embedded_windows),
            },
        )

        return boundaries

    def _get_overlap_sentences(self, sentences: List[str]) -> List[str]:
        """Get sentences for overlap with next chunk."""
        overlap_count = 0
        overlap_tokens = 0

        # Count sentences from end until overlap_tokens is reached
        for sentence in reversed(sentences):
            sentence_tokens = self._count_tokens(sentence)
            if overlap_tokens + sentence_tokens > self.overlap_tokens:
                break
            overlap_tokens += sentence_tokens
            overlap_count += 1

        return sentences[-overlap_count:] if overlap_count > 0 else []

    def _build_chunks_adaptive(
        self,
        *,
        run: TextRun,
        sentences: List[str],
        boundaries: List[LateBoundary],
        document_id: UUID,
        document_title: str | None,
        document_ref: str | None,
        doc_type: str | None,
        list_run_map: dict[int, tuple[int, int]] | None = None,
        chunker_label: str = "late",
    ) -> List[Mapping[str, Any]]:
        """Build adaptive text chunks with structure-aware metadata."""
        chunks: list[Mapping[str, Any]] = []
        parent_ref = self._resolve_text_parent_ref(run)
        section_path = list(run.section_path) if run.section_path else []
        run_map = list_run_map or {}

        total_chunks = len(boundaries)
        for idx, boundary in enumerate(boundaries):
            chunk_sentences = sentences[boundary.start_idx : boundary.end_idx]
            chunk_text = " ".join(chunk_sentences)
            if not chunk_text.strip():
                continue
            list_header = self._resolve_list_header(
                sentences,
                run_map,
                boundary.start_idx,
            )
            chunk_position = self._format_chunk_position(idx, total_chunks)
            prefix = build_chunk_prefix(
                document_ref=document_ref or document_title,
                doc_type=doc_type,
                section_path=section_path,
                chunk_position=chunk_position,
                list_header=list_header,
            )
            if idx == 0 and prefix:
                self._log_prefix_sample(
                    document_id=document_id,
                    document_ref=document_ref or document_title,
                    doc_type=doc_type,
                    prefix=prefix,
                    section_path=section_path,
                    chunk_position=chunk_position,
                )
            if prefix:
                chunk_text = f"{prefix}{chunk_text}"
            locator = f"run:{run.index}:{boundary.start_idx}:{boundary.end_idx}:{idx}"
            chunk_id = self._build_adaptive_chunk_id(
                document_id=document_id,
                chunk_text=chunk_text,
                kind="text",
                parent_ref=parent_ref,
                locator=locator,
            )
            chunk = {
                "chunk_id": chunk_id,
                "text": chunk_text,
                "parent_ref": parent_ref,
                "section_path": section_path,
                "metadata": {
                    "chunker": chunker_label,
                    "kind": "text",
                    "sentence_range": (boundary.start_idx, boundary.end_idx),
                    "similarity_score": boundary.similarity_score,
                    "model": self.model,
                    "dimension": self.dimension,
                },
            }
            if run.page_index is not None:
                chunk["page_index"] = run.page_index
            chunks.append(chunk)

        return chunks

    def _resolve_text_parent_ref(self, run: TextRun) -> str:
        if run.section_path:
            return ">".join(run.section_path)
        return f"section:{run.index}"

    def _build_adaptive_chunk_id(
        self,
        *,
        document_id: UUID,
        chunk_text: str,
        kind: str,
        parent_ref: str,
        locator: str,
    ) -> str:
        if self.use_content_based_ids:
            normalized_text = chunk_text.lower().strip()
            namespaced_content = f"{document_id}:{kind}:{parent_ref}:{normalized_text}"
            chunk_hash = hashlib.sha256(namespaced_content.encode("utf-8")).hexdigest()
            return f"sha256-{chunk_hash[:32]}"
        locator_value = f"{kind}:{parent_ref}:{locator}"
        return str(uuid5(document_id, f"chunk:{locator_value}"))

    def _chunk_by_sections_adaptive(
        self,
        parsed: Any,
        context: Any,
        *,
        document_title: str | None = None,
        document_ref: str | None = None,
        doc_type: str | None = None,
    ) -> List[Mapping[str, Any]]:
        """Adaptive fallback chunking when document exceeds max_tokens."""
        logger.warning(
            "late_chunker_fallback_to_sections",
            extra={
                "document_id": str(context.metadata.document_id),
                "chunker_mode": "adaptive",
            },
        )
        chunks: list[Mapping[str, Any]] = []
        namespace_id = self._resolve_chunk_namespace_id(context)
        runs = self._build_text_runs(parsed)
        min_tokens = max(40, self.target_tokens // 6)
        runs = self._merge_short_runs(runs, min_tokens=min_tokens)
        for run in runs:
            if not run.text:
                continue
            sentences = self._split_sentences(run.text)
            if not sentences:
                continue
            list_run_map, list_run_tokens = self._build_list_run_map(sentences)
            boundaries = self._detect_boundaries(
                sentences,
                context,
                list_run_map=list_run_map,
                list_run_tokens=list_run_tokens,
            )
            boundaries = self._merge_list_run_boundaries(
                boundaries,
                list_run_map=list_run_map,
                list_run_tokens=list_run_tokens,
            )
            boundaries = self._merge_heading_only_boundaries(sentences, boundaries)
            run_chunks = self._build_chunks_adaptive(
                run=run,
                sentences=sentences,
                boundaries=boundaries,
                document_id=namespace_id,
                document_title=document_title,
                document_ref=document_ref,
                doc_type=doc_type,
                list_run_map=list_run_map,
                chunker_label="late-fallback",
            )
            if run_chunks:
                chunks.extend(run_chunks)
        return chunks

    def _build_asset_chunks(
        self,
        parsed: Any,
        context: Any,
        *,
        document_title: str | None = None,
        document_ref: str | None = None,
        doc_type: str | None = None,
    ) -> List[Mapping[str, Any]]:
        assets = getattr(parsed, "assets", None)
        if not assets:
            return []
        chunks: list[Mapping[str, Any]] = []
        document_id = context.metadata.document_id
        namespace_id = self._resolve_chunk_namespace_id(context)

        total_assets = len(assets)
        for index, asset in enumerate(assets):
            chunk_text = self._build_asset_text(asset, index)
            if not chunk_text:
                continue
            chunk_position = self._format_chunk_position(
                index,
                total_assets,
                label="Asset",
            )
            prefix = build_chunk_prefix(
                document_ref=document_ref or document_title,
                doc_type=doc_type,
                section_path=None,
                chunk_position=chunk_position,
                list_header=None,
            )
            if index == 0 and prefix:
                self._log_prefix_sample(
                    document_id=document_id,
                    document_ref=document_ref or document_title,
                    doc_type=doc_type,
                    prefix=prefix,
                    section_path=(),
                    chunk_position=chunk_position,
                )
            if prefix:
                chunk_text = f"{prefix}{chunk_text}"
            parent_ref = self._resolve_asset_parent_ref(asset, index, document_id)
            chunk_id = self._build_adaptive_chunk_id(
                document_id=namespace_id,
                chunk_text=chunk_text,
                kind="asset",
                parent_ref=parent_ref,
                locator=f"asset:{index}",
            )
            chunk = {
                "chunk_id": chunk_id,
                "text": chunk_text,
                "parent_ref": parent_ref,
                "section_path": [],
                "metadata": {
                    "chunker": "late",
                    "kind": "asset",
                    "model": self.model,
                    "dimension": self.dimension,
                },
            }
            page_index = getattr(asset, "page_index", None)
            if page_index is not None:
                chunk["page_index"] = page_index
            chunks.append(chunk)

        return chunks

    def _build_asset_text(self, asset: Any, index: int) -> str:
        parts: list[str] = []
        metadata = getattr(asset, "metadata", None)
        if isinstance(metadata, Mapping):
            candidates = metadata.get("caption_candidates")
            if isinstance(candidates, Sequence) and not isinstance(
                candidates, (str, bytes, bytearray)
            ):
                for candidate in candidates:
                    if not isinstance(candidate, (list, tuple)) or len(candidate) != 2:
                        continue
                    label, value = candidate
                    value_text = str(value).strip()
                    if not value_text:
                        continue
                    label_text = str(label).strip()
                    if label_text:
                        label_text = label_text.replace("_", " ").title()
                        parts.append(f"{label_text}: {value_text}")
                    else:
                        parts.append(value_text)
        context_before = getattr(asset, "context_before", None)
        if context_before:
            parts.append(f"Context Before: {context_before}")
        context_after = getattr(asset, "context_after", None)
        if context_after:
            parts.append(f"Context After: {context_after}")
        file_uri = getattr(asset, "file_uri", None)
        if file_uri:
            uri_text = str(file_uri)
            name = uri_text.split("/")[-1].split("\\")[-1] or uri_text
            parts.append(f"Filename: {name}")
        if not parts:
            media_type = getattr(asset, "media_type", "asset")
            parts.append(f"Asset {index + 1} ({media_type})")
        return "\n".join(part.strip() for part in parts if part.strip())

    def _resolve_asset_parent_ref(
        self,
        asset: Any,
        index: int,
        document_id: UUID,
    ) -> str:
        from common.assets import deterministic_asset_path
        from documents.contract_utils import normalize_string

        metadata = getattr(asset, "metadata", None)
        raw_locator = None
        if isinstance(metadata, Mapping):
            raw_locator = metadata.get("locator")
        locator = ""
        if raw_locator is not None:
            locator = normalize_string(str(raw_locator))
        if not locator:
            locator = f"asset-index:{index}"
        asset_id = uuid5(document_id, deterministic_asset_path(document_id, locator))
        return str(asset_id)

    def _build_chunks(
        self,
        *,
        sentences: List[str],
        boundaries: List[LateBoundary],
        block_map: dict,
        document_id: UUID,
        parsed: Any,
        document_title: str | None,
        document_ref: str | None,
        doc_type: str | None,
        list_run_map: dict[int, tuple[int, int]] | None = None,
    ) -> List[Mapping[str, Any]]:
        """Build chunks with metadata from boundaries."""
        chunks = []
        run_map = list_run_map or {}

        total_chunks = len(boundaries)
        for idx, boundary in enumerate(boundaries):
            # Extract chunk text
            chunk_sentences = sentences[boundary.start_idx : boundary.end_idx]
            chunk_text = " ".join(chunk_sentences)
            # Determine parent_ref and section_path from first sentence
            # (simplified - can be improved with block_map lookups)
            parent_ref = f"late-chunk:{idx}"
            section_path = []

            list_header = self._resolve_list_header(
                sentences,
                run_map,
                boundary.start_idx,
            )
            chunk_position = self._format_chunk_position(idx, total_chunks)
            prefix = build_chunk_prefix(
                document_ref=document_ref or document_title,
                doc_type=doc_type,
                section_path=section_path,
                chunk_position=chunk_position,
                list_header=list_header,
            )
            if idx == 0 and prefix:
                self._log_prefix_sample(
                    document_id=document_id,
                    document_ref=document_ref or document_title,
                    doc_type=doc_type,
                    prefix=prefix,
                    section_path=section_path,
                    chunk_position=chunk_position,
                )
            if prefix:
                chunk_text = f"{prefix}{chunk_text}"

            # Generate chunk_id (Phase 2: content-based SHA256 or uuid5)
            if self.use_content_based_ids:
                # Phase 2: Content-based deterministic ID (SHA256)
                # Namespace by document_id to prevent collisions across documents/tenants
                normalized_text = chunk_text.lower().strip()
                namespaced_content = f"{document_id}:{normalized_text}"
                chunk_hash = hashlib.sha256(
                    namespaced_content.encode("utf-8")
                ).hexdigest()
                # Use first 32 chars (128 bits) as UUID-compatible ID
                chunk_id = f"sha256-{chunk_hash[:32]}"
            else:
                # Phase 1: UUID5 based on document_id + locator
                locator = f"late:{idx}:{boundary.start_idx}:{boundary.end_idx}"
                chunk_id = str(uuid5(document_id, f"chunk:{locator}"))

            # Build chunk dict
            chunk = {
                "chunk_id": chunk_id,
                "text": chunk_text,
                "parent_ref": parent_ref,
                "section_path": section_path,
                "metadata": {
                    "chunker": "late",
                    "sentence_range": (boundary.start_idx, boundary.end_idx),
                    "similarity_score": boundary.similarity_score,
                    "model": self.model,
                    "dimension": self.dimension,
                },
            }

            chunks.append(chunk)

        return chunks

    def _chunk_by_sections(
        self,
        parsed: Any,
        context: Any,
        *,
        document_title: str | None = None,
        document_ref: str | None = None,
        doc_type: str | None = None,
    ) -> List[Mapping[str, Any]]:
        """
        Fallback: Chunk by sections when document exceeds max_tokens.

        This is a simplified fallback - in production, we'd process each section
        with Late Chunking independently.
        """
        logger.warning(
            "late_chunker_fallback_to_sections",
            extra={"document_id": str(context.metadata.document_id)},
        )

        chunks: list[Mapping[str, Any]] = []
        namespace_id = self._resolve_chunk_namespace_id(context)
        runs = self._build_text_runs(parsed)
        min_tokens = max(40, self.target_tokens // 6)
        runs = self._merge_short_runs(runs, min_tokens=min_tokens)
        for run in runs:
            if not run.text:
                continue
            sentences = self._split_sentences(run.text)
            if not sentences:
                continue
            list_run_map, list_run_tokens = self._build_list_run_map(sentences)
            boundaries = self._detect_boundaries(
                sentences,
                context,
                list_run_map=list_run_map,
                list_run_tokens=list_run_tokens,
            )
            boundaries = self._merge_list_run_boundaries(
                boundaries,
                list_run_map=list_run_map,
                list_run_tokens=list_run_tokens,
            )
            boundaries = self._merge_heading_only_boundaries(sentences, boundaries)
            run_chunks = self._build_chunks_adaptive(
                run=run,
                sentences=sentences,
                boundaries=boundaries,
                document_id=namespace_id,
                document_title=document_title,
                document_ref=document_ref,
                doc_type=doc_type,
                list_run_map=list_run_map,
                chunker_label="late-fallback",
            )
            if run_chunks:
                chunks.extend(run_chunks)

        return chunks

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
    def _apply_chunk_counts(chunks: List[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
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
        chunks: List[Mapping[str, Any]],
        document_text: str,
        context: Any,
    ) -> List[Mapping[str, Any]]:
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
            "late_chunker_contextual_enrichment",
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
    def _log_prefix_sample(
        *,
        document_id: UUID,
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
                    "document_id": str(document_id),
                    "document_ref": document_ref,
                    "doc_type": doc_type,
                    "section_path": list(section_path),
                    "chunk_position": chunk_position,
                    "prefix": prefix.strip(),
                },
            )
        except Exception:
            pass

    def _build_list_run_map(
        self, sentences: List[str]
    ) -> tuple[dict[int, tuple[int, int]], dict[tuple[int, int], int]]:
        runs = find_numbered_list_runs(sentences)
        run_map: dict[int, tuple[int, int]] = {}
        run_tokens: dict[tuple[int, int], int] = {}
        token_counts = [self._count_tokens(sentence) for sentence in sentences]
        token_prefix: list[int] = [0]
        for count in token_counts:
            token_prefix.append(token_prefix[-1] + count)
        for start, end in runs:
            run_tokens[(start, end)] = token_prefix[end] - token_prefix[start]
            for idx in range(start, end):
                run_map[idx] = (start, end)
        return run_map, run_tokens

    @staticmethod
    def _resolve_list_header(
        sentences: List[str],
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

    def _merge_list_run_boundaries(
        self,
        boundaries: List[LateBoundary],
        *,
        list_run_map: dict[int, tuple[int, int]] | None = None,
        list_run_tokens: dict[tuple[int, int], int] | None = None,
    ) -> List[LateBoundary]:
        if not boundaries:
            return boundaries
        run_map = list_run_map or {}
        run_tokens = list_run_tokens or {}
        merged: list[LateBoundary] = []
        idx = 0
        while idx < len(boundaries):
            boundary = boundaries[idx]
            start_idx = boundary.start_idx
            end_idx = boundary.end_idx
            run = run_map.get(max(end_idx - 1, 0))
            if run and end_idx < run[1]:
                run_token_count = run_tokens.get(run, 0)
                if run_token_count <= self.target_tokens:
                    logger.warning(
                        "late_chunker_list_split_merged",
                        extra={
                            "run_start": run[0],
                            "run_end": run[1],
                            "run_tokens": run_token_count,
                            "boundary_end": end_idx,
                        },
                    )
                    end_idx = run[1]
            while idx + 1 < len(boundaries) and boundaries[idx + 1].start_idx < end_idx:
                idx += 1
                end_idx = max(end_idx, boundaries[idx].end_idx)
            merged.append(
                LateBoundary(
                    start_idx=start_idx,
                    end_idx=end_idx,
                    similarity_score=boundary.similarity_score,
                )
            )
            idx += 1
        return merged

    def _resolve_chunk_namespace_id(self, context: Any) -> UUID:
        document_id = context.metadata.document_id
        raw_version_id = getattr(context.metadata, "document_version_id", None)
        if raw_version_id in {None, ""}:
            return document_id
        try:
            return (
                raw_version_id
                if isinstance(raw_version_id, UUID)
                else UUID(str(raw_version_id))
            )
        except (TypeError, ValueError, AttributeError):
            return document_id

    def _merge_heading_only_boundaries(
        self,
        sentences: List[str],
        boundaries: List[LateBoundary],
    ) -> List[LateBoundary]:
        if len(boundaries) < 2:
            return boundaries

        merged: list[LateBoundary] = []
        idx = 0
        while idx < len(boundaries):
            boundary = boundaries[idx]
            if idx < len(boundaries) - 1:
                chunk_sentences = sentences[boundary.start_idx : boundary.end_idx]
                chunk_text = " ".join(chunk_sentences).strip()
                if self._is_heading_only_chunk(chunk_text):
                    next_boundary = boundaries[idx + 1]
                    merged.append(
                        LateBoundary(
                            start_idx=boundary.start_idx,
                            end_idx=next_boundary.end_idx,
                            similarity_score=next_boundary.similarity_score,
                        )
                    )
                    idx += 2
                    continue
            merged.append(boundary)
            idx += 1

        return merged

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
