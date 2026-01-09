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
import logging
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Sequence, Tuple
from uuid import UUID, uuid5

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
        # Phase 2: SOTA Embedding-based Similarity
        use_embedding_similarity: bool = False,
        window_size: int = 3,
        batch_size: int = 16,
        use_content_based_ids: bool = True,
        adaptive_enabled: bool = True,
        asset_chunks_enabled: bool = True,
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
            use_embedding_similarity: Enable embedding-based window similarity (Phase 2, default: False)
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

        # Phase 2: SOTA features
        self.use_embedding_similarity = use_embedding_similarity
        self.window_size = window_size
        self.batch_size = batch_size
        self.use_content_based_ids = use_content_based_ids
        self.adaptive_enabled = adaptive_enabled
        self.asset_chunks_enabled = asset_chunks_enabled

    def chunk(
        self,
        parsed: Any,  # ParsedResult
        context: Any,  # DocumentProcessingContext
    ) -> List[Mapping[str, Any]]:
        """
        Chunk document using Late Chunking strategy.

        Args:
            parsed: ParsedResult with text_blocks
            context: DocumentProcessingContext with metadata

        Returns:
            List of chunk dicts with chunk_id, text, parent_ref, metadata
        """
        if not self.adaptive_enabled:
            return self._chunk_legacy(parsed, context)
        return self._chunk_adaptive(parsed, context)

    def _chunk_legacy(
        self,
        parsed: Any,
        context: Any,
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
            return self._chunk_by_sections(parsed, context)

        # 3. Split into sentences
        sentences = self._split_sentences(full_text)

        # 4. Detect chunk boundaries (Phase 1: Jaccard, Phase 2: Embedding-based)
        boundaries = self._detect_boundaries(sentences, context)

        # 5. Build chunks with metadata
        chunks = self._build_chunks(
            sentences=sentences,
            boundaries=boundaries,
            block_map=block_map,
            document_id=context.metadata.document_id,
            parsed=parsed,
        )

        logger.info(
            "late_chunker_completed",
            extra={
                "chunk_count": len(chunks),
                "document_id": str(context.metadata.document_id),
                "token_count": token_count,
            },
        )

        return chunks

    def _chunk_adaptive(
        self,
        parsed: Any,
        context: Any,
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
            chunks = self._chunk_by_sections_adaptive(parsed, context)
        else:
            runs = self._build_text_runs(parsed)
            chunks = []
            for run in runs:
                run_chunks = self._chunk_run(run, context)
                if run_chunks:
                    chunks.extend(run_chunks)

        if self.asset_chunks_enabled:
            asset_chunks = self._build_asset_chunks(parsed, context)
            if asset_chunks:
                chunks.extend(asset_chunks)

        logger.info(
            "late_chunker_completed",
            extra={
                "chunk_count": len(chunks),
                "document_id": str(context.metadata.document_id),
                "token_count": token_count,
                "chunker_mode": "adaptive",
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
    ) -> List[Mapping[str, Any]]:
        if not run.text:
            return []
        sentences = self._split_sentences(run.text)
        if not sentences:
            return []
        boundaries = self._detect_boundaries(sentences, context)
        return self._build_chunks_adaptive(
            run=run,
            sentences=sentences,
            boundaries=boundaries,
            document_id=context.metadata.document_id,
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
            response = client.embed([text], model=self.model)
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
        import re

        # Simple sentence splitter (can be replaced with NLTK/spaCy later)
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

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
                response = client.embed(texts, model=self.model)
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
                logger.error(
                    "late_chunker_batch_embed_failed",
                    extra={
                        "document_id": str(context.metadata.document_id),
                        "batch_start": batch_start,
                        "error": str(exc),
                    },
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
    ) -> List[LateBoundary]:
        """
        Detect chunk boundaries using similarity.

        Phase 1 (use_embedding_similarity=False):
          - Uses Jaccard similarity (text-based, deterministic)
          - Fast but semantically limited

        Phase 2 (use_embedding_similarity=True):
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
            return self._detect_boundaries_embedding(sentences, context)

        # Phase 1: Jaccard similarity (fallback)
        return self._detect_boundaries_jaccard(sentences)

    def _detect_boundaries_jaccard(
        self,
        sentences: List[str],
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

        for i, sentence in enumerate(sentences):
            sentence_tokens = self._count_tokens(sentence)
            current_tokens += sentence_tokens

            # Check if we've reached target size
            if current_tokens >= self.target_tokens:
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
            logger.warning(
                "late_chunker_embedding_failed_fallback_jaccard",
                extra={
                    "document_id": str(context.metadata.document_id),
                    "error": str(exc),
                },
            )
            # Fallback to Jaccard on embedding failure
            return self._detect_boundaries_jaccard(sentences)

        # 3. Detect boundaries using cosine similarity between windows
        boundaries = []
        current_start = 0
        current_tokens = 0

        for i, sentence in enumerate(sentences):
            sentence_tokens = self._count_tokens(sentence)
            current_tokens += sentence_tokens

            # Check if we've reached target size
            if current_tokens >= self.target_tokens:
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
    ) -> List[Mapping[str, Any]]:
        """Build adaptive text chunks with structure-aware metadata."""
        chunks: list[Mapping[str, Any]] = []
        parent_ref = self._resolve_text_parent_ref(run)
        section_path = list(run.section_path) if run.section_path else []

        for idx, boundary in enumerate(boundaries):
            chunk_sentences = sentences[boundary.start_idx : boundary.end_idx]
            chunk_text = " ".join(chunk_sentences)
            if not chunk_text.strip():
                continue
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
                    "chunker": "late",
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
        document_id = context.metadata.document_id
        for idx, block in enumerate(parsed.text_blocks):
            text = block.text.strip()
            if not text:
                continue
            section_path = list(block.section_path) if block.section_path else []
            parent_ref = (
                ">".join(block.section_path) if block.section_path else f"block:{idx}"
            )
            chunk_text = text[:2048]
            chunk_id = self._build_adaptive_chunk_id(
                document_id=document_id,
                chunk_text=chunk_text,
                kind="text",
                parent_ref=parent_ref,
                locator=f"fallback:{idx}",
            )
            chunk = {
                "chunk_id": chunk_id,
                "text": chunk_text,
                "parent_ref": parent_ref,
                "section_path": section_path,
                "metadata": {
                    "chunker": "late-fallback",
                    "kind": "text",
                },
            }
            if block.page_index is not None:
                chunk["page_index"] = block.page_index
            chunks.append(chunk)
        return chunks

    def _build_asset_chunks(
        self,
        parsed: Any,
        context: Any,
    ) -> List[Mapping[str, Any]]:
        assets = getattr(parsed, "assets", None)
        if not assets:
            return []
        chunks: list[Mapping[str, Any]] = []
        document_id = context.metadata.document_id

        for index, asset in enumerate(assets):
            chunk_text = self._build_asset_text(asset, index)
            if not chunk_text:
                continue
            parent_ref = self._resolve_asset_parent_ref(asset, index, document_id)
            chunk_id = self._build_adaptive_chunk_id(
                document_id=document_id,
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
    ) -> List[Mapping[str, Any]]:
        """Build chunks with metadata from boundaries."""
        chunks = []

        for idx, boundary in enumerate(boundaries):
            # Extract chunk text
            chunk_sentences = sentences[boundary.start_idx : boundary.end_idx]
            chunk_text = " ".join(chunk_sentences)

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

            # Determine parent_ref and section_path from first sentence
            # (simplified - can be improved with block_map lookups)
            parent_ref = f"late-chunk:{idx}"
            section_path = []

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

        # Simple fallback: use block-based chunking (like SimpleDocumentChunker)
        chunks = []
        for idx, block in enumerate(parsed.text_blocks):
            text = block.text.strip()
            if not text:
                continue

            locator = f"fallback:{idx}"
            chunk_id = str(uuid5(context.metadata.document_id, f"chunk:{locator}"))

            chunk = {
                "chunk_id": chunk_id,
                "text": text[:2048],  # Truncate like SimpleDocumentChunker
                "parent_ref": f"block:{idx}",
                "section_path": list(block.section_path) if block.section_path else [],
                "metadata": {
                    "chunker": "late-fallback",
                    "kind": block.kind,
                },
            }

            chunks.append(chunk)

        return chunks
