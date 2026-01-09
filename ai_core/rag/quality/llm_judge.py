"""LLM-as-Judge quality evaluation for RAG chunks.

Phase 1 of quality metrics: Evaluate chunk quality without ground truth labels.

Metrics:
- Coherence (0-100): Does the chunk form a coherent semantic unit?
- Completeness (0-100): Is the chunk self-contained?
- Reference Resolution (0-100): Are references resolved or contextualized?
- Redundancy (0-100, inverted): Does the chunk avoid unnecessary repetition?

Model: quality-eval label (resolves via MODEL_ROUTING.yaml → gpt-5-nano default)
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, List, Mapping

from ai_core.llm.routing import resolve as resolve_model_label

logger = logging.getLogger(__name__)


CHUNK_QUALITY_PROMPT = """Evaluate this text chunk on the following criteria (0-100 scale):

Chunk:
{chunk_text}

Parent reference: {parent_ref}

Criteria:
1. **Coherence**: Does the chunk form a coherent semantic unit? (0=fragmented, 100=perfectly coherent)
2. **Completeness**: Can this chunk be understood in isolation? (0=needs context, 100=self-contained)
3. **Reference Resolution**: Are references (e.g., "this", "the above") resolved or clear? (0=many unresolved, 100=all clear)
4. **Redundancy**: Does the chunk repeat information unnecessarily? (0=very redundant, 100=concise)

Output format (JSON):
{{
  "coherence": <0-100>,
  "completeness": <0-100>,
  "reference_resolution": <0-100>,
  "redundancy": <0-100>,
  "reasoning": "<brief explanation>"
}}
"""


@dataclass(frozen=True)
class ChunkQualityScore:
    """Quality score for a single chunk."""

    chunk_id: str
    coherence: float  # 0-100
    completeness: float  # 0-100
    reference_resolution: float  # 0-100
    redundancy: float  # 0-100, inverted (high = low redundancy)
    overall: float  # Average of all metrics
    reasoning: str = ""

    def to_dict(self) -> dict:
        """Convert to dict for storage."""
        return {
            "chunk_id": self.chunk_id,
            "coherence": self.coherence,
            "completeness": self.completeness,
            "reference_resolution": self.reference_resolution,
            "redundancy": self.redundancy,
            "overall": self.overall,
            "reasoning": self.reasoning,
        }


class ChunkQualityEvaluator:
    """
    LLM-as-Judge for chunk quality scoring.

    Scores each chunk on coherence, completeness, reference resolution, and redundancy.
    No ground truth needed - provides immediate quality feedback.

    Example:
        >>> from ai_core.rag.quality import ChunkQualityEvaluator
        >>> evaluator = ChunkQualityEvaluator(model="quality-eval")  # MODEL_ROUTING.yaml label
        >>> chunks = [{"chunk_id": "uuid-123", "text": "...", "parent_ref": "section:intro"}]
        >>> scores = evaluator.evaluate(chunks, context)
        >>> print(f"Overall quality: {scores[0].overall}/100")
    """

    def __init__(
        self,
        *,
        model: str = "quality-eval",  # MODEL_ROUTING.yaml label
        timeout: int = 60,
        sample_rate: float = 1.0,
        max_workers: int = 4,
    ):
        """
        Initialize Chunk Quality Evaluator.

        Args:
            model: Model name or MODEL_ROUTING.yaml label (e.g., "quality-eval")
            timeout: Timeout for LLM call in seconds
            sample_rate: Sample rate for evaluation (0.0-1.0, 1.0 = all chunks)
            max_workers: Maximum parallel workers for evaluation
        """
        # Resolve MODEL_ROUTING.yaml label to actual model name
        try:
            self.model = resolve_model_label(model)
            logger.debug(
                "quality_eval_model_resolved",
                extra={"label": model, "resolved_model": self.model},
            )
        except ValueError:
            # If not a label, assume it's already a model name (backward compat)
            self.model = model
            logger.debug(
                "quality_eval_model_direct",
                extra={"model": model},
            )

        self.timeout = timeout
        self.sample_rate = sample_rate
        self.max_workers = max(1, int(max_workers))

    def evaluate(
        self,
        chunks: List[Mapping[str, Any]],
        context: Any = None,
    ) -> List[ChunkQualityScore]:
        """
        Evaluate chunk quality using LLM-as-Judge.

        Args:
            chunks: List of chunk dicts with chunk_id, text, parent_ref
            context: Optional DocumentProcessingContext for logging

        Returns:
            List of ChunkQualityScore, one per chunk
        """
        scores: list[ChunkQualityScore | None] = [None] * len(chunks)
        indices_to_eval: list[int] = []

        # Sample chunks based on sample_rate
        if self.sample_rate < 1.0:
            import random

        for index, chunk in enumerate(chunks):
            if self.sample_rate < 1.0 and random.random() > self.sample_rate:
                # Skip this chunk, return default score
                scores[index] = self._default_score(chunk["chunk_id"])
                continue
            indices_to_eval.append(index)

        max_workers = (
            min(self.max_workers, len(indices_to_eval)) if indices_to_eval else 0
        )

        if max_workers <= 1:
            for index in indices_to_eval:
                chunk = chunks[index]
                try:
                    scores[index] = self._evaluate_chunk(chunk, context)
                except Exception as exc:
                    logger.error(
                        "quality_eval_failed",
                        extra={
                            "chunk_id": chunk.get("chunk_id", "unknown"),
                            "error": str(exc),
                        },
                        exc_info=exc,
                    )
                    # Return default score on failure
                    scores[index] = self._default_score(chunk["chunk_id"])
        else:
            # Preserve input ordering while evaluating in parallel.
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map = {
                    executor.submit(self._evaluate_chunk, chunks[index], context): index
                    for index in indices_to_eval
                }
                for future in as_completed(future_map):
                    index = future_map[future]
                    chunk = chunks[index]
                    try:
                        scores[index] = future.result()
                    except Exception as exc:
                        logger.error(
                            "quality_eval_failed",
                            extra={
                                "chunk_id": chunk.get("chunk_id", "unknown"),
                                "error": str(exc),
                            },
                            exc_info=exc,
                        )
                        # Return default score on failure
                        scores[index] = self._default_score(chunk["chunk_id"])

        scores = [
            (
                score
                if score is not None
                else self._default_score(chunks[index]["chunk_id"])
            )
            for index, score in enumerate(scores)
        ]

        logger.info(
            "quality_eval_completed",
            extra={
                "chunk_count": len(chunks),
                "evaluated_count": len([s for s in scores if s.coherence > 0]),
                "mean_overall": (
                    sum(s.overall for s in scores) / len(scores) if scores else 0
                ),
            },
        )

        return scores

    def _evaluate_chunk(
        self,
        chunk: Mapping[str, Any],
        context: Any = None,
    ) -> ChunkQualityScore:
        """Evaluate a single chunk."""
        from ai_core.infra.config import get_config
        from ai_core.infra.circuit_breaker import get_litellm_circuit_breaker
        from ai_core.llm.client import LlmUpstreamError
        from litellm import completion

        cfg = get_config()
        completion_kwargs = {"api_base": cfg.litellm_base_url}
        if cfg.litellm_api_key:
            completion_kwargs["api_key"] = cfg.litellm_api_key

        # Build prompt
        prompt = CHUNK_QUALITY_PROMPT.format(
            chunk_text=chunk.get("text", ""),
            parent_ref=chunk.get("parent_ref", "unknown"),
        )

        # Call LLM
        breaker = get_litellm_circuit_breaker()
        if not breaker.allow_request():
            raise LlmUpstreamError(
                "LiteLLM circuit breaker open",
                status=503,
                code="circuit_open",
            )
        try:
            response = completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                timeout=self.timeout,
                **completion_kwargs,
            )
        except Exception:
            breaker.record_failure(reason="litellm_completion")
            raise
        else:
            breaker.record_success()

        # Parse JSON response
        result = json.loads(response.choices[0].message.content)

        # Extract scores
        coherence = float(result.get("coherence", 0))
        completeness = float(result.get("completeness", 0))
        reference_resolution = float(result.get("reference_resolution", 0))
        redundancy = float(result.get("redundancy", 0))
        reasoning = result.get("reasoning", "")

        # Compute overall score
        overall = (coherence + completeness + reference_resolution + redundancy) / 4.0

        # Clamp to 0-100
        coherence = max(0, min(100, coherence))
        completeness = max(0, min(100, completeness))
        reference_resolution = max(0, min(100, reference_resolution))
        redundancy = max(0, min(100, redundancy))
        overall = max(0, min(100, overall))

        logger.debug(
            "quality_eval_chunk",
            extra={
                "chunk_id": chunk.get("chunk_id", "unknown"),
                "overall": overall,
                "coherence": coherence,
                "completeness": completeness,
            },
        )

        return ChunkQualityScore(
            chunk_id=chunk.get("chunk_id", "unknown"),
            coherence=coherence,
            completeness=completeness,
            reference_resolution=reference_resolution,
            redundancy=redundancy,
            overall=overall,
            reasoning=reasoning,
        )

    def _default_score(self, chunk_id: str) -> ChunkQualityScore:
        """Return default score (used when evaluation fails or is skipped)."""
        return ChunkQualityScore(
            chunk_id=chunk_id,
            coherence=0.0,
            completeness=0.0,
            reference_resolution=0.0,
            redundancy=0.0,
            overall=0.0,
            reasoning="Evaluation skipped or failed",
        )

    def add_quality_to_chunks(
        self,
        chunks: List[Mapping[str, Any]],
        scores: List[ChunkQualityScore],
    ) -> List[Mapping[str, Any]]:
        """
        Add quality scores to chunk metadata.

        Args:
            chunks: Original chunks
            scores: Quality scores (must match chunks 1:1)

        Returns:
            Chunks with quality metadata added
        """
        if len(chunks) != len(scores):
            logger.warning(
                "quality_score_mismatch",
                extra={
                    "chunk_count": len(chunks),
                    "score_count": len(scores),
                },
            )
            return chunks

        updated_chunks = []
        for chunk, score in zip(chunks, scores):
            # Copy chunk dict
            chunk_copy = dict(chunk)

            # Add quality metadata
            if "metadata" not in chunk_copy:
                chunk_copy["metadata"] = {}

            chunk_copy["metadata"]["quality"] = {
                "coherence": score.coherence,
                "completeness": score.completeness,
                "reference_resolution": score.reference_resolution,
                "redundancy": score.redundancy,
                "overall": score.overall,
                "reasoning": score.reasoning,
                "evaluated_by": "llm-judge-v1",
                "evaluated_at": datetime.now(timezone.utc).isoformat(),
            }

            updated_chunks.append(chunk_copy)

        return updated_chunks


def compute_quality_statistics(scores: List[ChunkQualityScore]) -> dict:
    """Compute statistics from quality scores."""
    if not scores:
        return {
            "count": 0,
            "mean_coherence": 0.0,
            "mean_completeness": 0.0,
            "mean_reference_resolution": 0.0,
            "mean_redundancy": 0.0,
            "mean_overall": 0.0,
        }

    return {
        "count": len(scores),
        "mean_coherence": sum(s.coherence for s in scores) / len(scores),
        "mean_completeness": sum(s.completeness for s in scores) / len(scores),
        "mean_reference_resolution": sum(s.reference_resolution for s in scores)
        / len(scores),
        "mean_redundancy": sum(s.redundancy for s in scores) / len(scores),
        "mean_overall": sum(s.overall for s in scores) / len(scores),
        "min_overall": min(s.overall for s in scores),
        "max_overall": max(s.overall for s in scores),
    }
