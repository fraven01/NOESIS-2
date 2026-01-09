"""Quality metrics and evaluation for RAG chunk quality.

This package implements phased quality metrics:
- Phase 1: LLM-as-Judge scoring (coherence, completeness, resolution, redundancy)
- Phase 2: Pseudo query generation for weak labels
- Phase 3: Human-in-the-Loop review CLI
- Phase 4: Retrieval metrics (MRR, NDCG@k)

See README.md for evaluation strategy details.
"""

from __future__ import annotations

__all__ = [
    "ChunkQualityEvaluator",
    "ChunkQualityScore",
    "compute_quality_statistics",
]

from .llm_judge import (
    ChunkQualityEvaluator,
    ChunkQualityScore,
    compute_quality_statistics,
)

# PseudoQueryGenerator and RetrievalMetricsEvaluator will be added in Phase 2-4
