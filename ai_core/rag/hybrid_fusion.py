"""Hybrid candidate fusion helpers (RRF)."""

from __future__ import annotations

import time
from typing import Callable, Dict, Mapping, Sequence

from .score_fusion import FusionResult, fuse_candidates as _fuse_candidates


def fuse_candidates(
    *,
    vector_rows: Sequence[object],
    lexical_rows: Sequence[object],
    query_vec: str | None,
    query_embedding_empty: bool,
    alpha: float,
    min_sim: float,
    top_k: int,
    tenant: str,
    case_id: str | None,
    normalized_filters: Mapping[str, object | None],
    fallback_limit_used: float | None,
    distance_score_mode: str,
    rrf_k: int,
    extract_score_from_row: Callable[..., object | None],
    normalise_result_row: Callable[..., tuple],
    ensure_chunk_metadata_contract: Callable[..., Dict[str, object]],
    logger,
) -> FusionResult:
    started_at = time.perf_counter()
    result = _fuse_candidates(
        vector_rows=vector_rows,
        lexical_rows=lexical_rows,
        query_vec=query_vec,
        query_embedding_empty=query_embedding_empty,
        alpha=alpha,
        min_sim=min_sim,
        top_k=top_k,
        tenant=tenant,
        case_id=case_id,
        normalized_filters=normalized_filters,
        fallback_limit_used=fallback_limit_used,
        distance_score_mode=distance_score_mode,
        rrf_k=rrf_k,
        extract_score_from_row=extract_score_from_row,
        normalise_result_row=normalise_result_row,
        ensure_chunk_metadata_contract=ensure_chunk_metadata_contract,
        logger=logger,
    )
    try:
        duration_ms = int(round((time.perf_counter() - started_at) * 1000))
        logger.info(
            "rag.hybrid.fusion_summary",
            extra={
                "tenant_id": tenant,
                "case_id": case_id,
                "duration_ms": duration_ms,
                "vector_candidates": len(vector_rows),
                "lexical_candidates": len(lexical_rows),
                "fused_candidates": result.fused_candidates,
            },
        )
    except Exception:
        pass
    return result


__all__ = ["fuse_candidates", "FusionResult"]
