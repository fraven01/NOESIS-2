from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, cast

from . import metrics
from .filters import strict_match
from .schemas import Chunk


@dataclass(frozen=True)
class FusionResult:
    chunks: List[Chunk]
    fused_candidates: int
    below_cutoff: int
    returned_after_cutoff: int
    scores: List[Dict[str, float]] | None


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
    def _safe_chunk_identifier(value: object | None) -> str | None:
        if value is None:
            return None
        try:
            text = str(value)
        except Exception:
            return None
        return text

    def _safe_float(value: object | None) -> float | None:
        if value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(number) or math.isinf(number):
            return None
        return number

    candidates: Dict[str, Dict[str, object]] = {}
    vector_ranks: Dict[str, int] = {}
    lexical_ranks: Dict[str, int] = {}
    for rank, row in enumerate(vector_rows, start=1):
        score_candidate = extract_score_from_row(row, kind="vector")
        (
            chunk_id,
            text_value,
            metadata,
            doc_hash,
            document_id,
            collection_id,
            score_raw,
        ) = normalise_result_row(row, kind="vector")
        score_source = score_candidate if score_candidate is not None else score_raw
        raw_value: float | None = None
        if score_source is not None:
            try:
                raw_value = float(score_source)
            except (TypeError, ValueError):
                raw_value = None
            else:
                if math.isnan(raw_value) or math.isinf(raw_value):
                    raw_value = None
        vector_score_missing = raw_value is None
        text_value = text_value or ""
        key = str(chunk_id) if chunk_id is not None else f"row-{len(candidates)}"
        chunk_identifier = chunk_id if chunk_id is not None else key

        metadata_dict = dict(metadata or {})

        entry = candidates.setdefault(
            key,
            {
                "chunk_id": key,
                "content": text_value,
                "metadata": metadata_dict,
                "doc_hash": doc_hash,
                "document_id": document_id,
                "collection_id": collection_id,
                "vscore": 0.0,
                "lscore": 0.0,
                "_allow_below_cutoff": False,
            },
        )
        entry["chunk_id"] = chunk_identifier
        if entry.get("collection_id") is None and collection_id is not None:
            entry["collection_id"] = collection_id
        if not entry.get("metadata"):
            entry["metadata"] = metadata_dict
        if entry.get("document_id") is None and document_id is not None:
            entry["document_id"] = document_id
        if entry.get("doc_hash") is None and doc_hash is not None:
            entry["doc_hash"] = doc_hash
        if vector_score_missing:
            entry["_allow_below_cutoff"] = True
        else:
            vector_ranks.setdefault(key, rank)

    # Only mark candidates as allowed-below-cutoff for exceptional cases.
    # We no longer blanket-allow all lexical candidates when a trigram
    # fallback occurred with alpha=0.0; that decision is deferred to the
    # dedicated cutoff fallback stage below which selects only the best
    # needed candidates up to top_k.

    for rank, row in enumerate(lexical_rows, start=1):
        score_candidate = extract_score_from_row(row, kind="lexical")
        (
            chunk_id,
            text_value,
            metadata,
            doc_hash,
            document_id,
            collection_id,
            score_raw,
        ) = normalise_result_row(row, kind="lexical")
        score_source = score_candidate if score_candidate is not None else score_raw
        raw_value: float | None = None
        if score_source is not None:
            try:
                raw_value = float(score_source)
            except (TypeError, ValueError):
                raw_value = None
            else:
                if math.isnan(raw_value) or math.isinf(raw_value):
                    raw_value = None
        lexical_score_missing = raw_value is None
        text_value = text_value or ""
        key = str(chunk_id) if chunk_id is not None else f"row-{len(candidates)}"
        chunk_identifier = chunk_id if chunk_id is not None else key

        metadata_dict = dict(metadata or {})
        entry = candidates.setdefault(
            key,
            {
                "chunk_id": key,
                "content": text_value,
                "metadata": metadata_dict,
                "doc_hash": doc_hash,
                "document_id": document_id,
                "collection_id": collection_id,
                "vscore": 0.0,
                "lscore": 0.0,
                "_allow_below_cutoff": False,
            },
        )
        entry["chunk_id"] = chunk_identifier
        if entry.get("collection_id") is None and collection_id is not None:
            entry["collection_id"] = collection_id

        if not entry.get("metadata"):
            entry["metadata"] = metadata_dict
        if entry.get("document_id") is None and document_id is not None:
            entry["document_id"] = document_id
        if entry.get("doc_hash") is None and doc_hash is not None:
            entry["doc_hash"] = doc_hash

        # Permit bypassing the min_sim cutoff only when the lexical score is
        # structurally missing (row shape/NaN), not merely because a trigram
        # fallback happened. The proper promotion of below-cutoff items is
        # handled later by the cutoff fallback logic which respects top_k.
        if lexical_score_missing:
            entry["_allow_below_cutoff"] = True
        else:
            lexical_ranks.setdefault(key, rank)

    has_vector_signal = (
        bool(vector_rows) and (query_vec is not None) and (not query_embedding_empty)
    )
    has_lexical_signal = bool(lexical_rows)
    if has_vector_signal and has_lexical_signal:
        dense_weight = alpha
        lexical_weight = 1.0 - alpha
    elif has_vector_signal:
        dense_weight = 1.0
        lexical_weight = 0.0
    elif has_lexical_signal:
        dense_weight = 0.0
        lexical_weight = 1.0
    else:
        dense_weight = 0.0
        lexical_weight = 0.0

    rrf_k = max(1, int(rrf_k))
    rrf_scale = float(rrf_k + 1)

    def _rrf_component(rank_value: Optional[int], weight: float) -> float:
        if rank_value is None or rank_value <= 0 or weight <= 0.0:
            return 0.0
        return weight / (rrf_k + rank_value)

    for key, entry in candidates.items():
        v_rank = vector_ranks.get(key)
        l_rank = lexical_ranks.get(key)
        vscore_raw = _rrf_component(v_rank, dense_weight)
        lscore_raw = _rrf_component(l_rank, lexical_weight)
        entry["vscore"] = vscore_raw * rrf_scale
        entry["lscore"] = lscore_raw * rrf_scale
        entry["fused"] = float(entry["vscore"]) + float(entry["lscore"])

    try:
        logger.debug(
            "rag.hybrid.candidates.compiled",
            extra={
                "tenant_id": tenant,
                "case_id": case_id,
                "count": len(candidates),
                "entries": [
                    {
                        "chunk_id": _safe_chunk_identifier(entry.get("chunk_id")),
                        "vscore": _safe_float(entry.get("vscore")),
                        "lscore": _safe_float(entry.get("lscore")),
                        "allow_below_cutoff": bool(
                            entry.get("_allow_below_cutoff", False)
                        ),
                    }
                    for entry in candidates.values()
                ],
            },
        )
    except Exception:
        pass
    fused_candidates = len(candidates)
    logger.info(
        "rag.hybrid.debug.fusion",
        extra={
            "tenant_id": tenant,
            "case_id": case_id,
            "candidates": fused_candidates,
            "has_vec": bool(vector_rows),
            "has_lex": bool(lexical_rows),
        },
    )
    results: List[Tuple[Chunk, bool]] = []
    for key, entry in candidates.items():
        allow_below_cutoff = bool(entry.pop("_allow_below_cutoff", False))
        raw_meta = dict(cast(Mapping[str, object] | None, entry.get("metadata")) or {})
        # Be tolerant to legacy result metadata that may still use
        # "tenant"/"case" keys instead of "tenant_id"/"case_id".
        candidate_tenant = cast(
            Optional[str], raw_meta.get("tenant_id") or raw_meta.get("tenant")
        )
        candidate_case = cast(
            Optional[str], raw_meta.get("case_id") or raw_meta.get("case")
        )
        reasons: List[str] = []
        try:
            vector_preview = float(entry.get("vscore", 0.0))
        except (TypeError, ValueError):
            vector_preview = 0.0
        if math.isnan(vector_preview) or math.isinf(vector_preview):
            vector_preview = 0.0
        try:
            lexical_preview = float(entry.get("lscore", 0.0))
        except (TypeError, ValueError):
            lexical_preview = 0.0
        if math.isnan(lexical_preview) or math.isinf(lexical_preview):
            lexical_preview = 0.0
        try:
            fused_preview = float(entry.get("fused", 0.0))
        except (TypeError, ValueError):
            fused_preview = 0.0
        fused_preview = max(0.0, min(1.0, fused_preview))
        if tenant is not None:
            if candidate_tenant is None:
                reasons.append("tenant_missing")
            elif candidate_tenant != tenant:
                reasons.append("tenant_mismatch")
        if case_id is not None:
            if candidate_case is None:
                reasons.append("case_missing")
            elif candidate_case != case_id:
                reasons.append("case_mismatch")

        if case_id is None:
            strict_ok = (tenant is None) or (
                candidate_tenant is not None and candidate_tenant == tenant
            )
        else:
            strict_ok = strict_match(raw_meta, tenant, case_id)

        if not strict_ok or reasons:
            logger.info(
                "rag.strict.reject",
                tenant_id=tenant,
                case_id=case_id,
                candidate_tenant_id=candidate_tenant,
                candidate_case_id=candidate_case,
                doc_hash=entry.get("doc_hash"),
                document_id=entry.get("document_id"),
                chunk_id=entry.get("chunk_id"),
                reasons=reasons or ["unknown"],
                vector_score=vector_preview,
                lexical_score=lexical_preview,
                fused_score=fused_preview,
                allow_below_cutoff=bool(entry.get("_allow_below_cutoff", False)),
            )
            continue

        doc_hash = entry.get("doc_hash")
        document_id = entry.get("document_id")
        meta = ensure_chunk_metadata_contract(
            raw_meta,
            tenant_id=tenant,
            case_id=case_id,
            filters=normalized_filters,
            chunk_id=entry.get("chunk_id"),
            document_id=document_id,
            collection_id=entry.get("collection_id"),
        )
        if doc_hash and not meta.get("hash"):
            meta["hash"] = doc_hash
        if document_id is not None and "id" not in meta:
            meta["id"] = str(document_id)
        try:
            vscore = float(entry.get("vscore", 0.0))
        except (TypeError, ValueError):
            vscore = 0.0
        if math.isnan(vscore) or math.isinf(vscore):
            vscore = 0.0
        try:
            lscore = float(entry.get("lscore", 0.0))
        except (TypeError, ValueError):
            lscore = 0.0
        if math.isnan(lscore) or math.isinf(lscore):
            lscore = 0.0
        try:
            fused = float(entry.get("fused", 0.0))
        except (TypeError, ValueError):
            fused = 0.0
        fused = max(0.0, min(1.0, fused))
        meta["vscore"] = vscore
        meta["lscore"] = lscore
        meta["fused"] = fused
        meta["score"] = fused
        results.append(
            (
                Chunk(content=str(entry.get("content", "")), meta=meta),
                allow_below_cutoff,
            )
        )

    if results:
        normalized_results: List[Tuple[Chunk, bool]] = []
        discarded_entries = 0
        for entry in results:
            chunk_candidate: Chunk | None = None
            allow_flag = False
            if isinstance(entry, tuple):
                if entry:
                    candidate_chunk = entry[0]
                    if isinstance(candidate_chunk, Chunk):
                        chunk_candidate = candidate_chunk
                        if len(entry) > 1:
                            allow_flag = bool(entry[1])
            elif isinstance(entry, Chunk):
                chunk_candidate = entry
            if chunk_candidate is None:
                discarded_entries += 1
                continue
            normalized_results.append((chunk_candidate, allow_flag))
        if discarded_entries:
            try:
                logger.warning(
                    "rag.hybrid.result_shape_unexpected",
                    extra={
                        "tenant_id": tenant,
                        "case_id": case_id,
                        "discarded_entries": discarded_entries,
                        "kept": len(normalized_results),
                    },
                )
            except Exception:
                pass
        results = normalized_results

    results.sort(key=lambda item: float(item[0].meta.get("fused", 0.0)), reverse=True)
    try:
        logger.debug(
            "rag.hybrid.results.pre_min_sim",
            extra={
                "tenant_id": tenant,
                "case_id": case_id,
                "min_sim": min_sim,
                "results": [
                    {
                        "chunk_id": _safe_chunk_identifier(chunk.meta.get("chunk_id")),
                        "fused": _safe_float(chunk.meta.get("fused")),
                        "vscore": _safe_float(chunk.meta.get("vscore")),
                        "lscore": _safe_float(chunk.meta.get("lscore")),
                        "allow_below_cutoff": bool(allow),
                    }
                    for chunk, allow in results
                ],
            },
        )
    except Exception:
        pass
    below_cutoff_count = 0
    filtered_out_details: List[Dict[str, object | None]] = []
    below_cutoff_chunks: List[Chunk] = []
    filtered_results: List[Chunk] = []
    selected_chunk_keys: set[str] = set()
    if min_sim > 0.0:
        for chunk, allow in results:
            fused_value = float(chunk.meta.get("fused", 0.0))
            if math.isnan(fused_value) or math.isinf(fused_value):
                fused_value = 0.0

            is_below_cutoff = fused_value < min_sim
            if is_below_cutoff:
                below_cutoff_count += 1

            if allow or not is_below_cutoff:
                filtered_results.append(chunk)
                chunk_key = (
                    _safe_chunk_identifier(chunk.meta.get("chunk_id"))
                    or f"id:{id(chunk)}"
                )
                selected_chunk_keys.add(chunk_key)
                if is_below_cutoff:
                    # These chunks are returned to the caller but still count as
                    # below-cutoff for telemetry and metrics purposes.
                    continue

            if is_below_cutoff and not allow:
                below_cutoff_chunks.append(chunk)
                filtered_out_details.append(
                    {
                        "chunk_id": _safe_chunk_identifier(chunk.meta.get("chunk_id")),
                        "fused": _safe_float(fused_value),
                    }
                )
        if below_cutoff_count > 0:
            metrics.RAG_QUERY_BELOW_CUTOFF_TOTAL.labels(tenant_id=tenant).inc(
                float(below_cutoff_count)
            )
    else:
        filtered_results = [chunk for chunk, _ in results]
    if not selected_chunk_keys:
        selected_chunk_keys = {
            _safe_chunk_identifier(chunk.meta.get("chunk_id")) or f"id:{id(chunk)}"
            for chunk in filtered_results
        }
    try:
        logger.debug(
            "rag.hybrid.results.post_min_sim",
            extra={
                "tenant_id": tenant,
                "case_id": case_id,
                "min_sim": min_sim,
                "returned": len(filtered_results),
                "filtered_out": filtered_out_details,
                "kept": [
                    {
                        "chunk_id": _safe_chunk_identifier(chunk.meta.get("chunk_id")),
                        "fused": _safe_float(chunk.meta.get("fused")),
                    }
                    for chunk in filtered_results
                ],
            },
        )
    except Exception:
        pass
    fallback_promoted: List[Dict[str, object | None]] = []
    fallback_attempted = False
    # Only run the cutoff fallback when the trigram fallback has been applied.
    # This prevents vector-only queries from reintroducing candidates that were
    # explicitly filtered out by the min_sim threshold, which is required by
    # the hybrid search tests that expect an empty result set in that case.
    cutoff_fallback_enabled = fallback_limit_used is not None
    if (
        min_sim > 0.0
        and len(filtered_results) < top_k
        and results
        and cutoff_fallback_enabled
    ):
        fallback_attempted = True
        needed = top_k - len(filtered_results)
        cutoff_candidates = {
            (
                _safe_chunk_identifier(chunk.meta.get("chunk_id")) or f"id:{id(chunk)}"
            ): chunk
            for chunk in below_cutoff_chunks
        }
        for chunk, _ in results:
            if needed <= 0:
                break
            chunk_key = (
                _safe_chunk_identifier(chunk.meta.get("chunk_id")) or f"id:{id(chunk)}"
            )
            if chunk_key in selected_chunk_keys:
                continue
            if chunk_key not in cutoff_candidates:
                continue
            new_meta = dict(chunk.meta)
            new_meta["cutoff_fallback"] = True
            updated_chunk = chunk.model_copy(update={"meta": new_meta})
            filtered_results.append(updated_chunk)
            selected_chunk_keys.add(chunk_key)
            fallback_promoted.append(
                {
                    "chunk_id": chunk_key,
                    "fused": _safe_float(chunk.meta.get("fused")),
                }
            )
            needed -= 1
        if fallback_promoted:
            promoted_ids = {
                entry.get("chunk_id")
                for entry in fallback_promoted
                if entry.get("chunk_id") is not None
            }
            if promoted_ids:
                filtered_out_details = [
                    detail
                    for detail in filtered_out_details
                    if detail.get("chunk_id") not in promoted_ids
                ]
    limited_results = filtered_results[:top_k]
    if fallback_attempted:
        try:
            logger.info(
                "rag.hybrid.cutoff_fallback",
                extra={
                    "tenant_id": tenant,
                    "case_id": case_id,
                    "requested_min_sim": min_sim,
                    "returned": len(limited_results),
                    "below_cutoff": below_cutoff_count,
                    "promoted": fallback_promoted,
                },
            )
        except Exception:
            pass
    elif not limited_results and results and min_sim > 0.0:
        try:
            logger.info(
                "rag.hybrid.cutoff_fallback",
                extra={
                    "tenant_id": tenant,
                    "case_id": case_id,
                    "requested_min_sim": min_sim,
                    "returned": len(limited_results),
                    "below_cutoff": below_cutoff_count,
                    "promoted": [],
                },
            )
        except Exception:
            pass

    try:
        top_fused = (
            float(limited_results[0].meta.get("fused", 0.0)) if limited_results else 0.0
        )
        top_v = (
            float(limited_results[0].meta.get("vscore", 0.0))
            if limited_results
            else 0.0
        )
        top_l = (
            float(limited_results[0].meta.get("lscore", 0.0))
            if limited_results
            else 0.0
        )
    except Exception:
        top_fused = top_v = top_l = 0.0

    logger.info(
        "rag.hybrid.debug.after_cutoff",
        extra={
            "tenant_id": tenant,
            "case_id": case_id,
            "returned": len(limited_results),
            "top_fused": top_fused,
            "top_vscore": top_v,
            "top_lscore": top_l,
            "min_sim": min_sim,
            "alpha": alpha,
            "distance_score_mode": distance_score_mode,
        },
    )

    per_result_scores: List[Dict[str, float]] = []
    for chunk in limited_results:
        fused_score = chunk.meta.get("fused", 0.0)
        vector_score = chunk.meta.get("vscore", 0.0)
        lexical_score = chunk.meta.get("lscore", 0.0)
        try:
            fused_value = float(fused_score)
        except (TypeError, ValueError):
            fused_value = 0.0
        try:
            vector_value = float(vector_score)
        except (TypeError, ValueError):
            vector_value = 0.0
        try:
            lexical_value = float(lexical_score)
        except (TypeError, ValueError):
            lexical_value = 0.0
        per_result_scores.append(
            {
                "fused": fused_value,
                "vector": vector_value,
                "lexical": lexical_value,
            }
        )

    return FusionResult(
        chunks=limited_results,
        fused_candidates=fused_candidates,
        below_cutoff=below_cutoff_count,
        returned_after_cutoff=len(filtered_results),
        scores=per_result_scores if per_result_scores else None,
    )
