from __future__ import annotations


from typing import Any, Callable, Dict, Iterable, List, Tuple

import time

from django.conf import settings

from ai_core.rag import metrics
from ai_core.rag.normalization import normalise_text
from common.logging import get_logger

try:
    from ai_core.rag.vector_store import get_default_router
except ImportError:  # pragma: no cover - optional dependency for demo mode
    get_default_router = None  # type: ignore[assignment]


logger = get_logger(__name__)


QueryState = Dict[str, Any]
Meta = Dict[str, Any]
GraphResult = Dict[str, Any]


_QUERY_KEYS: Tuple[str, ...] = ("query", "question", "q", "text")


def _extract_query(state: QueryState) -> str | None:
    for key in _QUERY_KEYS:
        value = state.get(key)
        if isinstance(value, str):
            value = value.strip()
        if value:
            return str(value)
    return None


def _coerce_top_k(state: QueryState, default: int = 5) -> int:
    candidate = state.get("top_k") or state.get("k")
    if candidate is None:
        return default
    try:
        top_k = int(candidate)
    except (TypeError, ValueError):
        return default
    return max(1, top_k)


def _truncate(text: str, limit: int = 500) -> str:
    if len(text) <= limit:
        return text
    return text[:limit]


def _chunk_matches(chunks: Iterable[Any]) -> List[Dict[str, Any]]:
    matches: List[Dict[str, Any]] = []
    for index, chunk in enumerate(chunks):
        content = getattr(chunk, "content", "") or ""
        meta = getattr(chunk, "meta", {}) or {}
        identifier = meta.get("id") or meta.get("hash") or f"chunk-{index}"
        score = meta.get("score")
        try:
            score_value = float(score) if score is not None else 0.0
        except (TypeError, ValueError):
            score_value = 0.0
        vscore = float(meta.get("vscore", 0.0) or 0.0)
        lscore = float(meta.get("lscore", 0.0) or 0.0)
        fused = float(meta.get("fused", score_value) or score_value)
        matches.append(
            {
                "id": str(identifier),
                "score": fused,
                "fused": fused,
                "vscore": vscore,
                "lscore": lscore,
                "text": _truncate(str(content)),
                "metadata": dict(meta),
            }
        )
    return matches


def _demo_matches(query: str, tenant_id: str, *, top_k: int) -> List[Dict[str, Any]]:
    demo_corpus = [
        {
            "id": "demo-1",
            "score": 0.42,
            "text": _truncate(
                (
                    "Demo knowledge base entry describing how retrieval works for "
                    "tenant '%s'. Query: '%s'."
                )
                % (tenant_id, query)
            ),
            "metadata": {"tenant_id": tenant_id, "source": "demo"},
        },
        {
            "id": "demo-2",
            "score": 0.36,
            "text": _truncate(
                "Second demo snippet outlining the behaviour of the RAG demo node."
            ),
            "metadata": {"tenant_id": tenant_id, "source": "demo"},
        },
    ]
    return demo_corpus[:top_k]


def _resolve_tenant_router(
    for_tenant: Callable[..., Any],
    tenant_id: Any,
    tenant_schema: Any | None,
) -> Tuple[Any | None, str | None]:
    """Call ``for_tenant`` defensively across common signatures."""

    attempts: List[Tuple[Tuple[Any, ...], Dict[str, Any]]] = []
    if tenant_schema is not None:
        attempts.extend(
            [
                ((tenant_id, tenant_schema), {}),
                ((), {"tenant_id": tenant_id, "tenant_schema": tenant_schema}),
            ]
        )
    attempts.extend(
        [
            ((tenant_id,), {}),
            ((), {"tenant_id": tenant_id}),
            ((), {}),
        ]
    )

    last_type_error: str | None = None
    for args, kwargs in attempts:
        try:
            return for_tenant(*args, **kwargs), None
        except TypeError as exc:
            last_type_error = str(exc)
            continue
        except Exception as exc:  # pragma: no cover - defensive fallback
            return None, str(exc)

    if last_type_error is None:
        return None, None
    return None, last_type_error


def run(state: QueryState, meta: Meta) -> Tuple[QueryState, GraphResult]:
    query = _extract_query(state)
    if not query:
        return state, {
            "ok": False,
            "query": None,
            "matches": [],
            "error": "missing_query",
        }

    tenant_id = meta.get("tenant_id") or meta.get("tenant")
    if not tenant_id:
        raise ValueError("tenant_id missing in meta")

    top_k = _coerce_top_k(state)
    project_id = state.get("project_id")

    matches: List[Dict[str, Any]] = []
    router_error: str | None = None
    retrieved_chunks: List[Any] = []
    hybrid_result = None
    latency_ms = 0.0

    normalized_query = normalise_text(query)
    search_input = normalized_query or query.strip()

    index_kind = str(getattr(settings, "RAG_INDEX_KIND", "HNSW")).upper()
    alpha = float(getattr(settings, "RAG_HYBRID_ALPHA", 0.7))
    min_sim = float(getattr(settings, "RAG_MIN_SIM", 0.15))
    ef_search = int(getattr(settings, "RAG_HNSW_EF_SEARCH", 80))
    probes = int(getattr(settings, "RAG_IVF_PROBES", 64))

    if get_default_router is not None:
        filters: Dict[str, Any] = {"tenant_id": tenant_id}
        if project_id:
            filters["project_id"] = project_id

        tenant_schema = meta.get("tenant_schema") or meta.get("schema")

        try:
            router = get_default_router()
        except Exception as exc:  # pragma: no cover - defensive fallback
            router = None
            router_error = str(exc)

        if router is not None:
            scoped_router = router
            for_tenant = getattr(router, "for_tenant", None)
            if callable(for_tenant):
                scoped_router, scoped_error = _resolve_tenant_router(
                    for_tenant, tenant_id, tenant_schema
                )
                if scoped_router is None and scoped_error:
                    router_error = scoped_error
                elif scoped_router is None:
                    scoped_router = router

            case_id = meta.get("case_id") or meta.get("case")
            start_ts = time.perf_counter()
            try:
                hybrid_callable = getattr(scoped_router, "hybrid_search", None)
                router_hybrid = getattr(router, "hybrid_search", None)
                if callable(hybrid_callable):
                    try:
                        hybrid_result = hybrid_callable(
                            search_input,
                            tenant_id=str(tenant_id),
                            case_id=case_id,
                            top_k=top_k,
                            filters=filters,
                            alpha=alpha,
                            min_sim=min_sim,
                        )
                    except TypeError:
                        hybrid_result = hybrid_callable(
                            search_input,
                            case_id=case_id,
                            top_k=top_k,
                            filters=filters,
                            alpha=alpha,
                            min_sim=min_sim,
                        )
                    retrieved_chunks = list(getattr(hybrid_result, "chunks", []))
                elif callable(router_hybrid):
                    scope_name = getattr(scoped_router, "_scope", None)
                    hybrid_result = router_hybrid(
                        search_input,
                        tenant_id=str(tenant_id),
                        case_id=case_id,
                        top_k=top_k,
                        filters=filters,
                        scope=scope_name or router.default_scope,
                        alpha=alpha,
                        min_sim=min_sim,
                    )
                    retrieved_chunks = list(getattr(hybrid_result, "chunks", []))
                else:
                    search_method = getattr(scoped_router, "search", None)
                    if not callable(search_method):
                        raise AttributeError("router missing search")
                    retrieved_chunks = list(
                        search_method(
                            search_input,
                            tenant_id=tenant_id,
                            case_id=case_id,
                            top_k=top_k,
                            filters=filters,
                        )
                    )
                    hybrid_result = None
                if hybrid_result is None:
                    from ai_core.rag.vector_client import (
                        HybridSearchResult as _HybridSearchResult,
                    )

                    hybrid_result = _HybridSearchResult(
                        chunks=retrieved_chunks,
                        vector_candidates=len(retrieved_chunks),
                        lexical_candidates=0,
                        fused_candidates=len(retrieved_chunks),
                        duration_ms=0.0,
                        alpha=alpha,
                        min_sim=min_sim,
                        vec_limit=top_k,
                        lex_limit=top_k,
                    )
                matches = _chunk_matches(retrieved_chunks)
            except Exception as exc:  # pragma: no cover - defensive fallback
                router_error = str(exc)
                retrieved_chunks = []
                matches = []
            finally:
                latency_ms = (time.perf_counter() - start_ts) * 1000
        else:
            matches = _demo_matches(query, str(tenant_id), top_k=top_k)
    else:
        matches = _demo_matches(query, str(tenant_id), top_k=top_k)

    tenant_label = str(tenant_id)
    if hybrid_result is not None and getattr(hybrid_result, "duration_ms", None) == 0.0:
        try:
            hybrid_result.duration_ms = latency_ms
        except Exception:  # pragma: no cover - defensive attribute set
            pass
    below_cutoff = 0
    returned_after_cutoff = len(retrieved_chunks)
    query_embedding_empty = False
    no_hit_due_to_cutoff = False
    if hybrid_result is not None:
        metrics.RAG_QUERY_TOTAL.labels(
            tenant=tenant_label, index_kind=index_kind, hybrid="true"
        ).inc()
        metrics.RAG_QUERY_LATENCY_MS.labels(
            tenant=tenant_label, index_kind=index_kind, hybrid="true"
        ).observe(latency_ms)
        metrics.RAG_QUERY_CANDIDATES.labels(
            tenant=tenant_label, type="semantic"
        ).observe(float(hybrid_result.vector_candidates))
        metrics.RAG_QUERY_CANDIDATES.labels(
            tenant=tenant_label, type="lexical"
        ).observe(float(hybrid_result.lexical_candidates))

        top1_fused = 0.0
        top1_vscore = 0.0
        top1_lscore = 0.0
        below_cutoff = int(getattr(hybrid_result, "below_cutoff", 0))
        returned_after_cutoff = int(
            getattr(hybrid_result, "returned_after_cutoff", len(retrieved_chunks))
        )
        query_embedding_empty = bool(
            getattr(hybrid_result, "query_embedding_empty", False)
        )
        if retrieved_chunks:
            top_meta = retrieved_chunks[0].meta or {}
            top1_fused = float(top_meta.get("fused", top_meta.get("score", 0.0)) or 0.0)
            top1_vscore = float(top_meta.get("vscore", 0.0) or 0.0)
            top1_lscore = float(top_meta.get("lscore", 0.0) or 0.0)
            metrics.RAG_QUERY_TOP1_SIM.labels(tenant=tenant_label).observe(top1_fused)
        else:
            metrics.RAG_QUERY_NO_HIT.labels(tenant=tenant_label).inc()
            no_hit_due_to_cutoff = (
                below_cutoff > 0
                and hybrid_result.fused_candidates > 0
                and returned_after_cutoff == 0
            )

        logger.info(
            "rag.hybrid_query",
            extra={
                "tenant": tenant_label,
                "index_kind": index_kind,
                "alpha": alpha,
                "min_sim": min_sim,
                "ef_search": ef_search if index_kind == "HNSW" else None,
                "probes": probes if index_kind == "IVFFLAT" else None,
                "latency_ms": latency_ms,
                "db_latency_ms": float(hybrid_result.duration_ms),
                "top1_fused": top1_fused,
                "top1_vscore": top1_vscore,
                "top1_lscore": top1_lscore,
                "topk": top_k,
                "vec_limit": hybrid_result.vec_limit,
                "lex_limit": hybrid_result.lex_limit,
                "vector_candidates": hybrid_result.vector_candidates,
                "lexical_candidates": hybrid_result.lexical_candidates,
                "below_cutoff": below_cutoff,
                "returned_after_cutoff": returned_after_cutoff,
                "query_embedding_empty": query_embedding_empty,
                "hybrid": True,
                "case_id": meta.get("case_id") or meta.get("case"),
                "project_id": project_id,
                "query_length": len(query),
                "normalized_query_length": len(search_input),
                "query_norm_len": len(normalized_query),
                "cutoff": min_sim if no_hit_due_to_cutoff else None,
            },
        )

    warnings: List[str] = []
    if no_hit_due_to_cutoff:
        warnings.append("no_hit_above_threshold")
    if not matches and not no_hit_due_to_cutoff:
        matches = _demo_matches(query, str(tenant_id), top_k=top_k)
        warnings.append("no_vector_matches_demo_fallback")

    response_meta: Dict[str, Any] = {
        "index_kind": index_kind,
        "alpha": alpha,
        "min_sim": min_sim,
        "latency_ms": latency_ms,
    }
    if hybrid_result is not None:
        response_meta["db_latency_ms"] = float(hybrid_result.duration_ms)
        response_meta["vector_candidates"] = getattr(
            hybrid_result, "vector_candidates", None
        )
        response_meta["lexical_candidates"] = getattr(
            hybrid_result, "lexical_candidates", None
        )
        response_meta["below_cutoff"] = below_cutoff
        response_meta["returned_after_cutoff"] = returned_after_cutoff
        response_meta["query_embedding_empty"] = query_embedding_empty

    new_state = dict(state)
    new_state["rag_demo"] = {
        "query": query,
        "top_k": top_k,
        "retrieved_count": len(matches),
    }

    result: GraphResult = {
        "ok": True,
        "query": query,
        "matches": matches,
        "meta": response_meta,
    }
    if router_error:
        result["error"] = router_error

    if warnings:
        result["warnings"] = warnings

    return new_state, result


__all__ = ["run"]
