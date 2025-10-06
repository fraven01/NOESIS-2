from __future__ import annotations

from typing import Any, Dict, Mapping, MutableMapping

from common.logging import get_logger

from ai_core.nodes._hybrid_params import parse_hybrid_parameters
from ai_core.rag.embedding_config import get_embedding_configuration
from ai_core.rag.profile_resolver import resolve_embedding_profile
from ai_core.rag.schemas import Chunk
from ai_core.rag.vector_store import VectorStoreRouter, get_default_router


logger = get_logger(__name__)


_ROUTER: VectorStoreRouter | None = None


def _get_router() -> VectorStoreRouter:
    global _ROUTER
    if _ROUTER is None:
        _ROUTER = get_default_router()
    return _ROUTER


def _reset_router_for_tests() -> None:
    global _ROUTER
    _ROUTER = None


def _ensure_mapping(value: object, *, field: str) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return value
    msg = f"{field} must be a mapping when provided"
    raise ValueError(msg)


def _extract_score(meta: Mapping[str, Any]) -> float:
    raw_score = meta.get("score")
    try:
        score = float(raw_score)
    except (TypeError, ValueError):
        score = 0.0
    return max(0.0, min(1.0, score))


def _extract_id(meta: Mapping[str, Any]) -> str | None:
    raw = meta.get("id")
    if raw is None:
        return None
    text = str(raw).strip()
    return text or None


def _chunk_to_match(chunk: Chunk) -> Dict[str, Any]:
    metadata = dict(chunk.meta or {})
    match: Dict[str, Any] = {
        "id": _extract_id(metadata),
        "text": chunk.content or "",
        "score": _extract_score(metadata),
        "source": metadata.get("source", ""),
        "hash": metadata.get("hash"),
    }
    extra_meta = {
        key: value
        for key, value in metadata.items()
        if key not in {"source", "score", "hash", "id"}
    }
    if extra_meta:
        match["meta"] = extra_meta
    return match


def _merge_duplicate(existing: Dict[str, Any], candidate: Dict[str, Any]) -> None:
    current_score = float(existing.get("score", 0.0))
    new_score = float(candidate.get("score", 0.0))
    if new_score > current_score:
        existing.update(candidate)
    elif new_score == current_score:
        existing_meta = existing.get("meta") or {}
        candidate_meta = candidate.get("meta") or {}
        if candidate_meta:
            merged = dict(existing_meta)
            for key, value in candidate_meta.items():
                merged.setdefault(key, value)
            existing["meta"] = merged


def _deduplicate_matches(matches: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    aggregated: dict[str, Dict[str, Any]] = {}
    ordered_keys: list[str] = []

    for index, match in enumerate(matches):
        identifier = match.get("id")
        if identifier is None:
            identifier = f"__chunk_{index}"
        if identifier in aggregated:
            _merge_duplicate(aggregated[identifier], match)
            continue
        aggregated[identifier] = match
        ordered_keys.append(identifier)

    ordered_matches = [aggregated[key] for key in ordered_keys]
    ordered_matches.sort(
        key=lambda item: (-float(item.get("score", 0.0)), str(item.get("id") or ""))
    )
    return ordered_matches


def _coerce_float_value(value: object, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _coerce_int_value(value: object, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _resolve_routing_metadata(
    *,
    tenant_id: str,
    process: str | None,
    doc_class: str | None,
) -> Dict[str, str | None]:
    profile_id = resolve_embedding_profile(
        tenant_id=tenant_id, process=process, doc_class=doc_class
    )
    configuration = get_embedding_configuration()
    profile_config = configuration.embedding_profiles[profile_id]
    return {
        "profile": profile_id,
        "vector_space_id": profile_config.vector_space,
    }


def run(
    state: MutableMapping[str, Any],
    meta: Mapping[str, Any],
    *,
    top_k: int | None = None,
) -> tuple[MutableMapping[str, Any], Dict[str, Any]]:
    query = str(state.get("query") or "")
    filters = _ensure_mapping(state.get("filters"), field="filters")
    process = state.get("process")
    doc_class = state.get("doc_class")

    tenant_id = meta.get("tenant_id") or meta.get("tenant")
    if not tenant_id:
        raise ValueError("tenant_id required")
    tenant_id = str(tenant_id)

    tenant_schema = meta.get("tenant_schema")
    case_id = meta.get("case_id") or meta.get("case")

    hybrid_config = parse_hybrid_parameters(state, override_top_k=top_k)

    router = _get_router()
    tenant_client: Any = router
    for_tenant = getattr(router, "for_tenant", None)
    if callable(for_tenant):
        try:
            tenant_client = for_tenant(tenant_id, tenant_schema)
        except TypeError:
            tenant_client = for_tenant(tenant_id)

    logger.debug(
        "Executing hybrid retrieval",
        extra={
            "tenant_id": tenant_id,
            "case_id": case_id,
            "top_k": hybrid_config.top_k,
            "alpha": hybrid_config.alpha,
            "min_sim": hybrid_config.min_sim,
        },
    )

    hybrid_result = tenant_client.hybrid_search(
        query,
        case_id=case_id,
        top_k=hybrid_config.top_k,
        filters=filters,
        alpha=hybrid_config.alpha,
        min_sim=hybrid_config.min_sim,
        vec_limit=hybrid_config.vec_limit,
        lex_limit=hybrid_config.lex_limit,
        trgm_limit=hybrid_config.trgm_limit,
        max_candidates=hybrid_config.max_candidates,
        process=process,
        doc_class=doc_class,
    )

    chunks = list(getattr(hybrid_result, "chunks", []) or [])
    matches = [_chunk_to_match(chunk) for chunk in chunks]
    deduplicated = _deduplicate_matches(matches)
    final_matches = deduplicated[: hybrid_config.top_k]

    routing_meta = _resolve_routing_metadata(
        tenant_id=tenant_id, process=process, doc_class=doc_class
    )

    alpha_value = _coerce_float_value(
        getattr(hybrid_result, "alpha", hybrid_config.alpha), hybrid_config.alpha
    )
    min_sim_value = _coerce_float_value(
        getattr(hybrid_result, "min_sim", hybrid_config.min_sim), hybrid_config.min_sim
    )
    vector_candidates = _coerce_int_value(
        getattr(hybrid_result, "vector_candidates", 0) or 0, 0
    )
    lexical_candidates = _coerce_int_value(
        getattr(hybrid_result, "lexical_candidates", 0) or 0, 0
    )
    meta_payload = {
        "alpha": alpha_value,
        "min_sim": min_sim_value,
        "top_k_effective": len(final_matches),
        "max_candidates_effective": hybrid_config.max_candidates,
        "vector_candidates": vector_candidates,
        "lexical_candidates": lexical_candidates,
        "routing": routing_meta,
    }

    state["matches"] = final_matches
    state["snippets"] = final_matches

    return state, {"matches": final_matches, "meta": meta_payload}


__all__ = ["run", "_reset_router_for_tests"]
