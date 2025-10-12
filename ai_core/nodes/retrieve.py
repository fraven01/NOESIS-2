from __future__ import annotations

import time
from typing import Any, Dict, Mapping

from common.logging import get_logger

from ai_core.nodes._hybrid_params import parse_hybrid_parameters
from ai_core.rag.embedding_config import get_embedding_configuration
from ai_core.rag.profile_resolver import resolve_embedding_profile
from ai_core.rag.schemas import Chunk
from ai_core.rag.vector_store import VectorStoreRouter, get_default_router
from ai_core.rag.visibility import coerce_bool_flag
from ai_core.tool_contracts import (
    ContextError,
    InconsistentMetadataError,
    InputError,
    NotFoundError,
    ToolContext,
)
from pydantic import BaseModel, ConfigDict


class RetrieveInput(BaseModel):
    """Structured input parameters for the retrieval tool."""

    query: str = ""
    filters: Mapping[str, Any] | None = None
    process: str | None = None
    doc_class: str | None = None
    visibility: str | None = None
    visibility_override_allowed: bool | None = None
    hybrid: Mapping[str, Any] | None = None
    top_k: int | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @classmethod
    def from_state(
        cls, state: Mapping[str, Any], *, top_k: int | None = None
    ) -> "RetrieveInput":
        """Build a :class:`RetrieveInput` instance from a legacy state mapping."""

        data: Dict[str, Any] = {
            "query": state.get("query", ""),
            "filters": state.get("filters"),
            "process": state.get("process"),
            "doc_class": state.get("doc_class"),
            "visibility": state.get("visibility"),
            "visibility_override_allowed": state.get("visibility_override_allowed"),
            "hybrid": state.get("hybrid"),
        }
        state_top_k: Any | None = None
        if top_k is not None:
            state_top_k = top_k
        else:
            candidate = state.get("top_k")
            if isinstance(candidate, str):
                if candidate.strip():
                    state_top_k = candidate
            elif candidate is not None:
                state_top_k = candidate
        if state_top_k is not None:
            data["top_k"] = state_top_k
        return cls(**data)


class RetrieveRouting(BaseModel):
    """Routing metadata resolved for the retrieval run."""

    profile: str
    vector_space_id: str

    model_config = ConfigDict(extra="forbid")


class RetrieveMeta(BaseModel):
    """Metadata emitted by the retrieval tool."""

    routing: RetrieveRouting
    took_ms: int
    alpha: float
    min_sim: float
    top_k_effective: int
    max_candidates_effective: int
    vector_candidates: int
    lexical_candidates: int
    deleted_matches_blocked: int
    visibility_effective: str

    model_config = ConfigDict(extra="forbid")


class RetrieveOutput(BaseModel):
    """Structured output payload returned by the retrieval tool."""

    matches: list[Dict[str, Any]]
    meta: RetrieveMeta

    model_config = ConfigDict(extra="forbid")


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
    raise InputError(msg, field=field)


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


def _ensure_chunk_metadata(
    chunks: list[Chunk], *, tenant_id: str, case_id: str | None
) -> None:
    for index, chunk in enumerate(chunks):
        meta = chunk.meta or {}
        if "tenant_id" in meta and "case_id" in meta:
            continue
        logger.error(
            "rag.retrieve.inconsistent_metadata",
            extra={
                "tenant_id": tenant_id,
                "case_id": case_id,
                "chunk_index": index,
                "keys": sorted(meta.keys()),
            },
        )
        raise InconsistentMetadataError("reindex required")


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


def run(context: ToolContext, params: RetrieveInput) -> RetrieveOutput:
    """Execute the retrieval tool based on the provided context and parameters."""

    started_at = time.perf_counter()

    tenant_id = str(getattr(context, "tenant_id", "") or "").strip()
    if not tenant_id:
        raise ContextError("tenant_id required", field="tenant_id")

    tenant_schema = (
        str(context.tenant_schema).strip()
        if context.tenant_schema is not None
        else None
    )
    case_id = str(context.case_id).strip() if context.case_id is not None else None

    filters = _ensure_mapping(params.filters, field="filters")
    process = params.process
    doc_class = params.doc_class
    requested_visibility = params.visibility

    hybrid_mapping = _ensure_mapping(params.hybrid, field="hybrid")
    if hybrid_mapping is None:
        raise InputError("hybrid configuration required", field="hybrid")

    hybrid_state: Dict[str, Any] = {"hybrid": dict(hybrid_mapping)}
    try:
        hybrid_config = parse_hybrid_parameters(
            hybrid_state, override_top_k=params.top_k
        )
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise InputError(str(exc), field="hybrid") from exc

    override_flag = params.visibility_override_allowed
    if override_flag is None:
        override_flag = context.visibility_override_allowed
    visibility_override_allowed = coerce_bool_flag(override_flag)

    router = _get_router()
    tenant_client: Any = router
    scoped_client = False
    for_tenant = getattr(router, "for_tenant", None)
    if callable(for_tenant):
        scoped_client = True
        try:
            tenant_client = for_tenant(tenant_id, tenant_schema)
        except TypeError:
            tenant_client = for_tenant(tenant_id)
        if tenant_client is router:
            scoped_client = False

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
        params.query,
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
        visibility=requested_visibility,
        visibility_override_allowed=visibility_override_allowed,
    )

    vector_candidates = _coerce_int_value(
        getattr(hybrid_result, "vector_candidates", 0) or 0, 0
    )
    lexical_candidates = _coerce_int_value(
        getattr(hybrid_result, "lexical_candidates", 0) or 0, 0
    )
    chunks = list(getattr(hybrid_result, "chunks", []) or [])
    _ensure_chunk_metadata(chunks, tenant_id=tenant_id, case_id=case_id)
    if not chunks and vector_candidates == 0 and lexical_candidates == 0:
        logger.info(
            "rag.retrieve.no_matches",
            extra={"tenant_id": tenant_id, "case_id": case_id},
        )
        if not scoped_client:
            raise NotFoundError("No matching documents were found for the query.")
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
    deleted_matches_blocked = _coerce_int_value(
        getattr(hybrid_result, "deleted_matches_blocked", 0) or 0,
        0,
    )
    visibility_effective = str(
        getattr(hybrid_result, "visibility", "active") or "active"
    )

    took_ms = int(round((time.perf_counter() - started_at) * 1000))
    if took_ms < 0:  # pragma: no cover - guard against clock issues
        took_ms = 0

    meta_payload = {
        "alpha": alpha_value,
        "min_sim": min_sim_value,
        "top_k_effective": len(final_matches),
        "max_candidates_effective": hybrid_config.max_candidates,
        "vector_candidates": vector_candidates,
        "lexical_candidates": lexical_candidates,
        "routing": routing_meta,
        "deleted_matches_blocked": deleted_matches_blocked,
        "visibility_effective": visibility_effective,
        "took_ms": took_ms,
    }

    return RetrieveOutput(matches=final_matches, meta=RetrieveMeta(**meta_payload))


__all__ = [
    "RetrieveInput",
    "RetrieveRouting",
    "RetrieveMeta",
    "RetrieveOutput",
    "run",
    "_reset_router_for_tests",
]
