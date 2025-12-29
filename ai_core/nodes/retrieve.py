from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from inspect import Signature, signature
from typing import Any, Callable, Dict, Iterable, Mapping

from common.logging import get_logger

from ai_core.nodes._hybrid_params import parse_hybrid_parameters
from ai_core.rag.embedding_config import get_embedding_configuration
from ai_core.rag.profile_resolver import resolve_embedding_profile
from ai_core.rag.schemas import Chunk
from ai_core.rag.vector_store import VectorStoreRouter, get_default_router
from ai_core.rag.selector_utils import normalise_selector_value
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
    """Structured input parameters for the retrieval tool.

    BREAKING CHANGE (Option A - Strict Separation):
    Business domain IDs (collection_id, workflow_id) have been REMOVED.
    These are now read from ToolContext.business.

    Permission flags (visibility_override_allowed) have been REMOVED.
    These are now read from ToolContext.visibility_override_allowed.

    Golden Rule: Tool-Inputs contain only functional parameters.
    Context contains Scope, Business, and Runtime Permissions.
    """

    query: str = ""
    filters: Mapping[str, Any] | None = None
    process: str | None = None
    doc_class: str | None = None
    visibility: str | None = None
    hybrid: Mapping[str, Any] | None = None
    top_k: int | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @classmethod
    def from_state(
        cls, state: Mapping[str, Any], *, top_k: int | None = None
    ) -> "RetrieveInput":
        """Build a :class:`RetrieveInput` instance from a legacy state mapping.

        BREAKING CHANGE: collection_id, workflow_id, and visibility_override_allowed
        are no longer extracted from state. These must be provided via ToolContext.
        """

        data: Dict[str, Any] = {
            "query": state.get("query", ""),
            "filters": state.get("filters"),
            "process": state.get("process"),
            "doc_class": state.get("doc_class"),
            "visibility": state.get("visibility"),
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
    process: str | None = None
    doc_class: str | None = None
    collection_id: str | None = None
    workflow_id: str | None = None

    model_config = ConfigDict(extra="forbid")


class RetrieveMeta(BaseModel):
    """Metadata emitted by the retrieval tool."""

    routing: RetrieveRouting
    took_ms: int
    alpha: float
    min_sim: float
    top_k_effective: int
    matches_returned: int
    max_candidates_effective: int
    vector_candidates: int
    lexical_candidates: int
    deleted_matches_blocked: int
    visibility_effective: str
    diversify_strength: float

    model_config = ConfigDict(extra="forbid")


class RetrieveOutput(BaseModel):
    """Structured output payload returned by the retrieval tool."""

    matches: list[Dict[str, Any]]
    meta: RetrieveMeta

    model_config = ConfigDict(extra="forbid")


logger = get_logger(__name__)


_ROUTER: VectorStoreRouter | None = None
_ROUTER_ADAPTORS: Dict[int, "_RouterAdaptor"] = {}


@dataclass(frozen=True)
class _RouterAdaptor:
    factory: Callable[[str, str | None], Any]
    is_scoped: bool
    requires_warning: bool


def _get_router() -> VectorStoreRouter:
    global _ROUTER
    if _ROUTER is None:
        _ROUTER = get_default_router()
    return _ROUTER


def _reset_router_for_tests() -> None:
    global _ROUTER
    _ROUTER = None
    _ROUTER_ADAPTORS.clear()


def _build_router_adaptor(router: VectorStoreRouter) -> _RouterAdaptor:
    for_tenant = getattr(router, "for_tenant", None)
    if not callable(for_tenant):
        return _RouterAdaptor(
            factory=lambda tenant_id, tenant_schema: router,
            is_scoped=False,
            requires_warning=True,
        )

    router_name = type(router).__name__
    scope_error = (
        f"{router_name}.for_tenant must accept (tenant_id) or"
        " (tenant_id, tenant_schema)"
    )

    try:
        sig: Signature = signature(for_tenant)
    except (TypeError, ValueError) as exc:
        raise ContextError(scope_error, field="router") from exc

    try:
        sig.bind("tenant", "schema")
    except TypeError:
        pass
    else:
        return _RouterAdaptor(
            factory=lambda tenant_id, tenant_schema: for_tenant(
                tenant_id, tenant_schema
            ),
            is_scoped=True,
            requires_warning=False,
        )

    try:
        sig.bind("tenant")
    except TypeError as exc:
        raise ContextError(scope_error, field="router") from exc

    return _RouterAdaptor(
        factory=lambda tenant_id, tenant_schema: for_tenant(tenant_id),
        is_scoped=True,
        requires_warning=False,
    )


def _get_router_adaptor(router: VectorStoreRouter) -> _RouterAdaptor:
    adaptor = _ROUTER_ADAPTORS.get(id(router))
    if adaptor is None:
        adaptor = _build_router_adaptor(router)
        _ROUTER_ADAPTORS[id(router)] = adaptor
    return adaptor


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


def _coerce_positive_int(value: object) -> int | None:
    try:
        candidate = int(str(value))
    except (TypeError, ValueError):
        return None
    if candidate < 0:
        return None
    return candidate


def _coerce_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _format_range(start: int | None, end: int | None, prefix: str) -> str | None:
    if start is None:
        return None
    if end is None or end == start:
        return f"{prefix}{start}"
    return f"{prefix}{start}-{end}"


def _build_citation(metadata: Mapping[str, object]) -> str | None:
    source = _coerce_str(metadata.get("source"))
    if not source:
        return None

    explicit_locator = _coerce_str(
        metadata.get("citation")
        or metadata.get("source_location")
        or metadata.get("locator")
        or metadata.get("location")
    )
    if explicit_locator:
        if explicit_locator.lower().startswith(source.lower()):
            return explicit_locator
        return f"{source} · {explicit_locator}"

    location_parts: list[str] = []

    page_value = _coerce_positive_int(metadata.get("page"))
    if page_value is None:
        page_value = _coerce_positive_int(metadata.get("page_number"))
    if page_value is None:
        page_index = _coerce_positive_int(metadata.get("page_index"))
        if page_index is not None:
            page_value = page_index + 1
    if page_value is not None:
        location_parts.append(f"S.{page_value}")

    line_start = _coerce_positive_int(
        metadata.get("line_start")
        or metadata.get("start_line")
        or metadata.get("line_begin")
    )
    line_end = _coerce_positive_int(
        metadata.get("line_end")
        or metadata.get("end_line")
        or metadata.get("line_finish")
    )
    line_range = _format_range(line_start, line_end, "Z.")
    if line_range:
        location_parts.append(line_range)

    char_start = _coerce_positive_int(
        metadata.get("char_start")
        or metadata.get("start_char")
        or metadata.get("char_begin")
    )
    char_end = _coerce_positive_int(
        metadata.get("char_end")
        or metadata.get("end_char")
        or metadata.get("char_finish")
    )
    char_range = _format_range(char_start, char_end, "Zeichen ")
    if char_range:
        location_parts.append(char_range)

    for key in ("section", "heading", "title", "chapter"):
        section_value = _coerce_str(metadata.get(key))
        if section_value:
            location_parts.append(section_value)

    if not location_parts:
        chunk_id = _coerce_str(metadata.get("chunk_id"))
        doc_id = _coerce_str(metadata.get("document_id"))
        external_id = _coerce_str(metadata.get("external_id"))
        hash_id = _coerce_str(metadata.get("hash"))
        if chunk_id:
            short = chunk_id[:8] if len(chunk_id) > 12 else chunk_id
            location_parts.append(f"Chunk {short}")
        elif doc_id:
            location_parts.append(f"Dok-ID {doc_id}")
        elif external_id:
            location_parts.append(f"Dok {external_id}")
        elif hash_id:
            short = hash_id[:8] if len(hash_id) > 12 else hash_id
            location_parts.append(f"Hash {short}")

    if location_parts:
        return " · ".join([source, *location_parts])

    return source


def _chunk_to_match(chunk: Chunk) -> Dict[str, Any]:
    metadata = dict(chunk.meta or {})
    match: Dict[str, Any] = {
        "id": _extract_id(metadata),
        "text": chunk.content or "",
        "score": _extract_score(metadata),
        "source": metadata.get("source", ""),
        "hash": metadata.get("hash"),
    }
    citation = _build_citation(metadata)
    if citation:
        match["citation"] = citation
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


def _normalise_identifier(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _match_key(match: Mapping[str, Any], index: int) -> tuple[str, ...]:
    meta = match.get("meta")
    if isinstance(meta, Mapping):
        chunk_identifier = _normalise_identifier(meta.get("chunk_id"))
        if chunk_identifier:
            return ("chunk_id", chunk_identifier)

    hash_identifier = _normalise_identifier(match.get("hash"))
    if hash_identifier:
        return ("hash", hash_identifier)

    doc_identifier = _normalise_identifier(match.get("id"))
    text_value = match.get("text")
    if doc_identifier and isinstance(text_value, str):
        return ("id:text", doc_identifier, text_value)
    if doc_identifier:
        return ("id:index", doc_identifier, str(index))

    return ("index", str(index))


def _deduplicate_matches(matches: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    aggregated: dict[tuple[str, ...], Dict[str, Any]] = {}
    ordered_keys: list[tuple[str, ...]] = []

    for index, match in enumerate(matches):
        identifier = _match_key(match, index)
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


def _extract_document_id(match: Mapping[str, Any]) -> str | None:
    doc_id = _coerce_str(match.get("id"))
    if doc_id:
        return doc_id
    meta = match.get("meta")
    if isinstance(meta, Mapping):
        return _coerce_str(meta.get("document_id"))
    return None


def _filter_matches_by_permissions(
    matches: list[Dict[str, Any]],
    *,
    context: ToolContext,
    permission_type: str = "VIEW",
) -> list[Dict[str, Any]]:
    user_id = context.scope.user_id
    if not user_id:
        return matches

    doc_ids = [_extract_document_id(match) for match in matches]
    doc_ids = [doc_id for doc_id in doc_ids if doc_id]
    if not doc_ids:
        return matches

    from contextlib import nullcontext
    from django_tenants.utils import schema_context

    from customers.tenant_context import TenantContext
    from django.contrib.auth import get_user_model
    from documents.authz import DocumentAuthzService

    tenant = TenantContext.resolve_identifier(context.scope.tenant_id, allow_pk=True)
    tenant_schema = (
        context.scope.tenant_schema
        or (tenant.schema_name if tenant else None)
        or context.scope.tenant_id
    )

    context_manager = (
        schema_context(tenant_schema) if tenant_schema else nullcontext()
    )
    with context_manager:
        User = get_user_model()
        user = User.objects.filter(pk=user_id).first()
        if user is None:
            return []

        allowed_ids = set(
            str(doc_id)
            for doc_id in DocumentAuthzService.accessible_documents_queryset(
                user=user,
                tenant=tenant,
                permission_type=permission_type,
            )
            .filter(id__in=doc_ids)
            .values_list("id", flat=True)
        )

    filtered: list[Dict[str, Any]] = []
    for match in matches:
        doc_id = _extract_document_id(match)
        if not doc_id or doc_id in allowed_ids:
            filtered.append(match)
    return filtered


_TOKEN_PATTERN = re.compile(r"[\w\u00C0-\u024F]+", re.UNICODE)


def _normalise_strength(value: float) -> float:
    return max(0.0, min(1.0, value))


def _tokenise(text: str | None) -> set[str]:
    if not isinstance(text, str) or not text:
        return set()
    tokens = {match.group(0).lower() for match in _TOKEN_PATTERN.finditer(text)}
    return tokens


def _similarity(tokens_a: set[str], tokens_b: set[str]) -> float:
    if not tokens_a or not tokens_b:
        return 0.0
    overlap = tokens_a.intersection(tokens_b)
    if not overlap:
        return 0.0
    denominator = math.sqrt(len(tokens_a) * len(tokens_b))
    if denominator == 0:
        return 0.0
    return len(overlap) / denominator


def _apply_diversification(
    matches: list[Dict[str, Any]], *, top_k: int, strength: float
) -> list[Dict[str, Any]]:
    if len(matches) <= 1:
        return matches

    limit = min(max(1, top_k), len(matches))
    if limit <= 1:
        return matches

    normalised_strength = _normalise_strength(float(strength))
    if normalised_strength <= 0.0:
        return matches

    lambda_param = 1.0 - normalised_strength

    relevance_scores = [float(match.get("score", 0.0)) for match in matches]
    token_sets = [
        _tokenise(match.get("text") or match.get("content") or "") for match in matches
    ]

    ordered_indices = list(range(len(matches)))
    ordered_indices.sort(key=lambda idx: (-relevance_scores[idx], idx))

    selected: list[int] = []
    candidate_pool = ordered_indices.copy()

    while candidate_pool and len(selected) < limit:
        if not selected:
            selected.append(candidate_pool.pop(0))
            continue

        best_index: int | None = None
        best_score = float("-inf")
        for idx in candidate_pool:
            diversity_penalty = 0.0
            if selected:
                diversity_penalty = max(
                    _similarity(token_sets[idx], token_sets[chosen])
                    for chosen in selected
                )
            mmr_score = lambda_param * relevance_scores[idx] - (
                (1.0 - lambda_param) * diversity_penalty
            )
            if mmr_score > best_score or (
                math.isclose(mmr_score, best_score) and idx < (best_index or idx)
            ):
                best_score = mmr_score
                best_index = idx

        if best_index is None:
            best_index = candidate_pool[0]
        selected.append(best_index)
        candidate_pool.remove(best_index)

    remaining = [idx for idx in ordered_indices if idx not in selected]
    final_order = selected + remaining
    return [matches[idx] for idx in final_order]


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
    collection_id: str | None,
    workflow_id: str | None,
) -> Dict[str, str | None]:
    sanitized_process = normalise_selector_value(process)
    sanitized_doc_class = normalise_selector_value(doc_class)
    sanitized_workflow = normalise_selector_value(workflow_id)
    sanitized_collection = None
    if collection_id is not None:
        text = str(collection_id).strip()
        sanitized_collection = text or None
    profile_id = resolve_embedding_profile(
        tenant_id=tenant_id,
        process=process,
        doc_class=doc_class,
        collection_id=sanitized_collection,
        workflow_id=workflow_id,
    )
    configuration = get_embedding_configuration()
    profile_config = configuration.embedding_profiles[profile_id]
    return {
        "profile": profile_id,
        "vector_space_id": profile_config.vector_space,
        "process": sanitized_process,
        "doc_class": sanitized_doc_class,
        "collection_id": sanitized_collection,
        "workflow_id": sanitized_workflow,
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
    collection_id = context.business.collection_id
    workflow_id = context.business.workflow_id
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

    visibility_override_allowed = coerce_bool_flag(context.visibility_override_allowed)

    router = _get_router()
    adaptor = _get_router_adaptor(router)
    if adaptor.requires_warning:
        logger.warning(
            "rag.retrieve.router_incompatible",
            extra={
                "tenant_id": tenant_id,
                "case_id": case_id,
                "router": type(router).__name__,
            },
        )
    tenant_client = adaptor.factory(tenant_id, tenant_schema)
    scoped_client = adaptor.is_scoped and tenant_client is not router

    logger.debug(
        "Executing hybrid retrieval",
        extra={
            "tenant_id": tenant_id,
            "case_id": case_id,
            "process": process,
            "doc_class": doc_class,
            "collection_id": collection_id,
            "workflow_id": workflow_id,
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
        collection_id=collection_id,
        workflow_id=workflow_id,
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
    parent_requests: Dict[str, set[str]] = {}
    for chunk in chunks:
        meta_map = chunk.meta or {}
        doc_identifier = _coerce_str(meta_map.get("document_id"))
        if not doc_identifier:
            continue
        parent_candidates = meta_map.get("parent_ids")
        if isinstance(parent_candidates, Iterable) and not isinstance(
            parent_candidates, (str, bytes)
        ):
            normalised_ids: list[str] = []
            for candidate in parent_candidates:
                parent_id = _coerce_str(candidate)
                if parent_id:
                    normalised_ids.append(parent_id)
            if normalised_ids:
                parent_requests.setdefault(doc_identifier, set()).update(normalised_ids)
                meta_map["parent_ids"] = normalised_ids
        elif "parent_ids" in meta_map:
            meta_map["parent_ids"] = []
    if not chunks and vector_candidates == 0 and lexical_candidates == 0:
        logger.info(
            "rag.retrieve.no_matches",
            extra={"tenant_id": tenant_id, "case_id": case_id},
        )
        if not scoped_client:
            raise NotFoundError("No matching documents were found for the query.")

    parent_context: Dict[str, Dict[str, Any]] = {}
    if parent_requests:
        fetch_parents = getattr(tenant_client, "fetch_parent_context", None)
        if callable(fetch_parents):
            payload = {
                doc_id: sorted(parent_ids)
                for doc_id, parent_ids in parent_requests.items()
            }
            try:
                if scoped_client:
                    parent_context = fetch_parents(payload)  # type: ignore[misc]
                else:
                    parent_context = fetch_parents(tenant_id, payload)  # type: ignore[misc]
            except TypeError:
                try:
                    parent_context = fetch_parents(payload)  # type: ignore[misc]
                except Exception as exc:  # pragma: no cover - defensive logging
                    logger.warning(
                        "rag.retrieve.parent_lookup_failed",
                        extra={
                            "tenant_id": tenant_id,
                            "case_id": case_id,
                            "doc_count": len(parent_requests),
                            "error": str(exc),
                        },
                    )
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning(
                    "rag.retrieve.parent_lookup_failed",
                    extra={
                        "tenant_id": tenant_id,
                        "case_id": case_id,
                        "doc_count": len(parent_requests),
                        "error": str(exc),
                    },
                )

    matches = [_chunk_to_match(chunk) for chunk in chunks]
    deduplicated = _deduplicate_matches(matches)
    diversified = _apply_diversification(
        deduplicated,
        top_k=hybrid_config.top_k,
        strength=hybrid_config.diversify_strength,
    )
    final_matches = diversified[: hybrid_config.top_k]
    final_matches = _filter_matches_by_permissions(
        final_matches,
        context=context,
        permission_type="VIEW",
    )

    if parent_context:
        for match in final_matches:
            meta_section = match.get("meta")
            if meta_section is None:
                meta_section = {}
                match["meta"] = meta_section
            elif not isinstance(meta_section, dict):
                try:
                    meta_section = dict(meta_section)
                except Exception:
                    meta_section = {}
                match["meta"] = meta_section
            doc_identifier = _coerce_str(meta_section.get("document_id"))
            if not doc_identifier:
                continue
            parent_candidates = meta_section.get("parent_ids")
            if not isinstance(parent_candidates, Iterable) or isinstance(
                parent_candidates, (str, bytes)
            ):
                continue
            ordered_ids: list[str] = []
            for candidate in parent_candidates:
                parent_id = _coerce_str(candidate)
                if parent_id:
                    ordered_ids.append(parent_id)
            if not ordered_ids:
                continue
            doc_parents = parent_context.get(doc_identifier)
            if not isinstance(doc_parents, Mapping):
                continue
            parents_payload = [
                doc_parents[parent_id]
                for parent_id in ordered_ids
                if parent_id in doc_parents
            ]
            if parents_payload:
                meta_section["parents"] = parents_payload
                meta_section["parent_ids"] = ordered_ids

    routing_meta = _resolve_routing_metadata(
        tenant_id=tenant_id,
        process=process,
        doc_class=doc_class,
        collection_id=collection_id,
        workflow_id=workflow_id,
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
        "top_k_effective": hybrid_config.top_k,
        "matches_returned": len(final_matches),
        "max_candidates_effective": hybrid_config.max_candidates,
        "vector_candidates": vector_candidates,
        "lexical_candidates": lexical_candidates,
        "routing": routing_meta,
        "deleted_matches_blocked": deleted_matches_blocked,
        "visibility_effective": visibility_effective,
        "took_ms": took_ms,
        "diversify_strength": hybrid_config.diversify_strength,
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
