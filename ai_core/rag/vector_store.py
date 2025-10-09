"""Vector store abstractions for routing Retrieval-Augmented Generation data."""

from __future__ import annotations

import atexit
import inspect
import logging
from typing import Dict, Iterable, Mapping, NoReturn, Protocol, TYPE_CHECKING, Any

from psycopg2 import OperationalError

from ai_core.rag.schemas import Chunk
from ai_core.rag.limits import clamp_fraction, get_limit_setting
from ai_core.rag.visibility import DEFAULT_VISIBILITY, Visibility
from common.logging import get_log_context
from ai_core.infra import tracing
from . import metrics
from .router_validation import (
    RouterInputError,
    SearchValidationResult,
    emit_router_validation_failure,
    validate_search_inputs,
)

if TYPE_CHECKING:
    from ai_core.rag.vector_client import HybridSearchResult

logger = logging.getLogger(__name__)

_EXTENDED_VISIBILITY = {Visibility.ALL, Visibility.DELETED}


class NullVectorStore:
    """Fallback store that returns empty results when no backend is available."""

    def __init__(self, scope: str) -> None:
        self._scope = scope
        self.name = f"null:{scope}"
        logger.warning(
            "Null vector store initialised; hybrid search will return no results",
            extra={"scope": scope},
        )

    def search(
        self,
        query: str,
        tenant_id: str,
        *,
        case_id: str | None = None,
        top_k: int = 5,
        filters: Mapping[str, object | None] | None = None,
    ) -> list[Chunk]:
        logger.debug(
            "Null vector store search invoked",
            extra={"scope": self._scope, "tenant_id": tenant_id},
        )
        return []

    def hybrid_search(
        self,
        query: str,
        tenant_id: str,
        *,
        case_id: str | None = None,
        top_k: int = 5,
        filters: Mapping[str, object | None] | None = None,
        alpha: float | None = None,
        min_sim: float | None = None,
        vec_limit: int | None = None,
        lex_limit: int | None = None,
        trgm_limit: float | None = None,
        trgm_threshold: float | None = None,
        max_candidates: int | None = None,
        process: str | None = None,
        doc_class: str | None = None,
        visibility: object | None = None,
        visibility_override_allowed: bool = False,
    ) -> "HybridSearchResult":
        from .vector_client import HybridSearchResult

        logger.debug(
            "Null vector store hybrid search invoked",
            extra={"scope": self._scope, "tenant_id": tenant_id},
        )

        try:
            effective_top_k = int(top_k)
        except (TypeError, ValueError):
            effective_top_k = 5
        effective_top_k = min(max(1, effective_top_k), 10)

        alpha_default = float(get_limit_setting("RAG_HYBRID_ALPHA", 0.7))
        min_sim_default = float(get_limit_setting("RAG_MIN_SIM", 0.15))

        try:
            alpha_value = float(alpha) if alpha is not None else alpha_default
        except (TypeError, ValueError):
            alpha_value = alpha_default

        try:
            min_sim_value = (
                float(min_sim) if min_sim is not None else min_sim_default
            )
        except (TypeError, ValueError):
            min_sim_value = min_sim_default

        def _coerce_limit(value: object | None) -> int:
            try:
                return int(value) if value is not None else effective_top_k
            except (TypeError, ValueError):
                return effective_top_k

        vec_limit_value = _coerce_limit(vec_limit)
        lex_limit_value = _coerce_limit(lex_limit)

        try:
            max_candidates_value = (
                int(max_candidates) if max_candidates is not None else effective_top_k
            )
        except (TypeError, ValueError):
            max_candidates_value = effective_top_k
        max_candidates_value = max(effective_top_k, max_candidates_value)

        vec_limit_value = min(max_candidates_value, max(vec_limit_value, effective_top_k))
        lex_limit_value = min(max_candidates_value, max(lex_limit_value, effective_top_k))

        if isinstance(visibility, Visibility):
            visibility_mode = visibility
        else:
            try:
                visibility_text = str(visibility or "").strip().lower()
            except Exception:
                visibility_text = ""
            try:
                visibility_mode = Visibility(visibility_text)
            except ValueError:
                visibility_mode = DEFAULT_VISIBILITY

        result = HybridSearchResult(
            chunks=[],
            vector_candidates=0,
            lexical_candidates=0,
            fused_candidates=0,
            duration_ms=0.0,
            alpha=float(alpha_value),
            min_sim=float(min_sim_value),
            vec_limit=vec_limit_value,
            lex_limit=lex_limit_value,
            visibility=visibility_mode.value,
        )
        result.deleted_matches_blocked = 0
        return result

    def upsert_chunks(self, chunks: Iterable[Chunk]) -> int:
        chunk_list = list(chunks)
        logger.debug(
            "Null vector store upsert invoked",
            extra={"scope": self._scope, "chunk_count": len(chunk_list)},
        )
        return 0

    def health_check(self) -> bool:
        return False

    def close(self) -> None:
        return None


def _coerce_int(value: object | None) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: object | None) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _emit_retrieval_span(
    *,
    tenant_id: str,
    scope: str,
    case_id: str | None,
    context: Mapping[str, object | None],
    result: Any,
) -> None:
    trace_id = (get_log_context().get("trace_id") or "").strip()
    if not trace_id:
        return

    metadata: dict[str, object | None] = {
        "tenant_id": tenant_id,
        "scope": scope,
        "case_id": case_id,
        "process": context.get("process"),
        "doc_class": context.get("doc_class"),
        "visibility_requested": context.get("visibility_requested"),
        "visibility_source": context.get("visibility_source"),
        "visibility_effective": context.get("visibility_effective"),
        "visibility_override_allowed": bool(context.get("visibility_override_allowed")),
    }

    requested_top_k = context.get("top_k_requested")
    if requested_top_k is not None:
        coerced = _coerce_int(requested_top_k)
        if coerced is not None:
            metadata["top_k_requested"] = coerced

    effective_top_k = context.get("top_k_effective")
    if effective_top_k is not None:
        coerced = _coerce_int(effective_top_k)
        if coerced is not None:
            metadata["top_k_effective"] = coerced

    max_candidates_effective = context.get("max_candidates_effective")
    if max_candidates_effective is not None:
        coerced = _coerce_int(max_candidates_effective)
        if coerced is not None:
            metadata["max_candidates_effective"] = coerced

    alpha_value = context.get("alpha_effective")
    coerced_alpha = _coerce_float(alpha_value)
    if coerced_alpha is not None:
        metadata["alpha"] = coerced_alpha

    min_sim_value = context.get("min_sim_effective")
    coerced_min_sim = _coerce_float(min_sim_value)
    if coerced_min_sim is not None:
        metadata["min_sim"] = coerced_min_sim

    trgm_value = context.get("trgm_limit_effective")
    coerced_trgm = _coerce_float(trgm_value)
    if coerced_trgm is not None:
        metadata["trgm_limit"] = coerced_trgm

    vector_candidates = _coerce_int(getattr(result, "vector_candidates", None))
    if vector_candidates is not None:
        metadata["vector_candidates"] = vector_candidates

    lexical_candidates = _coerce_int(getattr(result, "lexical_candidates", None))
    if lexical_candidates is not None:
        metadata["lexical_candidates"] = lexical_candidates

    fused_candidates = _coerce_int(getattr(result, "fused_candidates", None))
    if fused_candidates is not None:
        metadata["fused_candidates"] = fused_candidates

    duration_ms = _coerce_float(getattr(result, "duration_ms", None))
    if duration_ms is not None:
        metadata["duration_ms"] = duration_ms

    below_cutoff = _coerce_int(getattr(result, "below_cutoff", None))
    if below_cutoff is not None:
        metadata["below_cutoff"] = below_cutoff

    returned_after_cutoff = _coerce_int(getattr(result, "returned_after_cutoff", None))
    if returned_after_cutoff is not None:
        metadata["returned_after_cutoff"] = returned_after_cutoff

    deleted_matches_blocked = _coerce_int(getattr(result, "deleted_matches_blocked", 0))
    metadata["deleted_matches_blocked"] = (
        deleted_matches_blocked if deleted_matches_blocked is not None else 0
    )

    visibility = getattr(result, "visibility", None)
    if isinstance(visibility, str):
        metadata["visibility_effective"] = visibility

    query_embedding_empty = getattr(result, "query_embedding_empty", None)
    if isinstance(query_embedding_empty, bool):
        metadata["query_embedding_empty"] = query_embedding_empty

    tracing.emit_span(
        trace_id=trace_id,
        node_name="rag.hybrid.search",
        metadata=metadata,
    )


def _raise_router_error(error: RouterInputError) -> NoReturn:
    """Emit tracing/logging metadata before re-raising router errors."""

    emit_router_validation_failure(error)
    logger.warning(
        "rag.router.invalid_search_input",
        extra={
            "code": error.code,
            "field": error.field or "-",
            **{k: v for k, v in error.context.items() if v is not None},
        },
    )
    raise error


class VectorStore(Protocol):
    """Protocol describing the persistence layer used for RAG retrieval.

    Implementations are responsible for persisting and retrieving :class:`Chunk`
    instances. They do not perform tenant validation or scope routing – that is
    handled by :class:`VectorStoreRouter`.
    """

    def upsert_chunks(self, chunks: Iterable[Chunk]) -> int:
        """Insert or update chunks and return the number of stored items."""

    def search(
        self,
        query: str,
        tenant_id: str,
        *,
        case_id: str | None = None,
        top_k: int = 5,
        filters: Mapping[str, object | None] | None = None,
    ) -> list[Chunk]:
        """Return the most relevant chunks for a query."""

    def hybrid_search(
        self,
        query: str,
        tenant_id: str,
        *,
        case_id: str | None = None,
        top_k: int = 5,
        filters: Mapping[str, object | None] | None = None,
        alpha: float | None = None,
        min_sim: float | None = None,
        vec_limit: int | None = None,
        lex_limit: int | None = None,
        trgm_limit: float | None = None,
        trgm_threshold: float | None = None,
        max_candidates: int | None = None,
    ) -> "HybridSearchResult":
        """Execute a hybrid semantic/lexical search."""

    def close(self) -> None:
        """Release underlying resources if applicable."""


class TenantScopedVectorStore(Protocol):
    """Protocol for clients that are already bound to a tenant context."""

    def search(
        self,
        query: str,
        tenant_id: str | None = None,
        *,
        case_id: str | None = None,
        top_k: int = 5,
        filters: Mapping[str, object | None] | None = None,
    ) -> list[Chunk]:
        """Search within the tenant scope."""

    def upsert_chunks(self, chunks: Iterable[Chunk]) -> int:
        """Insert or update chunks within the tenant scope."""

    def close(self) -> None:
        """Release underlying resources if applicable."""

    def hybrid_search(
        self,
        query: str,
        *,
        case_id: str | None = None,
        top_k: int = 5,
        filters: Mapping[str, object | None] | None = None,
        alpha: float | None = None,
        min_sim: float | None = None,
        vec_limit: int | None = None,
        lex_limit: int | None = None,
        trgm_limit: float | None = None,
        trgm_threshold: float | None = None,
        max_candidates: int | None = None,
    ) -> "HybridSearchResult":
        """Execute a hybrid search within the tenant scope."""


class VectorStoreRouter:
    """Route vector store operations to scoped backends.

    Args:
        stores: Mapping of scope names to :class:`VectorStore` implementations.
        default_scope: Name of the scope that receives upsert operations and
            serves as fallback for unknown scopes.
        tenant_scopes: Optional explicit mapping of tenant identifiers to scope
            names. Useful when large tenants are isolated in dedicated silos.
        schema_scopes: Optional mapping of tenant schema names to scope names.

    The router guarantees tenant enforcement, filter normalisation and a
    defensive cap on ``top_k`` values (minimum 1, maximum 10).
    """

    def __init__(
        self,
        stores: Mapping[str, VectorStore],
        default_scope: str = "global",
        *,
        tenant_scopes: Mapping[str, str] | None = None,
        schema_scopes: Mapping[str, str] | None = None,
    ):
        if default_scope not in stores:
            msg = "default_scope '%s' is not present in provided stores"
            raise ValueError(msg % default_scope)
        self._stores = dict(stores)
        self._default_scope = default_scope
        self._tenant_scopes = {
            str(key): value for key, value in (tenant_scopes or {}).items()
        }
        self._schema_scopes = {
            str(key): value for key, value in (schema_scopes or {}).items()
        }
        logger.debug(
            "VectorStoreRouter initialised",
            extra={"default_scope": default_scope, "scopes": list(self._stores)},
        )

    @property
    def default_scope(self) -> str:
        """Return the fallback scope name."""

        return self._default_scope

    def _get_store(self, scope: str) -> VectorStore:
        if scope in self._stores:
            return self._stores[scope]
        logger.debug("Scope '%s' missing, falling back to default", scope)
        return self._stores[self._default_scope]

    def _resolve_scope(
        self, tenant_id: str | None, tenant_schema: str | None
    ) -> str | None:
        if tenant_schema and tenant_schema in self._schema_scopes:
            return self._schema_scopes[tenant_schema]
        if tenant_id and tenant_id in self._tenant_scopes:
            return self._tenant_scopes[tenant_id]
        return None

    def _apply_visibility_guard(
        self,
        *,
        validation: SearchValidationResult,
        tenant: str,
        scope: str,
        override_allowed: bool,
    ) -> Visibility:
        requested_visibility = validation.visibility
        context = validation.context
        context["visibility_requested"] = requested_visibility.value
        context["visibility_source"] = validation.visibility_source
        context["visibility_override_allowed"] = override_allowed

        effective_visibility = requested_visibility
        if requested_visibility in _EXTENDED_VISIBILITY and not override_allowed:
            effective_visibility = DEFAULT_VISIBILITY
            logger.debug(
                "rag.visibility.override_denied",
                extra={
                    "tenant": tenant,
                    "requested": requested_visibility.value,
                    "scope": scope,
                },
            )

        context["visibility_effective"] = effective_visibility.value
        return effective_visibility

    def search(
        self,
        query: str,
        tenant_id: str,
        *,
        case_id: str | None = None,
        top_k: int = 5,
        filters: Mapping[str, object | None] | None = None,
        scope: str = "global",
        process: str | None = None,
        doc_class: str | None = None,
        visibility: object | None = None,
        visibility_override_allowed: bool = False,
    ) -> list[Chunk]:
        """Search within the given scope while enforcing tenant and limits.

        ``top_k`` is always capped to the inclusive range [1, 10]. Empty strings
        in ``filters`` are normalised to ``None`` so that backends can treat
        them uniformly.
        """

        try:
            validation = validate_search_inputs(
                tenant_id=tenant_id,
                process=process,
                doc_class=doc_class,
                top_k=top_k,
                visibility=visibility,
            )
        except RouterInputError as exc:
            _raise_router_error(exc)

        tenant = validation.tenant_id
        override_allowed = bool(visibility_override_allowed)
        effective_visibility = self._apply_visibility_guard(
            validation=validation,
            tenant=tenant,
            scope=scope,
            override_allowed=override_allowed,
        )
        validation_context = validation.context

        requested_top_k = validation.top_k
        requested_top_k = validation.top_k
        capped_top_k = validation.effective_top_k
        top_k_source = validation.top_k_source
        normalised_filters: dict[str, object | None] | None = None
        if filters is not None:
            normalised_filters = {}
            for key, value in filters.items():
                if isinstance(value, str):
                    normalised_filters[key] = value or None
                else:
                    normalised_filters[key] = value
        if normalised_filters is None:
            normalised_filters = {}
        normalised_filters["visibility"] = effective_visibility.value

        logger.debug(
            "Vector search",
            extra={
                "tenant_id": tenant,
                "scope": scope,
                "process": validation_context.get("process"),
                "doc_class": validation_context.get("doc_class"),
                "top_k_requested": (
                    requested_top_k if requested_top_k is not None else capped_top_k
                ),
                "top_k_effective": capped_top_k,
                "top_k_source": top_k_source,
                "case_id": case_id,
            },
        )

        store = self._get_store(scope)
        hybrid = getattr(store, "hybrid_search", None)
        if callable(hybrid):
            result = hybrid(
                query,
                tenant,
                case_id=case_id,
                top_k=capped_top_k,
                filters=normalised_filters,
            )
            if result is not None:
                setattr(result, "visibility", effective_visibility.value)
                return list(getattr(result, "chunks", result))
        return store.search(
            query,
            tenant,
            case_id=case_id,
            top_k=capped_top_k,
            filters=normalised_filters,
        )

    def hybrid_search(
        self,
        query: str,
        tenant_id: str,
        *,
        case_id: str | None = None,
        top_k: int = 5,
        filters: Mapping[str, object | None] | None = None,
        scope: str = "global",
        alpha: float | None = None,
        min_sim: float | None = None,
        vec_limit: int | None = None,
        lex_limit: int | None = None,
        trgm_limit: float | None = None,
        trgm_threshold: float | None = None,
        max_candidates: int | None = None,
        process: str | None = None,
        doc_class: str | None = None,
        visibility: object | None = None,
        visibility_override_allowed: bool = False,
    ) -> "HybridSearchResult":
        try:
            validation = validate_search_inputs(
                tenant_id=tenant_id,
                process=process,
                doc_class=doc_class,
                top_k=top_k,
                max_candidates=max_candidates,
                visibility=visibility,
            )
        except RouterInputError as exc:
            _raise_router_error(exc)

        tenant = validation.tenant_id
        validation_context = validation.context

        override_allowed = bool(visibility_override_allowed)
        effective_visibility = self._apply_visibility_guard(
            validation=validation,
            tenant=tenant,
            scope=scope,
            override_allowed=override_allowed,
        )
        validation_context["visibility_effective"] = effective_visibility.value
        validation_context["top_k_requested"] = validation.top_k

        normalized_top_k = validation.effective_top_k
        top_k_source = validation.top_k_source
        max_candidates_value = validation.effective_max_candidates
        max_candidates_source = validation.max_candidates_source
        validation_context["top_k_effective"] = normalized_top_k
        validation_context["top_k_source"] = top_k_source
        validation_context["max_candidates_effective"] = max_candidates_value
        validation_context["max_candidates_source"] = max_candidates_source
        normalised_filters: dict[str, object | None] | None = None
        if filters is not None:
            normalised_filters = {}
            for key, value in filters.items():
                if isinstance(value, str):
                    normalised_filters[key] = value or None
                else:
                    normalised_filters[key] = value
        if normalised_filters is None:
            normalised_filters = {}
        normalised_filters["visibility"] = effective_visibility.value

        alpha_default = float(get_limit_setting("RAG_HYBRID_ALPHA", 0.7))
        min_sim_default = float(get_limit_setting("RAG_MIN_SIM", 0.15))
        trgm_default = float(get_limit_setting("RAG_TRGM_LIMIT", 0.30))
        alpha_value, alpha_source = clamp_fraction(
            alpha, default=alpha_default, return_source=True
        )
        min_sim_value, min_sim_source = clamp_fraction(
            min_sim, default=min_sim_default, return_source=True
        )

        trgm_requested = trgm_limit if trgm_limit is not None else trgm_threshold
        trgm_value, trgm_source = clamp_fraction(
            trgm_requested, default=trgm_default, return_source=True
        )

        validation_context["alpha_effective"] = alpha_value
        validation_context["min_sim_effective"] = min_sim_value
        validation_context["trgm_limit_effective"] = trgm_value

        logger.debug(
            "rag.hybrid.params",
            extra={
                "tenant": tenant,
                "scope": scope,
                "process": validation_context.get("process"),
                "doc_class": validation_context.get("doc_class"),
                "case_id": case_id,
                "top_k": normalized_top_k,
                "top_k_source": top_k_source,
                "alpha": alpha_value,
                "alpha_source": alpha_source,
                "min_sim": min_sim_value,
                "min_sim_source": min_sim_source,
                "trgm_limit": trgm_value,
                "trgm_limit_source": trgm_source,
                "max_candidates": max_candidates_value,
                "max_candidates_source": max_candidates_source,
            },
        )

        store = self._get_store(scope)
        hybrid = getattr(store, "hybrid_search", None)
        if callable(hybrid):
            protocol_hybrid = getattr(VectorStore, "hybrid_search", None)
            store_hybrid = getattr(type(store), "hybrid_search", None)
            if store_hybrid is protocol_hybrid:
                hybrid = None
        if callable(hybrid):
            hybrid_kwargs = {
                "case_id": case_id,
                "top_k": normalized_top_k,
                "filters": normalised_filters,
                "alpha": alpha_value,
                "min_sim": min_sim_value,
                "vec_limit": vec_limit,
                "lex_limit": lex_limit,
                "trgm_limit": trgm_value,
                "max_candidates": max_candidates_value,
                "visibility": effective_visibility.value,
                "visibility_override_allowed": override_allowed,
            }
            try:
                signature = inspect.signature(hybrid)
            except (TypeError, ValueError):
                signature = None
            if signature is not None:
                accepts_var_kwargs = any(
                    parameter.kind is inspect.Parameter.VAR_KEYWORD
                    for parameter in signature.parameters.values()
                )
                if not accepts_var_kwargs:
                    allowed_keywords = {
                        name
                        for name, parameter in signature.parameters.items()
                        if parameter.kind
                        in (
                            inspect.Parameter.POSITIONAL_OR_KEYWORD,
                            inspect.Parameter.KEYWORD_ONLY,
                        )
                    }
                    # visibility is handled via dedicated parameters on the client
                    # even if older implementations haven't been updated yet. keep
                    # these keys so PgVectorClient.hybrid_search can consume them.
                    allowed_keywords.update(
                        {"visibility", "visibility_override_allowed"}
                    )
                    hybrid_kwargs = {
                        key: value
                        for key, value in hybrid_kwargs.items()
                        if key in allowed_keywords
                    }
            result = hybrid(
                query,
                tenant,
                **hybrid_kwargs,
            )
            if result is not None:
                setattr(result, "visibility", effective_visibility.value)
                validation_context["visibility_effective"] = getattr(
                    result, "visibility", effective_visibility.value
                )
                _emit_retrieval_span(
                    tenant_id=tenant,
                    scope=scope,
                    case_id=case_id,
                    context=validation_context,
                    result=result,
                )
                return result
            logger.warning(
                "rag.hybrid.router.no_result",
                extra={
                    "scope": scope,
                    "tenant_id": tenant,
                    "store": getattr(store, "name", scope),
                },
            )

        fallback_chunks = store.search(
            query,
            tenant,
            case_id=case_id,
            top_k=normalized_top_k,
            filters=normalised_filters,
        )
        from .vector_client import (
            HybridSearchResult as _HybridSearchResult,
        )  # noqa: WPS433

        fallback_max = max_candidates_value
        effective_vec = int(vec_limit if vec_limit is not None else normalized_top_k)
        effective_lex = int(lex_limit if lex_limit is not None else normalized_top_k)
        effective_vec = min(fallback_max, max(normalized_top_k, effective_vec))
        effective_lex = min(fallback_max, max(normalized_top_k, effective_lex))
        fallback_result = _HybridSearchResult(
            chunks=list(fallback_chunks),
            vector_candidates=len(fallback_chunks),
            lexical_candidates=0,
            fused_candidates=len(fallback_chunks),
            duration_ms=0.0,
            alpha=float(alpha_value),
            min_sim=float(min_sim_value),
            vec_limit=effective_vec,
            lex_limit=effective_lex,
            visibility=effective_visibility.value,
        )
        validation_context["visibility_effective"] = effective_visibility.value
        _emit_retrieval_span(
            tenant_id=tenant,
            scope=scope,
            case_id=case_id,
            context=validation_context,
            result=fallback_result,
        )
        return fallback_result

    def upsert_chunks(
        self,
        chunks: Iterable[Chunk],
        *,
        scope: str | None = None,
        tenant_id: str | None = None,
    ) -> int:
        """Delegate writes to the configured scope (default if omitted)."""

        target_scope = scope or self._default_scope
        chunk_list = list(chunks)
        expected_tenant = str(tenant_id).strip() if tenant_id is not None else None
        for chunk in chunk_list:
            tenant_meta = str(chunk.meta.get("tenant_id") or "").strip()
            if not tenant_meta:
                raise ValueError("chunk metadata must include tenant_id")
            if expected_tenant is not None and tenant_meta != expected_tenant:
                raise ValueError(
                    "Chunk tenant_id '%s' does not match expected tenant '%s'"
                    % (tenant_meta, expected_tenant)
                )
        logger.debug("Upserting chunks", extra={"scope": target_scope})
        return self._get_store(target_scope).upsert_chunks(chunk_list)

    def close(self) -> None:
        """Close all scoped stores if they expose a ``close`` method."""

        for scope, store in self._stores.items():
            close = getattr(store, "close", None)
            if callable(close):
                logger.debug("Closing vector store scope", extra={"scope": scope})
                close()

    def health_check(self) -> dict[str, bool]:
        """Run health checks for each configured scope."""

        results: dict[str, bool] = {}
        for scope, store in self._stores.items():
            check = getattr(store, "health_check", None)
            if not callable(check):
                results[scope] = True
                metrics.RAG_HEALTH_CHECKS.labels(scope=scope, status="success").inc()
                continue
            try:
                healthy = bool(check())
                results[scope] = healthy
                metrics.RAG_HEALTH_CHECKS.labels(
                    scope=scope,
                    status="success" if healthy else "failure",
                ).inc()
            except Exception:  # pragma: no cover - defensive logging
                logger.exception(
                    "Vector store health check failed", extra={"scope": scope}
                )
                results[scope] = False
                metrics.RAG_HEALTH_CHECKS.labels(scope=scope, status="failure").inc()
        return results

    def for_tenant(
        self, tenant_id: str, tenant_schema: str | None = None
    ) -> TenantScopedVectorStore:
        """Return a client bound to a specific tenant context."""

        if not tenant_id:
            raise ValueError("tenant_id is required for tenant routing")
        scope = self._resolve_scope(str(tenant_id), tenant_schema)
        return _TenantScopedClient(self, tenant_id=str(tenant_id), scope=scope)


class _TenantScopedClient:
    """Wrapper that binds vector store operations to a tenant context."""

    def __init__(
        self,
        router: VectorStoreRouter,
        *,
        tenant_id: str,
        scope: str | None,
    ) -> None:
        self._router = router
        self._tenant_id = tenant_id
        self._scope = scope

    def search(
        self,
        query: str,
        tenant_id: str | None = None,
        *,
        case_id: str | None = None,
        top_k: int = 5,
        filters: Mapping[str, object | None] | None = None,
        process: str | None = None,
        doc_class: str | None = None,
        visibility: object | None = None,
        visibility_override_allowed: bool = False,
    ) -> list[Chunk]:
        if tenant_id is not None:
            assert (
                tenant_id == self._tenant_id
            ), "Tenant scoped client cannot search as different tenant"
        return self._router.search(
            query,
            tenant_id=self._tenant_id,
            case_id=case_id,
            top_k=top_k,
            filters=filters,
            scope=self._scope or self._router.default_scope,
            process=process,
            doc_class=doc_class,
            visibility=visibility,
            visibility_override_allowed=visibility_override_allowed,
        )

    def hybrid_search(
        self,
        query: str,
        *,
        case_id: str | None = None,
        top_k: int = 5,
        filters: Mapping[str, object | None] | None = None,
        alpha: float | None = None,
        min_sim: float | None = None,
        vec_limit: int | None = None,
        lex_limit: int | None = None,
        trgm_limit: float | None = None,
        trgm_threshold: float | None = None,
        max_candidates: int | None = None,
        process: str | None = None,
        doc_class: str | None = None,
        visibility: object | None = None,
        visibility_override_allowed: bool = False,
    ) -> "HybridSearchResult":
        return self._router.hybrid_search(
            query,
            tenant_id=self._tenant_id,
            case_id=case_id,
            top_k=top_k,
            filters=filters,
            scope=self._scope or self._router.default_scope,
            alpha=alpha,
            min_sim=min_sim,
            vec_limit=vec_limit,
            lex_limit=lex_limit,
            trgm_limit=trgm_limit,
            trgm_threshold=trgm_threshold,
            max_candidates=max_candidates,
            process=process,
            doc_class=doc_class,
            visibility=visibility,
            visibility_override_allowed=visibility_override_allowed,
        )

    def upsert_chunks(self, chunks: Iterable[Chunk]) -> int:
        chunk_list = list(chunks)
        coerced: list[Chunk] = []
        for chunk in chunk_list:
            meta = dict(chunk.meta)
            tenant_meta_raw = meta.get("tenant_id")
            tenant_meta = str(tenant_meta_raw).strip() if tenant_meta_raw else ""
            if tenant_meta and tenant_meta != self._tenant_id:
                msg = "Chunk tenant_id '%s' does not match scoped tenant '%s'" % (
                    tenant_meta,
                    self._tenant_id,
                )
                raise ValueError(msg)
            meta["tenant_id"] = self._tenant_id
            coerced.append(
                Chunk(content=chunk.content, meta=meta, embedding=chunk.embedding)
            )
        return self._router.upsert_chunks(
            coerced,
            scope=self._scope,
            tenant_id=self._tenant_id,
        )

    def health_check(self) -> dict[str, bool]:
        return self._router.health_check()

    def close(self) -> None:
        self._router.close()


def get_default_router() -> VectorStoreRouter:
    """Return a router configured with the default pgvector backend."""

    stores_config: Dict[str, Dict[str, object]] = {}
    default_scope: str | None = None
    tenant_scopes: Dict[str, str] = {}
    schema_scopes: Dict[str, str] = {}

    try:  # pragma: no cover - requires Django settings
        from django.conf import settings  # type: ignore

        configured = getattr(settings, "RAG_VECTOR_STORES", None)
        if isinstance(configured, dict):
            stores_config = configured
        default_scope = getattr(settings, "RAG_VECTOR_DEFAULT_SCOPE", None)
    except Exception:
        stores_config = {}

    if not stores_config:
        stores_config = {"global": {"backend": "pgvector"}}

    stores: Dict[str, VectorStore] = {}
    for scope_name, config in stores_config.items():
        backend = str(config.get("backend", "")).lower()
        if backend != "pgvector":
            raise ValueError(
                f"Unsupported vector store backend '{backend}' for scope '{scope_name}'"
            )
        stores[scope_name] = _build_pgvector_store(scope_name, config)
        if config.get("default") and default_scope is None:
            default_scope = scope_name
        for tenant_value in config.get("tenants", []):
            tenant_scopes[str(tenant_value)] = scope_name
        for schema_value in config.get("schemas", []):
            schema_scopes[str(schema_value)] = scope_name

    if default_scope is None:
        default_scope = "global" if "global" in stores else next(iter(stores))

    router = VectorStoreRouter(
        stores,
        default_scope=default_scope,
        tenant_scopes=tenant_scopes,
        schema_scopes=schema_scopes,
    )
    return router


def _build_pgvector_store(scope: str, config: Mapping[str, object]) -> VectorStore:
    from .vector_client import PgVectorClient, get_default_client

    dsn = config.get("dsn")
    kwargs: Dict[str, object] = {}

    for key in (
        "schema",
        "minconn",
        "maxconn",
        "statement_timeout_ms",
        "retries",
        "retry_base_delay_ms",
    ):
        if key in config:
            value = config[key]
            if (
                key
                in {
                    "minconn",
                    "maxconn",
                    "statement_timeout_ms",
                    "retries",
                    "retry_base_delay_ms",
                }
                and value is not None
            ):
                kwargs[key] = int(value)
            else:
                kwargs[key] = value

    if dsn:
        logger.info("Initialising pgvector store for scope %s via explicit DSN", scope)
        try:
            return PgVectorClient(str(dsn), **kwargs)
        except OperationalError as exc:
            logger.warning(
                "Falling back to null vector store for scope %s due to connection error",
                scope,
                extra={"scope": scope, "error": str(exc)},
            )
            return NullVectorStore(scope)

    env_var = str(config.get("dsn_env", "RAG_DATABASE_URL"))
    fallback_env_var = str(config.get("fallback_env", "DATABASE_URL"))
    try:
        logger.info(
            "Initialising pgvector store for scope %s via env var %s", scope, env_var
        )
        return PgVectorClient.from_env(
            env_var=env_var,
            fallback_env_var=fallback_env_var,
            **kwargs,
        )
    except RuntimeError as exc:
        logger.info(
            "Falling back to shared pgvector client for scope %s; env vars %s/%s unset",
            scope,
            env_var,
            fallback_env_var,
        )
        try:
            return get_default_client()
        except (RuntimeError, OperationalError) as shared_exc:
            logger.warning(
                "Shared pgvector client unavailable for scope %s; using null vector store",
                scope,
                extra={"scope": scope, "error": str(shared_exc)},
            )
            return NullVectorStore(scope)
    except OperationalError as exc:
        logger.warning(
            "Failed to initialise pgvector store for scope %s; using null vector store",
            scope,
            extra={"scope": scope, "error": str(exc)},
        )
        return NullVectorStore(scope)


def reset_default_router() -> None:
    """Reset cached router state and close stores if needed."""

    # Routers are currently built on demand without caching. The hook ensures
    # compatibility with potential future caching and mirrors the client reset
    # helper used in tests.
    logger.debug("reset_default_router called - no cached router to clear")


__all__ = [
    "NullVectorStore",
    "VectorStore",
    "VectorStoreRouter",
    "get_default_router",
    "TenantScopedVectorStore",
]


atexit.register(reset_default_router)
