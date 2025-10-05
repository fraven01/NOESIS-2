"""Vector store abstractions for routing Retrieval-Augmented Generation data."""

from __future__ import annotations

import atexit
import inspect
import logging
from typing import Dict, Iterable, Mapping, NoReturn, Protocol, TYPE_CHECKING

from ai_core.rag.schemas import Chunk
from ai_core.rag.limits import clamp_fraction, get_limit_setting
from . import metrics
from .router_validation import (
    RouterInputError,
    emit_router_validation_failure,
    validate_search_inputs,
)

if TYPE_CHECKING:
    from ai_core.rag.vector_client import HybridSearchResult

logger = logging.getLogger(__name__)


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
            )
        except RouterInputError as exc:
            _raise_router_error(exc)

        tenant = validation.tenant_id
        validation_context = validation.context
        requested_top_k = validation.top_k
        capped_top_k = validation.effective_top_k
        top_k_source = validation.top_k_source
        normalised_filters = None
        if filters is not None:
            normalised_filters = {}
            for key, value in filters.items():
                if isinstance(value, str):
                    normalised_filters[key] = value or None
                else:
                    normalised_filters[key] = value

        logger.debug(
            "Vector search",
            extra={
                "tenant_id": tenant,
                "scope": scope,
                "process": validation_context.get("process"),
                "doc_class": validation_context.get("doc_class"),
                "top_k_requested": requested_top_k
                if requested_top_k is not None
                else capped_top_k,
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
    ) -> "HybridSearchResult":
        try:
            validation = validate_search_inputs(
                tenant_id=tenant_id,
                process=process,
                doc_class=doc_class,
                top_k=top_k,
                max_candidates=max_candidates,
            )
        except RouterInputError as exc:
            _raise_router_error(exc)

        tenant = validation.tenant_id
        sanitized_top_k = validation.top_k
        sanitized_max = validation.max_candidates
        validation_context = validation.context

        normalized_top_k = validation.effective_top_k
        top_k_source = validation.top_k_source
        max_candidates_value = validation.effective_max_candidates
        max_candidates_source = validation.max_candidates_source
        normalised_filters = None
        if filters is not None:
            normalised_filters = {}
            for key, value in filters.items():
                if isinstance(value, str):
                    normalised_filters[key] = value or None
                else:
                    normalised_filters[key] = value

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
                return result
            logger.warning(
                "rag.hybrid.router.no_result",
                extra={
                    "scope": scope,
                    "tenant": tenant,
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
        return _HybridSearchResult(
            chunks=list(fallback_chunks),
            vector_candidates=len(fallback_chunks),
            lexical_candidates=0,
            fused_candidates=len(fallback_chunks),
            duration_ms=0.0,
            alpha=float(alpha_value),
            min_sim=float(min_sim_value),
            vec_limit=effective_vec,
            lex_limit=effective_lex,
        )

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
            tenant_meta = str(chunk.meta.get("tenant") or "").strip()
            if not tenant_meta:
                raise ValueError("chunk metadata must include tenant")
            if expected_tenant is not None and tenant_meta != expected_tenant:
                raise ValueError(
                    "Chunk tenant '%s' does not match expected tenant '%s'"
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
        )

    def upsert_chunks(self, chunks: Iterable[Chunk]) -> int:
        chunk_list = list(chunks)
        coerced: list[Chunk] = []
        for chunk in chunk_list:
            meta = dict(chunk.meta)
            tenant_meta_raw = meta.get("tenant")
            tenant_meta = str(tenant_meta_raw).strip() if tenant_meta_raw else ""
            if tenant_meta and tenant_meta != self._tenant_id:
                msg = "Chunk tenant '%s' does not match scoped tenant '%s'" % (
                    tenant_meta,
                    self._tenant_id,
                )
                raise ValueError(msg)
            meta["tenant"] = self._tenant_id
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
        return PgVectorClient(str(dsn), **kwargs)

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
    except RuntimeError:
        logger.info(
            "Falling back to shared pgvector client for scope %s; env vars %s/%s unset",
            scope,
            env_var,
            fallback_env_var,
        )
        return get_default_client()


def reset_default_router() -> None:
    """Reset cached router state and close stores if needed."""

    # Routers are currently built on demand without caching. The hook ensures
    # compatibility with potential future caching and mirrors the client reset
    # helper used in tests.
    logger.debug("reset_default_router called - no cached router to clear")


__all__ = [
    "VectorStore",
    "VectorStoreRouter",
    "get_default_router",
    "TenantScopedVectorStore",
]


atexit.register(reset_default_router)
