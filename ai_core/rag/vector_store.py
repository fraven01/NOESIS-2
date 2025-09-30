"""Vector store abstractions for routing Retrieval-Augmented Generation data."""

from __future__ import annotations

import atexit
import logging
from typing import Dict, Iterable, Mapping, Protocol, TYPE_CHECKING

from ai_core.rag.schemas import Chunk
from . import metrics

if TYPE_CHECKING:
    from ai_core.rag.vector_client import HybridSearchResult

logger = logging.getLogger(__name__)


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
    ) -> list[Chunk]:
        """Search within the given scope while enforcing tenant and limits.

        ``top_k`` is always capped to the inclusive range [1, 10]. Empty strings
        in ``filters`` are normalised to ``None`` so that backends can treat
        them uniformly.
        """

        if not tenant_id:
            raise ValueError("tenant_id is required for vector store access")

        capped_top_k = max(1, min(top_k, 10))
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
                "tenant_id": tenant_id,
                "scope": scope,
                "top_k_requested": top_k,
                "top_k_effective": capped_top_k,
                "case_id": case_id,
            },
        )

        store = self._get_store(scope)
        hybrid = getattr(store, "hybrid_search", None)
        if callable(hybrid):
            result = hybrid(
                query,
                tenant_id,
                case_id=case_id,
                top_k=capped_top_k,
                filters=normalised_filters,
            )
            return list(getattr(result, "chunks", result))
        return store.search(
            query,
            tenant_id,
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
    ) -> "HybridSearchResult":
        if not tenant_id:
            raise ValueError("tenant_id is required for vector store access")

        capped_top_k = max(1, min(top_k, 10))
        normalised_filters = None
        if filters is not None:
            normalised_filters = {}
            for key, value in filters.items():
                if isinstance(value, str):
                    normalised_filters[key] = value or None
                else:
                    normalised_filters[key] = value

        store = self._get_store(scope)
        hybrid = getattr(store, "hybrid_search", None)
        if callable(hybrid):
            return hybrid(
                query,
                tenant_id,
                case_id=case_id,
                top_k=capped_top_k,
                filters=normalised_filters,
                alpha=alpha,
                min_sim=min_sim,
                vec_limit=vec_limit,
                lex_limit=lex_limit,
            )

        fallback_chunks = store.search(
            query,
            tenant_id,
            case_id=case_id,
            top_k=capped_top_k,
            filters=normalised_filters,
        )
        from .vector_client import HybridSearchResult as _HybridSearchResult  # noqa: WPS433

        effective_vec = int(vec_limit if vec_limit is not None else capped_top_k)
        effective_lex = int(lex_limit if lex_limit is not None else capped_top_k)
        return _HybridSearchResult(
            chunks=list(fallback_chunks),
            vector_candidates=len(fallback_chunks),
            lexical_candidates=0,
            fused_candidates=len(fallback_chunks),
            duration_ms=0.0,
            alpha=float(alpha if alpha is not None else 1.0),
            min_sim=float(min_sim if min_sim is not None else 0.0),
            vec_limit=effective_vec,
            lex_limit=effective_lex,
        )

    def upsert_chunks(
        self, chunks: Iterable[Chunk], *, scope: str | None = None
    ) -> int:
        """Delegate writes to the configured scope (default if omitted)."""

        target_scope = scope or self._default_scope
        chunk_list = list(chunks)
        for chunk in chunk_list:
            tenant_meta = str(chunk.meta.get("tenant") or "").strip()
            if not tenant_meta:
                raise ValueError("chunk metadata must include tenant")
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
    ) -> list[Chunk]:
        effective_tenant = tenant_id or self._tenant_id
        return self._router.search(
            query,
            tenant_id=effective_tenant,
            case_id=case_id,
            top_k=top_k,
            filters=filters,
            scope=self._scope or self._router.default_scope,
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
        )

    def upsert_chunks(self, chunks: Iterable[Chunk]) -> int:
        chunk_list = list(chunks)
        coerced: list[Chunk] = []
        for chunk in chunk_list:
            meta = dict(chunk.meta)
            if not meta.get("tenant"):
                meta["tenant"] = self._tenant_id
                coerced.append(
                    Chunk(content=chunk.content, meta=meta, embedding=chunk.embedding)
                )
            else:
                coerced.append(chunk)
        return self._router.upsert_chunks(
            coerced,
            scope=self._scope,
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
