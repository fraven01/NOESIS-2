"""Vector store abstractions for routing Retrieval-Augmented Generation data.

This module defines a `VectorStore` protocol along with a `VectorStoreRouter`
that can dispatch calls to scoped backends. The router enforces tenant
identifiers and caps retrieval sizes to keep usage predictable.

Example:
    >>> from ai_core.rag.schemas import Chunk
    >>> class InMemoryStore(VectorStore):
    ...     def __init__(self):
    ...         self._chunks: list[Chunk] = []
    ...     def upsert_chunks(self, chunks: Iterable[Chunk]) -> int:
    ...         self._chunks.extend(chunks)
    ...         return len(self._chunks)
    ...     def search(self, query: str, tenant_id: str, *, case_id: str | None = None,
    ...                top_k: int = 5, filters: Mapping[str, str | None] | None = None
    ...                ) -> list[Chunk]:
    ...         return self._chunks[:top_k]
    >>> router = VectorStoreRouter({"global": InMemoryStore()})
    >>> router.upsert_chunks([Chunk(content="hello", meta={"tenant": "t"})])
    1
    >>> router.search("hi", tenant_id="t")
    [Chunk(content='hello', meta={'tenant': 't'}, embedding=None)]
"""

from __future__ import annotations

import logging
from typing import Iterable, Mapping, Protocol

from ai_core.rag.schemas import Chunk

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
        filters: Mapping[str, str | None] | None = None,
    ) -> list[Chunk]:
        """Return the most relevant chunks for a query."""

    def close(self) -> None:
        """Release underlying resources if applicable."""


class VectorStoreRouter:
    """Route vector store operations to scoped backends.

    Args:
        stores: Mapping of scope names to :class:`VectorStore` implementations.
        default_scope: Name of the scope that receives upsert operations and
            serves as fallback for unknown scopes.

    The router guarantees tenant enforcement, filter normalisation and a
    defensive cap on ``top_k`` values (minimum 1, maximum 10).
    """

    def __init__(self, stores: Mapping[str, VectorStore], default_scope: str = "global"):
        if default_scope not in stores:
            msg = "default_scope '%s' is not present in provided stores"
            raise ValueError(msg % default_scope)
        self._stores = dict(stores)
        self._default_scope = default_scope
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

    def search(
        self,
        query: str,
        tenant_id: str,
        *,
        case_id: str | None = None,
        top_k: int = 5,
        filters: Mapping[str, str | None] | None = None,
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
            normalised_filters = {key: value or None for key, value in filters.items()}

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
        return store.search(
            query,
            tenant_id,
            case_id=case_id,
            top_k=capped_top_k,
            filters=normalised_filters,
        )

    def upsert_chunks(self, chunks: Iterable[Chunk]) -> int:
        """Delegate writes to the default scope store."""

        logger.debug("Upserting chunks", extra={"scope": self._default_scope})
        return self._stores[self._default_scope].upsert_chunks(chunks)

    def close(self) -> None:
        """Close all scoped stores if they expose a ``close`` method."""

        for scope, store in self._stores.items():
            close = getattr(store, "close", None)
            if callable(close):
                logger.debug("Closing vector store scope", extra={"scope": scope})
                close()


def get_default_router() -> VectorStoreRouter:
    """Return a router configured with the default pgvector backend."""

    from .vector_client import get_default_client

    client = get_default_client()
    return VectorStoreRouter({"global": client}, default_scope="global")


__all__ = ["VectorStore", "VectorStoreRouter", "get_default_router"]
