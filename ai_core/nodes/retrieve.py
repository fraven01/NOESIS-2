from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from common.logging import get_logger

from ai_core.rag.vector_store import get_default_router, VectorStoreRouter


logger = get_logger(__name__)


_ROUTER: VectorStoreRouter | None = None


def _get_router() -> VectorStoreRouter:
    global _ROUTER
    if _ROUTER is None:
        _ROUTER = get_default_router()
    return _ROUTER


def _reset_router_for_tests() -> None:
    """Reset the cached router singleton.

    Intended for use in tests that patch the default router factory.
    """

    global _ROUTER
    _ROUTER = None


def run(
    state: Dict[str, Any],
    meta: Dict[str, str],
    *,
    top_k: int = 5,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Search the vector store router for relevant snippets.

    Parameters
    ----------
    state:
        Must contain ``query`` with the search string.
    meta:
        Contains ``tenant`` and ``case`` identifiers.
    top_k:
        Number of snippets to return.
    """
    query = state.get("query", "")
    tenant_id = meta.get("tenant") or meta.get("tenant_id")
    if not tenant_id:
        raise ValueError("tenant_id required")
    case_id = meta.get("case") or meta.get("case_id")
    tenant_schema = meta.get("tenant_schema") or meta.get("schema")
    router = _get_router()
    tenant_client = router
    filters: Dict[str, Optional[str]] | None = None
    for_tenant = getattr(router, "for_tenant", None)
    if callable(for_tenant):
        tenant_client = for_tenant(tenant_id, tenant_schema)
        # Tenant-scoped clients already enforce the tenant context, so we only
        # inject the optional case filter here to avoid redundant constraints.
        filters = {"case": case_id} if case_id else None
    else:
        filters = {"tenant": tenant_id}
        if case_id:
            filters["case"] = case_id
    logger.debug(
        "Executing RAG retrieval",
        extra={"tenant_id": tenant_id, "case_id": case_id, "top_k": top_k},
    )
    search_kwargs = {"case_id": case_id, "top_k": top_k, "filters": filters}
    if tenant_client is router:
        chunks = tenant_client.search(
            query,
            tenant_id=tenant_id,
            **search_kwargs,
        )
    else:
        chunks = tenant_client.search(
            query,
            **search_kwargs,
        )
    snippets = []
    for chunk in chunks:
        chunk_meta = chunk.meta or {}
        snippet: Dict[str, Any] = {
            "text": chunk.content,
            "source": chunk_meta.get("source", ""),
            "score": chunk_meta.get("score"),
            "hash": chunk_meta.get("hash"),
            "id": chunk_meta.get("id"),
        }
        extra_meta = {
            key: value
            for key, value in chunk_meta.items()
            if key not in {"source", "score", "hash", "id"}
        }
        if extra_meta:
            snippet["meta"] = extra_meta
        snippets.append(snippet)
    new_state = dict(state)
    new_state["snippets"] = snippets
    confidence = 1.0 if snippets else 0.0
    return new_state, {"snippets": snippets, "confidence": confidence}
