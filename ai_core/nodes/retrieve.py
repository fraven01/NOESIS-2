from __future__ import annotations

from typing import Any, Dict, Tuple

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
    router = _get_router()
    logger.debug("Executing RAG retrieval", extra={"tenant_id": tenant_id, "top_k": top_k})
    chunks = router.search(
        query,
        tenant_id=tenant_id,
        case_id=case_id,
        top_k=top_k,
        filters=None,
    )
    snippets = [{"text": c.content, "source": c.meta.get("source", "")} for c in chunks]
    new_state = dict(state)
    new_state["snippets"] = snippets
    confidence = 1.0 if snippets else 0.0
    return new_state, {"snippets": snippets, "confidence": confidence}
