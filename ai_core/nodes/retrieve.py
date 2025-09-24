from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from ai_core.rag.vector_client import PgVectorClient


def run(
    state: Dict[str, Any],
    meta: Dict[str, str],
    *,
    client: Optional[PgVectorClient],
    top_k: int = 5,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Search the vector client for relevant snippets.

    Parameters
    ----------
    state:
        Must contain ``query`` with the search string.
    meta:
        Contains ``tenant`` and ``case`` identifiers.
    client:
        Vector client used for search.
    top_k:
        Number of snippets to return.
    """
    if client is None:
        raise ValueError("A PgVectorClient instance is required for retrieval")
    query = state.get("query", "")
    filters = {"tenant": meta.get("tenant"), "case": meta.get("case")}
    chunks = client.search(query, filters=filters, top_k=top_k)
    snippets = [{"text": c.content, "source": c.meta.get("source", "")} for c in chunks]
    new_state = dict(state)
    new_state["snippets"] = snippets
    confidence = 1.0 if snippets else 0.0
    return new_state, {"snippets": snippets, "confidence": confidence}
