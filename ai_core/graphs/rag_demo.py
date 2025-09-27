from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

try:
    from ai_core.rag.vector_store import get_default_router
except ImportError:  # pragma: no cover - optional dependency for demo mode
    get_default_router = None  # type: ignore[assignment]


QueryState = Dict[str, Any]
Meta = Dict[str, Any]
GraphResult = Dict[str, Any]


_QUERY_KEYS: Tuple[str, ...] = ("query", "question", "q", "text")


def _extract_query(state: QueryState) -> str | None:
    for key in _QUERY_KEYS:
        value = state.get(key)
        if isinstance(value, str):
            value = value.strip()
        if value:
            return str(value)
    return None


def _coerce_top_k(state: QueryState, default: int = 5) -> int:
    candidate = state.get("top_k") or state.get("k")
    if candidate is None:
        return default
    try:
        top_k = int(candidate)
    except (TypeError, ValueError):
        return default
    return max(1, top_k)


def _truncate(text: str, limit: int = 500) -> str:
    if len(text) <= limit:
        return text
    return text[:limit]


def _chunk_matches(chunks: Iterable[Any]) -> List[Dict[str, Any]]:
    matches: List[Dict[str, Any]] = []
    for index, chunk in enumerate(chunks):
        content = getattr(chunk, "content", "") or ""
        meta = getattr(chunk, "meta", {}) or {}
        identifier = meta.get("id") or meta.get("hash") or f"chunk-{index}"
        score = meta.get("score")
        try:
            score_value = float(score) if score is not None else 0.0
        except (TypeError, ValueError):
            score_value = 0.0
        matches.append(
            {
                "id": str(identifier),
                "score": score_value,
                "text": _truncate(str(content)),
                "metadata": dict(meta),
            }
        )
    return matches


def _demo_matches(query: str, tenant_id: str, *, top_k: int) -> List[Dict[str, Any]]:
    demo_corpus = [
        {
            "id": "demo-1",
            "score": 0.42,
            "text": _truncate(
                (
                    "Demo knowledge base entry describing how retrieval works for "
                    "tenant '%s'. Query: '%s'."
                )
                % (tenant_id, query)
            ),
            "metadata": {"tenant_id": tenant_id, "source": "demo"},
        },
        {
            "id": "demo-2",
            "score": 0.36,
            "text": _truncate(
                "Second demo snippet outlining the behaviour of the RAG demo node."
            ),
            "metadata": {"tenant_id": tenant_id, "source": "demo"},
        },
    ]
    return demo_corpus[:top_k]


def run(state: QueryState, meta: Meta) -> Tuple[QueryState, GraphResult]:
    query = _extract_query(state)
    if not query:
        return state, {
            "ok": False,
            "query": None,
            "matches": [],
            "error": "missing_query",
        }

    tenant_id = meta.get("tenant_id") or meta.get("tenant")
    if not tenant_id:
        raise ValueError("tenant_id missing in meta")

    top_k = _coerce_top_k(state)
    project_id = state.get("project_id")

    matches: List[Dict[str, Any]]
    router_error: str | None = None

    if get_default_router is not None:
        filters = {"tenant_id": tenant_id}
        if project_id:
            filters["project_id"] = project_id

        try:
            router = get_default_router()
        except Exception as exc:  # pragma: no cover - defensive fallback
            router = None
            router_error = str(exc)

        if router is not None:
            scoped_router = router
            for_tenant = getattr(router, "for_tenant", None)
            if callable(for_tenant):
                try:
                    scoped_router = for_tenant(tenant_id)
                except TypeError:
                    try:
                        scoped_router = for_tenant(tenant_id=tenant_id)
                    except TypeError:
                        scoped_router = for_tenant()
                except Exception as exc:  # pragma: no cover - defensive fallback
                    router_error = str(exc)
                    scoped_router = None

            search = (
                getattr(scoped_router, "search", None)
                if scoped_router is not None
                else None
            )
            if callable(search):
                case_id = meta.get("case_id") or meta.get("case")
                try:
                    chunks = search(
                        query,
                        tenant_id=tenant_id,
                        case_id=case_id,
                        top_k=top_k,
                        filters=filters,
                    )
                    matches = _chunk_matches(chunks)
                except Exception as exc:  # pragma: no cover - defensive fallback
                    router_error = str(exc)
                    matches = _demo_matches(query, str(tenant_id), top_k=top_k)
            else:
                if router_error is None:
                    router_error = "router missing search"
                matches = _demo_matches(query, str(tenant_id), top_k=top_k)
        else:
            matches = _demo_matches(query, str(tenant_id), top_k=top_k)
    else:
        matches = _demo_matches(query, str(tenant_id), top_k=top_k)

    new_state = dict(state)
    new_state["rag_demo"] = {
        "query": query,
        "top_k": top_k,
        "retrieved_count": len(matches),
    }

    result: GraphResult = {"ok": True, "query": query, "matches": matches}
    if router_error:
        result["error"] = router_error

    return new_state, result


__all__ = ["run"]
