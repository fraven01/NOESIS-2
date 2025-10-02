import pytest

from ai_core.graphs import rag_demo
from ai_core.rag.vector_client import HybridSearchResult


class _DummyRouter:
    def __init__(self) -> None:
        self.calls = 0
        self.received_kwargs: dict | None = None

    def hybrid_search(self, query: str, **kwargs):
        self.calls += 1
        self.received_kwargs = dict(kwargs)
        top_k = int(kwargs.get("top_k", 0) or 0)
        vec_limit = kwargs.get("vec_limit", top_k)
        lex_limit = kwargs.get("lex_limit", top_k)
        return HybridSearchResult(
            chunks=[],
            vector_candidates=0,
            lexical_candidates=0,
            fused_candidates=0,
            duration_ms=0.0,
            alpha=float(kwargs.get("alpha", 0.0) or 0.0),
            min_sim=float(kwargs.get("min_sim", 0.0) or 0.0),
            vec_limit=int(vec_limit),
            lex_limit=int(lex_limit),
        )


@pytest.mark.django_db
def test_run_clamps_limits_before_router_call(monkeypatch, settings):
    settings.RAG_MAX_CANDIDATES = 25
    router = _DummyRouter()
    monkeypatch.setattr(rag_demo, "get_default_router", lambda: router)

    state = {
        "query": "Which limits?",
        "vec_limit": 9999,
        "lex_limit": 12345,
        "top_k": 10,
    }
    meta = {"tenant_id": "tenant-123"}

    new_state, result = rag_demo.run(state, meta)

    assert router.calls == 1
    assert router.received_kwargs is not None
    assert router.received_kwargs["vec_limit"] == 25
    assert router.received_kwargs["lex_limit"] == 25
    assert new_state["rag_demo"]["top_k"] == 10
    assert result["ok"] is True
