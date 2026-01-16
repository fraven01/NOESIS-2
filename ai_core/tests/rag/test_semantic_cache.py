from uuid import uuid4

from ai_core.contracts.business import BusinessContext
from ai_core.contracts.scope import ScopeContext
from ai_core.rag import semantic_cache


class _StubEmbeddingClient:
    def __init__(self, vector):
        self._vector = vector

    def embed(self, texts):
        class _Result:
            def __init__(self, vectors):
                self.vectors = vectors

        return _Result([self._vector for _ in texts])


def _tool_context():
    scope = ScopeContext(
        tenant_id="tenant",
        trace_id="trace-1",
        invocation_id=uuid4().hex,
        run_id=uuid4().hex,
        service_id="test-worker",
    )
    business = BusinessContext(case_id="case-1", collection_id="col-1")
    return scope.to_tool_context(business=business)


def test_semantic_cache_hit(monkeypatch):
    cache = semantic_cache.SemanticCache(
        enabled=True,
        ttl_s=60,
        max_items=10,
        similarity_threshold=0.8,
    )
    monkeypatch.setattr(
        semantic_cache,
        "get_embedding_client",
        lambda: _StubEmbeddingClient([1.0, 0.0]),
    )
    context = _tool_context()

    lookup = cache.lookup("hello", context)
    assert not lookup.hit
    cache.store("hello", context, {"answer": "cached"}, embedding=lookup.embedding)

    lookup2 = cache.lookup("hello", context)
    assert lookup2.hit
    assert lookup2.response["answer"] == "cached"
