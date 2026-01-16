from uuid import uuid4

from ai_core.contracts.business import BusinessContext
from ai_core.contracts.scope import ScopeContext
from ai_core.rag import rerank, strategy


def _tool_context():
    scope = ScopeContext(
        tenant_id="tenant",
        trace_id="trace-1",
        invocation_id=uuid4().hex,
        run_id=uuid4().hex,
        service_id="test-worker",
    )
    business = BusinessContext(case_id="case-1")
    return scope.to_tool_context(business=business)


def test_generate_query_variants_fallback():
    context = _tool_context()
    result = strategy.generate_query_variants("what is arbitration", context)
    assert result.queries
    assert "what is arbitration" in result.queries
    assert len(result.queries) <= strategy.DEFAULT_MAX_VARIANTS


def test_rerank_chunks_heuristic():
    context = _tool_context()
    chunks = [
        {"id": "a", "text": "low", "score": 0.1},
        {"id": "b", "text": "high", "score": 0.9},
    ]
    result = rerank.rerank_chunks(
        chunks,
        "test query",
        context,
        top_k=1,
        mode="heuristic",
    )
    assert len(result.chunks) == 1
    assert result.chunks[0]["id"] == "b"
