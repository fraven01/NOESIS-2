from __future__ import annotations

from ai_core.contracts import BusinessContext, ScopeContext
from ai_core.services import rag_query


def test_execute_allows_global_scope_without_case_id(monkeypatch) -> None:
    scope = ScopeContext(
        tenant_id="tenant-1",
        trace_id="trace-1",
        invocation_id="invoke-1",
        run_id="run-1",
    )
    business = BusinessContext(case_id=None, collection_id=None)
    tool_context = scope.to_tool_context(business=business)

    captured: dict[str, object] = {}

    def fake_run(state, meta):
        captured["state"] = state
        captured["meta"] = meta
        return state, {"answer": "ok"}

    monkeypatch.setattr(rag_query, "run_retrieval_augmented_generation", fake_run)

    service = rag_query.RagQueryService()
    service.execute(tool_context=tool_context, question="hello")

    meta = captured["meta"]
    assert meta["business_context"] == {}
    assert "case_id" not in meta["tool_context"]["business"]


def test_execute_with_case_scope(monkeypatch) -> None:
    scope = ScopeContext(
        tenant_id="tenant-1",
        trace_id="trace-1",
        invocation_id="invoke-1",
        run_id="run-1",
    )
    business = BusinessContext(case_id="case-1", collection_id=None)
    tool_context = scope.to_tool_context(business=business)

    captured: dict[str, object] = {}

    def fake_run(state, meta):
        captured["meta"] = meta
        return state, {"answer": "ok"}

    monkeypatch.setattr(rag_query, "run_retrieval_augmented_generation", fake_run)

    service = rag_query.RagQueryService()
    service.execute(tool_context=tool_context, question="hello")

    meta = captured["meta"]
    assert meta["business_context"]["case_id"] == "case-1"
    assert "collection_id" not in meta["business_context"]
    assert meta["tool_context"]["business"]["case_id"] == "case-1"


def test_execute_with_collection_scope(monkeypatch) -> None:
    scope = ScopeContext(
        tenant_id="tenant-1",
        trace_id="trace-1",
        invocation_id="invoke-1",
        run_id="run-1",
    )
    business = BusinessContext(case_id=None, collection_id="collection-1")
    tool_context = scope.to_tool_context(business=business)

    captured: dict[str, object] = {}

    def fake_run(state, meta):
        captured["meta"] = meta
        return state, {"answer": "ok"}

    monkeypatch.setattr(rag_query, "run_retrieval_augmented_generation", fake_run)

    service = rag_query.RagQueryService()
    service.execute(tool_context=tool_context, question="hello")

    meta = captured["meta"]
    assert meta["business_context"]["collection_id"] == "collection-1"
    assert "case_id" not in meta["business_context"]
    assert meta["tool_context"]["business"]["collection_id"] == "collection-1"


def test_execute_with_case_and_collection_scope(monkeypatch) -> None:
    scope = ScopeContext(
        tenant_id="tenant-1",
        trace_id="trace-1",
        invocation_id="invoke-1",
        run_id="run-1",
    )
    business = BusinessContext(case_id="case-1", collection_id="collection-1")
    tool_context = scope.to_tool_context(business=business)

    captured: dict[str, object] = {}

    def fake_run(state, meta):
        captured["meta"] = meta
        return state, {"answer": "ok"}

    monkeypatch.setattr(rag_query, "run_retrieval_augmented_generation", fake_run)

    service = rag_query.RagQueryService()
    service.execute(tool_context=tool_context, question="hello")

    meta = captured["meta"]
    assert meta["business_context"]["case_id"] == "case-1"
    assert meta["business_context"]["collection_id"] == "collection-1"
    assert meta["tool_context"]["business"]["case_id"] == "case-1"
    assert meta["tool_context"]["business"]["collection_id"] == "collection-1"
