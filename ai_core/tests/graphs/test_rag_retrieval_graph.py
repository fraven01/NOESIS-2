from __future__ import annotations

from typing import Any, Mapping

from ai_core.contracts import BusinessContext, ScopeContext
from ai_core.graphs.technical.rag_retrieval import RagRetrievalGraph
from ai_core.nodes import retrieve
from ai_core.rag import rerank as rag_rerank


def _tool_context() -> Any:
    scope = ScopeContext(
        tenant_id="tenant-1",
        trace_id="trace-1",
        invocation_id="invoke-1",
        run_id="run-1",
    )
    business = BusinessContext(collection_id="collection-1", case_id="case-1")
    return scope.to_tool_context(business=business)


def _retrieve_output(matches: list[dict[str, Any]], *, took_ms: int = 5) -> Any:
    routing = retrieve.RetrieveRouting(
        profile="profile-1",
        vector_space_id="vector-1",
        process=None,
        doc_class=None,
        collection_id="collection-1",
        workflow_id=None,
    )
    meta = retrieve.RetrieveMeta(
        routing=routing,
        took_ms=took_ms,
        alpha=0.7,
        min_sim=0.0,
        top_k_effective=10,
        matches_returned=len(matches),
        max_candidates_effective=10,
        vector_candidates=5,
        lexical_candidates=5,
        deleted_matches_blocked=0,
        visibility_effective="active",
        diversify_strength=0.0,
    )
    return retrieve.RetrieveOutput(matches=matches, meta=meta)


def test_multi_query_dedupe() -> None:
    context = _tool_context()

    class StubRetrieve:
        def __call__(self, _context, params):
            if params.query == "alpha":
                return _retrieve_output(
                    [
                        {"id": "a", "text": "alpha-1", "score": 0.7},
                        {"id": "b", "text": "alpha-2", "score": 0.6},
                    ],
                    took_ms=4,
                )
            return _retrieve_output(
                [
                    {"id": "a", "text": "beta-1", "score": 0.95},
                    {"id": "c", "text": "beta-2", "score": 0.5},
                ],
                took_ms=6,
            )

    graph = RagRetrievalGraph(retrieve_node=StubRetrieve())
    state = {
        "schema_id": "noesis.graphs.rag_retrieval",
        "schema_version": "1.0.0",
        "tool_context": context,
        "queries": ["alpha", "beta"],
        "retrieve": {"hybrid": {"alpha": 0.7, "top_k": 10}},
        "use_rerank": False,
        "document_id": None,
    }

    result = graph.invoke(state)

    matches = result["matches"]
    assert len(matches) == 3
    assert matches[0]["id"] == "a"
    assert matches[0]["score"] == 0.95
    assert result["retrieval_meta"]["matches_returned"] == 3
    assert result["query_variants_used"] == ["alpha", "beta"]


def test_rerank_toggle_changes_order() -> None:
    context = _tool_context()
    base_matches = [
        {"id": "a", "text": "alpha-1", "score": 0.7},
        {"id": "b", "text": "alpha-2", "score": 0.6},
    ]

    class StubRetrieve:
        def __call__(self, _context, _params):
            return _retrieve_output(list(base_matches))

    def stub_rerank(chunks, query, _context, *, top_k=None):
        return rag_rerank.RerankResult(
            chunks=list(reversed([dict(chunk) for chunk in chunks])),
            mode="heuristic",
            prompt_version=None,
            scores=None,
            error=None,
        )

    graph = RagRetrievalGraph(
        retrieve_node=StubRetrieve(),
        rerank_node=stub_rerank,
    )
    state = {
        "schema_id": "noesis.graphs.rag_retrieval",
        "schema_version": "1.0.0",
        "tool_context": context,
        "queries": ["alpha"],
        "retrieve": {"hybrid": {"alpha": 0.7, "top_k": 10}},
        "use_rerank": True,
    }

    result = graph.invoke(state)

    assert result["snippets"][0]["id"] == "b"
    assert result["rerank_meta"]["mode"] == "heuristic"


def test_document_id_scoping_overrides_filters() -> None:
    context = _tool_context()
    captured_filters: list[Mapping[str, Any]] = []

    class StubRetrieve:
        def __call__(self, _context, params):
            captured_filters.append(params.filters or {})
            return _retrieve_output([])

    graph = RagRetrievalGraph(retrieve_node=StubRetrieve())
    state = {
        "schema_id": "noesis.graphs.rag_retrieval",
        "schema_version": "1.0.0",
        "tool_context": context,
        "queries": ["alpha"],
        "retrieve": {
            "filters": {"id": "legacy", "type": "framework"},
            "hybrid": {"alpha": 0.7, "top_k": 10},
        },
        "use_rerank": False,
        "document_id": "doc-123",
    }

    graph.invoke(state)

    assert captured_filters
    assert captured_filters[0]["id"] == "doc-123"
    assert captured_filters[0]["type"] == "framework"


def test_reference_expansion_adds_matches(monkeypatch) -> None:
    context = _tool_context()

    class StubRetrieve:
        def __call__(self, _context, params):
            if params.filters and params.filters.get("id") == "doc-ref":
                return _retrieve_output(
                    [
                        {
                            "id": "ref-chunk",
                            "text": "ref-1",
                            "score": 0.65,
                            "meta": {"document_id": "doc-ref"},
                        }
                    ]
                )
            return _retrieve_output(
                [
                    {
                        "id": "seed-chunk",
                        "text": "seed",
                        "score": 0.7,
                        "meta": {"reference_ids": ["doc-ref"]},
                    }
                ]
            )

    monkeypatch.setenv("RAG_REFERENCE_EXPANSION", "1")
    monkeypatch.setenv("RAG_REFERENCE_EXPANSION_LIMIT", "3")
    monkeypatch.setenv("RAG_REFERENCE_EXPANSION_TOP_K", "5")

    graph = RagRetrievalGraph(retrieve_node=StubRetrieve())
    state = {
        "schema_id": "noesis.graphs.rag_retrieval",
        "schema_version": "1.0.0",
        "tool_context": context,
        "queries": ["alpha"],
        "retrieve": {"hybrid": {"alpha": 0.7, "top_k": 10}},
        "use_rerank": False,
    }

    result = graph.invoke(state)

    ids = {match["id"] for match in result["matches"]}
    assert "seed-chunk" in ids
    assert "ref-chunk" in ids
    assert result["retrieval_meta"]["reference_expansion"]["reference_count"] == 1
