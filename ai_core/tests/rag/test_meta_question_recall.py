from typing import Any
from uuid import uuid4

import pytest

from ai_core.contracts.business import BusinessContext
from ai_core.contracts.scope import ScopeContext
from ai_core.graphs.technical import retrieval_augmented_generation
from ai_core.nodes import compose
from ai_core.nodes import retrieve
from ai_core.rag.schemas import Chunk
from ai_core.rag.standalone_question import StandaloneQuestionResult


def _scope_meta(
    tenant_id: str, case_id: str, *, budget_tokens: int | None = None
) -> dict[str, Any]:
    scope = ScopeContext(
        tenant_id=tenant_id,
        trace_id="trace-1",
        invocation_id=uuid4().hex,
        run_id=uuid4().hex,
        service_id="test-worker",
    )
    business = BusinessContext(case_id=case_id, workflow_id="rag.default")
    tool_context = scope.to_tool_context(business=business, budget_tokens=budget_tokens)
    return {
        "scope_context": scope.model_dump(mode="json"),
        "business_context": business.model_dump(mode="json"),
        "tool_context": tool_context.model_dump(mode="json"),
    }


def _anchor_output() -> retrieve.RetrieveOutput:
    matches = [
        {
            "id": "chunk-anchor",
            "text": "Anlage 1:",
            "score": 0.9,
            "source": "Anlage 1",
            "meta": {
                "chunk_id": "chunk-anchor",
                "document_id": "doc-1",
                "external_id": "Anlage 1",
                "section_path": [],
                "chunk_index": 0,
            },
        }
    ]
    meta = retrieve.RetrieveMeta(
        routing={"profile": "default", "vector_space_id": "rag/default@v1"},
        took_ms=12,
        alpha=0.5,
        min_sim=0.2,
        top_k_effective=1,
        matches_returned=1,
        max_candidates_effective=5,
        vector_candidates=3,
        lexical_candidates=2,
        deleted_matches_blocked=0,
        visibility_effective="active",
        diversify_strength=0.0,
    )
    return retrieve.RetrieveOutput(matches=matches, meta=meta)


def test_meta_question_document_expansion(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_RERANK_MODE", "off")
    captured: dict[str, Any] = {}

    class _TenantClient:
        def get_chunks_by_document(
            self,
            document_id: str,
            *,
            case_id: str | None = None,
            collection_id: str | None = None,
        ) -> list[Chunk]:
            captured["document_id"] = document_id
            captured["case_id"] = case_id
            captured["collection_id"] = collection_id
            return [
                Chunk(
                    content="Frage 1?",
                    meta={
                        "id": "chunk-1",
                        "chunk_id": "chunk-1",
                        "document_id": "doc-1",
                        "external_id": "Anlage 1",
                        "section_path": ["Fragen"],
                        "chunk_index": 1,
                    },
                ),
                Chunk(
                    content="Frage 2?",
                    meta={
                        "id": "chunk-2",
                        "chunk_id": "chunk-2",
                        "document_id": "doc-1",
                        "external_id": "Anlage 1",
                        "section_path": ["Fragen"],
                        "chunk_index": 2,
                    },
                ),
            ]

    class _Router:
        def for_tenant(self, tenant_id: str, tenant_schema: str | None = None):
            captured["tenant_id"] = tenant_id
            captured["tenant_schema"] = tenant_schema
            return _TenantClient()

    def _recording_retrieve(
        context: Any, params: retrieve.RetrieveInput
    ) -> retrieve.RetrieveOutput:
        return _anchor_output()

    monkeypatch.setattr(
        retrieval_augmented_generation,
        "get_default_router",
        lambda: _Router(),
    )
    monkeypatch.setattr(
        retrieval_augmented_generation.rag_standalone,
        "generate_standalone_question",
        lambda question, history, context: StandaloneQuestionResult(
            question=question, source="original"
        ),
    )

    graph = retrieval_augmented_generation.RetrievalAugmentedGenerationGraph(
        retrieve_node=_recording_retrieve,
        compose_node=lambda *args, **kwargs: compose.ComposeOutput(
            answer="ok", prompt_version="v-test"
        ),
        compose_extract_node=lambda *args, **kwargs: compose.ComposeOutput(
            answer="ok", prompt_version="v-test"
        ),
    )

    state, _result = graph.run(
        {"query": "Welche Fragen fuer Anlage 1"},
        _scope_meta("tenant-1", "case-1", budget_tokens=2000),
    )

    chunk_ids = [
        snippet.get("meta", {}).get("chunk_id")
        for snippet in state["snippets"]
        if isinstance(snippet, dict)
    ]
    assert "chunk-1" in chunk_ids
    assert "chunk-2" in chunk_ids
    assert captured["document_id"] == "doc-1"
    assert captured["case_id"] == "case-1"
    assert captured["collection_id"] is None


def test_meta_question_document_expansion_requires_anchor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAG_RERANK_MODE", "off")
    captured: dict[str, Any] = {}

    class _TenantClient:
        def get_chunks_by_document(
            self,
            document_id: str,
            *,
            case_id: str | None = None,
            collection_id: str | None = None,
        ) -> list[Chunk]:
            captured["called"] = True
            return []

    class _Router:
        def for_tenant(self, tenant_id: str, tenant_schema: str | None = None):
            return _TenantClient()

    def _recording_retrieve(
        context: Any, params: retrieve.RetrieveInput
    ) -> retrieve.RetrieveOutput:
        matches = [
            {
                "id": "chunk-1",
                "text": "Anlage 1 Fragenkatalog?",
                "score": 0.9,
                "source": "Anlage 1",
                "meta": {
                    "chunk_id": "chunk-1",
                    "document_id": "doc-1",
                    "external_id": "Anlage 1",
                    "section_path": ["Einleitung"],
                    "chunk_index": 0,
                },
            }
        ]
        meta = retrieve.RetrieveMeta(
            routing={"profile": "default", "vector_space_id": "rag/default@v1"},
            took_ms=12,
            alpha=0.5,
            min_sim=0.2,
            top_k_effective=1,
            matches_returned=1,
            max_candidates_effective=5,
            vector_candidates=3,
            lexical_candidates=2,
            deleted_matches_blocked=0,
            visibility_effective="active",
            diversify_strength=0.0,
        )
        return retrieve.RetrieveOutput(matches=matches, meta=meta)

    monkeypatch.setattr(
        retrieval_augmented_generation,
        "get_default_router",
        lambda: _Router(),
    )
    monkeypatch.setattr(
        retrieval_augmented_generation.rag_standalone,
        "generate_standalone_question",
        lambda question, history, context: StandaloneQuestionResult(
            question=question, source="original"
        ),
    )

    graph = retrieval_augmented_generation.RetrievalAugmentedGenerationGraph(
        retrieve_node=_recording_retrieve,
        compose_node=lambda *args, **kwargs: compose.ComposeOutput(
            answer="ok", prompt_version="v-test"
        ),
        compose_extract_node=lambda *args, **kwargs: compose.ComposeOutput(
            answer="ok", prompt_version="v-test"
        ),
    )

    state, _result = graph.run(
        {"query": "Welche Fragen fuer Anlage 1"},
        _scope_meta("tenant-1", "case-1", budget_tokens=2000),
    )

    assert captured.get("called") is None
    chunk_ids = [
        snippet.get("meta", {}).get("chunk_id")
        for snippet in state["snippets"]
        if isinstance(snippet, dict)
    ]
    assert chunk_ids == ["chunk-1"]
