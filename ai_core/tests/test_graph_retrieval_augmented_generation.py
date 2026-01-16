from typing import Any
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from ai_core.contracts.business import BusinessContext
from ai_core.contracts.scope import ScopeContext
from ai_core.graphs.technical import retrieval_augmented_generation
from ai_core.nodes import compose, retrieve
from ai_core.tool_contracts import ToolContext


def _scope_meta(tenant_id: str, case_id: str) -> dict[str, Any]:
    scope = ScopeContext(
        tenant_id=tenant_id,
        trace_id="trace-1",
        invocation_id=uuid4().hex,
        run_id=uuid4().hex,
        service_id="test-worker",
    )
    business = BusinessContext(case_id=case_id)
    tool_context = scope.to_tool_context(business=business)
    return {
        "scope_context": scope.model_dump(mode="json"),
        "business_context": business.model_dump(mode="json"),
        "tool_context": tool_context.model_dump(mode="json"),
    }


def _dummy_output() -> retrieve.RetrieveOutput:
    matches = [
        {
            "id": "doc-1",
            "text": "snippet",
            "score": 0.42,
            "source": "handbook.md",
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


def _fake_compose(context: ToolContext, params: compose.ComposeInput):
    assert params.snippets is not None
    return compose.ComposeOutput(answer="answer", prompt_version="v-test")


def test_graph_persists_history_with_thread_id() -> None:
    scope = ScopeContext(
        tenant_id="tenant-1",
        trace_id="trace-1",
        invocation_id=uuid4().hex,
        run_id=uuid4().hex,
        service_id="test-worker",
    )
    business = BusinessContext(case_id="case-1", thread_id="thread-1")
    tool_context = scope.to_tool_context(business=business)
    meta = {"tool_context": tool_context.model_dump(mode="json")}

    history = [
        {"role": "user", "content": "Prev Q"},
        {"role": "assistant", "content": "Prev A"},
    ]

    with patch(
        "ai_core.graphs.technical.retrieval_augmented_generation.ThreadAwareCheckpointer"
    ) as mock_checkpointer, patch(
        "ai_core.graphs.technical.retrieval_augmented_generation._build_compiled_graph"
    ) as mock_build:
        mock_checkpointer.return_value.load.return_value = {"chat_history": history}
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "state": {
                "question": "Next Q",
                "chat_history": list(history),
            },
            "result": {
                "answer": "Next A",
                "prompt_version": "v-test",
                "retrieval": {},
                "snippets": [],
            },
        }
        mock_build.return_value = mock_graph

        graph = retrieval_augmented_generation.RetrievalAugmentedGenerationGraph()
        graph.run({"question": "Next Q"}, meta)

        mock_checkpointer.return_value.load.assert_called_once()
        mock_graph.invoke.assert_called_once()
        saved_state = mock_checkpointer.return_value.save.call_args[0][1]
        saved_history = saved_state["chat_history"]

        assert len(saved_history) == 4
        assert saved_history[-2]["content"] == "Next Q"
        assert saved_history[-1]["content"] == "Next A"


def test_graph_runs_retrieve_then_compose() -> None:
    calls: list[str] = []

    def _recording_retrieve(
        context: ToolContext, params: retrieve.RetrieveInput
    ) -> retrieve.RetrieveOutput:
        calls.append("retrieve")
        assert context.scope.tenant_id == "tenant-42"
        assert isinstance(params, retrieve.RetrieveInput)
        return _dummy_output()

    def _recording_compose(context: ToolContext, params: compose.ComposeInput):
        calls.append("compose")
        assert context.scope.tenant_id == "tenant-42"
        return _fake_compose(context, params)

    graph = retrieval_augmented_generation.RetrievalAugmentedGenerationGraph(
        retrieve_node=_recording_retrieve,
        compose_node=_recording_compose,
    )

    state, result = graph.run(
        {"query": "what is arbitration", "question": "what is arbitration"},
        _scope_meta("tenant-42", "case-1"),
    )

    assert calls[0] == "retrieve"
    assert calls[-1] == "compose"
    assert all(call == "retrieve" for call in calls[:-1])
    assert state["answer"] == "answer"
    assert state["snippets"] == [
        {
            "id": "doc-1",
            "text": "snippet",
            "score": 0.42,
            "source": "handbook.md",
            "citation": "handbook.md",
        }
    ]
    snippets = result["snippets"]
    assert isinstance(snippets[0]["score"], float)
    assert snippets[0]["text"]
    assert snippets[0]["source"]
    assert isinstance(state["retrieval"], dict)
    retrieval = state["retrieval"]
    assert retrieval["alpha"] == 0.5
    assert retrieval["min_sim"] == 0.2
    assert retrieval["top_k_effective"] == 1
    assert retrieval["matches_returned"] == 1
    assert retrieval["visibility_effective"] == "active"
    assert retrieval["took_ms"] >= 12
    assert retrieval["routing"]["profile"] == "default"
    assert retrieval["routing"]["vector_space_id"] == "rag/default@v1"
    assert result["answer"] == "answer"
    assert result["prompt_version"] == "v-test"
    assert result["retrieval"] == state["retrieval"]
    assert result["snippets"] == state["snippets"]


def test_graph_normalises_tenant_alias() -> None:
    captured: dict[str, Any] = {}

    def _recording_retrieve(
        context: ToolContext, params: retrieve.RetrieveInput
    ) -> retrieve.RetrieveOutput:
        captured["tenant_id"] = context.scope.tenant_id
        return _dummy_output()

    graph = retrieval_augmented_generation.RetrievalAugmentedGenerationGraph(
        retrieve_node=_recording_retrieve,
        compose_node=_fake_compose,
    )

    meta = _scope_meta("tenant-alias", "case-1")
    state, result = graph.run(
        {"query": "what is arbitration", "question": "what is arbitration"},
        meta,
    )

    assert meta["scope_context"]["tenant_id"] == "tenant-alias"
    assert captured["tenant_id"] == "tenant-alias"
    assert state["answer"] == "answer"
    assert result["answer"] == "answer"
    retrieval = result["retrieval"]
    assert retrieval["alpha"] == 0.5
    assert retrieval["min_sim"] == 0.2
    assert retrieval["top_k_effective"] == 1
    assert retrieval["matches_returned"] == 1
    assert retrieval["visibility_effective"] == "active"
    assert retrieval["took_ms"] >= 12
    assert retrieval["routing"]["profile"] == "default"
    assert retrieval["routing"]["vector_space_id"] == "rag/default@v1"


def test_graph_fills_missing_snippet_fields() -> None:
    def _recording_retrieve(
        context: ToolContext, params: retrieve.RetrieveInput
    ) -> retrieve.RetrieveOutput:
        return _dummy_output()

    def _compose_with_sparse_snippets(
        context: ToolContext, params: compose.ComposeInput
    ):
        sparse_snippet = {"id": "doc-1", "score": "0.2"}
        return compose.ComposeOutput(
            answer="filled",
            prompt_version="v-test",
            snippets=[dict(sparse_snippet)],
        )

    graph = retrieval_augmented_generation.RetrievalAugmentedGenerationGraph(
        retrieve_node=_recording_retrieve,
        compose_node=_compose_with_sparse_snippets,
    )

    state, result = graph.run(
        {"query": "what is arbitration", "question": "what is arbitration"},
        _scope_meta("tenant", "case"),
    )

    snippet = result["snippets"][0]
    assert snippet["text"] == "snippet"
    assert snippet["source"] == "handbook.md"
    assert snippet["score"] == pytest.approx(0.2)
    assert state["snippets"][0]["text"] == "snippet"
    assert state["snippets"][0]["source"] == "handbook.md"


def test_build_graph_returns_new_instance() -> None:
    first = retrieval_augmented_generation.build_graph()
    second = retrieval_augmented_generation.build_graph()
    assert first is not second


def test_bootstrap_registers_rag_default() -> None:
    from ai_core.graph import bootstrap

    recorded: dict[str, Any] = {}

    def _record(name: str, runner: Any) -> None:
        recorded[name] = runner

    with patch("ai_core.graph.bootstrap.register", new=_record), patch(
        "ai_core.graph.bootstrap.module_runner", new=lambda module: module
    ):
        bootstrap.bootstrap()

    assert "rag.default" in recorded
    assert recorded["rag.default"] is recorded["retrieval_augmented_generation"]
