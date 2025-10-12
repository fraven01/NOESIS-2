from typing import Any, MutableMapping
from unittest.mock import patch

import pytest

from ai_core.graphs import retrieval_augmented_generation
from ai_core.nodes import retrieve
from ai_core.tool_contracts import ToolContext


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
        routing={"profile": "default", "vector_space_id": "global"},
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
    )
    return retrieve.RetrieveOutput(matches=matches, meta=meta)


def _fake_compose(state: MutableMapping[str, Any], meta: MutableMapping[str, Any]):
    new_state = dict(state)
    assert "snippets" in new_state
    assert "retrieval" in new_state
    new_state["answer"] = "answer"
    return new_state, {"answer": "answer", "prompt_version": "v-test"}


def test_graph_runs_retrieve_then_compose() -> None:
    calls: list[str] = []

    def _recording_retrieve(
        context: ToolContext, params: retrieve.RetrieveInput
    ) -> retrieve.RetrieveOutput:
        calls.append("retrieve")
        assert context.tenant_id == "tenant-42"
        assert isinstance(params, retrieve.RetrieveInput)
        return _dummy_output()

    def _recording_compose(
        state: MutableMapping[str, Any], meta: MutableMapping[str, Any]
    ):
        calls.append("compose")
        assert meta["tenant_id"] == "tenant-42"
        return _fake_compose(state, meta)

    graph = retrieval_augmented_generation.RetrievalAugmentedGenerationGraph(
        retrieve_node=_recording_retrieve,
        compose_node=_recording_compose,
    )

    state, result = graph.run({}, {"tenant_id": "tenant-42", "case_id": "case-1"})

    assert calls == ["retrieve", "compose"]
    assert state["answer"] == "answer"
    assert state["snippets"] == [
        {
            "id": "doc-1",
            "text": "snippet",
            "score": 0.42,
            "source": "handbook.md",
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
    assert retrieval["took_ms"] == 12
    assert retrieval["routing"]["profile"] == "default"
    assert retrieval["routing"]["vector_space_id"] == "global"
    assert result["answer"] == "answer"
    assert result["prompt_version"] == "v-test"
    assert result["retrieval"] == state["retrieval"]
    assert result["snippets"] == state["snippets"]


def test_graph_normalises_tenant_alias() -> None:
    captured: dict[str, Any] = {}

    def _recording_retrieve(
        context: ToolContext, params: retrieve.RetrieveInput
    ) -> retrieve.RetrieveOutput:
        captured["tenant_id"] = context.tenant_id
        return _dummy_output()

    graph = retrieval_augmented_generation.RetrievalAugmentedGenerationGraph(
        retrieve_node=_recording_retrieve,
        compose_node=_fake_compose,
    )

    meta = {"tenant_id": "tenant-alias", "case_id": "case-1"}
    state, result = graph.run({}, meta)

    assert meta["tenant_id"] == "tenant-alias"
    assert captured["tenant_id"] == "tenant-alias"
    assert state["answer"] == "answer"
    assert result["answer"] == "answer"
    retrieval = result["retrieval"]
    assert retrieval["alpha"] == 0.5
    assert retrieval["min_sim"] == 0.2
    assert retrieval["top_k_effective"] == 1
    assert retrieval["matches_returned"] == 1
    assert retrieval["visibility_effective"] == "active"
    assert retrieval["took_ms"] == 12
    assert retrieval["routing"]["profile"] == "default"
    assert retrieval["routing"]["vector_space_id"] == "global"


def test_graph_fills_missing_snippet_fields() -> None:
    def _recording_retrieve(
        context: ToolContext, params: retrieve.RetrieveInput
    ) -> retrieve.RetrieveOutput:
        return _dummy_output()

    def _compose_with_sparse_snippets(
        state: MutableMapping[str, Any], meta: MutableMapping[str, Any]
    ):
        new_state = dict(state)
        sparse_snippet = {"id": "doc-1", "score": "0.2"}
        new_state["snippets"] = [sparse_snippet]
        new_state["answer"] = "filled"
        return new_state, {
            "answer": "filled",
            "prompt_version": "v-test",
            "snippets": [dict(sparse_snippet)],
            "retrieval": new_state["retrieval"],
        }

    graph = retrieval_augmented_generation.RetrievalAugmentedGenerationGraph(
        retrieve_node=_recording_retrieve,
        compose_node=_compose_with_sparse_snippets,
    )

    state, result = graph.run({}, {"tenant_id": "tenant", "case_id": "case"})

    snippet = result["snippets"][0]
    assert snippet["text"] == "snippet"
    assert snippet["source"] == "handbook.md"
    assert snippet["score"] == pytest.approx(0.2)
    assert state["snippets"][0]["text"] == "snippet"
    assert state["snippets"][0]["source"] == "handbook.md"


def test_build_graph_returns_shared_instance() -> None:
    first = retrieval_augmented_generation.build_graph()
    second = retrieval_augmented_generation.build_graph()
    assert first is second is retrieval_augmented_generation.GRAPH


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
