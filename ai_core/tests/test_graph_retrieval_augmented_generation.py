from typing import Any, MutableMapping
from unittest.mock import patch

from ai_core.graphs import retrieval_augmented_generation
from ai_core.nodes import retrieve
from ai_core.tool_contracts import ToolContext


def _dummy_output() -> retrieve.RetrieveOutput:
    matches = [{"text": "snippet"}]
    meta = retrieve.RetrieveMeta(
        routing={"profile": "default", "vector_space_id": "global"},
        took_ms=0,
    )
    return retrieve.RetrieveOutput(matches=matches, meta=meta)


def _fake_compose(state: MutableMapping[str, Any], meta: MutableMapping[str, Any]):
    new_state = dict(state)
    assert "snippets" in new_state
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

    state, result = graph.run({}, {"tenant_id": "tenant-42"})

    assert calls == ["retrieve", "compose"]
    assert state["answer"] == "answer"
    assert state["snippets"] == [{"text": "snippet"}]
    assert result["answer"] == "answer"
    assert result["prompt_version"] == "v-test"


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

    meta = {"tenant_id": "tenant-alias"}
    state, result = graph.run({}, meta)

    assert meta["tenant_id"] == "tenant-alias"
    assert captured["tenant_id"] == "tenant-alias"
    assert state["answer"] == "answer"
    assert result["answer"] == "answer"


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
