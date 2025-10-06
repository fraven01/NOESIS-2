"""Tests for the production retrieval augmented generation graph."""

from __future__ import annotations

from typing import Any, Mapping, MutableMapping
from unittest.mock import patch

from ai_core.graphs import retrieval_augmented_generation


def _fake_retrieve(
    state: MutableMapping[str, Any], meta: Mapping[str, Any], *, top_k: int | None = None
):
    new_state = dict(state)
    new_state.setdefault("snippets", [])
    new_state["snippets"].append({"text": "snippet"})
    return new_state, {"matches": new_state["snippets"], "meta": {}}


def _fake_compose(state: MutableMapping[str, Any], meta: MutableMapping[str, Any]):
    new_state = dict(state)
    new_state["answer"] = "answer"
    return new_state, {"answer": "answer", "prompt_version": "v-test"}


def test_graph_runs_retrieve_then_compose() -> None:
    calls: list[str] = []

    def _recording_retrieve(state: MutableMapping[str, Any], meta: Mapping[str, Any], *, top_k: int | None = None):
        calls.append("retrieve")
        assert meta["tenant_id"] == "tenant-42"
        return _fake_retrieve(state, meta, top_k=top_k)

    def _recording_compose(state: MutableMapping[str, Any], meta: MutableMapping[str, Any]):
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
    assert result["answer"] == "answer"
    assert result["prompt_version"] == "v-test"


def test_graph_normalises_tenant_alias() -> None:
    captured_meta: dict[str, Any] = {}

    def _recording_retrieve(state: MutableMapping[str, Any], meta: Mapping[str, Any], *, top_k: int | None = None):
        captured_meta["tenant_id"] = meta.get("tenant_id")
        return _fake_retrieve(state, meta, top_k=top_k)

    graph = retrieval_augmented_generation.RetrievalAugmentedGenerationGraph(
        retrieve_node=_recording_retrieve,
        compose_node=_fake_compose,
    )

    meta = {"tenant": "tenant-alias"}
    state, result = graph.run({}, meta)

    assert meta["tenant_id"] == "tenant-alias"
    assert captured_meta["tenant_id"] == "tenant-alias"
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
