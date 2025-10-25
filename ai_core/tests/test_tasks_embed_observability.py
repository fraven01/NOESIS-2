from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, List

import pytest

from ai_core import tasks
from ai_core.rag.embeddings import EmbeddingBatchResult


def _wrap_observed_span(monkeypatch, calls: List[str]) -> None:
    metrics_cls = tasks._EmbedSpanMetrics

    def recorder(name: str):  # noqa: D401 - simple proxy to record span names
        calls.append(name)

        @contextmanager
        def _cm():
            metrics = metrics_cls()
            try:
                yield metrics
            except Exception:
                metrics.set("status", "error")
                raise
            finally:
                metrics.finalise()

        return _cm()

    monkeypatch.setattr(tasks, "_observed_embed_section", recorder)


def test_embed_emits_span_metadata(monkeypatch):
    meta = {"tenant_id": "tenant", "case_id": "case"}
    raw_chunks: Dict[str, Any] = {
        "chunks": [
            {"content": "one two", "meta": {"hash": "chunk-1"}},
            {"normalized": "three four five", "meta": {"hash": "chunk-2"}},
        ],
        "parents": {"p1": {"id": "p1"}},
    }

    monkeypatch.setattr(tasks.object_store, "read_json", lambda _: raw_chunks)
    monkeypatch.setattr(tasks.settings, "EMBEDDINGS_BATCH_SIZE", 2, raising=False)

    written: Dict[str, Any] = {}

    def fake_write_json(path: str, payload: Dict[str, Any]) -> None:  # noqa: D401
        written["path"] = path
        written["payload"] = payload

    monkeypatch.setattr(tasks.object_store, "write_json", fake_write_json)

    class _Client:
        batch_size = 2

        def embed(self, inputs):  # noqa: D401 - simple fake client
            fake_calls.append(list(inputs))
            return EmbeddingBatchResult(
                vectors=[[1.0, 0.0] for _ in inputs],
                model="openai/text-embedding-3-small",
                model_used="primary",
                attempts=2,
                timeout_s=None,
                retry_delays=(0.05,),
            )

        def dim(self):  # noqa: D401
            return 2

    fake_calls: List[List[str]] = []
    monkeypatch.setattr(tasks, "get_embedding_client", lambda: _Client())

    span_calls: List[str] = []
    _wrap_observed_span(monkeypatch, span_calls)

    metadata_calls: List[Dict[str, Any]] = []

    def fake_update_observation(**kwargs):  # noqa: D401
        metadata_calls.append(kwargs)

    monkeypatch.setattr(tasks, "update_observation", fake_update_observation)

    cost_calls: List[tuple[str, int]] = []

    def fake_cost(model: str, tokens: int | float) -> float:  # noqa: D401
        cost_calls.append((model, int(tokens)))
        return 0.01 * float(tokens)

    monkeypatch.setattr(tasks, "calculate_embedding_cost", fake_cost)

    monkeypatch.setattr("ai_core.infra.observability.tracing_enabled", lambda: False)

    result = tasks.embed(meta, "chunks.json")

    assert result["path"].endswith("vectors.json")
    assert written["payload"]["chunks"]

    assert set(span_calls) >= {
        "load",
        "chunk",
        "embed",
        "write",
    }

    embed_metadata = None
    for call in metadata_calls:
        meta_payload = call.get("metadata")
        if meta_payload and "embedding_model" in meta_payload:
            embed_metadata = meta_payload
            break

    assert embed_metadata is not None
    assert embed_metadata["chunks_count"] == 2
    assert embed_metadata["batch_size"] == 2
    assert embed_metadata["embedding_model"] == "openai/text-embedding-3-small"
    assert embed_metadata["retry.count"] == 1
    assert embed_metadata["retry.backoff_ms"] == pytest.approx(50.0)
    assert embed_metadata["cost.usd_embedding"] == pytest.approx(0.05)

    parent_counts = [
        meta_payload.get("parents_count")
        for meta_payload in (
            call.get("metadata") for call in metadata_calls if call.get("metadata")
        )
        if meta_payload.get("parents_count") is not None
    ]
    assert parent_counts.count(1) >= 2

    chunk_metadata = next(
        (
            call.get("metadata")
            for call in metadata_calls
            if call.get("metadata") and "token_count" in call.get("metadata")
        ),
        None,
    )
    assert chunk_metadata is not None
    assert chunk_metadata["token_count"] > 0

    assert cost_calls == [("openai/text-embedding-3-small", 5)]


def test_embed_error_updates_observation(monkeypatch):
    meta = {"tenant_id": "tenant", "case_id": "case"}
    chunks = {
        "chunks": [
            {"content": f"chunk {idx}", "meta": {"hash": f"chunk-{idx}"}}
            for idx in range(12)
        ]
    }

    monkeypatch.setattr(tasks.object_store, "read_json", lambda _: chunks)
    monkeypatch.setattr(tasks.object_store, "write_json", lambda *args, **kwargs: None)
    monkeypatch.setattr(tasks.settings, "EMBEDDINGS_BATCH_SIZE", 3, raising=False)

    class _FailingClient:
        batch_size = 3

        def embed(self, inputs):  # noqa: D401
            raise RuntimeError("boom")

        def dim(self):  # noqa: D401
            return 3

    monkeypatch.setattr(tasks, "get_embedding_client", lambda: _FailingClient())

    span_calls: List[str] = []
    _wrap_observed_span(monkeypatch, span_calls)

    metadata_calls: List[Dict[str, Any]] = []

    def fake_update_observation(**kwargs):  # noqa: D401
        metadata_calls.append(kwargs)

    monkeypatch.setattr(tasks, "update_observation", fake_update_observation)

    monkeypatch.setattr("ai_core.infra.observability.tracing_enabled", lambda: False)

    with pytest.raises(RuntimeError):
        tasks.embed(meta, "chunks.json")

    error_metadata = None
    for call in metadata_calls:
        payload = call.get("metadata")
        if payload and payload.get("failed_chunks_count") is not None:
            error_metadata = payload
            break

    assert error_metadata is not None
    assert error_metadata["status"] == "error"
    assert error_metadata["failed_chunks_count"] == 12
    assert error_metadata["failed_chunk_ids"] == [f"chunk-{idx}" for idx in range(10)]
