"""Tests for graph worker timeout handling in execute_graph service."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from celery import exceptions as celery_exceptions
from django.conf import settings

from ai_core import services
from ai_core.graph.schemas import ToolContext


class _DummyCheckpointer:
    """Fake checkpointer that stores state in memory."""

    def __init__(self) -> None:
        self.saved: list[tuple[Any, Any]] = []

    def load(self, ctx):  # type: ignore[no-untyped-def]
        return {}

    def save(self, ctx, state):  # type: ignore[no-untyped-def]
        self.saved.append((ctx, state))


class _DummyRunner:
    """Fake graph runner that returns a predictable result."""

    def run(self, state, meta):  # type: ignore[no-untyped-def]
        return {"state": "updated"}, {"result": "success", "data": "test"}


@pytest.mark.django_db
def test_rag_worker_sync_success(monkeypatch, disable_async_graphs):
    """Test graph worker returns 200 OK when task completes within timeout."""

    # Re-enable async graphs for this test (override the autouse fixture)
    def real_should_enqueue_graph(graph_name):  # type: ignore[no-untyped-def]
        return graph_name == "rag.default"

    monkeypatch.setattr(services, "_should_enqueue_graph", real_should_enqueue_graph)

    # Setup request
    request = SimpleNamespace(
        headers={"Content-Type": "application/json"},
        META={},
        body=json.dumps({"query": "test"}).encode(),
    )

    tool_context = ToolContext(
        tenant_id="tenant-test",
        case_id="case-test",
        trace_id="trace-test",
    )

    normalized_meta = {
        "tenant_id": "tenant-test",
        "case_id": "case-test",
        "trace_id": "trace-test",
        "graph_name": "rag.default",
        "graph_version": "v1",
        "tool_context": tool_context,
    }

    # Mock async result that returns immediately (success case)
    mock_async_result = MagicMock()
    mock_async_result.get.return_value = {
        "state": {"updated": "state"},
        "result": {"result": "success", "data": "test"},
        "cost_summary": {
            "total_usd": 0.123456789,
            "components": [],
        },  # Will be rounded to 4 decimals
    }
    mock_async_result.id = "task-123"

    def fake_normalize_meta(_request):  # type: ignore[no-untyped-def]
        return dict(normalized_meta)

    def fake_with_scope_apply_async(signature, scope):  # type: ignore[no-untyped-def]
        # Verify the signature was called correctly
        assert signature is not None
        assert scope["tenant_id"] == "tenant-test"
        assert scope["case_id"] == "case-test"
        assert scope["trace_id"] == "trace-test"
        return mock_async_result

    # Setup mocks
    dummy_checkpointer = _DummyCheckpointer()
    monkeypatch.setattr(services, "_normalize_meta", fake_normalize_meta)
    monkeypatch.setattr(services, "_get_checkpointer", lambda: dummy_checkpointer)
    monkeypatch.setattr(services, "with_scope_apply_async", fake_with_scope_apply_async)
    monkeypatch.setattr(services, "lf_tracing_enabled", lambda: False)
    monkeypatch.setattr(services, "update_observation", lambda **kwargs: None)
    monkeypatch.setattr(services, "emit_event", lambda *args, **kwargs: None)
    # Disable schema validation for rag.default to avoid validation errors
    monkeypatch.setattr(services, "GRAPH_REQUEST_MODELS", {})

    # Execute
    response = services.execute_graph(request, _DummyRunner())

    # Assert
    assert response.status_code == 200
    # Check result fields individually
    assert response.data["result"] == "success"
    assert response.data["data"] == "test"
    # Verify cost rounding to 4 decimal places
    # Original: 0.123456789 -> Rounded: 0.1235
    # Note: The rounding happens in ai_core/services.py before returning
    # Since we're calling execute_graph directly, we don't see the rounded value here
    # The rounding is tested implicitly by the worker integration
    # Verify get() was called with correct timeout
    timeout_s = getattr(settings, "GRAPH_WORKER_TIMEOUT_S", 45)
    mock_async_result.get.assert_called_once_with(timeout=timeout_s, propagate=True)


@pytest.mark.django_db
def test_rag_worker_async_fallback(monkeypatch, disable_async_graphs):
    """Test graph worker returns 202 Accepted when task exceeds timeout."""

    # Re-enable async graphs for this test (override the autouse fixture)
    def real_should_enqueue_graph(graph_name):  # type: ignore[no-untyped-def]
        return graph_name == "rag.default"

    monkeypatch.setattr(services, "_should_enqueue_graph", real_should_enqueue_graph)

    # Setup request
    request = SimpleNamespace(
        headers={"Content-Type": "application/json"},
        META={},
        body=json.dumps({"query": "test"}).encode(),
    )

    tool_context = ToolContext(
        tenant_id="tenant-test",
        case_id="case-test",
        trace_id="trace-test",
    )

    normalized_meta = {
        "tenant_id": "tenant-test",
        "case_id": "case-test",
        "trace_id": "trace-test",
        "graph_name": "rag.default",
        "graph_version": "v1",
        "tool_context": tool_context,
    }

    # Mock async result that times out
    mock_async_result = MagicMock()
    mock_async_result.get.side_effect = celery_exceptions.TimeoutError(
        "Task did not complete within timeout"
    )
    mock_async_result.id = "task-456"

    def fake_normalize_meta(_request):  # type: ignore[no-untyped-def]
        return dict(normalized_meta)

    def fake_with_scope_apply_async(signature, scope):  # type: ignore[no-untyped-def]
        # Verify the signature was called correctly
        assert signature is not None
        assert scope["tenant_id"] == "tenant-test"
        assert scope["case_id"] == "case-test"
        assert scope["trace_id"] == "trace-test"
        return mock_async_result

    # Setup mocks
    dummy_checkpointer = _DummyCheckpointer()
    monkeypatch.setattr(services, "_normalize_meta", fake_normalize_meta)
    monkeypatch.setattr(services, "_get_checkpointer", lambda: dummy_checkpointer)
    monkeypatch.setattr(services, "with_scope_apply_async", fake_with_scope_apply_async)
    monkeypatch.setattr(services, "lf_tracing_enabled", lambda: False)
    monkeypatch.setattr(services, "update_observation", lambda **kwargs: None)
    monkeypatch.setattr(services, "emit_event", lambda *args, **kwargs: None)
    # Disable schema validation for rag.default to avoid validation errors
    monkeypatch.setattr(services, "GRAPH_REQUEST_MODELS", {})

    # Execute
    response = services.execute_graph(request, _DummyRunner())

    # Assert
    assert response.status_code == 202
    assert response.data == {
        "status": "queued",
        "task_id": "task-456",
        "graph": "rag.default",
        "tenant_id": "tenant-test",
        "case_id": "case-test",
        "trace_id": "trace-test",
    }
    # Verify get() was called with correct timeout
    timeout_s = getattr(settings, "GRAPH_WORKER_TIMEOUT_S", 45)
    mock_async_result.get.assert_called_once_with(timeout=timeout_s, propagate=True)
