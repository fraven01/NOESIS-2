"""Tests for the file-based graph checkpointer."""

from __future__ import annotations

import json

from ai_core.contracts.business import BusinessContext
from ai_core.contracts.scope import ScopeContext
from ai_core.graph.core import FileCheckpointer, GraphContext, ThreadAwareCheckpointer
from ai_core.infra import object_store
from ai_core.tool_contracts.base import tool_context_from_scope


def test_load_returns_empty_when_state_file_absent(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(object_store, "BASE_PATH", tmp_path)
    checkpointer = FileCheckpointer()
    scope = ScopeContext(
        tenant_id="tenant-123",
        trace_id="trace-789",
        invocation_id="invocation-001",
        run_id="test-run",
    )
    business = BusinessContext(case_id="case-456")
    ctx = GraphContext(
        tool_context=tool_context_from_scope(scope, business),
        graph_name="info_intake",
    )

    assert checkpointer.load(ctx) == {}


def test_save_persists_state_under_sanitized_path(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(object_store, "BASE_PATH", tmp_path)
    checkpointer = FileCheckpointer()
    scope = ScopeContext(
        tenant_id="Tenant One",
        trace_id="trace-abc",
        invocation_id="invocation-002",
        run_id="test-run",
    )
    business = BusinessContext(case_id="Case:01")
    ctx = GraphContext(
        tool_context=tool_context_from_scope(scope, business),
        graph_name="retrieval_augmented_generation",
    )
    state = {"step": "complete", "nested": {"value": 3}}

    checkpointer.save(ctx, state)

    safe_tenant = object_store.sanitize_identifier(ctx.tenant_id)
    safe_case = object_store.sanitize_identifier(ctx.case_id)
    state_path = tmp_path / safe_tenant / safe_case / "state.json"

    assert state_path.exists()
    payload = json.loads(state_path.read_text())
    assert payload["state"] == state
    assert payload["graph_name"] == ctx.graph_name
    assert payload["graph_version"] == ctx.graph_version
    assert "tool_context" in payload
    assert checkpointer.load(ctx) == state


def test_thread_checkpointer_uses_thread_id(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(object_store, "BASE_PATH", tmp_path)
    checkpointer = ThreadAwareCheckpointer()
    scope = ScopeContext(
        tenant_id="Tenant One",
        trace_id="trace-xyz",
        invocation_id="invocation-003",
        run_id="test-run",
    )
    business = BusinessContext(case_id="Case:01", thread_id="thread-123")
    ctx = GraphContext(
        tool_context=tool_context_from_scope(scope, business),
        graph_name="retrieval_augmented_generation",
    )
    state = {"chat_history": [{"role": "user", "content": "hello"}]}

    checkpointer.save(ctx, state)

    safe_tenant = object_store.sanitize_identifier(ctx.tenant_id)
    safe_thread = object_store.sanitize_identifier("thread-123")
    state_path = tmp_path / safe_tenant / "threads" / safe_thread / "state.json"

    assert state_path.exists()
    payload = json.loads(state_path.read_text())
    assert payload["state"] == state
    assert checkpointer.load(ctx) == state
