"""Tests for the file-based graph checkpointer."""

from __future__ import annotations

import json

from ai_core.graph.core import FileCheckpointer, GraphContext
from ai_core.infra import object_store


def test_load_returns_empty_when_state_file_absent(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(object_store, "BASE_PATH", tmp_path)
    checkpointer = FileCheckpointer()
    ctx = GraphContext(
        tenant_id="tenant-123",
        case_id="case-456",
        trace_id="trace-789",
        graph_name="info_intake",
    )

    assert checkpointer.load(ctx) == {}


def test_save_persists_state_under_sanitized_path(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(object_store, "BASE_PATH", tmp_path)
    checkpointer = FileCheckpointer()
    ctx = GraphContext(
        tenant_id="Tenant One",
        case_id="Case:01",
        trace_id="trace-abc",
        graph_name="scope_check",
    )
    state = {"step": "complete", "nested": {"value": 3}}

    checkpointer.save(ctx, state)

    safe_tenant = object_store.sanitize_identifier(ctx.tenant_id)
    safe_case = object_store.sanitize_identifier(ctx.case_id)
    state_path = tmp_path / safe_tenant / safe_case / "state.json"

    assert state_path.exists()
    assert json.loads(state_path.read_text()) == state
    assert checkpointer.load(ctx) == state
