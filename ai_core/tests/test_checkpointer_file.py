"""Tests for the file-based graph checkpointer."""

from __future__ import annotations

import json

from datetime import datetime, timezone

import pytest

from ai_core.contracts.business import BusinessContext
from ai_core.contracts.plans import (
    Evidence,
    ImplementationPlan,
    PlanScope,
    derive_plan_key,
)
from ai_core.contracts.scope import ScopeContext
from ai_core.graph.core import FileCheckpointer, GraphContext, ThreadAwareCheckpointer
from ai_core.graph.state import load_plan_from_scope
from ai_core.infra import object_store
from ai_core.tool_contracts.base import ToolContext, tool_context_from_scope


def _build_plan_scope() -> PlanScope:
    return PlanScope(
        tenant_id="tenant-1",
        gremium_identifier="Board A",
        framework_profile_id=None,
        framework_profile_version="v1",
        case_id="case-1",
        workflow_id="workflow-1",
        run_id="run-1",
    )


def _build_plan(scope: PlanScope, *, evidence_count: int) -> ImplementationPlan:
    evidence = [
        Evidence(
            ref_type="url",
            ref_id=f"https://example.com/source-{idx}",
            summary="source",
        )
        for idx in range(evidence_count)
    ]
    return ImplementationPlan(
        plan_key=derive_plan_key(scope),
        scope=scope,
        evidence=evidence,
    )


def test_load_returns_empty_when_state_file_absent(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(object_store, "BASE_PATH", tmp_path)
    checkpointer = FileCheckpointer()
    scope = ScopeContext(
        tenant_id="tenant-123",
        trace_id="trace-789",
        invocation_id="invocation-001",
        run_id="test-run",
    )
    business = BusinessContext(workflow_id="workflow-456")
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
    business = BusinessContext(workflow_id="Workflow:01")
    ctx = GraphContext(
        tool_context=tool_context_from_scope(scope, business),
        graph_name="retrieval_augmented_generation",
    )
    state = {"step": "complete", "nested": {"value": 3}}

    checkpointer.save(ctx, state)

    safe_tenant = object_store.sanitize_identifier(ctx.tenant_id)
    safe_workflow = object_store.sanitize_identifier(ctx.workflow_id)
    safe_run = object_store.sanitize_identifier(ctx.run_id)
    state_path = (
        tmp_path
        / safe_tenant
        / "workflow-executions"
        / safe_workflow
        / safe_run
        / "state.json"
    )

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
    business = BusinessContext(workflow_id="Workflow:01", thread_id="thread-123")
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


def test_checkpointer_separates_runs_by_run_id(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(object_store, "BASE_PATH", tmp_path)
    checkpointer = FileCheckpointer()
    business = BusinessContext(workflow_id="Workflow:01")
    scope_one = ScopeContext(
        tenant_id="Tenant One",
        trace_id="trace-001",
        invocation_id="invocation-004",
        run_id="run-1",
    )
    scope_two = ScopeContext(
        tenant_id="Tenant One",
        trace_id="trace-002",
        invocation_id="invocation-005",
        run_id="run-2",
    )
    ctx_one = GraphContext(
        tool_context=tool_context_from_scope(scope_one, business),
        graph_name="retrieval_augmented_generation",
    )
    ctx_two = GraphContext(
        tool_context=tool_context_from_scope(scope_two, business),
        graph_name="retrieval_augmented_generation",
    )

    checkpointer.save(ctx_one, {"step": "one"})
    checkpointer.save(ctx_two, {"step": "two"})

    safe_tenant = object_store.sanitize_identifier(ctx_one.tenant_id)
    safe_workflow = object_store.sanitize_identifier(ctx_one.workflow_id)
    path_one = (
        tmp_path
        / safe_tenant
        / "workflow-executions"
        / safe_workflow
        / object_store.sanitize_identifier(ctx_one.run_id)
        / "state.json"
    )
    path_two = (
        tmp_path
        / safe_tenant
        / "workflow-executions"
        / safe_workflow
        / object_store.sanitize_identifier(ctx_two.run_id)
        / "state.json"
    )

    assert path_one.exists()
    assert path_two.exists()
    assert path_one != path_two
    assert checkpointer.load(ctx_one) == {"step": "one"}
    assert checkpointer.load(ctx_two) == {"step": "two"}


def test_checkpointer_uses_plan_key_when_present(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(object_store, "BASE_PATH", tmp_path)
    checkpointer = FileCheckpointer()
    scope = ScopeContext(
        tenant_id="Tenant One",
        trace_id="trace-003",
        invocation_id="invocation-008",
        run_id="run-3",
    )
    business = BusinessContext(workflow_id=None)
    ctx = GraphContext(
        tool_context=tool_context_from_scope(
            scope, business, metadata={"plan_key": "Plan: 01"}
        ),
        graph_name="retrieval_augmented_generation",
    )
    state = {"step": "plan-key"}

    checkpointer.save(ctx, state)

    safe_tenant = object_store.sanitize_identifier(ctx.tenant_id)
    safe_plan = object_store.sanitize_identifier("Plan: 01")
    state_path = (
        tmp_path / safe_tenant / "workflow-executions" / safe_plan / "state.json"
    )

    assert state_path.exists()
    payload = json.loads(state_path.read_text())
    assert payload["state"] == state
    assert checkpointer.load(ctx) == state


def test_checkpointer_requires_workflow_id(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(object_store, "BASE_PATH", tmp_path)
    scope = ScopeContext(
        tenant_id="tenant-123",
        trace_id="trace-abc",
        invocation_id="invocation-006",
        run_id="run-1",
    )
    business = BusinessContext(workflow_id=None)
    with pytest.raises(ValueError, match="workflow_id and run_id"):
        GraphContext(
            tool_context=tool_context_from_scope(scope, business),
            graph_name="info_intake",
        )


def test_checkpointer_requires_run_id(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(object_store, "BASE_PATH", tmp_path)
    scope = ScopeContext.model_construct(
        tenant_id="tenant-123",
        trace_id="trace-def",
        invocation_id="invocation-007",
        run_id=None,
        ingestion_run_id=None,
        user_id=None,
        service_id=None,
        tenant_schema=None,
        idempotency_key=None,
        timestamp=datetime.now(timezone.utc),
    )
    business = BusinessContext(workflow_id="workflow-1")
    with pytest.raises(ValueError, match="workflow_id and run_id"):
        GraphContext(
            tool_context=ToolContext.model_construct(
                scope=scope,
                business=business,
                locale=None,
                timeouts_ms=None,
                budget_tokens=None,
                safety_mode=None,
                auth=None,
                visibility_override_allowed=False,
                metadata={},
            ),
            graph_name="info_intake",
        )


def test_checkpointer_persists_plan_and_evidence(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(object_store, "BASE_PATH", tmp_path)
    checkpointer = FileCheckpointer()
    plan_scope = _build_plan_scope()
    plan = _build_plan(plan_scope, evidence_count=1)
    scope = ScopeContext(
        tenant_id=plan_scope.tenant_id,
        trace_id="trace-901",
        invocation_id="invocation-901",
        run_id=plan_scope.run_id,
    )
    business = BusinessContext(
        workflow_id=plan_scope.workflow_id,
        case_id=plan_scope.case_id,
    )
    ctx = GraphContext(
        tool_context=tool_context_from_scope(scope, business),
        graph_name="collection_search",
    )
    state = {"plan": plan.model_dump(mode="json")}

    checkpointer.save(ctx, state)

    safe_tenant = object_store.sanitize_identifier(ctx.tenant_id)
    safe_plan = object_store.sanitize_identifier(plan.plan_key)
    state_path = (
        tmp_path / safe_tenant / "workflow-executions" / safe_plan / "state.json"
    )
    payload = json.loads(state_path.read_text())

    assert payload["plan"]["plan_key"] == plan.plan_key
    assert payload["evidence"][0]["ref_id"] == plan.evidence[0].ref_id


def test_load_plan_from_scope_returns_latest(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(object_store, "BASE_PATH", tmp_path)
    checkpointer = FileCheckpointer()
    plan_scope = _build_plan_scope()
    scope = ScopeContext(
        tenant_id=plan_scope.tenant_id,
        trace_id="trace-902",
        invocation_id="invocation-902",
        run_id=plan_scope.run_id,
    )
    business = BusinessContext(
        workflow_id=plan_scope.workflow_id,
        case_id=plan_scope.case_id,
    )
    ctx = GraphContext(
        tool_context=tool_context_from_scope(scope, business),
        graph_name="collection_search",
    )

    plan = _build_plan(plan_scope, evidence_count=1)
    checkpointer.save(ctx, {"plan": plan.model_dump(mode="json")})

    updated_plan = _build_plan(plan_scope, evidence_count=2)
    checkpointer.save(ctx, {"plan": updated_plan.model_dump(mode="json")})

    loaded = load_plan_from_scope(plan_scope)

    assert loaded is not None
    loaded_plan, evidence = loaded
    assert loaded_plan.plan_key == updated_plan.plan_key
    assert len(evidence) == 2
    assert isinstance(evidence[0], Evidence)
