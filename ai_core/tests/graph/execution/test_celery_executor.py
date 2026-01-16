"""Tests for the Celery-based graph executor."""

from __future__ import annotations

from unittest.mock import MagicMock

from ai_core.contracts.business import BusinessContext
from ai_core.contracts.scope import ScopeContext
from ai_core.graph.execution.celery import CeleryGraphExecutor


def _tool_context_payload() -> dict[str, object]:
    scope = ScopeContext(
        tenant_id="tenant-1",
        trace_id="trace-1",
        invocation_id="inv-1",
        run_id="run-1",
        service_id="svc-1",
    )
    business = BusinessContext(case_id="case-1")
    context = scope.to_tool_context(business=business)
    return context.model_dump(mode="json")


def test_submit_schedules_graph() -> None:
    signature = MagicMock()
    async_result = MagicMock()
    async_result.id = "task-123"

    signature_factory = MagicMock(return_value=signature)
    apply_async = MagicMock(return_value=async_result)

    executor = CeleryGraphExecutor(
        signature_factory=signature_factory,
        apply_async=apply_async,
    )

    meta = {"tool_context": _tool_context_payload()}
    task_id = executor.submit("framework_analysis", {"input": {}}, meta)

    signature_factory.assert_called_once()
    apply_async.assert_called_once_with(
        signature, {"tenant_id": "tenant-1", "case_id": "case-1", "trace_id": "trace-1"}
    )
    assert task_id == "task-123"


def test_run_delegates_to_local_executor() -> None:
    local_executor = MagicMock()
    local_executor.run.return_value = ({"state": "ok"}, {"result": "ok"})

    executor = CeleryGraphExecutor(local_executor=local_executor)

    state, result = executor.run("framework_analysis", {"input": {}}, {"meta": "ok"})

    local_executor.run.assert_called_once_with(
        "framework_analysis", {"input": {}}, {"meta": "ok"}
    )
    assert state == {"state": "ok"}
    assert result == {"result": "ok"}
