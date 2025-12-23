from __future__ import annotations

from typing import Any, Mapping
from uuid import uuid4

from celery import current_app, exceptions as celery_exceptions
from django.conf import settings

from ai_core.contracts.scope import ScopeContext
from common.celery import with_scope_apply_async


def submit_worker_task(
    *,
    task_payload: Mapping[str, Any],
    scope: Mapping[str, Any],
    graph_name: str,
    ledger_identifier: str | None = None,
    initial_cost_total: float | None = None,
    timeout_s: float | None = None,
) -> tuple[dict[str, Any], bool]:
    """
    Enqueue a worker task and optionally wait for the synchronous result.

    Returns a tuple ``(payload, completed)`` where ``completed`` indicates
    whether the worker finished within the timeout window. When ``False``
    the payload only contains ``task_id`` so the caller can poll later.

    The scope dict should contain context IDs for traceability (Pre-MVP ID Contract):
    - tenant_id, case_id, trace_id (required)
    - user_id (optional, for User Request Hops)
    - workflow_id, run_id, ingestion_run_id (optional)
    """

    meta = dict(task_payload)
    # Extract scope fields including identity IDs (Pre-MVP ID Contract)
    scope_payload = {
        "tenant_id": scope.get("tenant_id"),
        "case_id": scope.get("case_id"),
        "trace_id": scope.get("trace_id"),
        "invocation_id": scope.get("invocation_id") or uuid4().hex,
        "run_id": scope.get("run_id"),
        "ingestion_run_id": scope.get("ingestion_run_id"),
        "workflow_id": scope.get("workflow_id"),
        "idempotency_key": scope.get("idempotency_key"),
        "collection_id": scope.get("collection_id"),
        "tenant_schema": scope.get("tenant_schema"),
        # Identity IDs (Pre-MVP ID Contract)
        "user_id": scope.get("user_id"),  # User Request Hop
        "service_id": scope.get("service_id"),  # S2S Hop
    }
    if not scope_payload.get("run_id") and not scope_payload.get("ingestion_run_id"):
        scope_payload["run_id"] = uuid4().hex

    scope_context = ScopeContext.model_validate(scope_payload).model_dump(mode="json")
    meta["scope_context"] = scope_context

    for key in (
        "tenant_id",
        "case_id",
        "trace_id",
        "invocation_id",
        "run_id",
        "ingestion_run_id",
        "workflow_id",
        "idempotency_key",
        "collection_id",
        "tenant_schema",
        "user_id",
        "service_id",
    ):
        meta.pop(key, None)

    task_state = meta.pop("state", {})

    signature = current_app.signature(
        "llm_worker.tasks.run_graph",
        kwargs={
            "graph_name": graph_name,
            "state": task_state,
            "meta": meta,
            "ledger_identifier": ledger_identifier,
            "initial_cost_total": initial_cost_total,
        },
        queue="agents",
    )

    async_result = with_scope_apply_async(signature, scope_context)

    timeout = timeout_s
    if timeout is None:
        timeout = getattr(settings, "GRAPH_WORKER_TIMEOUT_S", 45)

    try:
        task_result = async_result.get(timeout=timeout, propagate=True)
    except celery_exceptions.TimeoutError:
        return {"task_id": async_result.id}, False

    response_payload = dict(task_result)
    response_payload["task_id"] = async_result.id
    return response_payload, True


__all__ = ["submit_worker_task"]
