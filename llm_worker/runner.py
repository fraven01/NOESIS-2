from __future__ import annotations

from typing import Any, Mapping

from celery import current_app, exceptions as celery_exceptions
from django.conf import settings

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
    """

    meta = dict(task_payload)
    scope_payload = {
        "tenant_id": scope.get("tenant_id"),
        "case_id": scope.get("case_id"),
        "trace_id": scope.get("trace_id"),
    }
    for key, value in scope_payload.items():
        if value and key not in meta:
            meta[key] = value
    scope_context = {key: value for key, value in scope_payload.items() if value}

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
