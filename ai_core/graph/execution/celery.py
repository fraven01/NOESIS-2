"""Celery-based implementation of the GraphExecutor protocol."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Callable

from celery import current_app

from ai_core.graph.execution.contract import GraphExecutor
from ai_core.graph.execution.local import LocalGraphExecutor
from ai_core.tool_contracts.base import tool_context_from_meta
from common.celery import with_scope_apply_async

logger = logging.getLogger(__name__)

SignatureFactory = Callable[..., Any]
ApplyAsync = Callable[..., Any]


def _scope_from_meta(meta: Mapping[str, Any] | None) -> dict[str, str]:
    if not isinstance(meta, Mapping):
        return {}
    try:
        tool_context = tool_context_from_meta(meta)
    except (TypeError, ValueError):
        return {}

    scope = tool_context.scope
    business = tool_context.business
    payload = {
        "tenant_id": scope.tenant_id,
        "case_id": business.case_id,
        "trace_id": scope.trace_id,
    }
    return {k: str(v) for k, v in payload.items() if v}


class CeleryGraphExecutor(GraphExecutor):
    """Execute graphs via Celery for async submission."""

    def __init__(
        self,
        *,
        queue: str = "agents-high",
        signature_factory: SignatureFactory | None = None,
        apply_async: ApplyAsync | None = None,
        local_executor: GraphExecutor | None = None,
    ) -> None:
        self._queue = queue
        self._signature_factory = signature_factory or current_app.signature
        self._apply_async = apply_async or with_scope_apply_async
        self._local_executor = local_executor or LocalGraphExecutor()

    def run(
        self, name: str, input: dict[str, Any], meta: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Synchronously execute via local fallback."""
        logger.info(
            "graph.execution.celery.run.local_fallback",
            extra={"graph": name, "executor": "celery"},
        )
        return self._local_executor.run(name, input, meta)

    def submit(self, name: str, input: dict[str, Any], meta: dict[str, Any]) -> str:
        """Submit graph execution to the Celery worker queue."""
        signature = self._signature_factory(
            "llm_worker.tasks.run_graph",
            kwargs={
                "graph_name": name,
                "state": input,
                "meta": meta,
                "ledger_identifier": None,
                "initial_cost_total": None,
            },
            queue=self._queue,
        )
        scope = _scope_from_meta(meta)
        async_result = self._apply_async(signature, scope)
        task_id = getattr(async_result, "id", None)
        return str(task_id) if task_id is not None else ""
