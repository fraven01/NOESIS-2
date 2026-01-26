from __future__ import annotations

from typing import Any, Mapping

from celery import shared_task
from django.db import connection

from ai_core.graph.execution import LocalGraphExecutor
from ai_core.graphs.technical.cost_tracking import track_ledger_costs
from ai_core.graphs.technical.collection_search import (
    CollectionSearchGraphOutput,
)
from ai_core.ids.http_scope import normalize_task_context
from ai_core.infra.observability import emit_event
from ai_core.tool_contracts.base import tool_context_from_meta
from cases.integration import emit_case_lifecycle_for_collection_search
from common.celery import RetryableTask
from common.logging import get_logger
from llm_worker.graphs import run_score_results

logger = get_logger(__name__)


def _resolve_collection_search_timeout_s() -> float | None:
    try:
        from django.conf import settings
    except Exception:
        return None
    value = getattr(settings, "GRAPH_COLLECTION_SEARCH_TIMEOUT_S", None)
    try:
        timeout = float(value)
    except (TypeError, ValueError):
        return None
    if timeout <= 0:
        return None
    return timeout


@shared_task(
    base=RetryableTask,
    queue="agents-high",
    time_limit=600,
    soft_time_limit=540,
)
def run_graph(  # type: ignore[no-untyped-def]
    *,
    graph_name: str,
    state: Mapping[str, Any] | None,
    meta: Mapping[str, Any] | None,
    ledger_identifier: str | None = None,
    initial_cost_total: float | None = None,
) -> dict[str, Any]:
    """
    Execute a registered graph runner with the provided state and metadata.

    The task proxies the execution to a worker queue so web requests do not
    block on LiteLLM/network latency.

    Note: This task returns results (ignore_result=False by default) which are
    retrieved by the web layer using async_result.get(timeout=...) pattern.
    CELERY_RESULT_BACKEND must be configured for this to work.

    In 202-Fallback-Modus kann das Result-Backend deaktiviert sein;
    siehe GRAPH_WORKER_TIMEOUT_S.
    """
    logger.debug("celery.run_graph.entered")

    runner_state = dict(state or {})
    runner_meta = dict(meta or {})

    tool_context = tool_context_from_meta(runner_meta)

    scope_context = tool_context.scope

    # BREAKING CHANGE (Option A - Strict Separation):
    # Business IDs (workflow_id, collection_id) now in business_context
    business_context = tool_context.business

    tenant_id = scope_context.tenant_id
    case_id = business_context.case_id
    trace_id = scope_context.trace_id

    task_type = runner_meta.get("task_type", "rag_query")

    # Set tenant schema context for database queries
    # This ensures document models are accessed in the correct tenant schema
    if tenant_id and task_type != "score_results":
        try:
            from customers.models import Tenant

            tenant = Tenant.objects.get(schema_name=str(tenant_id))
            connection.set_tenant(tenant)
            logger.debug(
                "celery.run_graph.tenant_schema_set",
                extra={"tenant_schema": tenant.schema_name},
            )
        except Tenant.DoesNotExist:
            # Log warning but continue - some graphs might not need DB access
            logger.warning(
                "celery.run_graph.tenant_missing",
                extra={"tenant_id": tenant_id},
            )

    # Build ScopeContext via normalize_task_context (Pre-MVP ID Contract)
    # S2S Hop: service_id REQUIRED, user_id ABSENT
    scope = normalize_task_context(
        tenant_id=tenant_id,
        service_id="celery-agents-worker",
        trace_id=trace_id,
        invocation_id=scope_context.invocation_id,
        run_id=scope_context.run_id,
        ingestion_run_id=scope_context.ingestion_run_id,
        idempotency_key=scope_context.idempotency_key,
        tenant_schema=scope_context.tenant_schema,
    )
    # Inject updated scope/tool context into meta for graph execution
    updated_context = tool_context.model_copy(update={"scope": scope})
    runner_meta["scope_context"] = scope.model_dump(mode="json")
    runner_meta["tool_context"] = updated_context.model_dump(
        mode="json", exclude_none=True
    )
    with track_ledger_costs(initial_cost_total) as tracker:
        runner_meta["ledger_logger"] = tracker.record_ledger_meta
        try:
            if task_type == "score_results":
                config_payload = runner_meta.get("config")
                if not isinstance(config_payload, Mapping):
                    config_payload = {}
                data_payload = runner_meta.get("data")
                if not isinstance(data_payload, Mapping):
                    data_payload = {}

                config_payload = dict(config_payload)
                if tenant_id and "tenant_id" not in config_payload:
                    config_payload["tenant_id"] = tenant_id
                if case_id and "case_id" not in config_payload:
                    config_payload["case_id"] = case_id
                if trace_id and "trace_id" not in config_payload:
                    config_payload["trace_id"] = trace_id
                if "key_alias" in runner_meta and "key_alias" not in config_payload:
                    config_payload["key_alias"] = runner_meta["key_alias"]
                if "ledger_logger" in runner_meta:
                    config_payload["ledger_logger"] = runner_meta["ledger_logger"]

                result = run_score_results(data_payload, config=config_payload)
                new_state = runner_state
            else:
                graph_executor = LocalGraphExecutor()
                if graph_name == "collection_search":
                    timeout_s = _resolve_collection_search_timeout_s()
                else:
                    timeout_s = None
                if timeout_s is None:
                    new_state, result = graph_executor.run(
                        graph_name, runner_state, runner_meta
                    )
                else:
                    from concurrent.futures import (
                        ThreadPoolExecutor,
                        TimeoutError as FutureTimeout,
                    )

                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(
                            graph_executor.run, graph_name, runner_state, runner_meta
                        )
                        try:
                            new_state, result = future.result(timeout=timeout_s)
                        except FutureTimeout:
                            future.cancel()
                            result = CollectionSearchGraphOutput(
                                outcome="error",
                                search=None,
                                telemetry={"graph_timeout_s": timeout_s},
                                ingestion=None,
                                plan=None,
                                hitl=None,
                                error="graph_timeout",
                            ).model_dump(mode="json")
                            new_state = {"error": "graph_timeout"}
        finally:
            runner_meta.pop("ledger_logger", None)
            # Ensure ledger_logger is removed from the state if it was copied there
            if "new_state" in locals() and isinstance(new_state, dict):
                state_meta = new_state.get("meta")
                if isinstance(state_meta, dict):
                    state_meta.pop("ledger_logger", None)

    cost_summary = tracker.summary(ledger_identifier)
    # Round total_usd to 4 decimal places to reduce noise in logs/traces
    if cost_summary and "total_usd" in cost_summary:
        cost_summary["total_usd"] = round(cost_summary["total_usd"], 4)

    payload = {
        "state": new_state,
        "result": result,
        "cost_summary": cost_summary,
    }

    def _recursive_serialize(obj: Any) -> Any:
        from dataclasses import fields, is_dataclass
        from datetime import datetime
        import uuid
        from types import MappingProxyType
        from pydantic import BaseModel

        if isinstance(obj, dict):
            return {k: _recursive_serialize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set, frozenset)):
            return [_recursive_serialize(v) for v in obj]
        if isinstance(obj, MappingProxyType):
            return _recursive_serialize(dict(obj))
        if is_dataclass(obj) and not isinstance(obj, type):
            # Avoid asdict() because it uses deepcopy which fails on MappingProxyType
            return {
                f.name: _recursive_serialize(getattr(obj, f.name)) for f in fields(obj)
            }
        if isinstance(obj, BaseModel):
            return _recursive_serialize(obj.model_dump(mode="json"))
        if isinstance(obj, uuid.UUID):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()

        return obj

    try:
        payload = _recursive_serialize(payload)
    except Exception as exc:
        logger.warning(
            "celery.run_graph.serialization_failed",
            extra={"error": str(exc)},
        )

    lifecycle_result = emit_case_lifecycle_for_collection_search(
        graph_name=graph_name,
        tenant_id=tenant_id,
        case_id=case_id,
        state=new_state,
    )
    if lifecycle_result is not None:
        case = lifecycle_result.case
        lifecycle_payload = {
            "tenant_id": str(case.tenant_id),
            "case_id": case.external_id,
            "case_status": case.status,
            "case_phase": case.phase or "",
            "case_event_types": lifecycle_result.event_types,
            "collection_scope": lifecycle_result.collection_scope,
            "workflow_id": lifecycle_result.workflow_id,
            "graph_name": graph_name,
        }
        if lifecycle_result.trace_id:
            lifecycle_payload["trace_id"] = lifecycle_result.trace_id
        elif trace_id:
            lifecycle_payload["trace_id"] = trace_id

        payload["case_lifecycle"] = lifecycle_payload
        emit_event("case.lifecycle.collection_search", lifecycle_payload)

    return payload
