from __future__ import annotations

from typing import Any, Mapping

from celery import shared_task
from django.db import connection

from ai_core.graph.registry import get as get_graph_runner
from ai_core.graphs.technical.cost_tracking import track_ledger_costs
from ai_core.ids.http_scope import normalize_task_context
from ai_core.infra.observability import emit_event
from cases.integration import emit_case_lifecycle_for_collection_search
from common.celery import ScopedTask
from llm_worker.graphs import run_score_results


@shared_task(base=ScopedTask, queue="agents", accepts_scope=True)
def run_graph(  # type: ignore[no-untyped-def]
    *,
    graph_name: str,
    state: Mapping[str, Any] | None,
    meta: Mapping[str, Any] | None,
    ledger_identifier: str | None = None,
    initial_cost_total: float | None = None,
    tenant_id: str | None = None,
    case_id: str | None = None,
    trace_id: str | None = None,
    session_salt: str | None = None,
    **_scope: Any,
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
    import sys

    sys.stderr.write("DEBUG: run_graph ENTERED\n")
    sys.stderr.flush()

    # Set tenant schema context for database queries
    # This ensures document models are accessed in the correct tenant schema
    if tenant_id:
        try:
            from customers.models import Tenant

            tenant = Tenant.objects.get(schema_name=str(tenant_id))
            connection.set_tenant(tenant)
            sys.stderr.write(f"DEBUG: Tenant schema set to {tenant.schema_name}\n")
            sys.stderr.flush()
        except Tenant.DoesNotExist:
            # Log warning but continue - some graphs might not need DB access
            sys.stderr.write(
                f"WARNING: Tenant {tenant_id} not found, using default schema\n"
            )
            sys.stderr.flush()

    # Scope parameters (tenant_id, case_id, trace_id, session_salt) are accepted
    # so ScopedTask/with_scope_apply_async can attach masking context without
    # causing unexpected kwargs errors.

    runner_state = dict(state or {})
    runner_meta = dict(meta or {})

    scope_context = runner_meta.get("scope_context")
    if not isinstance(scope_context, Mapping):
        scope_context = {}

    # BREAKING CHANGE (Option A - Strict Separation):
    # Business IDs (workflow_id, collection_id) now in business_context
    business_context = runner_meta.get("business_context")
    if not isinstance(business_context, Mapping):
        business_context = {}

    # Build ScopeContext via normalize_task_context (Pre-MVP ID Contract)
    # S2S Hop: service_id REQUIRED, user_id ABSENT
    # BREAKING CHANGE (Option A): case_id is optional, only check tenant_id
    if tenant_id:
        scope = normalize_task_context(
            tenant_id=tenant_id,
            case_id=case_id,  # Optional after Option A
            service_id="celery-agents-worker",
            trace_id=trace_id or scope_context.get("trace_id"),
            invocation_id=scope_context.get("invocation_id"),
            workflow_id=business_context.get(
                "workflow_id"
            ),  # BREAKING CHANGE: from business_context
            run_id=scope_context.get("run_id"),
            ingestion_run_id=scope_context.get("ingestion_run_id"),
            idempotency_key=scope_context.get("idempotency_key"),
            tenant_schema=scope_context.get("tenant_schema"),
            collection_id=business_context.get(
                "collection_id"
            ),  # BREAKING CHANGE: from business_context
        )
        # Inject scope context into meta for graph execution
        runner_meta["scope_context"] = scope.model_dump(mode="json")
    task_type = runner_meta.get("task_type", "rag_query")

    with track_ledger_costs(initial_cost_total) as tracker:
        runner_meta["ledger_logger"] = tracker.record_ledger_meta
        try:
            if task_type == "score_results":
                control_meta = runner_meta.get("control")
                if not isinstance(control_meta, Mapping):
                    control_meta = {}
                data_payload = runner_meta.get("data")
                if not isinstance(data_payload, Mapping):
                    data_payload = {}
                result = run_score_results(
                    control=control_meta,
                    data=data_payload,
                    meta=runner_meta,
                )
                new_state = runner_state
            else:
                runner = get_graph_runner(graph_name)
                new_state, result = runner.run(runner_state, runner_meta)
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
        sys.stderr.write(f"DEBUG: Serialization failed: {exc}\n")
        sys.stderr.flush()

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
