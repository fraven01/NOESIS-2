from __future__ import annotations

from typing import Any, Mapping

from celery import shared_task

from ai_core.graph.registry import get as get_graph_runner
from ai_core.graphs.cost_tracking import track_ledger_costs
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

    # Scope parameters (tenant_id, case_id, trace_id, session_salt) are accepted
    # so ScopedTask/with_scope_apply_async can attach masking context without
    # causing unexpected kwargs errors. They are currently unused because the
    # runner receives fully prepared meta/state.

    runner_state = dict(state or {})
    runner_meta = dict(meta or {})
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

    cost_summary = tracker.summary(ledger_identifier)
    # Round total_usd to 4 decimal places to reduce noise in logs/traces
    if cost_summary and "total_usd" in cost_summary:
        cost_summary["total_usd"] = round(cost_summary["total_usd"], 4)

    return {
        "state": new_state,
        "result": result,
        "cost_summary": cost_summary,
    }
