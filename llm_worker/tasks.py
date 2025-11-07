from __future__ import annotations

from typing import Any, Mapping

from celery import shared_task

from ai_core.graph.registry import get as get_graph_runner
from ai_core.graphs.cost_tracking import track_ledger_costs
from common.celery import ScopedTask


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
    """

    # Scope parameters (tenant_id, case_id, trace_id, session_salt) are accepted
    # so ScopedTask/with_scope_apply_async can attach masking context without
    # causing unexpected kwargs errors. They are currently unused because the
    # runner receives fully prepared meta/state.

    runner = get_graph_runner(graph_name)
    runner_state = dict(state or {})
    runner_meta = dict(meta or {})

    with track_ledger_costs(initial_cost_total) as tracker:
        runner_meta["ledger_logger"] = tracker.record_ledger_meta
        try:
            new_state, result = runner.run(runner_state, runner_meta)
        finally:
            runner_meta.pop("ledger_logger", None)

    return {
        "state": new_state,
        "result": result,
        "cost_summary": tracker.summary(ledger_identifier),
    }
