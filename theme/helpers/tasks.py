from typing import Any, Mapping

from celery import exceptions as celery_exceptions

from ai_core.tasks.graph_tasks import run_business_graph
from ai_core.tool_contracts import ToolContext
from common.celery import with_scope_apply_async


def submit_business_graph(
    *,
    graph_name: str,
    tool_context: ToolContext,
    state: Mapping[str, Any],
    timeout_s: float | None = None,
    priority: str | None = None,
) -> tuple[dict[str, Any], bool]:
    """
    Submit a business graph for execution via the generic worker (M-1).

    Args:
        graph_name: Name of the graph to execute (must be registered).
        tool_context: Fully resolved ToolContext.
        state: Input state for the graph.
        timeout_s: If set, block and wait for result up to this many seconds.
        priority: 'high' or 'low' (default high).

    Returns:
        tuple (payload, completed).
        If completed=True, payload contains the graph result.
        If completed=False (timeout), payload contains {"task_id": ...}.
    """

    # 1. Prepare Meta
    # run_business_graph expects scope/business/tool contexts in meta
    # for strict separation and tool reconstruction.
    scope = tool_context.scope
    business = tool_context.business

    meta = {
        "scope_context": scope.model_dump(mode="json", exclude_none=True),
        "business_context": business.model_dump(mode="json", exclude_none=True),
        "tool_context": tool_context.model_dump(mode="json", exclude_none=True),
        "graph_name": graph_name,
    }

    # 2. Resolve Queue
    queue = "agents-high"
    if priority == "low":
        queue = "agents-low"

    # 3. Submit Task
    # We construct signature manually or use .s()?
    # run_business_graph is a SharedTask.
    signature = run_business_graph.s(
        graph_name=graph_name, state=dict(state), meta=meta
    ).set(queue=queue)

    # Use helper to inject scope headers for tracing (if supported)
    # with_scope_apply_async wraps apply_async
    # We pass the scope dict (infrastructure)
    async_result = with_scope_apply_async(signature, scope.model_dump(mode="json"))

    # 4. Handle Result Interactively?
    timeout = timeout_s
    if timeout is None:
        # Default behavior: Asynchronous return
        return {"task_id": async_result.id}, False

    # If timeout provided, wait
    try:
        task_result = async_result.get(timeout=timeout, propagate=True)
        # task_result from run_business_graph is {"status":..., "data":...}
        return dict(task_result), True
    except celery_exceptions.TimeoutError:
        return {"task_id": async_result.id}, False
