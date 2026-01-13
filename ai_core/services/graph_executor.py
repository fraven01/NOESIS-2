"""Graph execution service logic."""

from __future__ import annotations

from rest_framework.request import Request
from rest_framework.response import Response

from ai_core.commands.graph_execution import GraphExecutionCommand
from ai_core.graph.core import GraphRunner
from ai_core.graph.execution import (
    GraphExecutor,
    LocalGraphExecutor,
    RunnerGraphExecutor,
)
from ai_core.infra.observability import observe_span


@observe_span(name="graph.execute")
def execute_graph(
    request: Request,
    graph_runner: GraphRunner | None = None,
    *,
    executor: GraphExecutor | None = None,
) -> Response:
    """Delegate graph orchestration to the command layer."""
    command = GraphExecutionCommand()
    graph_executor = executor
    if graph_executor is None:
        if graph_runner is not None:
            graph_executor = RunnerGraphExecutor(graph_runner)
        else:
            graph_executor = LocalGraphExecutor()
    return command.execute(request, graph_executor=graph_executor)
