"""Graph execution service logic."""

from __future__ import annotations

from rest_framework.request import Request
from rest_framework.response import Response

from ai_core.commands.graph_execution import GraphExecutionCommand
from ai_core.graph.core import GraphRunner
from ai_core.infra.observability import observe_span


@observe_span(name="graph.execute")
def execute_graph(request: Request, graph_runner: GraphRunner) -> Response:
    """Delegate graph orchestration to the command layer."""
    command = GraphExecutionCommand()
    return command.execute(request, graph_runner_factory=lambda: graph_runner)
