"""GraphExecutor adapter for direct GraphRunner instances."""

from __future__ import annotations

from typing import Any

from ai_core.graph.core import GraphRunner
from ai_core.graph.execution.contract import GraphExecutor


class RunnerGraphExecutor(GraphExecutor):
    """Wrap a GraphRunner to satisfy the GraphExecutor interface."""

    def __init__(self, runner: GraphRunner) -> None:
        self._runner = runner

    def run(
        self, name: str, input: dict[str, Any], meta: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return self._runner.run(input, meta)

    def submit(self, name: str, input: dict[str, Any], meta: dict[str, Any]) -> str:
        raise NotImplementedError(
            "RunnerGraphExecutor does not support async submission."
        )
