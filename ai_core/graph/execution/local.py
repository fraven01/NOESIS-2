"""Local implementation of the GraphExecutor protocol.

This executor runs graphs synchronously in the current process using the graph registry.
"""

from __future__ import annotations

import logging
from typing import Any

from ai_core.graph import registry
from ai_core.graph.execution.contract import GraphExecutor

logger = logging.getLogger(__name__)


class LocalGraphExecutor(GraphExecutor):
    """Executes graphs in the same process via the registry."""

    def run(
        self, name: str, input: dict[str, Any], meta: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Synchronously execute the graph from the registry.

        Args:
            name: Name of the graph in ai_core.graph.registry.
            input: The input state dictionary.
            meta: The unified metadata dictionary.

        Returns:
            Tuple of (final_state, result).

        Raises:
            KeyError: If graph name is not found in registry.
        """
        runner = registry.get(name)
        logger.info(
            "graph.execution.local.run", extra={"graph": name, "executor": "local"}
        )
        return runner.run(input, meta)

    def submit(self, name: str, input: dict[str, Any], meta: dict[str, Any]) -> str:
        """Async submission is not supported by LocalGraphExecutor (sync-only).

        In the future, this could use a ThreadPoolExecutor or similar if needed,
        but for 'Local' execution in a request/response cycle, sync is usually intended.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "LocalGraphExecutor does not support async submission. Use run() or a Celery-based executor."
        )
