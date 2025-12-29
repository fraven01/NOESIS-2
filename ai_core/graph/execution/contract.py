"""Protocol definition for the graph execution boundary.

This module defines the contract that business graphs and views use to execute technical graphs.
"""

from __future__ import annotations

from typing import Protocol, Any, runtime_checkable


@runtime_checkable
class GraphExecutor(Protocol):
    """Execution boundary for invoking graphs."""

    def run(self, name: str, input: dict[str, Any], meta: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """Synchronously execute the named graph.

        Args:
            name: The registered name of the graph to execute (see ai_core.graph.registry).
            input: The input state dictionary.
            meta: The unified metadata dictionary (ScopeContext, etc.).

        Returns:
            A tuple of (final_state, result_payload).
        """
        ...

    def submit(self, name: str, input: dict[str, Any], meta: dict[str, Any]) -> str:
        """Submit the graph for asynchronous execution.

        Args:
            name: The registered name of the graph to execute.
            input: The input state dictionary.
            meta: The unified metadata dictionary.

        Returns:
            A task identifier (str) that can be used to track status.
        """
        ...
