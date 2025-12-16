"""
Standard contracts for Technical Graphs (Layer 3).

This module defines the unified input contracts and protocols for all technical
capabilities graphs. It serves as the cross-graph interface definition required
for externalization and standard worker invocation.
"""

from __future__ import annotations

from typing import Any, NotRequired, Protocol, TypedDict


class StandardGraphState(TypedDict):
    """
    Base state dictionary for all technical tier graphs.

    All technical graphs MUST accept a state dictionary adhering to this shape
    (extending it with specific payload fields).
    """

    context: dict[str, Any]  # Serialized ToolContext
    error: NotRequired[str]  # Standard error propagation


class TechnicalGraphRunnable(Protocol):
    """
    Protocol for a runnable technical graph (Layer 3).

    This protocol unifies the invocation of:
    1. Legacy graphs (manual state management, run methods)
    2. LangGraph workflows (StateGraph compiled runnables)

    The `invoke` method is the standard entry point for all future graph interactions.
    """

    def invoke(
        self, input: dict[str, Any], config: Any | None = None
    ) -> dict[str, Any]:
        """Execute the graph with the given input state."""
        ...


__all__ = ["StandardGraphState", "TechnicalGraphRunnable"]
