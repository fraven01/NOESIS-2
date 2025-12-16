from __future__ import annotations

"""Public tool-contract exports.

Canonical tool envelope models and ``ToolContext`` live in ``ai_core.tool_contracts.base``.
This package module keeps the historical exception types used by nodes/graphs while
re-exporting the canonical context model.
"""

from .base import ToolContext


class ToolError(Exception):
    """Base error for tool related issues."""


class InputError(ToolError):
    """Raised when input data validation fails."""

    def __init__(self, message: str, *, field: str | None = None) -> None:
        super().__init__(message)
        self.field = field


class NotFoundError(ToolError):
    """Raised when a requested resource cannot be located."""


class RateLimitedError(ToolError):
    """Raised when rate limits are exceeded."""


class TimeoutError(ToolError):
    """Raised when an external call exceeds its timeout."""


class UpstreamServiceError(ToolError):
    """Raised when a dependent service returns an error."""


class InternalToolError(ToolError):
    """Raised when an unexpected error occurs within the tool itself."""


class ContextError(ToolError):
    """Raised when the surrounding execution context is invalid."""

    def __init__(self, message: str, *, field: str | None = None) -> None:
        super().__init__(message)
        self.field = field


class InconsistentMetadataError(ToolError):
    """Raised when tool inputs contain inconsistent metadata."""


__all__ = [
    "ToolContext",
    "ToolError",
    "InputError",
    "ContextError",
    "InconsistentMetadataError",
    "NotFoundError",
    "RateLimitedError",
    "TimeoutError",
    "UpstreamServiceError",
    "InternalToolError",
]

