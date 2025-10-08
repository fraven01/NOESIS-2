from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, ConfigDict, Field


class ToolContext(BaseModel):
    """Context metadata required for tool invocations.

    Fields follow the Layer 2 contracts norm (see AGENTS.md):
    - tenant_id (required)
    - case_id (required)
    - trace_id/idempotency_key (optional)
    """

    tenant_id: str
    case_id: str
    trace_id: str | None = None
    idempotency_key: str | None = None
    tenant_schema: str | None = None
    visibility_override_allowed: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow", frozen=True)


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


__all__ = [
    "ToolContext",
    "ToolError",
    "InputError",
    "ContextError",
    "NotFoundError",
    "RateLimitedError",
    "TimeoutError",
    "UpstreamServiceError",
    "InternalToolError",
]
