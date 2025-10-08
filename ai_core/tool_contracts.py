"""Contracts and error taxonomy for tool interactions."""
from __future__ import annotations

from pydantic import BaseModel


class ToolContext(BaseModel):
    """Context metadata required for tool invocations."""

    tenant_id: str
    case_id: str
    trace_id: str | None = None
    idempotency_key: str | None = None


class ToolError(Exception):
    """Base error for tool related issues."""


class InputError(ToolError):
    """Raised when input data validation fails."""


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
