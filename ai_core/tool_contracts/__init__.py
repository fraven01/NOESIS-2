from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, ConfigDict, Field


class ToolError(Exception):
    """Base class for errors raised by tool implementations."""


class InputError(ToolError):
    """Raised when the provided tool input violates the contract."""

    def __init__(self, message: str, *, field: str | None = None) -> None:
        super().__init__(message)
        self.field = field


class ContextError(ToolError):
    """Raised when the surrounding execution context is invalid."""

    def __init__(self, message: str, *, field: str | None = None) -> None:
        super().__init__(message)
        self.field = field


class ToolContext(BaseModel):
    """Shared execution context provided to tool implementations."""

    tenant_id: str
    tenant_schema: str | None = None
    case_id: str | None = None
    trace_id: str | None = None
    visibility_override_allowed: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow", frozen=True)


__all__ = ["ContextError", "InputError", "ToolContext", "ToolError"]
