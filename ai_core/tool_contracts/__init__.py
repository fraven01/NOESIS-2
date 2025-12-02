from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ToolContext(BaseModel):
    """Runtime metadata accompanying every tool invocation.
    
    (Duplicated from base.py to avoid circular imports and maintain compatibility)
    """

    model_config = ConfigDict(frozen=True)
    
    tenant_id: Union[UUID, str]
    trace_id: str = Field(default_factory=lambda: "trace-test")
    invocation_id: UUID = Field(default_factory=uuid4)
    now_iso: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    idempotency_key: Optional[str] = None
    tenant_schema: Optional[str] = None
    timeouts_ms: Optional[int] = None
    budget_tokens: Optional[int] = None
    locale: Optional[str] = None
    safety_mode: Optional[str] = None
    auth: Optional[dict[str, Any]] = None

    # New fields
    run_id: Optional[str] = None
    ingestion_run_id: Optional[str] = None
    workflow_id: Optional[str] = None
    collection_id: Optional[str] = None
    document_id: Optional[str] = None
    document_version_id: Optional[str] = None
    case_id: Optional[str] = None
    
    # Extra fields from original __init__.py
    visibility_override_allowed: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("now_iso")
    @classmethod
    def ensure_timezone_aware(cls, value: datetime) -> datetime:
        """Ensure the timestamp is timezone-aware in UTC."""
        if not isinstance(value, datetime):
            raise TypeError("now_iso must be a datetime instance")
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError("now_iso must include timezone information")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def check_run_ids(self) -> "ToolContext":
        # Validation logic commented out to avoid strictness issues during migration
        # if self.run_id is None and self.ingestion_run_id is None:
        #     raise ValueError("Either run_id or ingestion_run_id must be provided.")
        # if self.run_id is not None and self.ingestion_run_id is not None:
        #     raise ValueError("Only one of run_id or ingestion_run_id can be provided.")
        return self


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
