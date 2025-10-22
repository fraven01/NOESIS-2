"""Pydantic models defining the shared tool contract envelopes."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any, ClassVar, Generic, Literal, Optional, TypeVar, Union

try:  # pragma: no cover - typing backport
    from typing import TypeAliasType
except ImportError:  # pragma: no cover - fallback for <3.12
    from typing_extensions import TypeAliasType
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ai_core.tools.errors import ToolErrorType

NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(gt=0)]

IT = TypeVar("IT", bound=BaseModel)
OT = TypeVar("OT", bound=BaseModel)


class ToolContext(BaseModel):
    """Runtime metadata accompanying every tool invocation."""

    model_config = ConfigDict(frozen=True)
    json_schema_extra: ClassVar[dict[str, Any]] = {
        "examples": [
            {
                "tenant_id": "5aa31da6-9278-4da0-9f1a-61b8d3edc5cc",
                "request_id": "8607a8d9-0f3f-43df-bf86-37e845e1574c",
                "trace_id": "trace-123",
                "invocation_id": "0f4e6712-6d04-4514-b6cb-943b0667d45c",
                "now_iso": "2024-05-03T12:34:56.123456+00:00",
                "locale": "de-DE",
                "timeouts_ms": 120000,
                "budget_tokens": 4096,
            }
        ]
    }

    tenant_id: UUID
    request_id: UUID
    trace_id: str
    invocation_id: UUID
    now_iso: datetime
    idempotency_key: Optional[str] = None
    timeouts_ms: Optional[PositiveInt] = None
    budget_tokens: Optional[int] = None
    locale: Optional[str] = None
    safety_mode: Optional[str] = None
    auth: Optional[dict[str, Any]] = None

    @field_validator("now_iso")
    @classmethod
    def ensure_timezone_aware(cls, value: datetime) -> datetime:
        """Ensure the timestamp is timezone-aware in UTC."""

        if not isinstance(value, datetime):  # pragma: no cover - defensive guard
            raise TypeError("now_iso must be a datetime instance")
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError("now_iso must include timezone information")
        return value.astimezone(timezone.utc)


class ToolErrorMeta(BaseModel):
    """Meta information for tool errors."""

    model_config = ConfigDict(frozen=True)
    json_schema_extra: ClassVar[dict[str, Any]] = {
        "examples": [
            {
                "took_ms": 250,
            }
        ]
    }

    took_ms: NonNegativeInt


class ToolErrorDetail(BaseModel):
    """Structured error payload returned by tools."""

    model_config = ConfigDict(frozen=True)
    json_schema_extra: ClassVar[dict[str, Any]] = {
        "examples": [
            {
                "type": "RATE_LIMIT",
                "message": "Quota exceeded",
                "code": "rate_limit",
                "details": {"limit": 100, "interval": "1m"},
                "retry_after_ms": 1500,
                "upstream_status": 429,
                "endpoint": "POST /tool",
            }
        ]
    }

    type: ToolErrorType
    message: str
    code: Optional[str] = None
    cause: Optional[str] = None
    details: Optional[dict[str, Any]] = None
    retry_after_ms: Optional[int] = None
    upstream_status: Optional[int] = None
    endpoint: Optional[str] = None
    attempt: Optional[int] = None


class ToolError(BaseModel, Generic[IT]):
    """Envelope describing a failed tool invocation."""

    model_config = ConfigDict(frozen=True)
    json_schema_extra: ClassVar[dict[str, Any]] = {
        "examples": [
            {
                "status": "error",
                "input": {"query": "hello"},
                "error": {
                    "type": "VALIDATION",
                    "message": "Missing required field",
                    "code": "missing_field",
                },
                "meta": {"took_ms": 10},
            }
        ]
    }

    status: Literal["error"] = "error"
    input: IT
    error: ToolErrorDetail
    meta: ToolErrorMeta


class ToolResultMeta(BaseModel):
    """Meta data attached to successful tool responses."""

    model_config = ConfigDict(frozen=True)
    json_schema_extra: ClassVar[dict[str, Any]] = {
        "examples": [
            {
                "took_ms": 180,
                "source_counts": {"documents": 3},
                "routing": {"embedding_profile": "default"},
                "cache_hit": False,
                "token_usage": {"prompt": 1200, "completion": 800},
            }
        ]
    }

    took_ms: NonNegativeInt
    source_counts: Optional[dict[str, int]] = None
    routing: Optional[dict[str, Any]] = None
    cache_hit: Optional[bool] = None
    token_usage: Optional[dict[str, int]] = None


class ToolResult(BaseModel, Generic[IT, OT]):
    """Envelope describing a successful tool invocation."""

    model_config = ConfigDict(frozen=True)
    json_schema_extra: ClassVar[dict[str, Any]] = {
        "examples": [
            {
                "status": "ok",
                "input": {"query": "hello"},
                "data": {"answer": "Hi there!"},
                "meta": {
                    "took_ms": 42,
                    "token_usage": {"prompt": 12, "completion": 20},
                },
            }
        ]
    }

    status: Literal["ok"] = "ok"
    input: IT
    data: OT
    meta: ToolResultMeta


ToolOutput = TypeAliasType(
    "ToolOutput",
    Annotated[
        Union[ToolResult[IT, OT], ToolError[IT]],
        Field(discriminator="status"),
    ],
    type_params=(IT, OT),
)

__all__ = [
    "NonNegativeInt",
    "PositiveInt",
    "ToolContext",
    "ToolErrorMeta",
    "ToolErrorDetail",
    "ToolError",
    "ToolResultMeta",
    "ToolResult",
    "ToolOutput",
]
