"""Task context contracts for Celery task execution."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ai_core.contracts.business import BusinessContext
from ai_core.contracts.scope import ScopeContext
from ai_core.tool_contracts.base import ToolContext, tool_context_from_meta


PriorityLevel = Literal["high", "low", "background", "bulk"]


class TaskContextMetadata(BaseModel):
    """Runtime metadata specific to Celery task execution."""

    key_alias: str | None = Field(
        default=None, description="Key alias used for upstream routing."
    )
    session_salt: str | None = Field(
        default=None,
        description="PII masking session salt for deterministic redaction.",
    )
    priority: PriorityLevel | None = Field(
        default=None, description="Task priority hint for routing."
    )
    task_id: str | None = Field(default=None, description="Celery task identifier.")
    queue: str | None = Field(default=None, description="Resolved Celery queue name.")
    retry_count: int | None = Field(
        default=None, description="Retry attempt count (0-based)."
    )

    model_config = ConfigDict(frozen=True)


class TaskContext(BaseModel):
    """Typed task context composed from scope, business, and runtime metadata."""

    scope: ScopeContext = Field(
        description="Infrastructure request context for the task."
    )
    business: BusinessContext = Field(
        description="Business identifiers for task execution."
    )
    metadata: TaskContextMetadata = Field(
        default_factory=TaskContextMetadata,
        description="Runtime metadata for Celery tasks.",
    )

    model_config = ConfigDict(frozen=True)


def _coerce_text(value: object | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        candidate = value.strip()
        return candidate or None
    try:
        return str(value).strip() or None
    except Exception:
        return None


def _coerce_priority(value: object | None) -> PriorityLevel | None:
    text = _coerce_text(value)
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"high", "low", "background", "bulk"}:
        return lowered  # type: ignore[return-value]
    return None


def _coerce_retry_count(value: object | None) -> int | None:
    if value is None:
        return None
    try:
        candidate = int(value)
    except (TypeError, ValueError):
        return None
    if candidate < 0:
        return None
    return candidate


def _extract_key_alias(
    tool_context: ToolContext, meta: Mapping[str, Any]
) -> str | None:
    key_alias = tool_context.metadata.get("key_alias")
    if key_alias:
        return _coerce_text(key_alias)
    return _coerce_text(meta.get("key_alias"))


def task_context_from_meta(meta: Mapping[str, Any]) -> TaskContext:
    """Build a TaskContext from Celery meta (scope/business + runtime metadata)."""

    tool_context = tool_context_from_meta(meta)
    metadata = TaskContextMetadata(
        key_alias=_extract_key_alias(tool_context, meta),
        session_salt=_coerce_text(meta.get("session_salt")),
        priority=_coerce_priority(meta.get("priority") or meta.get("task_priority")),
        task_id=_coerce_text(meta.get("task_id")),
        queue=_coerce_text(meta.get("queue")),
        retry_count=_coerce_retry_count(meta.get("retry_count")),
    )
    return TaskContext(
        scope=tool_context.scope,
        business=tool_context.business,
        metadata=metadata,
    )


__all__ = [
    "PriorityLevel",
    "TaskContext",
    "TaskContextMetadata",
    "task_context_from_meta",
]
