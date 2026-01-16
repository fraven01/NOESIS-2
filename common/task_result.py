"""Standard task result envelope for Celery tasks."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class TaskResult(BaseModel):
    """Standardized Celery task result format."""

    status: Literal["success", "error", "partial"]
    data: dict[str, Any]
    context_snapshot: dict[str, Any]
    task_name: str
    completed_at: datetime
    error: str | None = None

    model_config = ConfigDict(frozen=True)
