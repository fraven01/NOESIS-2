"""Persisted graph state envelope."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

from ai_core.tool_contracts import ToolContext


class PersistedGraphState(BaseModel):
    """Standardized format for persisted graph state."""

    tool_context: ToolContext
    state: dict[str, Any]
    graph_name: str
    graph_version: str = "v0"
    checkpoint_at: datetime

    model_config = ConfigDict(frozen=True)
