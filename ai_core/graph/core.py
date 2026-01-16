"""Core interfaces and file-based checkpoint support for graph execution."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from pydantic import ValidationError

from ai_core.infra.object_store import read_json, sanitize_identifier, write_json
from ai_core.graph.state import PersistedGraphState


if TYPE_CHECKING:  # pragma: no cover - typing-only imports
    from ai_core.tool_contracts import ToolContext

logger = logging.getLogger(__name__)


class GraphRunner(Protocol):
    """Protocol describing the callable surface of a graph runner."""

    def run(self, state: dict, meta: dict) -> tuple[dict, dict]:
        """Execute the graph and return the updated state and result payload."""


class Checkpointer(Protocol):
    """Protocol for persisting and retrieving graph execution state."""

    def load(self, ctx: "GraphContext") -> dict:
        """Return the stored state for the supplied context."""

    def save(self, ctx: "GraphContext", state: dict) -> None:
        """Persist the supplied state for the given context."""


@dataclass(frozen=True)
class GraphContext:
    """Immutable descriptor for a graph execution."""

    tool_context: "ToolContext"
    graph_name: str
    graph_version: str = "v0"

    def __post_init__(self) -> None:
        if not self.workflow_id or not self.run_id:
            raise ValueError(
                "workflow_id and run_id are required for workflow execution checkpointing"
            )

    @property
    def tenant_id(self) -> str:
        return self.tool_context.scope.tenant_id

    @property
    def case_id(self) -> str | None:
        return self.tool_context.business.case_id

    @property
    def trace_id(self) -> str:
        return self.tool_context.scope.trace_id

    @property
    def workflow_id(self) -> str | None:
        return self.tool_context.business.workflow_id

    @property
    def thread_id(self) -> str | None:
        return self.tool_context.business.thread_id

    @property
    def run_id(self) -> str | None:
        return (
            self.tool_context.scope.run_id or self.tool_context.scope.ingestion_run_id
        )

    @property
    def plan_key(self) -> str | None:
        value = self.tool_context.metadata.get("plan_key")
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return str(value)


class FileCheckpointer(Checkpointer):
    """Checkpoint implementation backed by the local object store."""

    def _path(self, ctx: GraphContext) -> str:
        safe_tenant = sanitize_identifier(ctx.tenant_id)
        plan_key = ctx.plan_key
        if plan_key:
            safe_plan_key = sanitize_identifier(plan_key)
            return f"{safe_tenant}/workflow-executions/{safe_plan_key}/state.json"

        workflow_id = ctx.workflow_id
        run_id = ctx.run_id
        safe_workflow = sanitize_identifier(workflow_id)
        safe_run = sanitize_identifier(run_id)
        return (
            f"{safe_tenant}/workflow-executions/{safe_workflow}/{safe_run}/state.json"
        )

    def load(self, ctx: GraphContext) -> dict:
        """Load a previously stored state or return an empty mapping."""

        try:
            path = self._path(ctx)
            data = read_json(path)
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError:
            logger.warning(
                "graph.checkpoint.corrupted",
                extra={
                    "graph": ctx.graph_name,
                    "tenant_id": ctx.tenant_id,
                    "case_id": ctx.case_id,
                    "workflow_id": ctx.workflow_id,
                    "run_id": ctx.run_id,
                    "plan_key": ctx.plan_key,
                },
            )
            write_json(path, {})
            return {}
        try:
            persisted = PersistedGraphState.model_validate(data)
        except ValidationError:
            logger.warning(
                "graph.checkpoint.invalid",
                extra={
                    "graph": ctx.graph_name,
                    "tenant_id": ctx.tenant_id,
                    "case_id": ctx.case_id,
                    "workflow_id": ctx.workflow_id,
                    "run_id": ctx.run_id,
                    "plan_key": ctx.plan_key,
                },
            )
            write_json(path, {})
            return {}
        return persisted.state

    def save(self, ctx: GraphContext, state: dict) -> None:
        """Persist the provided state for later retrieval."""

        if not isinstance(state, dict):  # pragma: no cover - defensive branch
            raise TypeError("state must be a dictionary")
        persisted = PersistedGraphState(
            tool_context=ctx.tool_context,
            state=state,
            graph_name=ctx.graph_name,
            graph_version=ctx.graph_version,
            checkpoint_at=datetime.now(timezone.utc),
        )
        write_json(self._path(ctx), persisted.model_dump(mode="json"))


class ThreadAwareCheckpointer(FileCheckpointer):
    """Checkpoint implementation that prefers chat thread IDs when available."""

    def _path(self, ctx: GraphContext) -> str:
        safe_tenant = sanitize_identifier(ctx.tenant_id)
        thread_id = ctx.thread_id
        if thread_id:
            safe_thread = sanitize_identifier(thread_id)
            return f"{safe_tenant}/threads/{safe_thread}/state.json"
        return super()._path(ctx)
