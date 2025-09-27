"""Core interfaces and file-based checkpoint support for graph execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ai_core.infra.object_store import read_json, sanitize_identifier, write_json


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

    tenant_id: str
    case_id: str
    trace_id: str
    graph_name: str
    graph_version: str = "v0"


class FileCheckpointer(Checkpointer):
    """Checkpoint implementation backed by the local object store."""

    def _path(self, ctx: GraphContext) -> str:
        safe_tenant = sanitize_identifier(ctx.tenant_id)
        safe_case = sanitize_identifier(ctx.case_id)
        return f"{safe_tenant}/{safe_case}/state.json"

    def load(self, ctx: GraphContext) -> dict:
        """Load a previously stored state or return an empty mapping."""

        try:
            data = read_json(self._path(ctx))
        except FileNotFoundError:
            return {}
        if not isinstance(data, dict):  # pragma: no cover - defensive branch
            raise TypeError("checkpoint data must be a dictionary")
        return data

    def save(self, ctx: GraphContext, state: dict) -> None:
        """Persist the provided state for later retrieval."""

        if not isinstance(state, dict):  # pragma: no cover - defensive branch
            raise TypeError("state must be a dictionary")
        write_json(self._path(ctx), state)
