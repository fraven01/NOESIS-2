"""In-memory registry for named graph runners."""

from __future__ import annotations

from typing import Dict

from .core import GraphRunner

_REGISTRY: Dict[str, GraphRunner] = {}

__all__ = ["register", "get"]


def register(name: str, runner: GraphRunner) -> None:
    """Register a graph runner under the provided name."""

    if not name:
        raise ValueError("graph name must be provided")
    _REGISTRY[name] = runner


def get(name: str) -> GraphRunner:
    """Return the graph runner registered under ``name``."""

    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"graph runner '{name}' is not registered") from exc
