"""Adapter utilities bridging legacy graph modules to runner protocols."""

from __future__ import annotations

from types import ModuleType
from typing import Callable

from .core import GraphRunner


class _ModuleRunner:
    """Simple runner wrapper delegating to a module-level ``run`` function."""

    def __init__(self, func: Callable[[dict, dict], tuple[dict, dict]]):
        self._func = func

    def run(self, state: dict, meta: dict) -> tuple[dict, dict]:
        """Invoke the wrapped module with the supplied arguments."""

        return self._func(state, meta)


def module_runner(module: ModuleType) -> GraphRunner:
    """Return a :class:`GraphRunner` wrapper for a legacy module."""

    run_attr = getattr(module, "run", None)
    if not callable(run_attr):
        module_name = getattr(module, "__name__", repr(module))
        raise AttributeError(
            f"module '{module_name}' does not expose a callable 'run' attribute"
        )
    return _ModuleRunner(run_attr)
