from typing import Callable, Dict, Union, Any

from .core import GraphRunner


class LazyGraphFactory:
    """Wrapper for a factory function that produces a graph runner."""

    def __init__(self, factory: Callable[[], Any]):
        self.factory = factory


_REGISTRY: Dict[str, Union[GraphRunner, LazyGraphFactory, Any]] = {}

__all__ = ["register", "get", "LazyGraphFactory"]


def register(name: str, runner: Union[GraphRunner, LazyGraphFactory, Any]) -> None:
    """Register a graph runner or factory under the provided name."""

    if not name:
        raise ValueError("graph name must be provided")
    _REGISTRY[name] = runner


def get(name: str) -> Any:
    """Return the graph runner registered under ``name``.

    If a LazyGraphFactory was registered, it is invoked on first access
    and the result is cached.
    """

    try:
        item = _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"graph runner '{name}' is not registered") from exc

    if isinstance(item, LazyGraphFactory):
        # It's a factory wrapper - build, cache, and return
        instance = item.factory()
        _REGISTRY[name] = instance
        return instance

    return item
