"""Shared object store protocol and factory helpers."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable


@runtime_checkable
class ObjectStore(Protocol):
    """Protocol describing the minimal contract for object store adapters."""

    @property
    def BASE_PATH(self) -> Path:
        """Root directory used by the object store implementation."""

    def read_bytes(self, path: str) -> bytes:
        """Load raw bytes from the object store."""

    def write_bytes(self, path: str, data: bytes) -> None:
        """Persist raw bytes into the object store."""

    def put_bytes(self, path: str, data: bytes) -> Path:
        """Persist bytes and return the absolute filesystem path."""

    def read_json(self, path: str) -> Any:
        """Load and deserialize JSON from the object store."""

    def write_json(self, path: str, obj: Any) -> Path:
        """Serialize and persist JSON into the object store."""

    def sanitize_identifier(self, value: str) -> str:
        """Return a filesystem safe identifier for tenant and case names."""

    def safe_filename(self, value: str) -> str:
        """Return a sanitized filename for derived assets."""


_ObjectStoreFactory: Callable[[], ObjectStore] | None = None
_BOOTSTRAP_MODULES: tuple[str, ...] = ("common.object_store_defaults",)


def set_default_object_store_factory(factory: Callable[[], ObjectStore]) -> None:
    """Register the callable producing the default :class:`ObjectStore`."""

    global _ObjectStoreFactory
    _ObjectStoreFactory = factory


def _bootstrap_default_factory() -> None:
    global _ObjectStoreFactory

    if _ObjectStoreFactory is not None:
        return

    for module_name in _BOOTSTRAP_MODULES:
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError:  # pragma: no cover - defensive guard
            continue

        if _ObjectStoreFactory is not None:
            break


def get_default_object_store() -> ObjectStore:
    """Return the configured default :class:`ObjectStore`."""

    if _ObjectStoreFactory is None:
        _bootstrap_default_factory()

    if _ObjectStoreFactory is None:
        raise RuntimeError("object_store_factory_not_configured")

    store = _ObjectStoreFactory()
    if not isinstance(store, ObjectStore):  # pragma: no cover - defensive
        raise TypeError("invalid_object_store_factory")
    return store


__all__ = [
    "ObjectStore",
    "get_default_object_store",
    "set_default_object_store_factory",
]


class FilesystemObjectStore:
    """Lazy proxy returning the default filesystem-backed object store."""

    def __new__(cls, *args: Any, **kwargs: Any) -> ObjectStore:  # noqa: D401
        from common.object_store_defaults import FilesystemObjectStore as _Impl

        return _Impl(*args, **kwargs)


__all__.append("FilesystemObjectStore")
