"""Shared object store protocol and factory helpers."""

from __future__ import annotations

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


def set_default_object_store_factory(factory: Callable[[], ObjectStore]) -> None:
    """Register the callable producing the default :class:`ObjectStore`."""

    global _ObjectStoreFactory
    _ObjectStoreFactory = factory


def get_default_object_store() -> ObjectStore:
    """Return the configured default :class:`ObjectStore`."""

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
