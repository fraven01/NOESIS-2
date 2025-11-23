"""Protocols for asset blob IO."""

from __future__ import annotations

from typing import Protocol, Tuple, runtime_checkable

__all__ = ["BlobWriter", "BlobReader", "BlobIO"]


@runtime_checkable
class BlobWriter(Protocol):
    """Write binary payloads and return storage metadata."""

    def put(self, payload: bytes) -> Tuple[str, str, int]:
        """Persist ``payload`` and return ``(uri, checksum, size)``."""


@runtime_checkable
class BlobReader(Protocol):
    """Retrieve binary payloads by URI."""

    def get(self, uri: str) -> bytes:
        """Return stored bytes for ``uri``."""


class BlobIO(BlobWriter, BlobReader, Protocol):
    """Convenience protocol covering both blob read and write operations."""

    pass
