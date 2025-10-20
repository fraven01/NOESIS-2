"""Storage interface and in-memory implementation for document blobs."""

from __future__ import annotations

import hashlib
from threading import RLock
from typing import Dict, Tuple

from .logging_utils import log_call, log_extra_exit, uri_kind_from_uri


class Storage:
    """Abstract storage interface for binary payloads."""

    def put(self, data: bytes) -> Tuple[str, str, int]:
        """Persist ``data`` and return ``(uri, sha256, size)``."""

        raise NotImplementedError

    def get(self, uri: str) -> bytes:
        """Return the stored payload for ``uri``."""

        raise NotImplementedError


class InMemoryStorage(Storage):
    """Thread-safe in-memory storage adapter using ``memory://`` URIs."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._store: Dict[str, bytes] = {}
        self._counter = 0

    @log_call("storage.put")
    def put(self, data: bytes) -> Tuple[str, str, int]:
        checksum = hashlib.sha256(data).hexdigest()
        size = len(data)
        with self._lock:
            self._counter += 1
            uri = f"memory://{self._counter:020d}"
            self._store[uri] = data
        log_extra_exit(
            uri_kind=uri_kind_from_uri(uri),
            sha256_prefix=checksum[:8],
            size_bytes=size,
        )
        return uri, checksum, size

    @log_call("storage.get")
    def get(self, uri: str) -> bytes:
        if not uri.startswith("memory://"):
            raise ValueError("storage_uri_unsupported")
        with self._lock:
            try:
                payload = self._store[uri]
            except KeyError as exc:  # pragma: no cover - defensive guard
                raise KeyError("storage_uri_missing") from exc
        log_extra_exit(
            uri_kind=uri_kind_from_uri(uri),
            size_bytes=len(payload),
        )
        return payload


__all__ = ["Storage", "InMemoryStorage"]
