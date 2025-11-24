"""Storage interface and in-memory implementation for document blobs."""

from __future__ import annotations

import hashlib
from threading import RLock
from typing import Dict, Sequence, Tuple
from urllib.parse import urlparse

import requests
from requests import RequestException

from ai_core.infra.blob_writers import ObjectStoreBlobWriter, S3BlobWriter

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

    def __init__(self, *, http_timeout: float = 10.0) -> None:
        self._lock = RLock()
        self._store: Dict[str, bytes] = {}
        self._counter = 0
        self._http_timeout = float(http_timeout)

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
        parsed = urlparse(uri)
        scheme = parsed.scheme.lower()
        if scheme in {"http", "https"}:
            try:
                response = requests.get(uri, timeout=self._http_timeout)
            except RequestException as exc:  # pragma: no cover - network guard
                raise ValueError("storage_uri_fetch_failed") from exc
            if response.status_code >= 400:
                raise ValueError("storage_uri_http_error")
            payload = response.content
            log_extra_exit(
                uri_kind=uri_kind_from_uri(uri),
                size_bytes=len(payload),
            )
            return payload
        if uri.startswith("memory://"):
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
        raise ValueError("storage_uri_unsupported")


class ObjectStoreStorage(Storage):
    """Object-store backed storage using :class:`ObjectStoreBlobWriter`."""

    def __init__(self, *, prefix: Sequence[str] | None = None) -> None:
        self._writer = ObjectStoreBlobWriter(
            prefix=tuple(prefix or ("documents", "uploads"))
        )

    @log_call("storage.put")
    def put(self, data: bytes) -> Tuple[str, str, int]:
        return self._writer.put(data)

    @log_call("storage.get")
    def get(self, uri: str) -> bytes:
        payload = self._writer.get(uri)
        log_extra_exit(uri_kind=uri_kind_from_uri(uri), size_bytes=len(payload))
        return payload


class S3Storage(Storage):
    """S3/MinIO backed storage adapter using :class:`S3BlobWriter`."""

    def __init__(
        self,
        *,
        bucket: str,
        prefix: Sequence[str] | None = None,
        endpoint_url: str | None = None,
        region_name: str | None = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
    ) -> None:
        self._writer = S3BlobWriter(
            bucket=bucket,
            prefix=prefix,
            endpoint_url=endpoint_url,
            region_name=region_name,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
        )

    @log_call("storage.put")
    def put(self, data: bytes) -> Tuple[str, str, int]:  # pragma: no cover - network
        return self._writer.put(data)

    @log_call("storage.get")
    def get(self, uri: str) -> bytes:  # pragma: no cover - network
        payload = self._writer.get(uri)
        log_extra_exit(uri_kind=uri_kind_from_uri(uri), size_bytes=len(payload))
        return payload


__all__ = ["Storage", "InMemoryStorage", "ObjectStoreStorage", "S3Storage"]
