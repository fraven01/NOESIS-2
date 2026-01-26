"""Default filesystem-backed object store implementation."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Callable

from common.object_store import ObjectStore, set_default_object_store_factory

BASE_PATH = Path(".ai_core_store")

LOGGER = logging.getLogger(__name__)


__all__ = [
    "BASE_PATH",
    "FilesystemObjectStore",
    "put_bytes",
    "read_bytes",
    "read_json",
    "safe_filename",
    "sanitize_identifier",
    "write_bytes",
    "write_json",
]


_FILENAME_PATTERN = re.compile(r"[^A-Za-z0-9._-]")


class FilesystemObjectStore(ObjectStore):
    """Filesystem-backed object store implementation."""

    def __init__(self, base_path_supplier: Callable[[], Path] | None = None) -> None:
        self._base_path_supplier = base_path_supplier or (lambda: BASE_PATH)

    @property
    def BASE_PATH(self) -> Path:  # noqa: N802 - match public contract
        return self._base_path_supplier()

    def _full_path(self, relative: str) -> Path:
        path = self.BASE_PATH / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def sanitize_identifier(self, value: str) -> str:
        value = str(value)
        if ".." in value or "/" in value or os.sep in value:
            raise ValueError("unsafe_identifier")

        sanitized = re.sub(r"[^A-Za-z0-9._-]", "_", value)[:128]
        if not sanitized:
            LOGGER.warning("object_store.unsafe_identifier", extra={"value": value})
            raise ValueError("unsafe_identifier")
        return sanitized

    def safe_filename(self, value: str) -> str:
        if not value:
            raise ValueError("invalid_filename")

        candidate = os.path.basename(str(value)).strip()
        if not candidate:
            raise ValueError("invalid_filename")

        sanitized = _FILENAME_PATTERN.sub("_", candidate)[:128]
        if not sanitized or not re.search(r"[A-Za-z0-9]", sanitized):
            raise ValueError("invalid_filename")

        return sanitized

    def put_bytes(self, path: str, data: bytes) -> Path:
        start = time.perf_counter()
        target = self._full_path(path)
        target.write_bytes(data)
        duration_ms = (time.perf_counter() - start) * 1000.0
        LOGGER.info(
            "object_store.put_bytes",
            extra={
                "path": path,
                "size_bytes": len(data),
                "duration_ms": duration_ms,
            },
        )
        return target

    def write_bytes(self, path: str, data: bytes) -> None:
        start = time.perf_counter()
        target = self.BASE_PATH / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        duration_ms = (time.perf_counter() - start) * 1000.0
        LOGGER.info(
            "object_store.write_bytes",
            extra={
                "path": path,
                "size_bytes": len(data),
                "duration_ms": duration_ms,
            },
        )

    def read_bytes(self, path: str) -> bytes:
        start = time.perf_counter()
        target = self.BASE_PATH / path
        data = target.read_bytes()
        duration_ms = (time.perf_counter() - start) * 1000.0
        LOGGER.info(
            "object_store.read_bytes",
            extra={
                "path": path,
                "size_bytes": len(data),
                "duration_ms": duration_ms,
            },
        )
        return data

    def read_json(self, path: str) -> Any:
        start = time.perf_counter()
        target = self.BASE_PATH / path
        size_bytes = None
        try:
            size_bytes = target.stat().st_size
        except OSError:
            size_bytes = None
        with target.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        duration_ms = (time.perf_counter() - start) * 1000.0
        LOGGER.info(
            "object_store.read_json",
            extra={
                "path": path,
                "size_bytes": size_bytes,
                "duration_ms": duration_ms,
            },
        )
        return payload

    def write_json(self, path: str, obj: Any) -> Path:
        start = time.perf_counter()
        target = self._full_path(path)
        with target.open("w", encoding="utf-8") as fh:
            json.dump(obj, fh)
        duration_ms = (time.perf_counter() - start) * 1000.0
        size_bytes = None
        try:
            size_bytes = target.stat().st_size
        except OSError:
            size_bytes = None
        LOGGER.info(
            "object_store.write_json",
            extra={
                "path": path,
                "size_bytes": size_bytes,
                "duration_ms": duration_ms,
            },
        )
        return target


_DEFAULT_OBJECT_STORE = FilesystemObjectStore()


def _default_factory() -> ObjectStore:
    return _DEFAULT_OBJECT_STORE


set_default_object_store_factory(_default_factory)


def sanitize_identifier(value: str) -> str:
    return _DEFAULT_OBJECT_STORE.sanitize_identifier(value)


def safe_filename(value: str) -> str:
    return _DEFAULT_OBJECT_STORE.safe_filename(value)


def put_bytes(path: str, data: bytes) -> Path:
    return _DEFAULT_OBJECT_STORE.put_bytes(path, data)


def write_bytes(path: str, data: bytes) -> None:
    _DEFAULT_OBJECT_STORE.write_bytes(path, data)


def read_bytes(path: str) -> bytes:
    return _DEFAULT_OBJECT_STORE.read_bytes(path)


def read_json(path: str) -> Any:
    return _DEFAULT_OBJECT_STORE.read_json(path)


def write_json(path: str, obj: Any) -> Path:
    return _DEFAULT_OBJECT_STORE.write_json(path, obj)
