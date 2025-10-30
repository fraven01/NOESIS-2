"""Default filesystem-backed object store implementation."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Callable

from common.object_store import ObjectStore, set_default_object_store_factory

BASE_PATH = Path(".ai_core_store")


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
        if ".." in value or "/" in value or os.sep in value:
            raise ValueError("unsafe_identifier")

        sanitized = re.sub(r"[^A-Za-z0-9._-]", "_", value)[:128]
        if not sanitized:
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
        target = self._full_path(path)
        target.write_bytes(data)
        return target

    def write_bytes(self, path: str, data: bytes) -> None:
        target = self.BASE_PATH / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    def read_bytes(self, path: str) -> bytes:
        target = self.BASE_PATH / path
        return target.read_bytes()

    def read_json(self, path: str) -> Any:
        target = self.BASE_PATH / path
        with target.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def write_json(self, path: str, obj: Any) -> Path:
        target = self._full_path(path)
        with target.open("w", encoding="utf-8") as fh:
            json.dump(obj, fh)
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
