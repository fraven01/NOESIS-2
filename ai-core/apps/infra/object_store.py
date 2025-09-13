"""Filesystem-backed object store helper."""

from __future__ import annotations

import os
from pathlib import Path


STORE_ROOT = Path(os.getenv("OBJSTORE_ROOT", ".objstore"))


def object_path(tenant: str, case: str, kind: str, name: str) -> str:
    return f"{tenant}/{case}/{kind}/{name}"


def put(path: str, data: bytes) -> str:
    """Write ``data`` to ``path`` and return the file URL."""
    full = STORE_ROOT / path
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_bytes(data)
    return str(full.resolve())


def signed_url(path: str) -> str:
    """Return a dummy signed URL for ``path``."""
    full = STORE_ROOT / path
    return str(full.resolve())


def ready() -> bool:
    """Return True if the filesystem store is writable."""
    try:
        STORE_ROOT.mkdir(parents=True, exist_ok=True)
        probe = STORE_ROOT / ".ready"
        probe.write_text("ok")
        probe.unlink()
        return True
    except Exception:  # pragma: no cover - filesystem
        return False
