from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

BASE_PATH = Path(".ai_core_store")


def sanitize_identifier(value: str) -> str:
    """Return a filesystem-safe representation of an identifier."""

    if ".." in value or "/" in value or os.sep in value:
        raise ValueError("unsafe identifier")

    sanitized = re.sub(r"[^A-Za-z0-9._-]", "_", value)[:128]
    if not sanitized:
        raise ValueError("unsafe identifier")
    return sanitized


def _full_path(relative: str) -> Path:
    path = BASE_PATH / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def put_bytes(path: str, data: bytes) -> Path:
    """Persist raw bytes to the object store."""
    target = _full_path(path)
    target.write_bytes(data)
    return target


def read_json(path: str) -> Any:
    """Read JSON from the object store."""
    target = BASE_PATH / path
    with target.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: str, obj: Any) -> Path:
    """Write JSON to the object store."""
    target = _full_path(path)
    with target.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh)
    return target
