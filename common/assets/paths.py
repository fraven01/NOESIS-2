"""Deterministic asset path helpers."""

from __future__ import annotations

import re
from uuid import UUID

__all__ = ["deterministic_asset_path"]

_PATH_SANITIZER = re.compile(r"[^a-zA-Z0-9._-]+")


def _safe_segment(value: str) -> str:
    cleaned = _PATH_SANITIZER.sub("-", value.strip())
    return cleaned or "asset"


def deterministic_asset_path(document_id: UUID, locator: str, *, prefix: str = "assets") -> str:
    """Return a deterministic path for an asset bound to ``document_id``.

    The resulting path is stable across runs and safe for filesystem or
    object-store usage.
    """

    segment = _safe_segment(locator or "")
    return f"{prefix}/{document_id}/{segment}"
