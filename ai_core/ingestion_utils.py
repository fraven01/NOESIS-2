from __future__ import annotations

import hashlib


def make_fallback_external_id(
    filename: str | None,
    size: int | None,
    first_bytes: bytes | None,
) -> str:
    """Derive a deterministic identifier for uploads without an external id."""

    digest = hashlib.sha256()
    digest.update((filename or "").encode("utf-8", "ignore"))
    digest.update(str(size or 0).encode("ascii"))
    digest.update((first_bytes or b"")[: 64 * 1024])
    return digest.hexdigest()
