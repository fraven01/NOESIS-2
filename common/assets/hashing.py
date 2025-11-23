"""Hashing helpers shared across ingestion components."""

from __future__ import annotations

import hashlib
import io
from typing import Optional

from PIL import Image, ImageOps


def sha256_bytes(payload: bytes) -> str:
    """Return the SHA256 hex digest for ``payload``."""

    return hashlib.sha256(payload).hexdigest()


def sha256_text(text: str, *, encoding: str = "utf-8") -> str:
    """Return the SHA256 hex digest for ``text`` using ``encoding``."""

    return sha256_bytes(text.encode(encoding))


def perceptual_hash(payload: bytes) -> Optional[str]:
    """Compute a simple average hash for image ``payload``.

    Returns ``None`` when the payload cannot be decoded as an image.
    """

    try:
        with Image.open(io.BytesIO(payload)) as img:
            grey = ImageOps.grayscale(img)
            resample_attr = getattr(Image, "Resampling", None)
            resample = getattr(resample_attr, "LANCZOS", getattr(Image, "LANCZOS", 1))
            resized = grey.resize((8, 8), resample=resample)
            pixels = list(resized.getdata())
    except Exception:  # pragma: no cover - fall back on unsupported formats
        return None

    if not pixels:
        return None

    average = sum(pixels) / len(pixels)
    bits = 0
    for value in pixels:
        bits = (bits << 1) | int(value >= average)

    return f"{bits:016x}"


__all__ = [
    "sha256_bytes",
    "sha256_text",
    "perceptual_hash",
]
