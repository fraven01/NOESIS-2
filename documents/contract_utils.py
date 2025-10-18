"""Normalization and validation utilities for document contracts."""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List, Optional


_INVISIBLE_CATEGORIES = {"Cf", "Cc", "Cs"}
_TAG_RE = re.compile(r"^[A-Za-z0-9._-]+$", re.ASCII)
_WORKFLOW_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$", re.ASCII)
_MEDIA_TYPE_RE = re.compile(r"^[\w.+-]+/[\w.+-]+$", re.ASCII)
_BCP47_SEGMENT_RE = re.compile(r"^[A-Za-z0-9]{1,8}$", re.ASCII)


def _strip_invisible(value: str) -> str:
    return "".join(ch for ch in value if unicodedata.category(ch) not in _INVISIBLE_CATEGORIES)


def normalize_string(value: str) -> str:
    """Normalize string input by applying NFKC, trimming invisibles and whitespace."""

    normalized = unicodedata.normalize("NFKC", value)
    normalized = _strip_invisible(normalized)
    return normalized.strip()


def normalize_optional_string(value: Optional[str]) -> Optional[str]:
    """Normalize optional strings and coerce empty values to ``None``."""

    if value is None:
        return None
    normalized = normalize_string(value)
    return normalized or None


def normalize_tenant(value: str) -> str:
    """Normalize and validate tenant identifiers."""

    normalized = normalize_string(value)
    if not normalized:
        raise ValueError("tenant_empty")
    if len(normalized) > 128:
        raise ValueError("tenant_too_long")
    return normalized


def normalize_workflow_id(value: str) -> str:
    """Normalize and validate workflow identifiers."""

    normalized = normalize_string(value)
    if not normalized:
        raise ValueError("workflow_empty")
    if len(normalized) > 128:
        raise ValueError("workflow_too_long")
    if not _WORKFLOW_ID_RE.fullmatch(normalized):
        raise ValueError("workflow_invalid_char")
    return normalized


def normalize_title(value: Optional[str]) -> Optional[str]:
    """Normalize optional document titles with maximum length enforcement."""

    normalized = normalize_optional_string(value)
    if normalized is None:
        return None
    if len(normalized) > 256:
        raise ValueError("title_too_long")
    return normalized


def normalize_tags(values: Iterable[str]) -> List[str]:
    """Normalize tag inputs, enforcing uniqueness and stable sorting."""

    normalized_values = set()
    for tag in values:
        normalized = normalize_string(tag)
        if not normalized:
            continue
        if len(normalized) > 64:
            raise ValueError("tag_too_long")
        if not _TAG_RE.fullmatch(normalized):
            raise ValueError("tag_invalid")
        normalized_values.add(normalized)
    return sorted(normalized_values)


def is_bcp47_like(value: str) -> bool:
    """Return ``True`` if the provided string resembles a BCP-47 language tag."""

    if not value:
        return False
    parts = value.split("-")
    if any(part == "" for part in parts):
        return False
    if not all(_BCP47_SEGMENT_RE.fullmatch(part) for part in parts):
        return False
    return any(any(ch.isalpha() for ch in part) for part in parts)


def normalize_media_type(value: str) -> str:
    """Normalize media types to lowercase ``type/subtype`` strings without parameters.

    Parameterized values such as ``text/html; charset=utf-8`` are rejected.
    """

    normalized = normalize_string(value)
    if not normalized:
        raise ValueError("media_type_empty")
    candidate = normalized.lower()
    if not _MEDIA_TYPE_RE.fullmatch(candidate):
        raise ValueError("media_type_invalid")
    return candidate


def is_image_mediatype(value: str) -> bool:
    """Return ``True`` when the media type represents an image payload."""

    try:
        normalized = normalize_media_type(value)
    except ValueError:
        return False
    return normalized.startswith("image/")


def validate_bbox(values: List[float]) -> List[float]:
    """Validate normalized bounding box coordinates."""

    if len(values) != 4:
        raise ValueError("bbox_invalid")
    try:
        x0, y0, x1, y1 = (float(v) for v in values)
    except (TypeError, ValueError) as exc:  # pragma: no cover - delegated error
        raise ValueError("bbox_invalid") from exc
    coordinates = [x0, y0, x1, y1]
    if any(coord < 0 or coord > 1 for coord in coordinates):
        raise ValueError("bbox_invalid")
    if x1 <= x0 or y1 <= y0:
        raise ValueError("bbox_invalid")
    return coordinates


def truncate_text(value: Optional[str], limit: int) -> Optional[str]:
    """Truncate the provided string to the byte limit without breaking UTF-8."""

    if value is None:
        return None
    if limit < 0:
        raise ValueError("text_limit_negative")
    data = value.encode("utf-8")
    if len(data) <= limit:
        return value
    truncated = data[:limit]
    while True:
        try:
            return truncated.decode("utf-8")
        except UnicodeDecodeError as exc:
            if exc.start == 0:
                return ""
            truncated = truncated[:exc.start]

