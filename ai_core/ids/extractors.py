"""Shared extraction utilities for ID normalization."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from uuid import uuid4

from common.constants import (
    META_INGESTION_RUN_ID_KEY,
    META_RUN_ID_KEY,
    X_INGESTION_RUN_ID_HEADER,
    X_RUN_ID_HEADER,
)


def _normalize_header_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return str(value).strip() or None


def _resolve_headers_meta(
    *,
    request: Any | None,
    headers: Mapping[str, Any] | None,
    meta: Mapping[str, Any] | None,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    resolved_headers = headers
    resolved_meta = meta

    if resolved_headers is None and request is not None:
        resolved_headers = getattr(request, "headers", None)
    if resolved_meta is None and request is not None:
        resolved_meta = getattr(request, "META", None)

    if not isinstance(resolved_headers, Mapping):
        resolved_headers = {}
    if not isinstance(resolved_meta, Mapping):
        resolved_meta = {}

    return resolved_headers, resolved_meta


def extract_runtime_ids(
    *,
    request: Any | None = None,
    headers: Mapping[str, Any] | None = None,
    meta: Mapping[str, Any] | None = None,
    generate_if_missing: bool = True,
) -> tuple[str | None, str | None]:
    """Extract run_id and ingestion_run_id with optional auto-generation."""
    resolved_headers, resolved_meta = _resolve_headers_meta(
        request=request, headers=headers, meta=meta
    )

    run_id = _normalize_header_value(
        getattr(request, "run_id", None) if request is not None else None
    )
    if run_id is None:
        run_id = _normalize_header_value(
            resolved_headers.get(X_RUN_ID_HEADER) or resolved_meta.get(META_RUN_ID_KEY)
        )

    ingestion_run_id = _normalize_header_value(
        getattr(request, "ingestion_run_id", None) if request is not None else None
    )
    if ingestion_run_id is None:
        ingestion_run_id = _normalize_header_value(
            resolved_headers.get(X_INGESTION_RUN_ID_HEADER)
            or resolved_meta.get(META_INGESTION_RUN_ID_KEY)
        )

    if generate_if_missing and not run_id and not ingestion_run_id:
        run_id = uuid4().hex

    return run_id, ingestion_run_id
