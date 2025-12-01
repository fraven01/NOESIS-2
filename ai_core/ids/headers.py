"""Header normalization helpers for correlation IDs."""

from __future__ import annotations

import re
from typing import Any, Callable, Iterable, Mapping, MutableMapping

from .contracts import normalize_trace_id

HeaderMap = Mapping[str, Any]

TRACE_ID_ALIASES: tuple[str, ...] = (
    "x-trace-id",
    "trace-id",
    "trace_id",
    "x_trace_id",
)
REQUEST_ID_ALIASES: tuple[str, ...] = (
    "request-id",
    "request_id",
    "x-request-id",
    "x_request_id",
)
TRACEPARENT_ALIASES: tuple[str, ...] = (
    "traceparent",
    "http_traceparent",
)
TENANT_ALIASES: tuple[str, ...] = (
    "x-tenant-id",
    "tenant-id",
    "tenant_id",
)
CASE_ALIASES: tuple[str, ...] = (
    "x-case-id",
    "case-id",
    "case_id",
)
IDEMPOTENCY_KEY_ALIASES: tuple[str, ...] = (
    "idempotency-key",
    "idempotency_key",
    "x-idempotency-key",
    "x_idempotency_key",
)


def _normalize_header_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    text = str(value).strip()
    return text or None


def _canonicalize_header_keys(headers: HeaderMap) -> dict[str, Any]:
    canonical: dict[str, Any] = {}
    for raw_key, value in headers.items():
        key = str(raw_key).lower()
        variants = {key, key.replace("_", "-"), key.replace("-", "_")}
        if key.startswith("http_"):
            trimmed = key.removeprefix("http_")
            variants.update(
                {
                    trimmed,
                    trimmed.replace("_", "-"),
                    trimmed.replace("-", "_"),
                }
            )
        for variant in variants:
            canonical[variant] = value
    return canonical


def _first_header(headers: HeaderMap, aliases: Iterable[str]) -> str | None:
    canonical = _canonicalize_header_keys(headers)
    for alias in aliases:
        candidate = canonical.get(alias.lower())
        normalized = _normalize_header_value(candidate)
        if normalized is not None:
            return normalized
    return None


def normalize_tenant_header(headers: HeaderMap) -> str | None:
    return _first_header(headers, TENANT_ALIASES)


_CASE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


def normalize_case_header(headers: HeaderMap) -> str | None:
    value = _first_header(headers, CASE_ALIASES)
    if value:
        if not _CASE_ID_PATTERN.match(value):
            raise ValueError(
                "Case header is required and must use the documented format."
            )
    return value


def normalize_idempotency_key(headers: HeaderMap) -> str | None:
    return _first_header(headers, IDEMPOTENCY_KEY_ALIASES)


def _parse_traceparent(value: str | None) -> tuple[str | None, str | None]:
    if not value:
        return None, None
    parts = value.strip().split("-")
    if len(parts) < 3:
        return None, None
    trace_part = parts[1] if len(parts) > 1 else None
    span_part = parts[2] if len(parts) > 2 else None
    trace_id = _normalize_w3c_id(trace_part)
    span_id = _normalize_w3c_id(span_part, span=True)
    return trace_id, span_id


def _normalize_w3c_id(value: str | None, *, span: bool = False) -> str | None:
    if not value:
        return None
    normalized = value.replace("-", "").strip().lower()
    if span and normalized:
        return normalized[:16]
    return normalized or None


def coerce_trace_id(
    headers: HeaderMap,
    *,
    warn: Callable[[str], None] | None = None,
) -> tuple[str, str | None]:
    meta: MutableMapping[str, Any] = {}
    span_id: str | None = None

    traceparent = _first_header(headers, TRACEPARENT_ALIASES)
    if traceparent:
        trace_id_from_parent, span_id = _parse_traceparent(traceparent)
        if trace_id_from_parent:
            meta["trace_id"] = trace_id_from_parent

    if "trace_id" not in meta:
        direct_trace = _first_header(headers, TRACE_ID_ALIASES)
        if direct_trace:
            meta["trace_id"] = direct_trace

    request_alias = _first_header(headers, REQUEST_ID_ALIASES)
    if "trace_id" not in meta and request_alias:
        meta["request_id"] = request_alias

    trace_id = normalize_trace_id(meta, warn=warn)
    return trace_id, span_id


__all__ = [
    "coerce_trace_id",
    "normalize_case_header",
    "normalize_idempotency_key",
    "normalize_tenant_header",
]
