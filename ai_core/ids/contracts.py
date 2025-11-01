"""Centralised ID contracts and validators."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence, TypedDict

K_REQUIRED_SPAN_ATTRS: Sequence[str] = (
    "tenant_id",
    "case_id",
    "trace_id",
    "workflow_id",
)
"""Common span attributes required for observability."""

REQUEST_ID_DEPRECATED = True
"""Flag indicating the deprecated `request_id` field is still mapped for compatibility."""


class MetaIds(TypedDict, total=False):
    """Metadata IDs provided by callers."""

    tenant_id: str
    trace_id: str
    workflow_id: str
    case_id: str
    span_id: str
    request_id: str


@dataclass(frozen=True)
class CorrelationIds:
    """Resolved correlation identifiers for telemetry and logging."""

    tenant_id: str
    trace_id: str
    workflow_id: str
    case_id: str | None = None
    span_id: str | None = None


@dataclass(frozen=True)
class DocumentRef:
    """Identifier of a document for lookups and storage."""

    tenant_id: str
    workflow_id: str
    document_id: str
    collection_id: str | None = None
    version: str | None = None


def _extract_value(meta: Mapping[str, Any], key: str) -> Any:
    try:
        return meta[key]  # type: ignore[index]
    except (KeyError, TypeError, AttributeError):
        return getattr(meta, key, None)


def require_ids(meta: Mapping[str, Any], required: Iterable[str] | None = None) -> None:
    """Ensure that *meta* contains all *required* identifiers."""

    required_ids = tuple(required or ("tenant_id", "trace_id", "workflow_id"))
    missing: list[str] = []

    for key in required_ids:
        value = _extract_value(meta, key)
        if value is None:
            missing.append(key)
            continue
        if isinstance(value, str) and value.strip() == "":
            missing.append(key)

    if missing:
        raise ValueError(f"Missing required id(s): {', '.join(missing)}")


def normalize_trace_id(
    meta: MutableMapping[str, Any],
    *,
    warn: Callable[[str], None] | None = None,
) -> str:
    """Normalise and return the ``trace_id`` value in *meta*.

    Supports the deprecated ``request_id`` key by remapping it to ``trace_id`` and
    emitting a deprecation warning.
    """

    warn_func = warn
    if warn_func is None:

        def _default_warn(message: str) -> None:
            warnings.warn(
                message,
                DeprecationWarning,
                stacklevel=2,
            )

        warn_func = _default_warn

    raw_trace = meta.get("trace_id")
    if raw_trace is None or (isinstance(raw_trace, str) and raw_trace.strip() == ""):
        request_id = meta.pop("request_id", None)
        if request_id:
            normalised_request = str(request_id).strip()
            if not normalised_request:
                raise ValueError("request_id cannot be empty when provided")
            meta["trace_id"] = normalised_request
            warn_func("`request_id` is deprecated, use `trace_id` instead.")
            raw_trace = normalised_request
        else:
            raise ValueError("trace_id is required")

    normalised = str(raw_trace).strip()
    if not normalised:
        raise ValueError("trace_id cannot be empty")

    meta["trace_id"] = normalised
    if "request_id" in meta and meta["request_id"] is None:
        meta.pop("request_id", None)
    return normalised


__all__ = [
    "CorrelationIds",
    "DocumentRef",
    "K_REQUIRED_SPAN_ATTRS",
    "MetaIds",
    "REQUEST_ID_DEPRECATED",
    "normalize_trace_id",
    "require_ids",
]
