"""Helpers for normalising request metadata and merging graph state."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, MutableMapping

from common.constants import (
    IDEMPOTENCY_KEY_HEADER,
    META_CASE_ID_KEY,
    META_IDEMPOTENCY_KEY,
    META_KEY_ALIAS_KEY,
    META_TENANT_ID_KEY,
    META_TENANT_SCHEMA_KEY,
    META_TRACE_ID_KEY,
    X_CASE_ID_HEADER,
    X_KEY_ALIAS_HEADER,
    X_TENANT_ID_HEADER,
    X_TENANT_SCHEMA_HEADER,
    X_TRACE_ID_HEADER,
)

from ai_core.infra.rate_limit import get_quota

REQUIRED_KEYS = {"tenant_id", "case_id", "trace_id", "graph_name", "graph_version"}


@dataclass(frozen=True)
class ToolContext:
    """Immutable context propagated to downstream tools and tasks."""

    tenant_id: str
    case_id: str
    trace_id: str
    idempotency_key: str | None = None

    def serialize(self) -> dict[str, str | None]:
        """Return a serialisable representation of the context."""

        return {
            "tenant_id": self.tenant_id,
            "case_id": self.case_id,
            "trace_id": self.trace_id,
            "idempotency_key": self.idempotency_key,
        }


def _coalesce(request: Any, header: str, meta_key: str) -> str | None:
    headers: Mapping[str, str] = getattr(request, "headers", {}) or {}
    meta: MutableMapping[str, Any] = getattr(request, "META", {}) or {}
    value = headers.get(header)
    if value is None:
        value = meta.get(meta_key)
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return value


def _resolve_graph_name(request: Any) -> str:
    explicit = getattr(request, "graph_name", None)
    if isinstance(explicit, str) and explicit:
        return explicit

    resolver = getattr(request, "resolver_match", None)
    if resolver is not None:
        url_name = getattr(resolver, "url_name", None)
        if isinstance(url_name, str) and url_name:
            return url_name
        view_name = getattr(resolver, "view_name", None)
        if isinstance(view_name, str) and view_name:
            return view_name

    path = getattr(request, "path", None)
    if isinstance(path, str) and path:
        candidate = path.rstrip("/").split("/")[-1]
        if candidate:
            return candidate
    raise ValueError("graph name could not be determined from request")


def normalize_meta(request: Any) -> dict:
    """Return a normalised metadata mapping for graph executions."""

    tenant_id = _coalesce(request, X_TENANT_ID_HEADER, META_TENANT_ID_KEY)
    case_id = _coalesce(request, X_CASE_ID_HEADER, META_CASE_ID_KEY)
    trace_id = _coalesce(request, X_TRACE_ID_HEADER, META_TRACE_ID_KEY)
    graph_name = _resolve_graph_name(request)
    graph_version = getattr(request, "graph_version", "v0")
    idempotency_key = _coalesce(request, IDEMPOTENCY_KEY_HEADER, META_IDEMPOTENCY_KEY)

    meta = {
        "tenant_id": tenant_id,
        "case_id": case_id,
        "trace_id": trace_id,
        "graph_name": graph_name,
        "graph_version": graph_version,
        "requested_at": datetime.now(timezone.utc).isoformat(),
        "rate_limit": {"quota": get_quota()},
    }

    tenant_schema = _coalesce(request, X_TENANT_SCHEMA_HEADER, META_TENANT_SCHEMA_KEY)
    if tenant_schema:
        meta["tenant_schema"] = tenant_schema

    key_alias = _coalesce(request, X_KEY_ALIAS_HEADER, META_KEY_ALIAS_KEY)
    if key_alias:
        meta["key_alias"] = key_alias

    missing = [key for key in REQUIRED_KEYS if not meta.get(key)]
    if missing:
        raise ValueError(f"missing required meta keys: {', '.join(sorted(missing))}")

    tool_context = ToolContext(
        tenant_id=meta["tenant_id"],
        case_id=meta["case_id"],
        trace_id=meta["trace_id"],
        idempotency_key=idempotency_key,
    )

    meta["tool_context"] = tool_context
    if idempotency_key:
        meta["idempotency_key"] = idempotency_key

    return meta


def merge_state(
    old: Mapping[str, Any] | None, incoming: Mapping[str, Any] | None
) -> dict:
    """Return a new state mapping with ``incoming`` values overwriting ``old``."""

    merged: dict[str, Any] = {}
    if old:
        merged.update(dict(old))
    if incoming:
        merged.update(dict(incoming))
    return merged

