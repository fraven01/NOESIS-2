"""Integration hooks between graph workers and case lifecycle services."""

from __future__ import annotations

from typing import Any, Mapping

from common.logging import get_logger

from cases.lifecycle import (
    CaseLifecycleUpdateResult,
    update_case_from_collection_search,
)

log = get_logger(__name__)

_COLLECTION_SEARCH_NAMES = {"collection_search", "CollectionSearchGraph"}


def emit_case_lifecycle_for_collection_search(
    *,
    graph_name: str,
    tenant_id: str | None,
    case_id: str | None,
    state: Mapping[str, Any] | None,
) -> CaseLifecycleUpdateResult | None:
    """Emit case lifecycle transitions for CollectionSearch graph runs."""

    if graph_name not in _COLLECTION_SEARCH_NAMES:
        return None
    if not tenant_id or not case_id:
        return None
    try:
        return update_case_from_collection_search(tenant_id, case_id, state)
    except Exception:  # pragma: no cover - defensive integration hook
        log.exception(
            "case_lifecycle_collection_search_failed",
            extra={
                "graph": graph_name,
                "tenant_id": tenant_id,
                "case_id": case_id,
            },
        )
        return None


__all__ = ["emit_case_lifecycle_for_collection_search"]
