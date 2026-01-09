"""Shared helpers for graph execution."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from importlib import import_module
from typing import Any

from django.conf import settings
from pydantic import ValidationError
from rest_framework.request import Request
from rest_framework.response import Response

from ai_core.graph.core import FileCheckpointer, GraphContext
from ai_core.graph.schemas import normalize_meta as _base_normalize_meta
from ai_core.graphs.technical.cost_tracking import coerce_cost_value
from ai_core.infra.resp import build_tool_error_payload
from ai_core.infra.serialization import to_jsonable
from common.constants import COLLECTION_ID_HEADER_CANDIDATES, META_COLLECTION_ID_KEY

from ..schemas import InfoIntakeRequest, RagQueryRequest

logger = logging.getLogger(__name__)

CHECKPOINTER = FileCheckpointer()
ASYNC_GRAPH_NAMES = frozenset(
    getattr(settings, "GRAPH_WORKER_GRAPHS", ("rag.default",))
)


def _should_enqueue_graph(graph_name: str) -> bool:
    return graph_name in ASYNC_GRAPH_NAMES


def _dump_jsonable(value: Any) -> Any:
    """Return a structure that json.dumps can serialise."""
    return to_jsonable(value)


def _get_checkpointer():  # type: ignore[no-untyped-def]
    try:
        views = import_module("ai_core.views")
        cp = getattr(views, "CHECKPOINTER", None)
        if cp is not None:
            return cp
    except Exception:
        pass
    return CHECKPOINTER


def _normalize_meta(request):  # type: ignore[no-untyped-def]
    try:
        views = import_module("ai_core.views")
        fn = getattr(views, "normalize_meta", None)
        if callable(fn):
            return fn(request)
    except Exception:
        pass
    return _base_normalize_meta(request)


def _apply_collection_header_bridge(
    request: Request, payload: Mapping[str, object] | None
) -> dict[str, object]:
    data = dict(payload or {})

    header_value: str | None = None
    headers = getattr(request, "headers", None)
    if isinstance(headers, Mapping):
        for candidate_key in COLLECTION_ID_HEADER_CANDIDATES:
            candidate = headers.get(candidate_key)
            if candidate is None:
                continue
            if not isinstance(candidate, str):
                candidate = str(candidate)
            candidate = candidate.strip()
            if candidate:
                header_value = candidate
                break
    if header_value is None:
        meta = getattr(request, "META", None)
        if isinstance(meta, Mapping):
            candidate = meta.get(META_COLLECTION_ID_KEY)
            if isinstance(candidate, str):
                header_value = candidate.strip() or None

    if not header_value:
        return data

    body_value = data.get("collection_id")
    body_present = False
    if isinstance(body_value, str):
        if body_value.strip():
            body_present = True
        else:
            body_value = None
    elif body_value not in (None, ""):
        body_present = True

    filters_value = data.get("filters")
    filter_has_list = False
    collection_scope = "none"
    if isinstance(filters_value, Mapping):
        candidates = filters_value.get("collection_ids")
        if candidates:
            filter_has_list = True
            collection_scope = "list"
        single_filter = filters_value.get("collection_id")
        if single_filter is not None:
            try:
                if str(single_filter).strip():
                    body_present = True
                    if not filter_has_list:
                        collection_scope = "single"
            except Exception:
                body_present = True
                if not filter_has_list:
                    collection_scope = "single"

    if not filter_has_list and body_present:
        collection_scope = "single"

    if not body_present and not filter_has_list:
        data["collection_id"] = header_value
    else:
        reason = "filter_list_present" if filter_has_list else "body_present"
        logger.debug(
            "collection header ignored due to %s (collection_scope=%s)",
            reason,
            collection_scope,
            extra={
                "reason": reason,
                "header_present": True,
                "collection_scope": collection_scope,
            },
        )

    return data


def _extract_initial_cost(meta: Mapping[str, Any]) -> float | None:
    cost_block = meta.get("cost")
    if isinstance(cost_block, Mapping):
        for key in ("total_usd", "usd", "total"):
            cost_value = cost_block.get(key)
            coerced = coerce_cost_value(cost_value)
            if coerced is not None:
                return coerced
    for key in ("cost_total_usd", "cost_usd", "total_cost_usd"):
        if key in meta:
            coerced = coerce_cost_value(meta[key])
            if coerced is not None:
                return coerced
    return None


def _extract_ledger_identifier(meta: Mapping[str, Any]) -> str | None:
    direct = meta.get("ledger_id") or meta.get("ledgerId")
    if direct:
        return str(direct)
    ledger_block = meta.get("ledger")
    if isinstance(ledger_block, Mapping):
        candidate = ledger_block.get("id") or ledger_block.get("ledger_id")
        if candidate:
            return str(candidate)
    return None


GRAPH_REQUEST_MODELS = {
    "info_intake": InfoIntakeRequest,
    "rag.default": RagQueryRequest,
}


def _log_graph_response_payload(payload: object, context: GraphContext) -> None:
    """Emit diagnostics about the response payload produced by a graph run."""

    try:
        payload_json = json.dumps(_dump_jsonable(payload), ensure_ascii=False)
    except TypeError:
        logger.exception(
            "graph.response_payload_serialization_error",
            extra={
                "graph": context.graph_name,
                "tenant_id": context.tenant_id,
                "case_id": context.case_id,
                "payload_type": type(payload).__name__,
            },
        )
        raise

    logger.info(
        "graph.response_payload",
        extra={
            "graph": context.graph_name,
            "tenant_id": context.tenant_id,
            "case_id": context.case_id,
            "payload_json": payload_json,
        },
    )


def _error_response(detail: str, code: str, status_code: int) -> Response:
    """Return a standardised error payload."""
    payload = build_tool_error_payload(
        message=detail,
        status_code=status_code,
        code=code,
    )
    return Response(payload, status=status_code)


def _format_validation_error(error: ValidationError) -> str:
    """Return a compact textual representation of a Pydantic validation error."""
    messages: list[str] = []
    for issue in error.errors():
        location = ".".join(str(part) for part in issue.get("loc", ()))
        message = issue.get("msg", "Invalid input")
        if location:
            messages.append(f"{location}: {message}")
        else:
            messages.append(message)
    return "; ".join(messages)
