from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from django.http import HttpResponse

from ai_core.tool_contracts.base import tool_context_from_meta
from ai_core.tool_contracts.base import ToolError, ToolErrorDetail, ToolErrorMeta
from ai_core.tools.errors import ToolErrorType
from pydantic import BaseModel
from common.constants import (
    IDEMPOTENCY_KEY_HEADER,
    X_CASE_ID_HEADER,
    X_KEY_ALIAS_HEADER,
    X_TENANT_ID_HEADER,
    X_TRACE_ID_HEADER,
)


Meta = Mapping[str, object | None]


class _EmptyToolInput(BaseModel):
    """Placeholder input for error envelopes returned by HTTP boundaries."""


def tool_error_type_from_status(status_code: int) -> ToolErrorType:
    if status_code == 429:
        return ToolErrorType.RATE_LIMIT
    if status_code == 504:
        return ToolErrorType.TIMEOUT
    if status_code == 502:
        return ToolErrorType.UPSTREAM
    if 400 <= status_code < 500:
        return ToolErrorType.VALIDATION
    return ToolErrorType.FATAL


def build_tool_error_payload(
    *,
    message: str,
    status_code: int,
    error_type: ToolErrorType | None = None,
    code: str | None = None,
    cause: str | None = None,
    details: dict[str, Any] | None = None,
    retry_after_ms: int | None = None,
    upstream_status: int | None = None,
    endpoint: str | None = None,
    attempt: int | None = None,
    took_ms: int | None = None,
) -> dict[str, Any]:
    detail = ToolErrorDetail(
        type=error_type or tool_error_type_from_status(status_code),
        message=message,
        code=code,
        cause=cause,
        details=details,
        retry_after_ms=retry_after_ms,
        upstream_status=upstream_status,
        endpoint=endpoint,
        attempt=attempt,
    )
    meta = ToolErrorMeta(took_ms=max(0, int(took_ms or 0)))
    payload = ToolError[_EmptyToolInput](
        input=_EmptyToolInput(),
        error=detail,
        meta=meta,
    )
    return payload.model_dump(mode="json")


def apply_std_headers(response: HttpResponse, meta: Meta) -> HttpResponse:
    """Attach standard metadata headers to successful responses only.

    The ``meta`` mapping may optionally include a ``traceparent`` entry. When
    provided, the corresponding W3C trace context header is forwarded alongside
    the standard ``X-*`` metadata headers.

    BREAKING CHANGE (Option A - Strict Separation):
    case_id is a business identifier, extracted from business_context.
    """

    if not 200 <= response.status_code < 300:
        return response

    context = tool_context_from_meta(meta)

    header_map = {
        X_TRACE_ID_HEADER: context.scope.trace_id,
        X_CASE_ID_HEADER: context.business.case_id,
        X_TENANT_ID_HEADER: context.scope.tenant_id,
        X_KEY_ALIAS_HEADER: meta.get("key_alias"),
        IDEMPOTENCY_KEY_HEADER: context.scope.idempotency_key,
        "traceparent": meta.get("traceparent"),
    }

    for header, value in header_map.items():
        if value:
            response[header] = value

    return response
