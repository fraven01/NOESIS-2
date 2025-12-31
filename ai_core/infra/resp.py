from __future__ import annotations

from collections.abc import Mapping

from django.http import HttpResponse

from ai_core.tool_contracts.base import tool_context_from_meta
from common.constants import (
    IDEMPOTENCY_KEY_HEADER,
    X_CASE_ID_HEADER,
    X_KEY_ALIAS_HEADER,
    X_TENANT_ID_HEADER,
    X_TRACE_ID_HEADER,
)


Meta = Mapping[str, object | None]


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
