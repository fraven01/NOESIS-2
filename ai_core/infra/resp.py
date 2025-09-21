from __future__ import annotations

from typing import Mapping

from django.http import HttpResponse

from common.constants import (
    X_CASE_ID_HEADER,
    X_KEY_ALIAS_HEADER,
    X_TENANT_ID_HEADER,
    X_TRACE_ID_HEADER,
)


Meta = Mapping[str, str]


def apply_std_headers(response: HttpResponse, meta: Meta) -> HttpResponse:
    """Attach standard metadata headers to successful responses only."""

    if not 200 <= response.status_code < 300:
        return response

    header_map = {
        X_TRACE_ID_HEADER: meta.get("trace_id"),
        X_CASE_ID_HEADER: meta.get("case"),
        X_TENANT_ID_HEADER: meta.get("tenant"),
        X_KEY_ALIAS_HEADER: meta.get("key_alias"),
    }

    for header, value in header_map.items():
        if value:
            response[header] = value

    return response
