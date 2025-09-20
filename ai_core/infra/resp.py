from __future__ import annotations

from typing import Mapping

from django.http import HttpResponse


Meta = Mapping[str, str]


def apply_std_headers(response: HttpResponse, meta: Meta) -> HttpResponse:
    """Attach standard metadata headers to successful responses only."""

    if not 200 <= response.status_code < 300:
        return response

    header_map = {
        "X-Trace-ID": meta.get("trace_id"),
        "X-Case-ID": meta.get("case"),
        "X-Tenant-ID": meta.get("tenant"),
        "X-Key-Alias": meta.get("key_alias"),
    }

    for header, value in header_map.items():
        if value:
            response[header] = value

    return response
