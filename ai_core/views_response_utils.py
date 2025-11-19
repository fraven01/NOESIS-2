"""Helpers for building HTTP responses with standard AI Core headers."""

from __future__ import annotations

from collections.abc import Mapping

from rest_framework.response import Response

from common.constants import IDEMPOTENCY_KEY_HEADER

from ai_core.infra.resp import apply_std_headers


Meta = Mapping[str, object | None]


def apply_response_headers(
    response: Response, meta: Meta, idempotency_key: str | None = None
) -> Response:
    """Attach standard response headers and propagate Idempotency-Key values.

    ``apply_std_headers`` only attaches metadata headers for successful
    responses (2xx). The crawler ingestion view needs to propagate the
    ``Idempotency-Key`` header even when guardrails reject a request, so this
    helper merges the behaviours and centralises the pattern for reuse.
    """

    resolved_key = idempotency_key or meta.get("idempotency_key")
    response = apply_std_headers(response, meta)
    if resolved_key:
        response[IDEMPOTENCY_KEY_HEADER] = resolved_key
    return response
