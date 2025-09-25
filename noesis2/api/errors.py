"""Shared error response contracts for OpenAPI documentation and responses."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, MutableMapping, Sequence

try:  # pragma: no cover - pydantic is optional at runtime
    from pydantic import BaseModel
except ModuleNotFoundError:  # pragma: no cover - fallback when pydantic is absent
    BaseModel = None  # type: ignore[assignment]

from drf_spectacular.utils import OpenApiExample, OpenApiResponse
from rest_framework import serializers


class ErrorResponseSerializer(serializers.Serializer):
    """Generic error payload emitted by API endpoints on failure."""

    detail = serializers.CharField(
        help_text="Human readable description of the failure."
    )
    code = serializers.CharField(help_text="Stable machine readable error identifier.")


if (
    BaseModel is not None
):  # pragma: no branch - executed only when pydantic is available

    class ErrorResponseModel(BaseModel):
        """Pydantic representation of :class:`ErrorResponseSerializer`."""

        detail: str
        code: str

else:  # pragma: no cover - exercised only when pydantic is not installed

    from typing import TypedDict

    class ErrorResponseModel(TypedDict):
        """Typed fallback ensuring hints remain intact without pydantic."""

        detail: str
        code: str


DEFAULT_ERROR_STATUSES: Sequence[int] = (400, 401, 403, 404, 503)
"""Default HTTP status codes documented on most endpoints."""

JSON_ERROR_STATUSES: Sequence[int] = (*DEFAULT_ERROR_STATUSES, 415)
"""Status codes for endpoints expecting JSON payloads."""

RATE_LIMIT_ERROR_STATUSES: Sequence[int] = (*DEFAULT_ERROR_STATUSES, 429)
"""Status codes for endpoints protected by rate limiting."""

RATE_LIMIT_JSON_ERROR_STATUSES: Sequence[int] = (*RATE_LIMIT_ERROR_STATUSES, 415)
"""Status codes for rate-limited endpoints expecting JSON payloads."""

CONFLICT_ERROR_STATUSES: Sequence[int] = (*DEFAULT_ERROR_STATUSES, 409)
"""Status codes for endpoints where conflicts are expected (e.g., uploads)."""

RATE_LIMIT_CONFLICT_ERROR_STATUSES: Sequence[int] = (*DEFAULT_ERROR_STATUSES, 409, 429)
"""Status codes for endpoints that can hit both conflict and rate limiting."""


_ERROR_RESPONSE_METADATA: Mapping[int, Mapping[str, object]] = {
    400: {
        "description": "Bad Request",
        "detail": "Tenant schema could not be resolved from headers.",
        "code": "tenant_not_found",
        "additional_examples": (
            {
                "name": "invalid_json",
                "summary": "JSON payload could not be parsed.",
                "detail": "Request payload contains invalid JSON.",
                "code": "invalid_json",
            },
        ),
    },
    401: {
        "description": "Unauthorized",
        "detail": "Authentication credentials were not provided.",
        "code": "unauthorized",
    },
    403: {
        "description": "Forbidden",
        "detail": "You do not have permission to perform this action.",
        "code": "forbidden",
    },
    404: {
        "description": "Not Found",
        "detail": "The requested resource was not found.",
        "code": "not_found",
    },
    415: {
        "description": "Unsupported Media Type",
        "detail": "Request payload must be encoded as application/json.",
        "code": "unsupported_media_type",
    },
    409: {
        "description": "Conflict",
        "detail": "A conflicting resource already exists.",
        "code": "conflict",
    },
    429: {
        "description": "Too Many Requests",
        "detail": "Rate limit exceeded for tenant.",
        "code": "rate_limit_exceeded",
    },
    503: {
        "description": "Service Unavailable",
        "detail": "Service temporarily unavailable.",
        "code": "service_unavailable",
    },
}


def _dedupe_statuses(statuses: Iterable[int]) -> Sequence[int]:
    """Return statuses without duplicates while preserving insertion order."""

    ordered: MutableMapping[int, None] = {}
    for status in statuses:
        ordered[int(status)] = None
    return tuple(ordered.keys())


def default_error_responses(
    statuses: Sequence[int] | None = None,
) -> Dict[int, OpenApiResponse]:
    """Return OpenAPI response objects for the given HTTP status codes."""

    if statuses is None:
        statuses = DEFAULT_ERROR_STATUSES

    responses: Dict[int, OpenApiResponse] = {}
    for status_code in _dedupe_statuses(statuses):
        metadata = _ERROR_RESPONSE_METADATA.get(status_code)
        if metadata is None:
            continue

        example = OpenApiExample(
            name=metadata["code"],
            summary=metadata["description"],
            value={"detail": metadata["detail"], "code": metadata["code"]},
            response_only=True,
        )

        examples = [example]

        extra_examples: Iterable[Mapping[str, str]] = metadata.get(
            "additional_examples", ()
        )  # type: ignore[assignment]
        for extra in extra_examples:
            examples.append(
                OpenApiExample(
                    name=extra["name"],
                    summary=extra.get("summary", extra["name"]),
                    value={"detail": extra["detail"], "code": extra["code"]},
                    response_only=True,
                )
            )

        responses[status_code] = OpenApiResponse(
            response=ErrorResponseSerializer,
            description=metadata["description"],
            examples=examples,
        )

    return responses


__all__ = [
    "CONFLICT_ERROR_STATUSES",
    "DEFAULT_ERROR_STATUSES",
    "JSON_ERROR_STATUSES",
    "ErrorResponseModel",
    "ErrorResponseSerializer",
    "RATE_LIMIT_CONFLICT_ERROR_STATUSES",
    "RATE_LIMIT_ERROR_STATUSES",
    "RATE_LIMIT_JSON_ERROR_STATUSES",
    "default_error_responses",
]
