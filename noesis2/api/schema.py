"""Utilities for decorating OpenAPI schema definitions.

This module centralises required tenant headers, optional idempotency
metadata, and trace helpers so the generated schema stays aligned with
multi-tenant platform contracts.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, MutableMapping, Sequence, Type, TypeVar

from drf_spectacular.types import OpenApiTypes
from drf_spectacular.utils import (
    OpenApiParameter,
    extend_schema as spectacular_extend_schema,
    extend_schema_view as spectacular_extend_schema_view,
)


from .errors import DEFAULT_ERROR_STATUSES, default_error_responses
from common.constants import (
    IDEMPOTENCY_KEY_HEADER,
    X_CASE_ID_HEADER,
    X_KEY_ALIAS_HEADER,
    X_TENANT_ID_HEADER,
    X_TENANT_SCHEMA_HEADER,
    X_TRACE_ID_HEADER,
)

TENANT_HEADER_COMPONENT_DEFINITIONS: Sequence[Dict[str, object]] = (
    {
        "component": "TenantSchemaHeader",
        "name": X_TENANT_SCHEMA_HEADER,
        "required": True,
        "description": "Active tenant schema identifier.",
        "response": False,
    },
    {
        "component": "TenantIdHeader",
        "name": X_TENANT_ID_HEADER,
        "required": True,
        "description": "Tenant identifier matching the active schema.",
        "response": False,
    },
    {
        "component": "CaseIdHeader",
        "name": X_CASE_ID_HEADER,
        "required": True,
        "description": "Case identifier propagated across AI Core requests.",
        "response": False,
    },
    {
        "component": "IdempotencyKeyHeader",
        "name": IDEMPOTENCY_KEY_HEADER,
        "required": False,
        "description": "Optional idempotency key for safely retrying write operations.",
        "response": False,
    },
    {
        "component": "KeyAliasHeader",
        "name": X_KEY_ALIAS_HEADER,
        "required": False,
        "description": "Optional key alias referencing customer-managed secrets.",
        "response": False,
    },
)

TRACE_ID_RESPONSE_HEADER_COMPONENT_NAME = "TraceIdResponseHeader"
TRACE_ID_RESPONSE_HEADER_COMPONENT = {
    "name": X_TRACE_ID_HEADER,
    "in": "header",
    "required": False,
    "description": "Correlation identifier echoed back in responses.",
    "schema": {"type": "string"},
    "style": "simple",
}
TRACE_ID_RESPONSE_HEADER_REF = (
    f"#/components/headers/{TRACE_ID_RESPONSE_HEADER_COMPONENT_NAME}"
)
TRACE_HEADER_EXTENSION_FLAG = "x-noesis-trace-response-header"

ADMIN_BEARER_AUTH_SCHEME = "bearerAuth"


def curl_code_sample(
    command: str, *, label: str = "cURL", lang: str = "bash"
) -> Dict[str, object]:
    """Return an ``x-codeSamples`` extension describing a curl invocation."""

    return {
        "x-codeSamples": [
            {
                "lang": lang,
                "label": label,
                "source": command,
            }
        ]
    }


def _build_parameter(parameter_definition: Dict[str, object]) -> OpenApiParameter:
    """Return an :class:`OpenApiParameter` constructed from the definition."""

    return OpenApiParameter(
        name=str(parameter_definition["name"]),
        type=OpenApiTypes.STR,
        location=OpenApiParameter.HEADER,
        required=bool(parameter_definition["required"]),
        description=str(parameter_definition["description"]),
        style="simple",
    )


def tenant_headers_parameters() -> List[OpenApiParameter]:
    """Return reusable tenant header parameters for schema decoration."""

    return [
        _build_parameter(definition)
        for definition in TENANT_HEADER_COMPONENT_DEFINITIONS
    ]


def tenant_header_components() -> Dict[str, Dict[str, object]]:
    """Return OpenAPI components for tenant aware headers."""

    components: Dict[str, Dict[str, object]] = {}
    for definition in TENANT_HEADER_COMPONENT_DEFINITIONS:
        components[str(definition["component"])] = {
            "name": definition["name"],
            "in": "header",
            "required": bool(definition["required"]),
            "description": definition["description"],
            "schema": {"type": "string"},
            "style": "simple",
        }
    return components


def trace_response_headers() -> Dict[str, Dict[str, str]]:
    """Return the reusable response header describing ``X-Trace-ID``."""

    return {X_TRACE_ID_HEADER: {"$ref": TRACE_ID_RESPONSE_HEADER_REF}}


def _prepare_parameters(
    parameters: Sequence[OpenApiParameter],
) -> List[OpenApiParameter]:
    """Make defensive copies of parameters so they are safe to reuse."""

    prepared: List[OpenApiParameter] = []
    for parameter in parameters:
        clone = OpenApiParameter(
            name=parameter.name,
            type=parameter.type,
            location=parameter.location,
            required=parameter.required,
            description=parameter.description,
            enum=parameter.enum,
            pattern=parameter.pattern,
            deprecated=parameter.deprecated,
            style=parameter.style,
            explode=parameter.explode,
            default=parameter.default,
            allow_blank=parameter.allow_blank,
            many=parameter.many,
            examples=parameter.examples,
            extensions=deepcopy(parameter.extensions),
            exclude=parameter.exclude,
            response=parameter.response,
        )
        prepared.append(clone)
    return prepared


def _prepare_responses(responses: object) -> Dict[int, object]:
    """Normalise the responses argument into a mapping keyed by status code."""

    if responses is None:
        return {}

    if isinstance(responses, MutableMapping):
        return dict(responses)

    return {200: responses}


def default_extend_schema(**kwargs):
    """Wrapper around :func:`drf_spectacular.utils.extend_schema`.

    All API views should use this helper instead of the raw decorator so
    that the tenant headers remain documented without repetition. The
    decorator accepts the same keyword arguments as
    :func:`extend_schema`. Additional flags are available:

    ``include_tenant_headers`` (default: ``True``)
        Controls whether the tenant header components are added.

    ``include_trace_header`` (default: ``False``)
        When ``True`` the ``X-Trace-ID`` response header component is
        attached to the operation.

    ``include_error_responses`` (default: ``True``)
        When enabled, the shared error responses defined in
        :func:`noesis2.api.errors.default_error_responses` are appended.

    ``error_statuses`` (default: ``DEFAULT_ERROR_STATUSES``)
        Optional iterable overriding the status codes used when documenting
        shared error responses.
    """

    include_tenant_headers: bool = kwargs.pop("include_tenant_headers", True)
    include_trace_header: bool = kwargs.pop("include_trace_header", False)
    include_error_responses: bool = kwargs.pop("include_error_responses", True)
    error_statuses: Sequence[int] | None = kwargs.pop(
        "error_statuses", DEFAULT_ERROR_STATUSES
    )

    parameters: Sequence[OpenApiParameter] = kwargs.pop("parameters", []) or []
    prepared_parameters: List[OpenApiParameter] = []
    if include_tenant_headers:
        prepared_parameters.extend(_prepare_parameters(tenant_headers_parameters()))
    if parameters:
        prepared_parameters.extend(_prepare_parameters(list(parameters)))

    responses = _prepare_responses(kwargs.pop("responses", None))
    if include_error_responses:
        for status_code, response in default_error_responses(error_statuses).items():
            responses.setdefault(status_code, response)

    if responses:
        kwargs["responses"] = responses

    extensions: Dict[str, object] = dict(kwargs.pop("extensions", {}) or {})
    if include_trace_header:
        extensions[TRACE_HEADER_EXTENSION_FLAG] = True

    if extensions:
        kwargs["extensions"] = extensions

    return spectacular_extend_schema(parameters=prepared_parameters, **kwargs)


ViewType = TypeVar("ViewType", bound=Type[object])


def default_extend_schema_view(**kwargs):
    """Apply :func:`default_extend_schema` to every handler on a view class.

    This helper mimics :func:`drf_spectacular.utils.extend_schema_view` but
    automatically wires :func:`default_extend_schema` for all exposed HTTP
    verbs and ViewSet actions. It ensures multi-tenant headers (and optional
    extras like the trace response header) stay documented without repeating
    decorators for each method.
    """

    def decorator(view: ViewType) -> ViewType:
        method_decorators: Dict[str, object] = {}

        # APIView subclasses expose ``http_method_names`` which lists the HTTP
        # verbs implemented on the class. Skip ``options`` because DRF provides
        # it implicitly.
        for method_name in getattr(view, "http_method_names", []):
            if method_name == "options":
                continue
            if getattr(view, method_name, None) is not None:
                method_decorators[method_name] = default_extend_schema(**kwargs)

        # ViewSets map HTTP verbs to action methods (``list``, ``retrieve``,
        # etc.). Ensure these handlers receive the same decoration so router
        # generated operations inherit the tenant headers.
        for action_name in (
            "list",
            "create",
            "retrieve",
            "update",
            "partial_update",
            "destroy",
        ):
            handler = getattr(view, action_name, None)
            if handler is not None:
                method_decorators[action_name] = default_extend_schema(**kwargs)

        if not method_decorators:
            return view

        return spectacular_extend_schema_view(**method_decorators)(view)

    return decorator


def inject_trace_response_header(generator, request, public, result):
    """Post-processing hook that appends the trace header to marked operations."""

    paths = result.get("paths", {})
    for operations in paths.values():
        for operation in operations.values():
            if not isinstance(operation, MutableMapping):
                continue
            if not operation.pop(TRACE_HEADER_EXTENSION_FLAG, False):
                continue

            responses = operation.setdefault("responses", {}) or {}
            if not responses:
                responses["default"] = {"description": "Trace metadata"}

            for response in responses.values():
                if not isinstance(response, MutableMapping):
                    continue
                headers = response.setdefault("headers", {})
                headers.update(trace_response_headers())

    return result
