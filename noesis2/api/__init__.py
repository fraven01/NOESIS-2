"""NOESIS 2 API schema helpers."""

from .authentication import (  # noqa: F401 re-exported for convenience
    LiteLLMAdminUser,
    LiteLLMMasterKeyAuthentication,
)
from .errors import (  # noqa: F401 re-exported for convenience
    CONFLICT_ERROR_STATUSES,
    DEFAULT_ERROR_STATUSES,
    JSON_ERROR_STATUSES,
    RATE_LIMIT_CONFLICT_ERROR_STATUSES,
    RATE_LIMIT_ERROR_STATUSES,
    RATE_LIMIT_JSON_ERROR_STATUSES,
    default_error_responses,
    ErrorResponseModel,
    ErrorResponseSerializer,
)
from .schema import (
    ADMIN_BEARER_AUTH_SCHEME,
    curl_code_sample,
    TRACE_ID_RESPONSE_HEADER_COMPONENT,
    TRACE_ID_RESPONSE_HEADER_COMPONENT_NAME,
    TRACE_ID_RESPONSE_HEADER_REF,
    default_extend_schema,
    default_extend_schema_view,
    inject_trace_response_header,
    tenant_header_components,
    tenant_headers_parameters,
    trace_response_headers,
)
from .versioning import (  # noqa: F401 re-exported for convenience
    DeprecationHeadersMixin,
    build_deprecation_headers,
    mark_deprecated_response,
)

__all__ = [
    "ADMIN_BEARER_AUTH_SCHEME",
    "curl_code_sample",
    "DeprecationHeadersMixin",
    "LiteLLMAdminUser",
    "LiteLLMMasterKeyAuthentication",
    "CONFLICT_ERROR_STATUSES",
    "DEFAULT_ERROR_STATUSES",
    "ErrorResponseModel",
    "ErrorResponseSerializer",
    "build_deprecation_headers",
    "JSON_ERROR_STATUSES",
    "RATE_LIMIT_CONFLICT_ERROR_STATUSES",
    "RATE_LIMIT_ERROR_STATUSES",
    "RATE_LIMIT_JSON_ERROR_STATUSES",
    "TRACE_ID_RESPONSE_HEADER_COMPONENT",
    "TRACE_ID_RESPONSE_HEADER_COMPONENT_NAME",
    "TRACE_ID_RESPONSE_HEADER_REF",
    "default_error_responses",
    "default_extend_schema",
    "default_extend_schema_view",
    "inject_trace_response_header",
    "mark_deprecated_response",
    "tenant_header_components",
    "tenant_headers_parameters",
    "trace_response_headers",
]
