"""Shared HTTP header and request metadata constants."""

# Canonical header names
X_TENANT_ID_HEADER = "X-Tenant-ID"
X_TENANT_SCHEMA_HEADER = "X-Tenant-Schema"
X_CASE_ID_HEADER = "X-Case-ID"
X_TRACE_ID_HEADER = "X-Trace-ID"
X_KEY_ALIAS_HEADER = "X-Key-Alias"
IDEMPOTENCY_KEY_HEADER = "Idempotency-Key"
X_RETRY_ATTEMPT_HEADER = "X-Retry-Attempt"

# Django request.META keys
META_TENANT_ID_KEY = "HTTP_X_TENANT_ID"
META_TENANT_SCHEMA_KEY = "HTTP_X_TENANT_SCHEMA"
META_CASE_ID_KEY = "HTTP_X_CASE_ID"
META_TRACE_ID_KEY = "HTTP_X_TRACE_ID"
META_KEY_ALIAS_KEY = "HTTP_X_KEY_ALIAS"
META_IDEMPOTENCY_KEY = "HTTP_IDEMPOTENCY_KEY"
META_RETRY_ATTEMPT_KEY = "HTTP_X_RETRY_ATTEMPT"

# Case-insensitive header lookups for task/request propagation
TRACE_ID_HEADER_CANDIDATES = (
    X_TRACE_ID_HEADER,
    "trace_id",
    "trace-id",
)
CASE_ID_HEADER_CANDIDATES = (
    X_CASE_ID_HEADER,
    "case_id",
    "case",
)
TENANT_ID_HEADER_CANDIDATES = (
    X_TENANT_ID_HEADER,
    "tenant",
    "tenant_id",
)
KEY_ALIAS_HEADER_CANDIDATES = (
    X_KEY_ALIAS_HEADER,
    "key_alias",
    "key-alias",
)

HEADER_CANDIDATE_MAP = {
    "trace_id": TRACE_ID_HEADER_CANDIDATES,
    "case_id": CASE_ID_HEADER_CANDIDATES,
    "tenant": TENANT_ID_HEADER_CANDIDATES,
    "key_alias": KEY_ALIAS_HEADER_CANDIDATES,
}
