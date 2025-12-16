# ID handling (code-backed reference)

This document summarizes how IDs are represented and normalized in the current codebase and points to the canonical implementations.

## Canonical header names

Header constants are defined in `common/constants.py`:

- `X-Tenant-ID`, `X-Tenant-Schema`, `X-Case-ID`, `X-Trace-ID`, `X-Workflow-ID`, `X-Collection-ID`, `X-Key-Alias`, `Idempotency-Key`

## Canonical runtime context models

- Scope context (request / task scope): `ai_core/contracts/scope.py:ScopeContext`
- Tool context (tool call envelope): `ai_core/tool_contracts/base.py:ToolContext`

Both models are immutable (`ConfigDict(frozen=True)`).

### Runtime identifier mutual exclusion

Both `ScopeContext` and the base `ToolContext` validate a mutual exclusion constraint:

- exactly one of `run_id` or `ingestion_run_id` is present

Code locations:

- `ai_core/contracts/scope.py:ScopeContext.validate_run_scope`
- `ai_core/tool_contracts/base.py:ToolContext.check_run_ids`

## Where IDs come from (normalizers)

### HTTP request normalization

Two normalizers are used in different call paths:

- Django `HttpRequest` → `ScopeContext`: `ai_core/ids/http_scope.py:normalize_request`
- Generic request objects (incl. DRF requests) → `ScopeContext`: `ai_core/graph/schemas.py:_build_scope_context`

### Trace ID normalization and aliases

Trace ID coercion accepts multiple input keys/aliases and supports the deprecated `request_id` key:

- `ai_core/ids/headers.py:coerce_trace_id`
- `ai_core/ids/contracts.py:normalize_trace_id`

### Case ID format validation

Case IDs are validated by a regex pattern in `ai_core/ids/headers.py` (`_CASE_ID_PATTERN`). The normalizers raise a `ValueError` when the format is invalid.

## AI Core views and graph meta

AI Core request handling builds a meta dictionary and persists key fields on the request:

- Header parsing and tenant enforcement: `ai_core/views.py:_prepare_request`
- Graph meta normalization: `ai_core/graph/schemas.py:normalize_meta`

`normalize_meta` rejects requests without `case_id`.

Graph execution uses:

- Graph protocol + context: `ai_core/graph/core.py` (`GraphRunner`, `GraphContext`)
- Graph execution orchestration: `ai_core/services/__init__.py:execute_graph`

## Case ID vs Case model

The `cases` app uses `Case.external_id` as the stable identifier surfaced via APIs:

- Model: `cases/models.py`
- Resolver: `cases/services.py:resolve_case`
- API viewset uses `lookup_field = "external_id"`: `cases/api.py`

## Collections: UUID vs logical key

Document collections have both:

- a UUID (`collection_id`) used as technical identifier, and
- a logical key/slug used for idempotent lookup/creation.

Code locations that encode this distinction:

- `documents/domain_service.py:CollectionIdConflictError` (conflict when an existing collection key maps to a different UUID)
- `documents/service_facade.py:ingest_document` (prefers `collection_key` over `collection_id` when deciding what to pass downstream)

## Test patterns in this repository

Examples of code that exercises these contracts:

- ToolContext/ScopeContext validation: `ai_core/tests/test_tool_context.py`
- Header normalization behavior: `ai_core/tests/test_ids_headers.py`
- AI Core view validation / trace headers: `ai_core/tests/test_views_min.py`, `tests/test_openapi_contract.py`
