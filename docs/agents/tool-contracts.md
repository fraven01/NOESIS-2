# Tool contract envelopes (code-backed reference)

This document mirrors the current implementation in `ai_core/tool_contracts/base.py` and `ai_core/tools/errors.py`.

## Code locations

- Tool envelopes (`ToolContext`, `ToolResult`, `ToolError`, `ToolOutput`): `ai_core/tool_contracts/base.py`
- Tool error identifiers (`ToolErrorType`): `ai_core/tools/errors.py`
- Scope → tool context helper: `ai_core/tool_contracts/base.py:tool_context_from_scope`
- Scope context model: `ai_core/contracts/scope.py:ScopeContext`

## `ToolContext`

`ToolContext` is the runtime metadata attached to tool invocations.

Implementation: `ai_core/tool_contracts/base.py:ToolContext`

Selected fields (see code for the full model):

- `tenant_id`: `UUID | str`
- `trace_id`: `str` (default factory)
- `invocation_id`: `UUID` (default factory)
- `now_iso`: `datetime` (timezone-aware; validator normalizes to UTC)
- `run_id` / `ingestion_run_id`: `str | None` (mutual exclusion validated in `check_run_ids`)
- `workflow_id`, `case_id`, `collection_id`, `document_id`, `document_version_id`: optional strings
- `idempotency_key`, `tenant_schema`: optional strings
- `timeouts_ms`, `budget_tokens`, `locale`: optional runtime hints
- `metadata`: `dict[str, Any]` (default empty mapping)

### Import location

- Canonical definition: `ai_core/tool_contracts/base.py:ToolContext`
- Convenience re-export: `ai_core/tool_contracts/__init__.py` re-exports the canonical `ToolContext` (no duplicate model)

## `ToolResult`, `ToolError`, `ToolOutput`

Implementation: `ai_core/tool_contracts/base.py`

- `ToolResult[IT, OT]` and `ToolError[IT]` are the two envelopes.
- `ToolOutput[IT, OT]` is a discriminated union over `status` (`"ok"` vs `"error"`).
- Both envelopes carry `meta.took_ms` (`ToolResultMeta` / `ToolErrorMeta`).
- `ToolResultMeta` includes optional fields such as `routing`, `token_usage`, `cache_hit`, `source_counts`.

## HTTP error responses

HTTP APIs in `ai_core`/`theme` return the `ToolError` envelope for error cases.
Use `ai_core/infra/resp.py:build_tool_error_payload` to build the payload and
return it via `Response`/`JsonResponse`. This keeps error responses aligned with
`ToolErrorType` and the shared contract.

## `ToolErrorType`

`ToolErrorDetail.type` is a `StrEnum` defined in `ai_core/tools/errors.py:ToolErrorType`:

- `RATE_LIMIT`
- `TIMEOUT`
- `UPSTREAM`
- `VALIDATION`
- `RETRYABLE`
- `FATAL`

## JSON schema export

All Pydantic models support `model_json_schema()`; this can be used to export JSON schema for the envelopes.
