# Tool contract envelopes (code-backed reference)

This document mirrors the current implementation in `ai_core/tool_contracts/base.py`
and `ai_core/tools/errors.py`, and notes the target standard where migration is
still in progress (see `docs/migrations/context-standardization-migration.md`).

## Code locations

- Tool envelopes (`ToolContext`, `ToolResult`, `ToolError`, `ToolOutput`): `ai_core/tool_contracts/base.py`
- Tool error identifiers (`ToolErrorType`): `ai_core/tools/errors.py`
- Scope -> tool context helper: `ai_core/tool_contracts/base.py:tool_context_from_scope`
- Scope context model: `ai_core/contracts/scope.py:ScopeContext`

## `ToolContext`

`ToolContext` is the runtime metadata attached to tool invocations.

Implementation: `ai_core/tool_contracts/base.py:ToolContext`

Selected fields (see code for the full model):

- `scope`: `ScopeContext` (tenant_id, trace_id, invocation_id, run_id/ingestion_run_id,
  user_id/service_id, tenant_schema, idempotency_key, timestamp)
- `business`: `BusinessContext` (case_id, collection_id, workflow_id, thread_id,
  document_id, document_version_id)
- `metadata`: `dict[str, Any]` (default empty mapping)
- `locale`, `timeouts_ms`, `budget_tokens`, `safety_mode`, `auth`,
  `visibility_override_allowed`: runtime hints and permissions

### Import location

- Canonical definition: `ai_core/tool_contracts/base.py:ToolContext`
- Convenience re-export: `ai_core/tool_contracts/__init__.py` re-exports the canonical `ToolContext` (no duplicate model)

## Standard tool/node signature (target)

Use `ToolContext` + Pydantic input/output models. Business IDs come from
`context.business`, not from tool inputs. Legacy `state, meta` node signatures
are not allowed (guardrail test enforces this).

```python
from pydantic import BaseModel, ConfigDict
from ai_core.tool_contracts import ToolContext, ToolOutput

class MyInput(BaseModel):
    query: str
    model_config = ConfigDict(extra="forbid")

class MyOutput(BaseModel):
    result: str

def run(context: ToolContext, params: MyInput) -> ToolOutput[MyInput, MyOutput]:
    case_id = context.business.case_id
    # business logic here
    return ToolOutput(input=params, output=MyOutput(result="ok"))
```

Boundary rules (target):

- Do not include business IDs in tool inputs; read them from `context.business`.
- WebSocket payloads must use Pydantic models with `extra="forbid"`; `user_pk` is not allowed; hybrid tuning fields are explicitly modeled.
- `user_id` must be a UUID string; invalid IDs raise `ContextError` and are mapped to 403 at the boundary.


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
