# Context Propagation Patterns

This document describes the standard patterns for propagating context across
boundaries in NOESIS-2. If code and docs diverge, follow code and update this
document. See `docs/migrations/context-standardization-migration.md` for the
planned standardization steps.

## Canonical models

- `ScopeContext` (infrastructure identity and runtime IDs)
- `BusinessContext` (domain identifiers)
- `ToolContext` (scope + business + runtime metadata)

Rules:

- Business IDs live in `BusinessContext` only.
- At least one runtime ID is required: `run_id` and/or `ingestion_run_id`.
- `user_id` and `service_id` are mutually exclusive; `user_id` is a UUID string.

## Boundary patterns

### HTTP boundary (current)

- `RequestContextMiddleware` builds `request.scope_context`.
- `normalize_meta()` attaches `scope_context`, `business_context`, `tool_context`.
- BusinessContext is built via `build_business_context_from_request()` using the
  same `_coalesce` helpers as `normalize_meta()`.
- Views should use `request.scope_context` + `build_business_context_from_request()`
  when constructing `ToolContext`.

### WebSocket boundary (current)

- Validate payloads with Pydantic models (`extra="forbid"`), forbid `user_pk`,
  and allow hybrid tuning fields explicitly.
- Use `build_websocket_context()` to extract user identity from the ASGI scope.
- Invalid `user.pk` -> `ContextError`, mapped to 403 at the boundary.

### Celery boundary

- Use `ContextTask`/`ScopedTask` to attach context to `meta`.
- Parse with `tool_context_from_meta(meta)` at the task boundary.

### Graph boundary (target)

- Parse context once at graph entry with `tool_context_from_meta(meta)`.
- Store context in graph state and reuse it in nodes.

### Tool/Node boundary (target)

- Signature: `run(context: ToolContext, params: InputModel) -> ToolOutput[...]`.
- Inputs contain functional parameters only; read IDs from `context.business`.

## Code pointers

- Scope model: `ai_core/contracts/scope.py`
- Business model: `ai_core/contracts/business.py`
- ToolContext: `ai_core/tool_contracts/base.py`
- HTTP normalization: `ai_core/graph/schemas.py`
- Graph entry: `ai_core/tool_contracts/base.py:tool_context_from_meta`
- WebSocket helpers: `theme/websocket_utils.py`, `theme/websocket_payloads.py`
