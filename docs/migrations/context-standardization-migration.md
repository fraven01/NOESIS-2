# Context Standardization Migration

This guide covers the migration away from legacy context patterns to the
ToolContext-first standard.

## Scope

The migration standardizes context across HTTP, WebSocket, Celery, graph
boundaries, and tool/node signatures.

## Phase 1 (docs)

- Fix runtime ID docs: `run_id` and/or `ingestion_run_id` (no XOR).
- Add `docs/architecture/context-propagation-patterns.md`.
- Update `docs/agents/tool-contracts.md` with ToolContext + Pydantic I/O standard.

## Phase 2 (helpers)

- Add `ai_core/tool_contracts/validation.py` with:
  - `require_business_field(business, field_name, ...)`
  - `require_runtime_id(scope, prefer=...)`
- Add `theme/websocket_utils.py` with `build_websocket_context()`.
- Add `theme/websocket_payloads.py` for WebSocket payload validation.

## Phase 3 (migration, breaking)

- **Tool/node signatures**:
  - `ai_core/nodes/compose.py` -> `run(context, params)` with Pydantic I/O.
  - Migrate prompt nodes: `assess`, `classify`, `extract`, `needs`, `draft_blocks`,
    plus `ai_core/nodes/_prompt_runner.py`.
- **Graph parsing**:
  - Parse `tool_context_from_meta(meta)` once in RAG graph entry.
- **Views**:
  - Remove `_tenant_context_from_request()` and use `request.scope_context`.
  - Add `build_business_context_from_request(request)` in `ai_core/graph/schemas.py`.
- **Ingestion context**:
  - Refactor `IngestionContextBuilder` to a function without breaking fallback logic.

## Phase 4 (tests)

- Expand `ai_core/tests/test_tool_context_adapter.py` to cover `tool_context_from_meta`
  (missing context, legacy scope fallback, metadata passthrough).
- Add guardrail: `ai_core/tests/test_tool_signature_guardrails.py` (enable after 3.1b).
- Keep boundary coverage in existing normalize meta and WebSocket tests:
  `ai_core/tests/test_normalize_meta.py`,
  `theme/tests/test_websocket_utils.py`,
  `theme/tests/test_websocket_payloads.py`.

## Phase 5 (cleanup)

- Remove deprecated helpers and imports.
- Finalize docs.

## Phase 6 (storage, done)

- Done: add `audit_meta` JSONB to Document and FrameworkProfile.
- Done: introduce `PersistedGraphState` for FileCheckpointer.
- Done: wrap Celery results in `TaskResult` and update consumers.

## Rollback

- Phase 1-2: revert doc/helper commits.
- Phase 3: restore old signatures and `_tenant_context_from_request()`.
- Phase 6: rollback DB migrations and TaskResult format changes.
