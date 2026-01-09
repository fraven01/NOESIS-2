# User Identity Cleanup (Pre-MVP, Clean-State)

Goal: Standardize user identity on UUID primary keys and propagate user attribution consistently across sync + async paths.

## Decisions (confirmed)

- User primary key is UUID; all user_id fields use UUID strings.
- Telemetry uses distinct fields: tenant_id for tenant, user_id for user.
- initiated_by_user_id is propagated explicitly in async tasks (audit meta only).
- Migration note: `users/migrations/0005_user_id_uuid_fix.py` truncates `users_user`
  with CASCADE when `id` is still bigint to allow a clean UUID PK switch (pre-MVP reset).

## Work Items

### 1) Contract + normalization alignment (BREAKING) — Done

Pointers:
- `ai_core/contracts/scope.py`
- `ai_core/ids/http_scope.py`
- `ai_core/tool_contracts/base.py`
- `docs/architecture/id-semantics.md`
- `docs/architecture/id-contract-review-checklist.md`

Acceptance:
- ScopeContext user_id documented/validated as UUID string.
- normalize_request extracts UUID user_id (no implicit int coercion).
- No documentation references int user_id or mixed types.

### 2) Persistence lookups use UUID (BREAKING) — Done

Pointers:
- `ai_core/adapters/db_documents_repository.py`
- `documents/*` (any other user lookups in audit paths)

Acceptance:
- No int() casts for user_id in persistence paths.
- User lookups accept UUID primary keys directly.
- Tests cover user attribution with UUID PK.

### 3) Telemetry: separate tenant_id vs user_id (BREAKING) — Done

Pointers:
- `ai_core/commands/graph_execution.py`
- `ai_core/ingestion_orchestration.py`
- `ai_core/llm/client.py`
- `ai_core/infra/observability.py`

Acceptance:
- Telemetry uses tenant_id and user_id as separate fields.
- No call sites pass tenant_id into user_id parameter.
- Traces/observability docs updated.

### 4) initiated_by_user_id propagation (BREAKING) — Done

Pointers:
- `ai_core/tasks.py`
- `llm_worker/tasks.py`
- `documents/upload_worker.py`
- `crawler/worker.py`
- `ai_core/contracts/audit_meta.py`

Acceptance:
- Async tasks propagate initiated_by_user_id explicitly in meta.
- audit_meta_from_scope used for persistence, with initiated_by_user_id passed through.
- ScopeContext remains free of initiated_by_user_id.

## Guardrails

- Add tests that fail when user_id is non-UUID.
- Add lint/rg check for `int(user_id)` in persistence paths.
