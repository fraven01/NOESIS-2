# ID semantics (code-backed reference)

This document describes how IDs are represented in the current codebase and points to the canonical implementations.

## Canonical definitions in code

- Header names: `common/constants.py`
- Scope context model (request/task scope): `ai_core/contracts/scope.py:ScopeContext`
- HTTP request normalization (User Request Hop): `ai_core/ids/http_scope.py:normalize_request`
- Celery task normalization (S2S Hop): `ai_core/ids/http_scope.py:normalize_task_context`
- Graph meta builder: `ai_core/graph/schemas.py:_build_scope_context`
- Trace ID normalization (incl. deprecated alias): `ai_core/ids/headers.py:coerce_trace_id`, `ai_core/ids/contracts.py:normalize_trace_id`
- Case model identifier used by APIs: `cases/models.py:Case.external_id` and `cases/api.py` (`lookup_field = "external_id"`)
- Audit metadata for persistence: `ai_core/contracts/audit_meta.py:audit_meta_from_scope`

## IDs used in this repository

| ID | Where it appears | Notes in code |
| --- | --- | --- |
| `tenant_id` | scope/meta/tool context | Required field in `ScopeContext` (`ai_core/contracts/scope.py`) |
| `case_id` | scope/meta/tool context | Optional at HTTP request level; required for tool invocations and graph meta (`ai_core/graph/schemas.py:normalize_meta`) |
| `workflow_id` | scope/meta/tool context | Accepted as optional in `ScopeContext`; some call paths default it |
| `run_id` | scope/tool context; graph execution | May co-exist with `ingestion_run_id` (e.g., when workflow triggers ingestion) |
| `ingestion_run_id` | scope/tool context; ingestion tasks | May co-exist with `run_id`; ingestion task entrypoints live in `ai_core/tasks.py` |
| `trace_id` | scope/meta/tool context | Normalized/coerced by `ai_core/ids/*`; generated when absent |
| `invocation_id` | scope/tool context | Generated when absent by the normalizers; new per "hop" (HTTP request or Celery task) |
| `user_id` | scope/tool context | User identity for User Request Hops; extracted from Django auth; mutually exclusive with `service_id` |
| `service_id` | scope/tool context | Service identity for S2S Hops (e.g., "celery-ingestion-worker"); mutually exclusive with `user_id` |
| `collection_id` | scope/tool context | UUID-string for scoped operations ("Aktenschrank"); optional |

## Identity IDs (Pre-MVP ID Contract)

`user_id` and `service_id` are mutually exclusive:

| Hop Type | `user_id` | `service_id` | Entry Point |
|----------|-----------|--------------|-------------|
| User Request Hop | REQUIRED (when auth) | ABSENT | `normalize_request()` |
| S2S Hop | ABSENT | REQUIRED | `normalize_task_context()` |
| Public Endpoint | ABSENT | ABSENT | `normalize_request()` |

## Audit Meta (Entity Persistence)

For persisting entities with traceability, use `audit_meta_from_scope()`:

```python
from ai_core.contracts.audit_meta import audit_meta_from_scope

audit_meta = audit_meta_from_scope(scope, created_by_user_id=scope.user_id)
# Returns dict with: trace_id, invocation_id, created_by_user_id, last_hop_service_id, etc.
```

| Key | Description |
|-----|-------------|
| `trace_id` | End-to-end correlation (constant across hops) |
| `invocation_id` | Per-hop identifier |
| `created_by_user_id` | Entity owner (set once at creation, immutable) |
| `initiated_by_user_id` | Who triggered the flow (causal tracking) |
| `last_hop_service_id` | Last service that wrote (from `scope.service_id`) |

Related reference: `docs/architecture/id-guide-for-agents.md`, `docs/architecture/id-contract-one-pager.md`.
