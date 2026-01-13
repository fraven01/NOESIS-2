# Pre-MVP ID Contract Review Checklist

This document provides a functional checklist to verify that the Pre-MVP ID Contract is correctly implemented across the codebase.

## 1. Core Contract Models

### ScopeContext (`ai_core/contracts/scope.py`)

- [ ] `tenant_id` is mandatory
- [ ] `trace_id` is mandatory
- [ ] `invocation_id` is mandatory
- [ ] `run_id` and `ingestion_run_id` may co-exist (no XOR validator)
- [ ] At least one of `run_id` or `ingestion_run_id` is required
- [ ] `user_id` and `service_id` fields exist
- [ ] `user_id` is a UUID string when present
- [ ] `user_id` and `service_id` are mutually exclusive (validator present)
- [ ] Business IDs are not present in ScopeContext (`case_id`, `collection_id`, etc.)

**Verification Command:**
```bash
npm run win:test:py:unit -- ai_core/tests/test_tool_context.py -v
```

### ToolContext (`ai_core/tool_contracts/base.py`)

- [ ] Inherits identity fields (`user_id`, `service_id`) from scope
- [ ] At least one runtime ID required (no XOR)
- [ ] `user_id` and `service_id` are mutually exclusive
- [ ] `tool_context_from_scope()` propagates all fields correctly

**Verification Command:**
```bash
npm run win:test:py:unit -- ai_core/tests/test_tool_context_adapter.py -v
```

### AuditMeta (`ai_core/contracts/audit_meta.py`)

- [ ] Model exists with fields: `trace_id`, `invocation_id`, `created_by_user_id`, `initiated_by_user_id`, `last_hop_service_id`
- [ ] `audit_meta_from_scope()` function exists and builds correctly
- [ ] `build_audit_meta()` helper exists for inline construction

**Verification Command:**
```bash
npm run win:test:py:unit -- ai_core/contracts/ -v
```

---

## 2. HTTP Request Normalization

### normalize_request() (`ai_core/ids/http_scope.py`)

- [ ] Extracts `user_id` from Django `request.user` when authenticated
- [ ] Normalizes `user_id` to UUID string (rejects non-UUID values)
- [ ] Sets `service_id = None` (HTTP = User Request Hop)
- [ ] Generates `run_id` if neither `run_id` nor `ingestion_run_id` provided
- [ ] Does NOT require `case_id` (business IDs extracted in `normalize_meta`)

**Verification Command:**
```bash
npm run win:test:py:unit -- ai_core/ids/tests/test_http_scope.py -v
```

### normalize_task_context() (`ai_core/ids/http_scope.py`)

- [ ] Requires `service_id` parameter (raises ValueError if missing)
- [ ] Sets `user_id = None` (S2S Hop = no user)
- [ ] Does not accept business IDs; build BusinessContext separately
- [ ] Generates IDs for missing trace_id, invocation_id, run_id

**Verification:** Check function signature and docstring.

---

## 3. Middleware

### RequestContextMiddleware (`ai_core/middleware/context.py`)

- [ ] Calls `normalize_request()` correctly
- [ ] Attaches `scope_context` to request
- [ ] Handles missing `case_id` without error (400 only for invalid format)
- [ ] Accepts both `run_id` and `ingestion_run_id` simultaneously

**Verification Command:**
```bash
npm run win:test:py:unit -- ai_core/tests/test_request_context_middleware.py -v
```

---

## 4. Graph Execution

### Graph Meta Normalization (`ai_core/graph/schemas.py`)

- [ ] Builds `ScopeContext` with identity fields
- [ ] Propagates `user_id` or `service_id` appropriately
- [ ] Does not require `case_id` globally; graphs validate required business IDs

### Service Entry Points (`ai_core/services/__init__.py`)

- [ ] `execute_graph()` builds scope with correct identity
- [ ] Task dispatch to Celery includes `service_id`

**Verification Command:**
```bash
npm run win:test:py:unit -- ai_core/tests/graphs/ -v
```

---

## 5. Celery Tasks

### Task Implementations (`ai_core/tasks/`)

For each Celery task that creates entities:

- [ ] Uses `normalize_task_context()` with explicit `service_id`
- [ ] Builds `audit_meta` using `audit_meta_from_scope()`
- [ ] Propagates `trace_id` from parent hop
- [ ] Propagates `initiated_by_user_id` explicitly in task meta
- [ ] Generates new `invocation_id` for the task hop

**Recommended Service IDs:**
| Task | service_id |
|------|------------|
| `run_ingestion_graph` | `"celery-ingestion-worker"` |
| `run_graph` (llm_worker) | `"celery-agents-worker"` |
| Crawler tasks | `"crawler-worker"` |

---

## 6. Entity Persistence

### Document Domain Service (`documents/domain_service.py`)

- [ ] Persists `audit_meta` on Document creation
- [ ] Sets `created_by_user_id` from scope (immutable after creation)
- [ ] Tracks `last_hop_service_id` from `scope.service_id`

### Framework Profile Persistence (`ai_core/graphs/business/framework_analysis_graph.py`)

- [ ] Persists `audit_meta` on FrameworkProfile creation
- [ ] Uses `audit_meta_from_scope()` in `persist_profile` node

---

## 7. API Endpoints

### Case API (`cases/`)

- [ ] Case creation works without `case_id` header (it's the result, not input)
- [ ] Case listing/retrieval works with tenant isolation

**Verification Command:**
```bash
npm run win:test:py:unit -- cases/tests/test_api.py -v
```

### AI Core Views (`ai_core/views.py`)

- [ ] RAG ingestion accepts optional `case_id`
- [ ] Graph execution validates required business IDs per graph

**Verification Command:**
```bash
npm run win:test:py:unit -- ai_core/tests/test_rag_ingestion_run.py -v
```

---

## 8. Documentation

### AGENTS.md (Root)

- [ ] Documents identity ID pattern (user_id/service_id)
- [ ] Documents run_id + ingestion_run_id co-existence
- [ ] References `normalize_task_context()` for S2S hops
- [ ] References `audit_meta_from_scope()` for persistence

### docs/architecture/id-semantics.md

- [ ] ID table includes `user_id` and `service_id`
- [ ] Audit Meta section with field descriptions
- [ ] Identity hop type table

### ai_core/graph/README.md

- [ ] Context & Identity section present
- [ ] `audit_meta_from_scope()` usage example

---

## 9. Full Test Suite

Run the complete unit test suite to verify no regressions:

```bash
npm run win:test:py:unit
```

Expected: All tests pass (1000+ passed, 0 failed)

---

## Quick Smoke Test Commands

```bash
# Core contracts
npm run win:test:py:unit -- ai_core/contracts/ ai_core/tool_contracts/ -v

# ID normalization
npm run win:test:py:unit -- ai_core/ids/tests/ -v

# Middleware
npm run win:test:py:unit -- ai_core/tests/test_request_context_middleware.py -v

# API endpoints
npm run win:test:py:unit -- cases/tests/test_api.py ai_core/tests/test_rag_ingestion_run.py -v

# Full unit suite
npm run win:test:py:unit
```

---

## Implementation Gap Analysis

If any check fails, refer to:

1. **Contract One-Pager**: `docs/architecture/id-contract-one-pager.md`
2. **ID Semantics**: `docs/architecture/id-semantics.md`
3. **Implementation Guide**: `docs/architecture/id-guide-for-agents.md`

---

## Sign-Off

| Area | Reviewer | Date | Status |
|------|----------|------|--------|
| Core Contracts | | | |
| HTTP Normalization | | | |
| Middleware | | | |
| Graph Execution | | | |
| Celery Tasks | | | |
| Entity Persistence | | | |
| API Endpoints | | | |
| Documentation | | | |
