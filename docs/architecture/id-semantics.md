# ID semantics (code-backed reference)

This document describes how IDs are represented in the current codebase and points to the canonical implementations.

## Canonical definitions in code

- Header names: `common/constants.py`
- Scope context model (request/task scope): `ai_core/contracts/scope.py:ScopeContext`
- HTTP request normalization: `ai_core/ids/http_scope.py:normalize_request` and `ai_core/graph/schemas.py:_build_scope_context`
- Trace ID normalization (incl. deprecated alias): `ai_core/ids/headers.py:coerce_trace_id`, `ai_core/ids/contracts.py:normalize_trace_id`
- Case model identifier used by APIs: `cases/models.py:Case.external_id` and `cases/api.py` (`lookup_field = "external_id"`)

## IDs used in this repository

| ID | Where it appears | Notes in code |
| --- | --- | --- |
| `tenant_id` | scope/meta/tool context | Required field in `ScopeContext` (`ai_core/contracts/scope.py`) |
| `case_id` | scope/meta/tool context | Validated for format in `ai_core/ids/headers.py`; required for AI Core graph meta (`ai_core/graph/schemas.py:normalize_meta`) |
| `workflow_id` | scope/meta/tool context | Accepted as optional in `ScopeContext`; some call paths default it (see `ai_core/views.py:_prepare_request` and `ai_core/services/__init__.py:execute_graph`) |
| `run_id` | scope/tool context; graph execution | Mutually exclusive with `ingestion_run_id` in `ScopeContext` and base `ToolContext` |
| `ingestion_run_id` | scope/tool context; ingestion tasks | Mutually exclusive with `run_id`; ingestion task entrypoints live in `ai_core/tasks.py` |
| `trace_id` | scope/meta/tool context | Normalized/coerced by `ai_core/ids/*`; generated when absent |
| `invocation_id` | scope/tool context | Generated when absent by the normalizers (see `ai_core/ids/http_scope.py` and `ai_core/graph/schemas.py`) |

Related reference: `docs/architecture/id-guide-for-agents.md`.
