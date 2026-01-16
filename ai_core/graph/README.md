# Graph Runtime Facade

This package hosts the lightweight runtime shim that bridges the legacy `run(state, meta)`
modules to a consistent execution contract.

## Components
- **Registry** - `register(name, runner)` and `get(name)` store `GraphRunner` instances in
  memory. Graphs are registered at startup via `graph.bootstrap.bootstrap()`, and tests can
  patch the registry directly when they need custom runners.
- **GraphRunner protocol** - every runner exposes `run(state: dict, meta: dict) -> (dict, dict)`
  and must return the next persisted state alongside the HTTP payload.
- **Checkpointer** - the default `FileCheckpointer` persists JSON snapshots under
  `.ai_core_store/<tenant>/workflow-executions/<workflow_id>/<run_id>/state.json`
  (or `.ai_core_store/<tenant>/workflow-executions/<plan_key>/state.json` when a derived
  `plan_key` is provided) using `sanitize_identifier`. The thread-aware variant stores
  under `.ai_core_store/<tenant>/threads/<thread_id>/state.json`.

## State & Meta Contract
- Required metadata fields: `tenant_id`, `trace_id`, `graph_name`, `graph_version`
  (default `"v0"`), plus `workflow_id` for persisted graph executions.
- Checkpointing requires `workflow_id` plus `run_id`/`ingestion_run_id`, unless a derived
  `plan_key` is supplied in context metadata (`workflow_execution` scope).
- `merge_state(old, incoming)` performs a shallow overwrite: keys from the request body
  replace existing entries; missing keys remain untouched.
- The file layout is tenant/workflow_execution scoped (or thread-scoped when `thread_id`
  is present); corrupted or non-dict payloads raise `TypeError` to trigger a 400 response
  in the views.

## Lifecycle
1. `apps.AiCoreConfig.ready()` imports and executes `graph.bootstrap.bootstrap()`.
2. `bootstrap()` wraps the remaining legacy module (`info_intake`) with
   `module_runner` und registriert die produktiven Graphen
   (`retrieval_augmented_generation`, `crawler.ingestion`).
3. If a runner is absent (e.g. in tests), register a stub runner in the registry
   or override the executor in the test harness.
