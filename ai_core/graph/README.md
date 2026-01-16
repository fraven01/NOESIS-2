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
  `.ai_core_store/<tenant>/<case>/state.json` using `sanitize_identifier`.

## State & Meta Contract
- Required metadata fields: `tenant_id`, `case_id`, `trace_id`, `graph_name`,
  `graph_version` (default `"v0"`).
- `merge_state(old, incoming)` performs a shallow overwrite: keys from the request body
  replace existing entries; missing keys remain untouched.
- The file layout is tenant/case scoped; corrupted or non-dict payloads raise
  `TypeError` to trigger a 400 response in the views.

## Lifecycle
1. `apps.AiCoreConfig.ready()` imports and executes `graph.bootstrap.bootstrap()`.
2. `bootstrap()` wraps the remaining legacy module (`info_intake`) with
   `module_runner` und registriert die produktiven Graphen
   (`retrieval_augmented_generation`, `crawler.ingestion`).
3. If a runner is absent (e.g. in tests), register a stub runner in the registry
   or override the executor in the test harness.
