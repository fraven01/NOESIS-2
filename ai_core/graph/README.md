# Graph Runtime Facade

This package hosts the lightweight runtime shim that bridges the legacy `run(state, meta)`
modules to a consistent execution contract.

## Components
- **Registry** – `register(name, runner)` and `get(name)` store `GraphRunner` instances in
  memory. `get_graph()` inside the views lazily registers module-based runners on first use,
  so patched runners in tests can still be injected.
- **GraphRunner protocol** – every runner exposes `run(state: dict, meta: dict) -> (dict, dict)`
  and must return the next persisted state alongside the HTTP payload.
- **Checkpointer** – the default `FileCheckpointer` persists JSON snapshots under
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
2. `bootstrap()` wraps the four legacy modules (`info_intake`, `scope_check`,
   `needs_mapping`, `system_description`) with `module_runner` and registers them.
3. If a runner is absent (e.g. in tests), `_GraphView.get_graph()` performs
   on-demand registration before executing the request.
