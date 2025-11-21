# Context Unification Refactoring Plan

## high_level_strategy
- Establish `ScopeContext` as the upstream canonical contract and derive downstream shapes from it without changing the middleware entry points or LangGraph boundaries.
- Preserve the existing architecture by layering adapters: convert HTTP headers → `ScopeContext`, then project into rich `ToolContext` and graph meta to minimise code churn.
- Gradually retire ad-hoc context builders by redirecting them to shared helpers so invariants (tenant/trace/run vs. ingestion run) are enforced once.
- Keep lean ToolContext and graph meta stable for callers while backfilling the missing fields via derived data, ensuring backward-compatible defaults.
- Use documentation and typing breadcrumbs to make the canonical shapes discoverable for future coding agents.

## canonical_shapes
- `ScopeContext` (`ai_core/contracts/scope.py`): canonical request/runtime scope with XOR `run_id`/`ingestion_run_id`, plus trace, invocation, tenant, case, workflow, idempotency, timestamp.
- Rich `ToolContext` (`ai_core/tool_contracts/base.py`): tool-facing derivative that extends `ScopeContext` fields with locale/budget/auth/timeouts.
- Normalized graph meta (`ai_core/graph/schemas.py::normalize_meta`): graph-facing derivative that should embed a serialised `ScopeContext` plus graph name/version and rate limit info.

## tasks
- title: Introduce ScopeContext-to-ToolContext adapter helpers
  rationale: Gives coding agents a single, documented path from canonical scope to tool invocations, avoiding manual field mapping and duplicative XOR checks.
  scope_of_changes: `ai_core/tool_contracts/base.py` (helper module or functions), `ai_core/contracts/scope.py` (factory/bridge), `ai_core/tests/` (new helper coverage), docs in `docs/agents/tool-contracts.md`.
  constraints: Do not alter existing ToolContext fields or validation; keep public API compatible; avoid touching middleware behavior.
  acceptance_criteria: Helper converts `ScopeContext` → rich `ToolContext` with preserved IDs and passes tests verifying XOR enforcement and field propagation.

- title: Expose ScopeContext on graph meta and serialize consistently
  rationale: Embedding the canonical scope in graph meta reduces divergent shapes and makes required IDs discoverable for graph authors.
  scope_of_changes: `ai_core/graph/schemas.py` (normalize_meta to accept/attach ScopeContext), graph entry tests under `ai_core/tests/graph/`, docstring/README updates in `ai_core/graphs/README.md`.
  constraints: Maintain existing meta keys for callers; keep lean ToolContext payload available; avoid breaking current graph inputs.
  acceptance_criteria: Graph meta includes serialized ScopeContext alongside existing keys; normalize_meta unit tests assert presence and correctness without removing prior keys.

- title: Align CollectionSearchGraph context payloads with canonical scope
  rationale: Removes parallel `_GraphIds` derivation and keeps run/ingestion/workflow IDs consistent with middleware-derived scope for ingestion flows.
  scope_of_changes: `ai_core/graphs/collection_search.py` (GraphContextPayload/_prepare_ids), related tests, any ingestion graph fixtures consuming meta_state.
  constraints: Preserve current behavior for missing IDs (auto-generation) while sourcing values from ScopeContext when available; do not modify business logic.
  acceptance_criteria: GraphContextPayload accepts/derives ScopeContext-compatible fields; tests confirm run_id/ingestion_run_id/trace_id alignment and no regression in auto-generation defaults.

- title: Deprecate lean ToolContext in favor of canonical derivatives
  rationale: Reduces parallel context shapes by routing lean usages through adapters that enrich or validate against ScopeContext before serialization.
  scope_of_changes: `ai_core/tool_contracts/__init__.py` (deprecation notice/adapter), graph helpers in `ai_core/graph/schemas.py` and `ai_core/graphs/retrieval_augmented_generation.py` to consume adapters, documentation updates in `docs/agents/tool-contracts.md`.
  constraints: Keep lean ToolContext model available for backward compatibility; avoid breaking serialized meta shape expected by clients; no broad refactors beyond adapter wiring.
  acceptance_criteria: Adapter path documented; lean ToolContext creation in graph helpers uses canonical adapter; tests cover adapter behavior and ensure existing fields remain intact.

- title: Centralize header/id normalization for ScopeContext
  rationale: Ensures one source of truth for tenant/case/trace/idempotency/run-id parsing, lowering drift between middleware and downstream builders.
  scope_of_changes: `ai_core/middleware/context.py` (delegate to shared normalization), `ai_core/ids/` helpers (shared functions), associated middleware tests, doc references in `docs/llm_contract_readiness.md`.
  constraints: Maintain middleware response headers and error semantics; avoid changes to TenantContext resolution; no API surface changes for callers.
  acceptance_criteria: Middleware builds ScopeContext via shared normalization helpers; tests assert identical behavior for header combinations and XOR validation; documentation references updated helper.

- title: Document canonical context flow for agents
  rationale: Improves discoverability by describing the end-to-end path from HTTP headers through ScopeContext to ToolContext and graph meta.
  scope_of_changes: `docs/agents/overview.md` or adjacent docs, add section referencing ScopeContext and adapters; ensure cross-links from `AGENTS.md` remain valid.
  constraints: No behavior changes; keep terminology consistent with Glossar; avoid editing unrelated sections.
  acceptance_criteria: Documentation section exists with clear mapping of context shapes and invariants; links pass markdown lint if present; reviewers can trace context propagation without code spelunking.
