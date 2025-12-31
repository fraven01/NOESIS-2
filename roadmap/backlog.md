# Backlog (prioritized)

This is a curated list of open work items for vibe-coding execution.
Top-to-bottom order within each section is priority order.
Prefer linking each item to concrete code paths (and optionally to an issue).

## Next up (highest leverage)

- [x] Drift #3: crawler ingestion persistence parity with upload path (pointers: `docs/architecture/drift3-critical-finding.md`, `crawler/worker.py`, `ai_core/tasks.py:run_ingestion_graph`, `ai_core/graphs/technical/universal_ingestion_graph.py`; acceptance: crawler path can persist/create document without relying on "parallel registration" in `crawler/worker.py`)
- [x] Workbench exception: allow UI/views to trigger technical graphs only in `DEBUG` (explicit guard at endpoint level; e.g. `ai_core/views.py:_GraphView`, `ai_core/views.py:CrawlerIngestionRunnerView`)
- [ ] Graph convergence: refactor `retrieval_augmented_generation.py` to real `langgraph` and migrate module-level singleton to factory (inventory `ai_core/graphs/technical/retrieval_augmented_generation.py`)
- [ ] Deprecation plan: `info_intake` graph + intake endpoint (`ai_core/graphs/technical/info_intake.py`, `ai_core/views.py:IntakeViewV1`, `docs/api/reference.md`)
- [ ] Review later: framework analysis graph convergence (`ai_core/graphs/business/framework_analysis_graph.py`)

## Code Quality & Architecture Cleanup (Pre-MVP Refactoring)

_Based on architectural analysis 2025-12-31. Pre-MVP; breaking changes and test updates are acceptable if they reduce agent friction and clarify contracts._

### P0 - Critical Quick Wins (High Impact, Medium-High Effort)

- [x] **ToolContext-First Context Access** BREAKING: Replace manual dict unpacking with typed `ToolContext` parsed from `meta["tool_context"]` (pointers: **50+ locations** in `ai_core/services/__init__.py:808-816,1251-1254,1626-1627`, `ai_core/services/crawler_runner.py:63-67,80-83,122-123`, `theme/views.py:934-947,1164-1176,1443-1452`; **Effort: HIGH** due to breadth of changes; implementation: add a helper like `tool_context_from_meta(meta)` or use `ToolContext.model_validate(meta["tool_context"])`, then use `context.scope.*` / `context.business.*`; acceptance: no direct `meta["scope_context"]` / `meta["business_context"]` access in application code outside boundary/normalization helpers, type-safe context use, tests updated; implementation: helper added + call sites migrated in `ai_core/services/__init__.py`, `ai_core/tasks.py`, `ai_core/services/crawler_runner.py`, `ai_core/services/crawler_state_builder.py`, `ai_core/views.py`, `ai_core/views_framework.py`, `ai_core/infra/resp.py`, `ai_core/views_response_utils.py`, `ai_core/graphs/technical/info_intake.py`, `ai_core/graphs/technical/retrieval_augmented_generation.py`, `documents/tasks.py`, `documents/upload_manager.py`, `documents/upload_worker.py`, `crawler/manager.py`, `crawler/tasks.py`, `llm_worker/tasks.py`, `llm_worker/graphs/hybrid_search_and_score.py`; tests pending)

- [ ] **Kill JSON Normalization Boilerplate** BREAKING: Remove `_make_json_safe()` and use Pydantic JSON serialization for mixed payloads (pointers: `ai_core/services/__init__.py:145-188` (43 lines!); implementation: replace calls with `TypeAdapter(Any).dump_python(value, mode="json")` or `pydantic_core.to_jsonable_python` to handle dataclasses/UUIDs/sets, update tests; acceptance: `_make_json_safe()` deleted, consistent JSON serialization, tests updated)

- [ ] **Standardize Error Handling** BREAKING: Converge on existing tool error contracts (pointers: 395 error sites across 81 files, 4 different patterns: typed exceptions in `ai_core/tool_contracts/__init__.py`, Pydantic ValidationError, custom graph exceptions, string-based codes in `theme/views.py`; implementation: define a single error-to-response mapping that emits `ToolErrorType` codes, translate ValidationError/custom exceptions at boundaries; acceptance: one error contract in responses, no string-based error returns, tests updated)

- [ ] **Fix Logging Chaos** BREAKING: Standardize logging strategy across codebase (pointers: `theme/views.py:1214` (**print() in production code!**), mixed `logger.info/warning/exception` without context, inconsistent structured logging; **Critical**: production `print()` statements are bugs, not just observability issues; implementation: enforce structured logging via `extra={}` everywhere, remove ALL print() statements, create logging standards doc `docs/observability/logging-standards.md`; acceptance: no print() in production code, all logs have `tenant_id/trace_id/invocation_id`, consistent log levels)

### Docs/Test touchpoints (checklist)

- [ ] ToolContext-First Context Access (owner: ai-core+theme; docs: `docs/audit/architecture-anti-patterns-2025-12-31.md`; tests: `ai_core/tests/test_graphs.py`, `ai_core/tests/test_meta_normalization.py`, `ai_core/tests/test_normalize_meta.py`, `ai_core/tests/test_graph_retrieval_augmented_generation.py`, `ai_core/tests/test_graph_worker_timeout.py`, `ai_core/tests/test_services_observability.py`, `ai_core/tests/test_ingestion_view.py`, `ai_core/tests/test_views.py`, `ai_core/tests/test_infra.py`, `ai_core/tests/test_repro_upload.py`, `ai_core/tests/test_tasks_embed_observability.py`; acceptance notes: context reads come from `meta["tool_context"]` only)
- [ ] Kill JSON Normalization Boilerplate (owner: ai-core; docs: `docs/audit/architecture-anti-patterns-2025-12-31.md`; tests: `ai_core/tests/test_services_json_safe.py`; acceptance notes: `_make_json_safe` removed, JSON serialization uses Pydantic adapters)
- [ ] Standardize Error Handling (owner: ai-core+theme; docs: `docs/agents/tool-contracts.md`, `docs/audit/architecture-anti-patterns-2025-12-31.md`; tests: `ai_core/tests/test_views.py`, `ai_core/tests/test_views_min.py`, `ai_core/tests/views/test_ingestion_run.py`, `ai_core/tests/test_rag_ingestion_run.py`, `ai_core/tests/test_rag_ingestion_status.py`; acceptance notes: responses emit `ToolErrorType` codes)
- [ ] Eliminate Pass-Through Glue Functions (owner: theme; docs: `docs/audit/architecture-anti-patterns-2025-12-31.md`; tests: `theme/tests/test_rag_tools_view.py`, `theme/tests/test_rag_tools_simulation.py`, `theme/tests/test_document_space_view.py`, `theme/tests/test_chat_submit_global_search.py`, `theme/tests/test_admin_domain_dropdown.py`, `ai_core/tests/test_views.py`, `ai_core/tests/test_views_min.py`; acceptance notes: query params validated via Pydantic models)
- [ ] Normalize the Normalizers (owner: theme+ai-core; docs: `docs/audit/architecture-anti-patterns-2025-12-31.md`; tests: `theme/tests/test_rag_tools_view.py`, `theme/tests/test_rag_tools_simulation.py`, `ai_core/tests/test_nodes.py`, `ai_core/tests/test_views.py`; acceptance notes: duplicate normalizers consolidated without behavior drift)
- [ ] Remove Fake Abstractions (owner: ai-core+documents; docs: `docs/audit/architecture-anti-patterns-2025-12-31.md`; tests: `ai_core/tests/test_services_observability.py`, `ai_core/tests/test_llm.py`, `documents/tests/test_pipeline.py`, `documents/tests/test_processing_graph_integration.py`, `documents/tests/test_processing_graph_e2e.py`; acceptance notes: direct usage or real DI, no wrapper-only classes)
- [ ] Fix Logging Chaos (owner: ai-core+theme; docs: `docs/audit/architecture-anti-patterns-2025-12-31.md`, `docs/observability/logging-standards.md` (new); tests: `ai_core/tests/test_services_observability.py`, `ai_core/tests/test_views.py`, `ai_core/tests/services/test_crawler_runner.py`, `theme/tests/test_rag_tools_view.py`; acceptance notes: no print statements, structured logging with context)
- [ ] Break Up God Files (owner: ai-core+theme; docs: `docs/audit/architecture-anti-patterns-2025-12-31.md`; tests: `ai_core/tests/conftest.py`, `ai_core/tests/test_views.py`, `ai_core/tests/test_ingestion_view.py`, `ai_core/tests/test_repro_upload.py`, `ai_core/tests/services/test_crawler_runner.py`, `theme/tests/test_rag_tools_view.py`, `theme/tests/test_document_space_view.py`, `theme/tests/test_chat_submit_global_search.py`; acceptance notes: modules split, imports updated, no file > 500 lines)
- [ ] Targeted Domain Enrichment (owner: ai-core; docs: `docs/audit/architecture-anti-patterns-2025-12-31.md`; tests: `ai_core/tests/tools/test_framework_contracts.py`, `ai_core/tests/graphs/test_framework_analysis_graph.py`; acceptance notes: domain methods replace duplicated logic)
- [ ] Service Layer Refactoring (owner: ai-core; docs: `docs/audit/architecture-anti-patterns-2025-12-31.md`; tests: `ai_core/tests/test_graph_worker_timeout.py`, `ai_core/tests/test_services_observability.py`, `ai_core/tests/test_ingestion_view.py`, `ai_core/tests/test_repro_upload.py`, `ai_core/tests/services/test_crawler_runner.py`; acceptance notes: command objects used, `execute_graph` <= 100 lines)
- [ ] State Dict -> Dataclasses (owner: ai-core; docs: `docs/audit/architecture-anti-patterns-2025-12-31.md`; tests: `ai_core/tests/services/test_crawler_runner.py`, `ai_core/tests/test_services_observability.py`; acceptance notes: structured state uses dataclasses/Pydantic)

### P1 - High Value Cleanups (Low-Medium Effort)

- [ ] **Eliminate Pass-Through Glue Functions**: Remove trivial helpers in favor of Pydantic validators (pointers: `theme/views.py:205-217` (_parse_bool, 12 lines), `:197-202` (_parse_limit), `:101-115` (_extract_user_id), `:220-235` (_human_readable_bytes); implementation: create `theme/validators.py` with Pydantic models for query params, move validation logic to `@field_validator`, inline formatting helpers or move to templates; note: exclude helpers that encapsulate IO or domain lookups (relocate instead of delete); acceptance: 10+ private helper functions removed, validation in Pydantic models, tests passing)

- [ ] **Normalize the Normalizers** BREAKING: Consolidate duplicate conversion functions without changing behavior (pointers: `theme/views.py:264` (_normalise_quality_mode), `:271` (_normalise_max_candidates); implementation: create `common/validators.py` with reusable Pydantic validators (e.g. `@field_validator` for strip/lower, clamp int), replace only identical normalize_* functions; acceptance: <10 distinct normalization patterns remain, shared validators, no behavior drift, tests updated)

- [ ] **Remove Fake Abstractions**: Delete ceremonial wrappers without value (pointers: `ai_core/services/__init__.py:307-326` (DocumentComponents - fake builder pattern), `:128-138` (_LedgerShim - fake strategy pattern); implementation: replace DocumentComponents with direct imports, replace_LedgerShim with proper dependency injection or remove if unused; acceptance: boilerplate classes deleted, direct usage, tests passing)

### P2 - Long-term Improvements (High Effort)

- [ ] **Break Up God Files** BREAKING: Split 2000+ line monoliths into focused modules (pointers: `ai_core/services/__init__.py` (2,034 lines: `execute_graph` L776-1231, `start_ingestion_run` L1234-1367, `handle_document_upload` L1577-2033), `theme/views.py` (2,055 lines); implementation: split into `ai_core/services/{graph_executor,ingestion,document_upload,components}.py`, split into `theme/views/{rag_tools,search,documents,validators}.py`; acceptance: no file >500 lines, clear module boundaries, imports updated, tests passing)

- [ ] **Targeted Domain Enrichment** BREAKING: Add behavior only where it removes duplication or clarifies invariants (pointers: `ai_core/tools/framework_contracts.py:16-23` (TypeEvidence), `:26-33` (AlternativeType), `:36-42` (ScopeIndicators), `:74-92` (ComponentLocation) - all frozen dataclasses with no methods; implementation: add small, local methods (e.g. `TypeEvidence.is_strong`, `TypeEvidence.merge_with()`), move duplicated logic out of services; acceptance: domain methods replace repeated logic, service layer thinner, tests updated)

- [ ] **Service Layer Refactoring**: Replace procedural god-functions with Command Pattern (pointers: `ai_core/services/__init__.py:execute_graph` (455 lines of procedural logic); implementation: create `ai_core/commands/graph_execution.py` with `GraphExecutionCommand` class encapsulating validation/execution/recording, use factory pattern for graph runners; acceptance: execute_graph <100 lines, commands testable in isolation, clear separation of concerns)

### Observability Cleanup

_Note: Critical logging issues (print() in production) moved to P0. Remaining observability work stays here._

### Hygiene

- [ ] **State Dict -> Dataclasses**: Replace manual dict building with typed dataclasses (pointers: `ai_core/services/crawler_runner.py:298-314` (synthesized_state, entry dicts built manually); implementation: create typed dataclasses for common state shapes, use Pydantic models for serialization; acceptance: no manual dict building for structured data, type-safe state management)

## Semantics / IDs

- [x] Semantics vNext: keep `case_id` required; standardize defaults (`general` for API views, `dev-case-local` for `/rag-tools`) and ensure that case exists per tenant (`ai_core/graph/schemas.py:normalize_meta`, `ai_core/views.py`, `theme/views.py`, `cases/*`)
- [ ] Hard-break ScopeContext business IDs: reject business identifiers in `ScopeContext` (pointers: `ai_core/contracts/scope.py:ScopeContext.forbid_business_ids`, `ai_core/tests/test_scope_context.py`; acceptance: ScopeContext validation raises when `case_id|collection_id|workflow_id|document_id|document_version_id` appear in input)
- [ ] ID propagation clarity: document and/or enforce the "runtime context injection" pattern for graphs (pointers: `docs/audit/contract_divergence_report.md#id-propagation-compliance`, `ai_core/graphs/upload_ingestion_graph.py`, `ai_core/graph/schemas.py`, `ai_core/contracts/scope.py`; acceptance: one canonical pattern, documented in code/doc pointers, so tenant/trace/run IDs are not "implicit knowledge")
- [x] Breaking meta contract (Contract Definition): enforce `scope_context` as the only ID source in graph/tool meta at contract level (pointers: `ai_core/graph/schemas.py:normalize_meta`, `ai_core/views.py:_prepare_request`, `llm_worker/runner.py:submit_worker_task`, `common/celery.py:ContextTask._from_meta`; acceptance: contract enforces `meta["scope_context"]` structure)
- [ ] Breaking meta contract (Implementation): migrate all 50+ dict unpacking call sites to use typed ToolContext (tracked as P0 "ToolContext-First Context Access"; acceptance: no direct `meta["scope_context"]` / `meta["business_context"]` access in application code)

## Layering / boundaries

- [ ] Graph package split: business vs technical graphs (finish relocation of remaining root graphs like `ai_core/graphs/info_intake.py`)
- [x] Enforce import direction (business -> technical) via a lightweight repo test (no new dependencies)
- [ ] Capability-first rule: graphs call explicit capabilities (`ai_core/nodes/`, `ai_core/tools/`, domain services) instead of embedding ad-hoc logic (TODOs: `roadmap/capability-first-todos.md`)

## Externalization readiness (later)

- [ ] Contract drift (defer until externalization): migrate `ai_core/rag/schemas.py:Chunk` off `@dataclass` (pointers: `docs/audit/contract_divergence_report.md#1-chunk-schema-non-compliance`, `ai_core/rag/schemas.py`, `ai_core/rag/vector_client.py` (mutating `chunk.meta`), `ai_core/api.py`, `ai_core/tasks.py`; rationale: RAG path stable, high call-site/mutation cost; revisit when external interfaces harden; acceptance: a Pydantic `ChunkV2` (frozen) exists, call sites migrated, legacy kept only behind explicit adapter until removed)
- [ ] Technical graph interface: versioned Pydantic I/O (avoid implicit state dicts) for externalization readiness
- [ ] Graph executor boundary: local vs celery vs remote execution (business graphs call executor; planned location `ai_core/graph/execution/`)
- [ ] Graph registry: once a real GraphExecutor exists, extend `ai_core/graph/registry.py` to support factories + versioning + optional request/response model metadata

## Domain boundary cleanup (later)

- [ ] Framework analysis graph: separate orchestration vs persistence boundary (`ai_core/graphs/framework_analysis_graph.py`, `documents/framework_models.py`)

## Observability / operations

- [ ] Review queue + retry handling across `agents`/`ingestion` (`llm_worker/tasks.py`, `ai_core/tasks.py`, `common/celery.py`)

## Agent navigation (docs-only drift)

- [ ] Fix stale docs links and naming drift around `docs/architektur/` vs `docs/architecture/` (pointers: `docs/audit/cleanup_report.md#link-fixes-required`, `docs/litellm/admin-gui.md`, `docs/development/onboarding.md`, `docs/documents/contracts-reference.md`, `docs/rag/overview.md`, `docs/crawler/overview.md`, `docs/architecture/architecture-reality.md`, `docs/roadmap/graphs.md`, `docs/architektur/langgraph-facade.md`; acceptance: no references to missing/renamed paths; LLM navigation lands on the canonical docs/code)

## Hygiene (lowest priority)

- [ ] Documentation hygiene: remove encoding artifacts in docs/strings (purely textual)
