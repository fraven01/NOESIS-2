# Backlog (prioritized)

This is a curated list of open work items for vibe-coding execution.
Top-to-bottom order within each section is priority order.
Prefer linking each item to concrete code paths (and optionally to an issue).

## Next up (highest leverage)

- [x] Drift #3: crawler ingestion persistence parity with upload path (pointers: `docs/architecture/drift3-critical-finding.md`, `crawler/worker.py`, `ai_core/tasks/graph_tasks.py:run_ingestion_graph`, `ai_core/graphs/technical/universal_ingestion_graph.py`; acceptance: crawler path can persist/create document without relying on "parallel registration" in `crawler/worker.py`)
- [x] Workbench exception: allow UI/views to trigger technical graphs only in `DEBUG` (explicit guard at endpoint level; e.g. `ai_core/views.py:_GraphView`, `ai_core/views.py:CrawlerIngestionRunnerView`)
- [x] Graph convergence: refactor `retrieval_augmented_generation.py` to real `langgraph` and migrate module-level singleton to factory (inventory `ai_core/graphs/technical/retrieval_augmented_generation.py`)
- [x] Deprecation plan: `info_intake` graph + intake endpoint (`ai_core/graphs/technical/info_intake.py`, `ai_core/views.py:IntakeViewV1`, `docs/api/reference.md`)
- [x] **BREAKING: API-2 global scope support for RAG query (allow `case_id=None`)**  
  - **Pointers:** `ai_core/views.py:482` (`_prepare_request`), `ai_core/views.py:626` (DEFAULT_CASE_ID fallback), `ai_core/views.py:671` (default-case bootstrap), `ai_core/views.py:346` (DEFAULT_CASE_ID constant)  
  - **Acceptance:** missing `X-Case-ID` no longer forces `case_id="general"`; `business_context.case_id` can be omitted in meta/tool_context; `/v1/ai/rag/query/` accepts requests without case/collection headers; tests added in `ai_core/tests/test_views.py` and `ai_core/tests/services/test_rag_query_service.py` covering global scope
- [x] **Workbench: Architecture Cleanup + RAG Scope Flexibilisierung (CONSOLIDATED)**  
  - **Status:** All tasks complete per `roadmap/consolidated-architecture-cleanup-plan.md`.  
  - **Pointers:** `theme/views_chat.py`, `theme/consumers.py`, `ai_core/views.py`, `ai_core/tests/test_views.py`, `theme/tests/test_chat_*`  
  - **Acceptance:** No `dev-case-local` fallback, both chat flows and API rely on `RagQueryService`, graph-specific follow-ups closed

- [x] **Review: RAG Service + Graph Execution integration** – ✅ APPROVED FOR PRODUCTION (2026-01-15). Integration is contract-compliant: `RagQueryService` returns `(state, result)` tuple, `GraphExecutionCommand` wraps only `result` in Response (preserves agentic caller contract), ToolContext propagation works, error handling comprehensive. **Documented technical debt** (non-blocking): (1) State persistence missing in RAG service path (Lines 302-309), (2) Cost tracking disabled (performance trade-off), (3) Test coverage gap for sync path, (4) Unreachable code (Lines 497-515). **Full review**: [`roadmap/rag-service-integration-review.md`](roadmap/rag-service-integration-review.md). **Follow-up issues**: Add state persistence, add sync path test, document cost tracking policy.

## Code Quality & Architecture Cleanup (Pre-MVP Refactoring)

_Based on architectural analysis 2025-12-31. Pre-MVP; breaking changes and test updates are acceptable if they reduce agent friction and clarify contracts._

### P0 - Critical Quick Wins (High Impact, Medium-High Effort)

- [x] **ToolContext-First Context Access** BREAKING: Replace manual dict unpacking with typed `ToolContext` parsed from `meta["tool_context"]` (pointers: **50+ locations** in `ai_core/services/__init__.py:808-816,1251-1254,1626-1627`, `ai_core/services/crawler_runner.py:63-67,80-83,122-123`, `theme/views.py:934-947,1164-1176,1443-1452`; **Effort: HIGH** due to breadth of changes; implementation: add a helper like `tool_context_from_meta(meta)` or use `ToolContext.model_validate(meta["tool_context"])`, then use `context.scope.*` / `context.business.*`; acceptance: no direct `meta["scope_context"]` / `meta["business_context"]` access in application code outside boundary/normalization helpers, type-safe context use, tests updated; implementation: helper added + call sites migrated in `ai_core/services/__init__.py`, `ai_core/tasks/`, `ai_core/services/crawler_runner.py`, `ai_core/services/crawler_state_builder.py`, `ai_core/views.py`, `ai_core/views_framework.py`, `ai_core/infra/resp.py`, `ai_core/views_response_utils.py`, `ai_core/graphs/technical/info_intake.py`, `ai_core/graphs/technical/retrieval_augmented_generation.py`, `documents/tasks.py`, `documents/upload_manager.py`, `documents/upload_worker.py`, `crawler/manager.py`, `crawler/tasks.py`, `llm_worker/tasks.py`, `llm_worker/graphs/hybrid_search_and_score.py`; tests green)

- [x] **Kill JSON Normalization Boilerplate** BREAKING: Removed `_make_json_safe()` and use `_dump_jsonable` (TypeAdapter(Any).dump_python(..., mode="json")) for mixed payloads (implementation: update call sites + tests; acceptance: `_make_json_safe()` deleted, consistent JSON serialization, tests updated)

- [x] **Standardize Error Handling** BREAKING: Converged on ToolErrorType-backed envelopes (implementation: `ai_core/infra/resp.py:build_tool_error_payload`, view/service boundaries emit ToolError envelope across `ai_core` + `theme`; acceptance: one error contract in responses, no string-based error returns, tests updated)

- [x] **Fix Logging Chaos** BREAKING: Standardized logging strategy (implementation: remove production `print()` usage, enforce structured logging via `extra={}`, add `docs/observability/logging-standards.md`; acceptance: no print() in production code, logs carry context where available, consistent log levels)

### Docs/Test touchpoints (checklist)

- [x] ToolContext-First Context Access (owner: ai-core+theme; docs: `docs/audit/architecture-anti-patterns-2025-12-31.md`; tests: `ai_core/tests/test_graphs.py`, `ai_core/tests/test_meta_normalization.py`, `ai_core/tests/test_normalize_meta.py`, `ai_core/tests/test_graph_retrieval_augmented_generation.py`, `ai_core/tests/test_graph_worker_timeout.py`, `ai_core/tests/test_services_observability.py`, `ai_core/tests/test_ingestion_view.py`, `ai_core/tests/test_views.py`, `ai_core/tests/test_infra.py`, `ai_core/tests/test_repro_upload.py`, `ai_core/tests/test_tasks_embed_observability.py`; acceptance notes: context reads come from `meta["tool_context"]` only)
- [x] Kill JSON Normalization Boilerplate (owner: ai-core; docs: `docs/audit/architecture-anti-patterns-2025-12-31.md`; tests: `ai_core/tests/test_services_json_safe.py`; acceptance notes: `_make_json_safe` removed, JSON serialization uses Pydantic adapters)
- [x] Standardize Error Handling (owner: ai-core+theme; docs: `docs/agents/tool-contracts.md`, `docs/audit/architecture-anti-patterns-2025-12-31.md`; tests: `ai_core/tests/test_views.py`, `ai_core/tests/test_views_min.py`, `ai_core/tests/views/test_ingestion_run.py`, `ai_core/tests/test_rag_ingestion_run.py`, `ai_core/tests/test_rag_ingestion_status.py`; acceptance notes: responses emit `ToolErrorType` codes)
- [x] Eliminate Pass-Through Glue Functions (owner: theme; docs: `docs/audit/architecture-anti-patterns-2025-12-31.md`; tests: `theme/tests/test_rag_tools_view.py`, `theme/tests/test_rag_tools_simulation.py`, `theme/tests/test_document_space_view.py`, `theme/tests/test_chat_submit_global_search.py`, `theme/tests/test_admin_domain_dropdown.py`, `ai_core/tests/test_views.py`, `ai_core/tests/test_views_min.py`; acceptance notes: query params validated via Pydantic models)
- [x] Normalize the Normalizers (owner: theme+ai-core; docs: `docs/audit/architecture-anti-patterns-2025-12-31.md`; tests: `theme/tests/test_rag_tools_view.py`, `theme/tests/test_rag_tools_simulation.py`, `ai_core/tests/test_nodes.py`, `ai_core/tests/test_views.py`; acceptance notes: duplicate normalizers consolidated without behavior drift)
- [x] Remove Fake Abstractions (owner: ai-core+documents; docs: `docs/audit/architecture-anti-patterns-2025-12-31.md`; tests: `ai_core/tests/test_services_observability.py`, `ai_core/tests/test_llm.py`, `documents/tests/test_pipeline.py`, `documents/tests/test_processing_graph_integration.py`, `documents/tests/test_processing_graph_e2e.py`; acceptance notes: direct usage or real DI, no wrapper-only classes)
- [x] Fix Logging Chaos (owner: ai-core+theme; docs: `docs/audit/architecture-anti-patterns-2025-12-31.md`, `docs/observability/logging-standards.md`; tests: `ai_core/tests/test_services_observability.py`, `ai_core/tests/test_views.py`, `ai_core/tests/services/test_crawler_runner.py`, `theme/tests/test_rag_tools_view.py`; acceptance notes: no print statements, structured logging with context)
- [x] Break Up God Files (owner: ai-core+theme; docs: `docs/audit/architecture-anti-patterns-2025-12-31.md`; tests: `ai_core/tests/conftest.py`, `ai_core/tests/test_views.py`, `ai_core/tests/test_ingestion_view.py`, `ai_core/tests/test_repro_upload.py`, `ai_core/tests/services/test_crawler_runner.py`, `theme/tests/test_rag_tools_view.py`, `theme/tests/test_document_space_view.py`, `theme/tests/test_chat_submit_global_search.py`; acceptance notes: modules split, imports updated, no file > 500 lines)
- [x] Targeted Domain Enrichment (owner: ai-core; docs: `docs/audit/architecture-anti-patterns-2025-12-31.md`; tests: `ai_core/tests/tools/test_framework_contracts.py`, `ai_core/tests/graphs/test_framework_analysis_graph.py`; acceptance notes: domain methods replace duplicated logic)
- [x] Service Layer Refactoring (owner: ai-core; docs: `docs/audit/architecture-anti-patterns-2025-12-31.md`; tests: `ai_core/tests/test_graph_worker_timeout.py`, `ai_core/tests/test_services_observability.py`, `ai_core/tests/test_ingestion_view.py`, `ai_core/tests/test_repro_upload.py`, `ai_core/tests/services/test_crawler_runner.py`; acceptance notes: command objects used, `execute_graph` <= 100 lines)
- [x] State Dict -> Dataclasses (owner: ai-core; docs: `docs/audit/architecture-anti-patterns-2025-12-31.md`; tests: `ai_core/tests/services/test_crawler_runner.py`, `ai_core/tests/test_services_observability.py`; acceptance notes: structured state uses dataclasses/Pydantic)

- [x] **Context Propagation Standardization** BREAKING: Standardize context propagation across ALL boundaries (HTTP, WebSocket, Celery, Graph, Tool) with typed helpers and validation.
  - **Phase 1 (docs)**: fix runtime ID docs (no XOR); add `docs/architecture/context-propagation-patterns.md`, `docs/migrations/context-standardization-migration.md`; update `docs/agents/tool-contracts.md` with ToolContext + Pydantic I/O standard + WS payload rules.
  - **Phase 2 (helpers)**: `ai_core/tool_contracts/validation.py` (require_business_field, require_runtime_id); `theme/websocket_utils.py` (build_websocket_context, UUID user_id enforced -> ContextError mapped to 403); `theme/websocket_payloads.py` (Pydantic payload w/ hybrid tuning fields, extra="forbid"); update `theme/consumers.py` to use helper + payload.
  - **Phase 3 (migration, CRITICAL)**: `ai_core/nodes/compose.py` signature -> ToolContext + Pydantic I/O; migrate prompt nodes `assess/classify/extract/needs/draft_blocks` + `_prompt_runner.py`; `ai_core/graphs/technical/retrieval_augmented_generation.py` parse context once; add `build_business_context_from_request()` in `ai_core/graph/schemas.py` and migrate views away from `_tenant_context_from_request()`; refactor `ai_core/ingestion_orchestration.py` (preserve fallback behavior).
  - **Phase 4 (tests)**: add `tests/integration/test_context_propagation.py`, `tests/negative/test_missing_context.py`; update `tests/chaos/test_tool_context_contracts.py`; add `tests/contracts/test_tool_signature_guardrails.py` (enable after 3.1b).
  - **Phase 5 (cleanup)**: remove deprecated code, update imports, finalize docs.
  - **Phase 6 (storage, approved now)**: add `audit_meta` JSONB to Document + FrameworkProfile + write/read paths; add `ai_core/graph/state.py` (PersistedGraphState) and update FileCheckpointer; add `common/task_result.py` and wrap `ai_core/tasks/` returns (after Q2 consumer investigation).
  - **Effort**: 8.5d core (Phases 1-5) + 2d Phase 6.
  - **Decisions**: Q1 tenant policy = no header-based resolving; Q2 investigate TaskResult consumers before 6.3; Q3 DB empty; Q4 validation helpers raise ContextError; Q5 WS payload extra="forbid" + hybrid tuning fields allowed; Q6 guardrail after 3.1b; Q7 UUID enforced (invalid -> ContextError mapped to 403).
  - **Acceptance**: docs aligned (no XOR references); helpers used across boundaries; no `(state, meta)` signatures in nodes/tools; RAG graph parses context once; WS payload validated; all tests pass; audit_meta + PersistedGraphState + TaskResult in place.
  - **Rollback**: Phase 1-2 revert commits; Phase 3 restore compose/prompt node signatures + `_tenant_context_from_request()`; Phase 6 rollback migrations and TaskResult wrapper changes.
  - **Plan**: `C:\Users\vendo\.claude\plans\idempotent-strolling-pearl.md`.

### P1 - High Value Cleanups (Low-Medium Effort)

- [x] **RAG vector_client refactor (phased, pre-MVP)**: execute the plan in `roadmap/rag-vector-client-refactor-roadmap.md` (pointers: `ai_core/rag/vector_client.py`, `ai_core/rag/delta.py`, `ai_core/rag/embedding_cache.py`; acceptance: Phase A/B/C completed as defined in the roadmap, tests updated, `ai_core/rag/vector_client.py` significantly reduced)

- [x] **Eliminate Pass-Through Glue Functions**: Remove trivial helpers in favor of Pydantic validators (pointers: `theme/validators.py`, `theme/views.py`; implementation: create `theme/validators.py` with Pydantic models for query params, move validation logic to `@field_validator`, inline formatting helpers or move to templates; note: exclude helpers that encapsulate IO or domain lookups (relocate instead of delete); acceptance: validation in Pydantic models, helpers removed, tests passing)

- [x] **Normalize the Normalizers** BREAKING: Consolidate duplicate conversion functions without changing behavior (pointers: `common/validators.py`, `ai_core/graphs/technical/collection_search.py`, `llm_worker/schemas.py`; implementation: reusable validation helpers for trimmed strings, optional strings, and string sequences; replace only identical normalize_* functions; acceptance: shared validators, no behavior drift, tests updated)

- [x] **Remove Fake Abstractions**: Delete ceremonial wrappers without value (pointers: `ai_core/services/__init__.py`, `ai_core/ingestion.py`, `ai_core/graphs/technical/universal_ingestion_graph.py`; implementation: direct imports of storage/captioner, use real ledger module; acceptance: boilerplate classes deleted, direct usage, tests passing)

- [x] **Remove hybrid ToolContext payload in workbench web search** BREAKING: Stop sending flattened `tenant_id`/`case_id`/etc alongside ToolContext when invoking collection search (pointers: `theme/views_web_search.py`, `ai_core/graphs/technical/collection_search.py`; acceptance: collection search reads from `meta["tool_context"]` (or ToolContext validation) only, no flattened fields in meta, tests updated)

- [x] **Drop legacy tenant/case metadata keys in LLM scoring** BREAKING: Remove `tenant`/`case` fallback keys and duplicated fields in scoring metadata (pointers: `llm_worker/graphs/score_results.py:_build_metadata`, `ai_core/llm/client.py:call`; acceptance: only `tenant_id`/`case_id` accepted in metadata, no legacy key fallback, tests updated)

- [x] **Enforce ingestion task signature (state/meta only)** BREAKING: Remove legacy args/kwargs fallback and `state` detection in ingestion enqueue (pointers: `ai_core/services/ingestion.py:_enqueue_ingestion_task`, `ai_core/tasks/ingestion_tasks.py`, `ai_core/views.py`; acceptance: ingestion tasks accept `state, meta` only, legacy args/kwargs removed, tests updated)

- [ ] **Simplify Adaptive Chunking Toggles (future)**: Keep only two flags and one "new" path (pointers: `ai_core/rag/chunking/hybrid_chunker.py`, `ai_core/rag/chunking/late_chunker.py`, `ai_core/rag/chunking/README.md`; implementation: route all chunking through HybridChunker, keep legacy path only when `RAG_ADAPTIVE_CHUNKING_ENABLED=false`, no alias settings; acceptance: two flags only, minimal metadata (`metadata.kind` + `parent_ref` for assets), unified chunk_id strategy, tests limited to core acceptance checks)

- [ ] **[MERGED into consolidated plan above]** RAG Scope Flexibilisierung → siehe konsolidierter Plan in Zeile 13

### P2 - Long-term Improvements (High Effort)

- [x] **Document Versioning (SOTA 5.3)** BREAKING: add per-upload `document_version_id` and latest-only RAG semantics; server-side version ID generation (no backfill; DB reset allowed); allow multiple `DocumentVersion` rows per `version` label (edit states). Pointers: `documents/models.py` (new `DocumentVersion` table + indexes), `documents/repository.py` (persist version rows + list versions), `documents/domain_service.py` + `documents/service_facade.py` + `documents/views.py` (History APIs), `ai_core/rag/ingestion_contracts.py:69` + `ai_core/api.py:452` + `ai_core/tasks/ingestion_tasks.py:1436` (ChunkMeta fields + validation), `ai_core/rag/vector_client.py` + `ai_core/rag/vector_store.py` + `ai_core/rag/query_builder.py` (filter by `document_version_id`/`is_latest`), `documents/tasks.py` or scheduler (cleanup job), `documents/tests/*` + `ai_core/tests/*` (coverage). Acceptance: each document update creates a new `DocumentVersion` row and chunks tagged with `document_version_id`/`is_latest=true` while previous versions flip to `is_latest=false`; `GET /documents/{doc_id}/versions` and `GET /documents/{doc_id}/versions/{version_id}/chunks` return expected results; RAG queries default to latest-only but can target explicit `document_version_id`; cleanup job runs daily and purges versions older than 30 days after soft-delete; diff view returns chunk-level added/removed/changed for two versions.

- [x] **Break Up God Files** BREAKING: Split 2000+ line monoliths into focused modules (pointers: `ai_core/services/__init__.py` (2,034 lines: `execute_graph` L776-1231, `start_ingestion_run` L1234-1367, `handle_document_upload` L1577-2033), `theme/views.py` (2,055 lines); implementation: split into `ai_core/services/{graph_executor,ingestion,document_upload,components}.py`, split into `theme/views/{rag_tools,search,documents,validators}.py`; acceptance: no file >500 lines, clear module boundaries, imports updated, tests passing)

- [x] **Targeted Domain Enrichment** BREAKING: Add behavior only where it removes duplication or clarifies invariants (pointers: `ai_core/tools/framework_contracts.py:16-23` (TypeEvidence), `:26-33` (AlternativeType), `:36-42` (ScopeIndicators), `:74-92` (ComponentLocation) - all frozen dataclasses with no methods; implementation: add small, local methods (e.g. `TypeEvidence.is_strong`, `TypeEvidence.merge_with()`), move duplicated logic out of services; acceptance: domain methods replace repeated logic, service layer thinner, tests updated)

- [x] **Service Layer Refactoring**: Replace procedural god-functions with Command Pattern (pointers: `ai_core/services/__init__.py:execute_graph` (455 lines of procedural logic); implementation: create `ai_core/commands/graph_execution.py` with `GraphExecutionCommand` class encapsulating validation/execution/recording, use factory pattern for graph runners; acceptance: execute_graph <100 lines, commands testable in isolation, clear separation of concerns)

### Observability Cleanup

_Note: Critical logging issues (print() in production) moved to P0. Remaining observability work stays here._

### Hygiene

- [x] **State Dict -> Dataclasses**: Replace manual dict building with typed dataclasses (pointers: `ai_core/services/crawler_runner.py:298-314` (synthesized_state, entry dicts built manually); implementation: create typed dataclasses for common state shapes, use Pydantic models for serialization; acceptance: no manual dict building for structured data, type-safe state management)

## Semantics / IDs

- [x] User identity cleanup (UUID PKs, telemetry separation, initiated_by_user_id propagation) (see `roadmap/backlog-user-identity.md`)
- [x] **Chat thread_id contract (BusinessContext)** BREAKING: add `thread_id` to BusinessContext + headers and propagate into normalize_meta/ToolContext (pointers: `common/constants.py`, `ai_core/contracts/business.py`, `ai_core/graph/schemas.py`, `ai_core/tool_contracts/base.py`, `theme/views_chat.py`, `roadmap/rag_roadmap.md`; acceptance: `X-Thread-ID` + `HTTP_X_THREAD_ID` supported, BusinessContext carries thread_id, ToolContext round-trips, chat uses thread_id for memory/checkpoint key, docs updated)
- [x] **RAG memory scoping decision (checkpointer key)**: chat threads use `thread_id` for checkpointer keys (pointers: `ai_core/graph/core.py:ThreadAwareCheckpointer`, `roadmap/rag_roadmap.md#L229`; acceptance: thread-aware path in place, documented in roadmap, note any migration/reset requirement)
- [x] Clean-state graph input schema for collection_search (BREAKING): move graph state to ToolContext only (pointers: `ai_core/graphs/technical/collection_search.py`, `ai_core/views.py`, `theme/views_web_search.py`, `ai_core/tests/graphs/test_collection_search_graph.py`; acceptance: state includes `tool_context` only, no flat context dict access, views build ToolContext, tests updated)
- [x] Enforce GraphIOSpec + versioned I/O for collection_search (BREAKING): add GraphIOSpec and validate schema_id/schema_version at boundary (pointers: `ai_core/graph/io.py`, `ai_core/graphs/technical/collection_search.py`, `ai_core/tests/graphs/test_collection_search_graph.py`; acceptance: GraphIOSpec attached to graph, boundary validation enforced, tests updated)
- [x] Remove legacy context surface (BREAKING): delete deprecated ToolContext properties and legacy GraphContext (pointers: `ai_core/tool_contracts/base.py`, `ai_core/graph/core.py`, `ai_core/commands/graph_execution.py`; acceptance: no `context.<id>` access, GraphContext removed or replaced, tests updated)

- [x] Semantics vNext: keep `case_id` required; standardize defaults (`general` for API views, `dev-case-local` for `/rag-tools`) and ensure that case exists per tenant (`ai_core/graph/schemas.py:normalize_meta`, `ai_core/views.py`, `theme/views.py`, `cases/*`)
- [x] Hard-break ScopeContext business IDs: reject business identifiers in `ScopeContext` (pointers: `ai_core/contracts/scope.py:ScopeContext.forbid_business_ids`, `ai_core/tests/test_scope_context.py`; acceptance: ScopeContext validation raises when `case_id|collection_id|workflow_id|document_id|document_version_id` appear in input)
- [x] ID propagation clarity: document and/or enforce the "runtime context injection" pattern for graphs (pointers: `docs/audit/contract_divergence_report.md#id-propagation-compliance`, `ai_core/graphs/README.md`, `ai_core/graph/schemas.py`, `ai_core/contracts/scope.py`; acceptance: one canonical pattern, documented in code/doc pointers, so tenant/trace/run IDs are not "implicit knowledge")
- [x] Breaking meta contract (Contract Definition): enforce `scope_context` as the only ID source in graph/tool meta at contract level (pointers: `ai_core/graph/schemas.py:normalize_meta`, `ai_core/views.py:_prepare_request`, `llm_worker/runner.py:submit_worker_task`, `common/celery.py:ContextTask._from_meta`; acceptance: contract enforces `meta["scope_context"]` structure)
- [x] Breaking meta contract (Implementation): migrate all 50+ dict unpacking call sites to use typed ToolContext (tracked as P0 "ToolContext-First Context Access"; acceptance: no direct `meta["scope_context"]` / `meta["business_context"]` access in application code)

## Layering / boundaries

- [x] Remove legacy graph shims after call sites are migrated (pointers: `ai_core/graphs/info_intake.py`, `ai_core/graphs/__init__.py`, `ai_core/views.py`; note: coordinate with the `info_intake` deprecation plan)
- [x] Enforce import direction (business -> technical) via a lightweight repo test (no new dependencies)
- [x] Dev Workbench: review UI capability boundary for web acquisition search (pointers: `theme/views_web_search.py`, `theme/views_rag_tools.py`, `ai_core/graphs/web_acquisition_graph.py`, `crawler/manager.py`; acceptance: explicit capability vs graph boundary and consistent naming in rag-tools UI)
- [ ] **RAG chat collection dropdown respects user access**: filter collections in `tool_chat` by user profile/membership rules (pointers: `theme/views_rag_tools.py:tool_chat`, `documents/authz.py:DocumentAuthzService`, `cases/models.py:CaseMembership`, `documents/models.py:DocumentCollectionMembership`; acceptance: logged-in users only see collections they can access; anonymous users see either none or tenant-wide default per policy; rule documented in `roadmap/rag_roadmap.md` decision notes)
- [x] Capability-first: framework analysis graph extractions (pointers: `ai_core/graphs/business/framework_analysis_graph.py`, `ai_core/services/framework_analysis.py`, `documents/services/framework_service.py`, `roadmap/capability-first-todos.md`; acceptance: extracted capabilities for gremium normalization, ToC extraction, LLM JSON parsing, component validation; persistence already via `documents/services/framework_service.py`)
- [x] Capability-first: collection search graph extractions (pointers: `ai_core/graphs/technical/collection_search.py`, `ai_core/rag/strategy.py`, `roadmap/capability-first-todos.md`; acceptance: scoring/strategy/HITL/auto-ingest logic moved to nodes/tools/services)
- [x] Capability-first: universal ingestion graph extractions (pointers: `ai_core/graphs/technical/universal_ingestion_graph.py`, `ai_core/graphs/web_acquisition_graph.py`, `ai_core/graphs/technical/document_service.py`, `roadmap/capability-first-todos.md`; acceptance: selection + blocked-domain logic moved to capability, normalization to NormalizedDocument routed through document_service capability, DI root optionally in factory)

## Externalization readiness (later)

- [x] Contract drift (hard break): migrate `Chunk` to Pydantic (frozen) and remove `Chunk.meta` mutation (pointers: `docs/audit/contract_divergence_report.md#1-chunk-schema-compliance`, `ai_core/rag/schemas.py`, `ai_core/rag/vector_client.py`, `ai_core/api.py`, `ai_core/tasks/ingestion_tasks.py`; acceptance: `Chunk` is Pydantic (frozen), call sites migrated to keyword-only construction, no in-place `Chunk.meta` mutation)
- [x] Technical graph interface: versioned Pydantic I/O for graph boundaries (pilot: `universal_ingestion_graph`, `web_acquisition_graph`, `retrieval_augmented_generation`, `crawler.ingestion` alias; acceptance: graphs declare input/output models and version tokens)
- [x] Technical graph interface: expand versioned Pydantic I/O to remaining graphs (completed: `framework_analysis_graph`; `info_intake` deferred to `docs/roadmap/rag-query.md`; `external_knowledge_graph` is `web_acquisition_graph` already migrated)
- [x] Graph executor boundary: wire `ai_core/graph/execution` into business graphs and add a Celery/remote executor implementation (acceptance: business graphs call executor interface; local + async executors covered)
- [x] Executor unification for technical graphs: route technical-graph invocations through `GraphExecutor` (pointers: `ai_core/graph/execution/*`, `ai_core/commands/graph_execution.py`, `llm_worker/tasks.py`, `ai_core/graph/registry.py`, `ai_core/graph/bootstrap.py`; acceptance: technical graphs are invoked via `GraphExecutor` only, no direct registry `get(...).run(...)` outside executor; local + celery executors covered by tests)
- [ ] Graph registry: add versioning + request/response model metadata (note: factory support already exists via `LazyGraphFactory`)

## Domain boundary cleanup (later)

- [x] Framework analysis graph: separate orchestration vs persistence boundary (`ai_core/graphs/framework_analysis_graph.py`, `documents/framework_models.py`)

## Observability / operations

- [x] Review queue + retry handling across `agents`/`ingestion` (`llm_worker/tasks.py`, `ai_core/tasks/`, `common/celery.py`; done: RetryableTask, queue routing, DLQ, circuit breaker, cleanup task)
- [x] Scope injection cleanup (PII/Logging): keep PII session scope and log context without passing scope kwargs into task run signatures (pointers: `common/celery.py:ScopedTask.__call__`, `common/celery.py:with_scope_apply_async`, `ai_core/ingestion.py:run_ingestion`; acceptance: tasks consume `state/meta` only, scope kwargs used only for PII/logging, no unexpected kwarg failures)

## Agent navigation (docs-only drift)

- [x] Fix stale docs links and naming drift around `docs/architektur/` vs `docs/architecture/` (pointers: `docs/litellm/admin-gui.md`, `docs/development/onboarding.md`, `docs/documents/contracts-reference.md`, `docs/rag/overview.md`, `docs/crawler/overview.md`, `docs/architecture/architecture-reality.md`, `docs/roadmap/graphs.md`, `docs/architecture/langgraph-facade.md`; acceptance: no references to missing/renamed paths; LLM navigation lands on the canonical docs/code)

## Hygiene (lowest priority)

- [ ] Documentation hygiene: remove encoding artifacts in docs/strings (purely textual)


# Backlog (prioritized)

This is a curated list of open work items for vibe-coding execution.
Top-to-bottom order within each section is priority order.
Prefer linking each item to concrete code paths (and optionally to an issue).

## Next up (highest leverage)

- [x] **Framework Analysis Graph Convergence**: Migriere Business-Graph von Custom DSL zu LangGraph `StateGraph` mit Patterns der technischen Graphen (TypedDict State, Protocols, Error Handling, Observability).
  - **Details:** [framework-analysis-convergence.md](framework-analysis-convergence.md)
  - **Pointers:** `ai_core/graphs/business/framework_analysis_graph.py`, `ai_core/graphs/technical/collection_search.py` (Referenz)
  - **Acceptance:** LangGraph StateGraph; TypedDict mit Reducers; Service Protocols für DI/Testing; Graceful Degradation statt Exception-Abbruch; `emit_event()` Observability; Boundary Validation mit schema_version

### Phase 0: Quick Wins (vor LangGraph-Migration)

- [x] **FA-0.0: Technical Graph "rag_retrieval" (multi-query RAG retrieval for framework analysis)**: New retrieval-only graph with multi-query batching, optional rerank, and document scoping to reuse internal RAG retrieval patterns.
  - **Pointers:** `ai_core/graphs/technical/rag_retrieval.py` (new), `ai_core/graphs/technical/retrieval_augmented_generation.py:_retrieve_step`, `ai_core/nodes/retrieve.py`, `ai_core/rag/rerank.py`
  - **Acceptance:** Graph boundary contract `schema_id="noesis.graphs.rag_retrieval"` + `schema_version="1.0.0"`; includes `tool_context`, `queries: list[str]`, `retrieve: RetrieveInput`, `use_rerank: bool`, `document_id: str | None`; `document_id` translated into retrieval filters; multi-query loop aggregates + dedupes matches across queries; optional rerank when `use_rerank=True`; outputs `matches`, `snippets`, `retrieval_meta`, `query_variants_used`, optional `rerank_meta`; `io_spec` attached; tests in `ai_core/tests/graphs/test_rag_retrieval_graph.py` cover multi-query dedupe, rerank toggle, and document_id scoping.


- [x] **FA-0.1: Retrieve-Calls konsolidieren (6->2)**: Ein initialer `rag_retrieval` Call mit top_k=100 fuer ToC/Preview, plus ein multi-query `rag_retrieval` Call fuer Komponenten.
  - **Pointers:** `ai_core/graphs/business/framework_analysis_graph.py`, `ai_core/graphs/technical/rag_retrieval.py`
  - **Acceptance:** Zwei Retrieval-Calls total; `state["all_chunks"]` von allen Nodes genutzt; Komponenten ueber multi-query Retrieval + Dedup; Latenz/Kosten reduziert.

- [x] **FA-0.2: Nodes konsolidieren (6→4)**: Überflüssige Nodes (`extract_toc`, `validate_components`, `finish`) in angrenzende Nodes integrieren.
  - **Pointers:** `ai_core/graphs/business/framework_analysis_graph.py:322-361`, `:440-460`, `:535-543`
  - **Acceptance:** Node-Struktur: `init_and_fetch → detect_type → locate_components → assemble_profile`

- [x] **FA-0.3: Early Exit bei Low Confidence**: Graph endet früh mit HITL-Request wenn `type_confidence < 0.5`.
  - **Pointers:** `ai_core/graphs/business/framework_analysis_graph.py:293-308`
  - **Acceptance:** Early exit mit `hitl_required=True` und `partial_results`

- [x] **FA-0.4: Query-Templates nach Agreement-Type**: Type-spezifische Queries (KBV, GBV, BV, DV) statt hardcoded.
  - **Pointers:** `ai_core/graphs/business/framework_analysis_graph.py:371-376`
  - **Acceptance:** `COMPONENT_QUERIES: dict[str, dict[str, str]]` mit Fallback auf generische Queries

- [x] **FA-0.5: LLM Retry-Logik**: Retry mit Exponential Backoff für `call_llm_json_prompt()`.
  - **Pointers:** `ai_core/services/framework_analysis_capabilities.py:73-88`
  - **Acceptance:** Max 3 Retries (1s, 2s, 4s); nur transient errors; Graceful Degradation nach Retries

- [x] **FA-0.6: Ungenutzten Validation-Prompt bereinigen**: Entweder LLM-Validation aktivieren oder Prompt löschen.
  - **Pointers:** `ai_core/prompts/framework/validate_component.v1.md`, `ai_core/graphs/business/framework_analysis_graph.py:440-460`
  - **Acceptance:** Keine toten Prompts im Repository

- [x] **FA-1.2: Framework Analysis boundary requires schema_version**: Enforce explicit `schema_version` (no default) like collection_search and update callers.

- [x] **FA-1.3: Unify FrameworkAnalysis output shape**: Wrapper returns `FrameworkAnalysisGraphOutput` (single boundary shape) instead of Draft.

- [x] **FA-1.4: Single output shape (GraphOutput only)**: FrameworkAnalysisGraph wrapper returns GraphOutput; Draft derived explicitly where needed.

- [x] **FA-4.1: Framework Analysis structured errors + partial_results**: Add error payloads and graceful degradation fields to graph output.

- [x] **FA-4.2: Framework Analysis timeout management**: Node + graph timeouts via runtime; timeouts degrade gracefully with structured errors.
  - **Pointers:** `ai_core/graphs/business/framework_analysis/graph.py`, `ai_core/graphs/business/framework_analysis/state.py`, `ai_core/tests/graphs/test_framework_analysis_graph.py`
  - **Acceptance:** `node_timeout_s`/`node_timeouts_s` and `graph_timeout_s` supported; timeouts return errors + HITL required.

- [x] **FA-5.1: Async node support**: Async graph execution path with `ainvoke` and graph timeout support.
  - **Pointers:** `ai_core/graphs/business/framework_analysis/graph.py`
  - **Acceptance:** `FrameworkAnalysisStateGraph.arun()` uses `ainvoke` with optional `graph_timeout_s`; async path returns same output shape.

- [x] **FA-6.1: Framework Analysis emit_event/update_observation**: Emit milestone events and metrics for graph execution and nodes.
  - **Pointers:** `ai_core/graphs/business/framework_analysis/graph.py`, `ai_core/graphs/business/framework_analysis/nodes.py`
  - **Acceptance:** `emit_event` fired for graph_started/completed and node milestones; `update_observation` captures completeness + hitl.
  - **Pointers:** `ai_core/tools/framework_contracts.py`, `ai_core/graphs/business/framework_analysis/graph.py`, `ai_core/graphs/business/framework_analysis/state.py`, `ai_core/graphs/README.md`
  - **Acceptance:** Output includes `errors` and `partial_results`; node errors captured in state without aborting; empty structure returned on failure with HITL required.
  - **Pointers:** `ai_core/graphs/business/framework_analysis_graph.py`, `ai_core/graphs/business/framework_analysis/io.py`, `ai_core/graphs/README.md`
  - **Acceptance:** Wrapper returns GraphOutput with schema_id/schema_version; docs reflect boundary output; tests pass.
  - **Pointers:** `ai_core/graphs/business/framework_analysis_graph.py`, `ai_core/services/framework_analysis.py`
  - **Acceptance:** Wrapper returns GraphOutput with schema_id/schema_version; Draft derived only where needed; tests updated if necessary.

- [x] **FA-2.1: Framework Analysis runtime DI in boundary**: Allow `runtime` overrides for retrieval/LLM services in graph input.
  - **Pointers:** `ai_core/graphs/business/framework_analysis/io.py`, `ai_core/graphs/business/framework_analysis_graph.py`, `ai_core/tests/graphs/test_framework_analysis_graph.py`
  - **Acceptance:** `runtime` accepted in boundary; retrieval/LLM overrides applied when provided; tests cover runtime acceptance.
  - **Pointers:** `ai_core/graphs/business/framework_analysis/io.py`, `ai_core/services/framework_analysis.py`, `ai_core/tests/graphs/test_framework_analysis_graph.py`
  - **Acceptance:** Missing `schema_version` fails boundary validation; callers pass explicit schema_id/schema_version; tests updated.

## Agentic AI-First System (target state)

### Pre AI-first blockers

- [x] **Workflow-execution-aware checkpointing**:
  - **Pointers:** `ai_core/graph/core.py:FileCheckpointer._path`, `ai_core/graph/core.py:GraphContext`, `common/object_store_defaults.py:sanitize_identifier`, `ai_core/graph/README.md`
  - **Acceptance:** Checkpoint paths are derived from workflow execution scope (tenant_id + workflow_id + run_id, or plan_key when available) instead of case_id; no "None" paths or case-less collisions; missing workflow_id/run_id raises a clear error; ThreadAwareCheckpointer still prefers thread_id; workflow_execution terminology used in docs; tests updated in `ai_core/tests/test_checkpointer_file.py` to cover unique paths per run_id, missing run_id errors, and thread_id path precedence

- [x] **Docs alignment for plan_key + workflow_execution**:
  - **Pointers:** `ai_core/graph/README.md`, `docs/roadmap/UTP.md`, `roadmap/agentic-ai-first-strategy.md`
  - **Acceptance:** plan_key (derived, not minted) replaces plan_id in planning docs; workflow_execution terminology used consistently; no references to execution_case_id or required case_id for graph meta

- [x] **Blueprint + ImplementationPlan contracts (Phase 1)**:
  - **Pointers:** `ai_core/contracts/plans/` (new package), `ai_core/contracts/plans/plan.py`, `ai_core/contracts/plans/blueprint.py`, `ai_core/contracts/plans/evidence.py`, `ai_core/graphs/technical/collection_search.py:ImplementationPlan`
  - **Acceptance:** Pydantic models for Blueprint, ImplementationPlan, Slot, Task, Gate, Deviation, Evidence, and Confidence; schema_version on plan models ("v0" or semver); Evidence uses ref_type/ref_id (url | repo_doc | repo_chunk | object_store | confluence | screenshot); optional slot_type or json_schema_ref; JSON schema export; deterministic plan_key derivation helper (UUIDv5 or hash) over plan scope tuple (derived, not minted) with canonicalization (ordered scope tuple, normalized gremium_identifier, choose framework_profile_id or framework_profile_version); round-trip serialization tests; tests forbid execution_case_id, minted plan_id, or any additional UUID fields; references updated in `roadmap/agentic-ai-first-strategy.md`; no new IDs introduced

- [x] **Vertical slice: Plan-driven Collection Search**:
  - **Pointers:** `ai_core/graphs/technical/collection_search.py:build_plan_node`, `ai_core/graphs/technical/collection_search.py:HitlDecision`, `ai_core/services/collection_search/hitl.py`, `ai_core/services/collection_search/strategy.py`, `ai_core/services/collection_search/scoring.py`, `ai_core/graphs/technical/universal_ingestion_graph.py`
  - **Acceptance:** `build_plan_node` emits the new plan schema with slots/tasks/gates/deviations/evidence using workflow_execution terminology; HITL decisions update slot values and gate outcomes (slot completion); `execute_plan` consumes task outputs and records evidence; graph output links evidence to ingested artifacts; tests updated in `ai_core/tests/graphs/test_collection_search_graph.py`

- [x] **Plan persistence and retrieval (vertical slice)**:
  - **Pointers:** `ai_core/graph/core.py:FileCheckpointer`, `ai_core/graph/state.py:PersistedGraphState`, `common/object_store_defaults.py`
  - **Acceptance:** Plan and evidence persist as part of the graph state envelope; deterministic plan_key resolves to the latest persisted state for the workflow execution (derived, not minted); retrieval helper returns validated plan + evidence; tests cover round-trip persistence; no new IDs introduced; no new DB tables in the slice

## Code Quality & Architecture Cleanup (Pre-MVP Refactoring)

### P0 - Critical Quick Wins (High Impact, Medium-High Effort)

- [x] **INC-20260119-001: Async-First Web Search (Gateway Timeout Prevention)**:
  - **Details:** Synchronous blocking view caused 504 Gateway Timeout when LLM latency exceeded 60s. View should return `202 Accepted` + `task_id` immediately; frontend polls for status or uses WebSockets (HTMX `hx-ws`).
  - **Pointers:** `theme/views_web_search.py:237` (`submit_business_graph(..., timeout_s=60)`), `theme/helpers/tasks.py:72` (`async_result.get(timeout=timeout)`)
  - **Acceptance:** `web_search` view returns `202 Accepted` with `task_id` for async execution; polling endpoint or WebSocket support for result retrieval; no synchronous blocking on graph execution; tests cover async flow

- [x] **INC-20260119-001: Remove DEBUG 360s LLM Timeout Override**:
  - **Details:** DEBUG mode enables 360s timeout for LLM calls, allowing unbounded blocking that exceeds infrastructure idle timeouts (60s firewall/gateway). Trace 56683b33 showed 88.5s LLM call causing cascading failures.
  - **Pointers:** `ai_core/services/collection_search/strategy.py:317-318` (`dev_timeout_s = 360.0`), `llm_worker/graphs/hybrid_search_and_score.py:1088-1090` (`timeout_s = max(..., 360)`)
  - **Acceptance:** DEBUG timeout override removed or capped at 90s max; LLM calls fail gracefully within infrastructure timeout limits; fallback strategies activated on timeout

- [x] **INC-20260119-001: DB Connection Management for Long-Running Tasks**:
  - **Details:** Celery worker held DB connection idle during 88.5s LLM call, causing `psycopg2.OperationalError` when connection was severed by firewall idle timeout (60s). Need `close_old_connections()` before/after blocking IO.
  - **Pointers:** `ai_core/graphs/technical/collection_search.py` (LLM call sites), `llm_worker/graphs/hybrid_search_and_score.py:_run_llm_rerank`, `tests/chaos/perf_smoke.py` (only current usage of `close_old_connections`)
  - **Acceptance:** `django.db.close_old_connections()` called before long-running LLM/IO operations; fresh connection acquired for persistence (`vector_sync`); tests cover connection recovery after long blocking operations

- [x] **Collection Search timeouts and stall protection**:
  - **Details:** `roadmap/collection-search-review.md`
  - **Pointers:** `ai_core/llm/client.py:391`, `ai_core/llm/client.py:738`, `ai_core/graphs/technical/collection_search.py:561`, `ai_core/graphs/technical/collection_search.py:650`, `llm_worker/graphs/hybrid_search_and_score.py:1116`, `llm_worker/graphs/score_results.py:353`
  - **Acceptance:** LLM client adds explicit connect/read timeouts (sync + streaming); parallel search uses a total timeout with partial results; hybrid score timeout handling is reachable and logged; tests updated to cover timeout behavior; see `roadmap/collection-search-review.md`

- [x] **Collection Search boundary contract cleanup (V1-V8)**:
  - **Details:** `roadmap/collection-search-review.md`
  - **Pointers:** `ai_core/graphs/technical/collection_search.py:140`, `ai_core/graphs/technical/collection_search.py:480`, `ai_core/graphs/technical/collection_search.py:636`, `ai_core/graphs/technical/collection_search.py:837`, `ai_core/graphs/technical/collection_search.py:867`, `llm_worker/graphs/hybrid_search_and_score.py:626`, `llm_worker/graphs/hybrid_search_and_score.py:1389`, `llm_worker/graphs/score_results.py:353`
  - **Acceptance:** Graph internals pass typed models across nodes (no dict round-trips); search output uses typed structures at boundaries; dead branch removed; redundant model_dump removed; config/control/meta shape simplified; hardcoded jurisdiction/purpose removed; tests updated to enforce contracts; see `roadmap/collection-search-review.md`

- [x] **Collection Search fail-fast reset (drop legacy fallbacks)**:
  - **Details:** `roadmap/collection-search-review.md`
  - **Pointers:** `ai_core/graphs/technical/collection_search.py:1458` (CollectionSearchAdapter.run tool_context meta fallback), `ai_core/graphs/technical/collection_search.py:1605` (_HybridExecutorAdapter HybridResult coercion), `llm_worker/tasks.py:111` (control -> config merge)
  - **Acceptance:** `CollectionSearchAdapter.run()` requires `tool_context` in boundary state (no meta fallback); `_HybridExecutorAdapter` rejects non-`HybridResult` payloads instead of coercing; `llm_worker/tasks.py` no longer merges legacy `control` into `config`; tests updated to cover hard failures (invalid inputs) and remove legacy payload paths; breaking change documented via this backlog item

### Docs/Test touchpoints (checklist)

### P1 - High Value Cleanups (Low-Medium Effort)

- [x] **Collection Search strategy quality improvements**: add structured JSON output with examples and richer fallback queries.
  - **Details:** `roadmap/collection-search-review.md`
  - **Pointers:** `ai_core/services/collection_search/strategy.py:139`, `ai_core/services/collection_search/strategy.py:226`
  - **Acceptance:** Strategy prompt includes JSON-only schema + few-shot example; fallback strategy uses non-trivial query variants; tests cover schema parsing and fallback behavior

- [x] **Collection Search adaptive embedding weights**: adjust embedding vs heuristic weights by quality_mode or context.
  - **Details:** `roadmap/collection-search-review.md`
  - **Pointers:** `ai_core/graphs/technical/collection_search.py:738`
  - **Acceptance:** Weight selection is driven by quality_mode (or explicit profile mapping) and recorded in telemetry; default remains unchanged when not configured

- [x] **Collection Search hard graph timeout (worker-safe)**: add a whole-graph timeout without signal.alarm (Windows-safe).
  - **Details:** `roadmap/collection-search-review.md`
  - **Pointers:** `ai_core/graphs/technical/collection_search.py`, `llm_worker/tasks.py:run_graph`, `ai_core/services/__init__.py`
  - **Acceptance:** Graph execution enforces a hard cap via worker limits or explicit timeout wrapper; timeout produces a deterministic error payload; no signal.alarm usage

- [x] **Collection Search runtime performance cleanup (async + retries)**:
  - **Details:** `roadmap/collection-search-review.md`
  - **Pointers:** `ai_core/graphs/technical/collection_search.py:search_node`, `ai_core/graphs/technical/collection_search.py:_execute_parallel_searches`, `ai_core/graphs/technical/collection_search.py:_embed_with_retry`
  - **Acceptance:** search_node runs as true async (no ThreadPoolExecutor + asyncio.run); parallel search uses a single event loop; embedding retry uses non-blocking wait or moves embedding to async node; tests updated for async execution path

- [x] **Collection Search LLM client unification (strategy generator)**:
  - **Details:** `roadmap/collection-search-review.md`
  - **Pointers:** `ai_core/services/collection_search/strategy.py:llm_strategy_generator`
  - **Acceptance:** strategy generator uses central LLM client with JSON-mode support, Langfuse spans, and unified cost/error handling; direct litellm calls removed; tests updated for JSON response parsing

- [x] **Collection Search embedding client injection**:
  - **Details:** `roadmap/collection-search-review.md`
  - **Pointers:** `ai_core/graphs/technical/collection_search.py:embedding_rank_node`, `ai_core/graphs/technical/collection_search.py:_embed_with_retry`
  - **Acceptance:** EmbeddingClient injected via runtime (like WebSearchWorker) and reused per graph; embedding rank node supports dependency injection for tests; no per-call from_settings()

- [x] **Collection Search purpose handling in embedding rank**:
  - **Details:** `roadmap/collection-search-review.md`
  - **Pointers:** `ai_core/graphs/technical/collection_search.py:embedding_rank_node`
  - **Acceptance:** purpose does not dilute query embedding (e.g., weighted vector combination or moved to hybrid scoring); rationale documented in code; tests cover purpose-heavy input

- [ ] **Collection Search schema version tolerance (plan evolution)**:
  - **Details:** `roadmap/collection-search-review.md`
  - **Pointers:** `ai_core/graphs/technical/collection_search.py:CollectionSearchGraphRequest`, `ai_core/contracts/plans/*`
  - **Acceptance:** Graph accepts minor-compatible schema updates without failing hard; strict rejection for incompatible major changes; tests cover minor version acceptance and major version rejection

- [ ] **Activate AgenticChunker LLM Boundary Detection**: Implement `_detect_boundaries_llm()` in `ai_core/rag/chunking/agentic_chunker.py:354-379` (~25 LOC). Infrastructure complete (prompt template, Pydantic models, rate limiter, fallback logic). Needed to improve fallback quality for long documents (`token_count > max_tokens`). Details: `ai_core/rag/chunking/README.md#agentic-chunking`.

- [ ] **Simplify Adaptive Chunking Toggles (deferred post-MVP)**: Keep only two flags and one "new" path. Deferred because current flexibility benefits future use cases. Review after production experience shows which flags are unused. (pointers: `ai_core/rag/chunking/hybrid_chunker.py`, `ai_core/rag/chunking/late_chunker.py`, `ai_core/rag/chunking/README.md`)

### P2 - Long-term Improvements (High Effort)

- [ ] **Recursive Chunking for Long Documents**: Replace truncation fallback (`text[:2048]` in `late_chunker.py:1002`) with recursive chunking. When `token_count > max_tokens`, split into sections and chunk each section independently with proper boundary detection instead of hard truncation. (pointers: `ai_core/rag/chunking/late_chunker.py:984-1023`)

- [ ] **Robust Sentence Tokenizer**: Current splitting is fragile (`text.split(".")`) – breaks on abbreviations ("Dr."), decimals ("3.14"), URLs. Extract shared tokenizer to `ai_core/rag/chunking/utils.py` using regex with negative lookahead or `nltk.sent_tokenize`. (pointers: `agentic_chunker.py:347`, `late_chunker.py` sentence splitting)

## SOTA Developer RAG Chat (Pre-MVP)

**Roadmap**: [rag-chat-sota.md](rag-chat-sota.md)
**Total Effort**: ~3-4 Sprints (Medium-High Complexity)

### P1 - Core Implementation (High Value)

- [x] **SOTA-1: Define RagResponse schema for structured CoT outputs**:
  - **Details:** Create Pydantic models `RagResponse`, `RagReasoning`, `SourceRef` for structured LLM outputs with Chain-of-Thought reasoning, relevance scores, and follow-up suggestions.
  - **Pointers:** `ai_core/rag/schemas.py` (new), `ai_core/nodes/compose.py:ComposeOutput`
  - **Acceptance:** Schema passes `model_json_schema()` export; unit tests for round-trip serialization; documented in `ai_core/rag/README.md`
  - **Effort:** S (0.5 Sprint)

- [x] **SOTA-2: Create answer.v2 prompt with JSON output enforcement**:
  - **Details:** New prompt template forcing CoT reasoning and JSON-only output matching `RagResponse` schema. Includes explicit steps: Analyze, Identify Gaps, Synthesize.
  - **Pointers:** `ai_core/prompts/retriever/answer.v2.md` (new), `ai_core/prompts/retriever/answer.v1.md` (reference)
  - **Acceptance:** Prompt produces valid JSON for 95%+ test cases; few-shot examples for edge cases; version tracked in `ai_core/infra/prompts.py`
  - **Effort:** S (0.5 Sprint)
  - **Depends on:** SOTA-1

- [x] **SOTA-3: Backend refactoring for structured compose output**:
  - **Details:** Update `compose.py` to use v2 prompt with `response_format={"type": "json_object"}`; parse JSON into `RagResponse`; graceful fallback on parse failure; propagate debug metadata (latency, tokens, model, cost) with DEBUG/staff-only visibility.
  - **Pointers:** `ai_core/nodes/compose.py:_run`, `ai_core/llm/client.py:call` (line 326), `ai_core/services/rag_query.py`, `theme/views_chat.py:chat_submit`
  - **Acceptance:** v2 path replaces v1 (no feature flag); fallback to v1 on JSON error; all fields propagated to view; integration tests; Langfuse spans include `prompt_version: v2`
  - **Effort:** M (1 Sprint)
  - **Depends on:** SOTA-1, SOTA-2

- [x] **SOTA-4: Frontend "Glass Box" chat message display**:
  - **Details:** New `chat_message_debug.html` partial with collapsible sections: Final Answer (default), Thinking Process, Sources & Evidence with relevance bars, Debug footer (staff only).
  - **Pointers:** `theme/templates/theme/partials/chat_message.html`, `theme/templates/theme/partials/chat_message_debug.html` (new), `theme/views_chat.py:chat_submit`
  - **Acceptance:** Tabs/toggles work with Alpine.js; relevance bars 0-100%; debug footer staff-only; suggested follow-ups as clickable chips; responsive layout
  - **Effort:** M-L (1.5 Sprints)
  - **Depends on:** SOTA-3

### P2 - Testing & Polish

- [ ] **SOTA-5: Test coverage for SOTA RAG Chat**:
  - **Details:** Unit tests for JSON parsing/fallback; integration tests for full request cycle; E2E with Playwright for UI interactions.
  - **Pointers:** `ai_core/tests/nodes/test_compose_v2.py` (new), `ai_core/tests/rag/test_schemas.py` (new), `theme/tests/test_chat_submit_v2.py` (new)
  - **Acceptance:** Test coverage >= 80% for new code; E2E test for collapsible sections; malformed JSON fallback tested
  - **Effort:** S (0.5 Sprint)
  - **Depends on:** SOTA-3, SOTA-4

## Observability Cleanup


### Hygiene

- [ ] **Extract Chunker Utils**: Deduplicate shared logic between `LateChunker` and `AgenticChunker` into `ai_core/rag/chunking/utils.py`: sentence splitting, chunk ID generation (`_build_adaptive_chunk_id`), token counting. (pointers: `late_chunker.py`, `agentic_chunker.py`)

## Semantics / IDs

## Layering / boundaries

- [ ] **Explicit ToolContext in HybridScoreExecutor Protocol**: Replace `tenant_context: Mapping[str, Any]` with explicit `tool_context: ToolContext` parameter in `HybridScoreExecutor.run()`. Current workaround embeds `tool_context` in `tenant_context` dict for `tool_context_from_meta()` reconstruction. Clean solution: extend Protocol signature, update `_HybridExecutorAdapter`, remove dict-embedding hack.
  - **Pointers:** `ai_core/graphs/technical/collection_search.py:HybridScoreExecutor`, `ai_core/graphs/technical/collection_search.py:hybrid_score_node`, `ai_core/graphs/technical/collection_search.py:_HybridExecutorAdapter`, `llm_worker/graphs/hybrid_search_and_score.py`
  - **Acceptance:** `HybridScoreExecutor.run()` accepts `tool_context: ToolContext`; `tenant_context` removed or reduced to minimal tenant-specific fields; `_HybridExecutorAdapter` passes `tool_context` directly to sub-graph meta; no dict-embedding workaround needed

## Externalization readiness (later)

- [ ] Graph registry: add versioning + request/response model metadata (note: factory support already exists via `LazyGraphFactory`)


## Observability / operations


## Agent navigation (docs-only drift)

## Hygiene (lowest priority)

- [ ] Documentation hygiene: remove encoding artifacts in docs/strings (purely textual)
