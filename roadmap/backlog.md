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
- [ ] Review later: framework analysis graph convergence (`ai_core/graphs/business/framework_analysis_graph.py`)
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
