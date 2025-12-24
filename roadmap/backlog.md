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

## Semantics / IDs

- [x] Semantics vNext: keep `case_id` required; standardize defaults (`general` for API views, `dev-case-local` for `/rag-tools`) and ensure that case exists per tenant (`ai_core/graph/schemas.py:normalize_meta`, `ai_core/views.py`, `theme/views.py`, `cases/*`)
- [ ] ID propagation clarity: document and/or enforce the "runtime context injection" pattern for graphs (pointers: `docs/audit/contract_divergence_report.md#id-propagation-compliance`, `ai_core/graphs/upload_ingestion_graph.py`, `ai_core/graph/schemas.py`, `ai_core/contracts/scope.py`; acceptance: one canonical pattern, documented in code/doc pointers, so tenant/trace/run IDs are not "implicit knowledge")
- [x] Breaking meta contract: enforce `scope_context` as the only ID source in graph/tool meta (pointers: `ai_core/graph/schemas.py:normalize_meta`, `ai_core/views.py:_prepare_request`, `llm_worker/runner.py:submit_worker_task`, `common/celery.py:ContextTask._from_meta`; acceptance: no top-level ID reads, all callers use `meta["scope_context"]`, tests updated and passing)

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
