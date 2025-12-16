# Backlog (prioritized)

This is a curated list of open work items for vibe-coding execution.
Top-to-bottom order within each section is priority order.
Prefer linking each item to concrete code paths (and optionally to an issue).

## Next up (highest leverage)

- [ ] Drift #3: crawler ingestion persistence parity with upload path (pointers: `docs/architecture/drift3-critical-finding.md`, `crawler/worker.py`, `ai_core/graphs/crawler_ingestion_graph.py`, `ai_core/services/__init__.py:_persist_via_repository`; acceptance: crawler path can persist/create document without relying on "parallel registration", with a migration-safe fallback)
- [ ] LangGraph build pattern: migrate module-level singleton graphs to factories (start: `ai_core/graphs/retrieval_augmented_generation.py`)
- [ ] Workbench exception: allow UI/views to trigger technical graphs only in `DEBUG` (explicit guard at endpoint level)
- [ ] Graph convergence: classify and migrate remaining custom graphs to real `langgraph` (inventory `ai_core/graphs/`, focus: `crawler_ingestion_graph.py`, `framework_analysis_graph.py`, `retrieval_augmented_generation.py`)

## Semantics / IDs

- [ ] Semantics vNext: keep `case_id` required; standardize dev default `case_id="local"` and ensure that case exists per tenant (`ai_core/graph/schemas.py:normalize_meta`, `cases/*`)
- [ ] ID propagation clarity: document and/or enforce the "runtime context injection" pattern for graphs (pointers: `docs/audit/contract_divergence_report.md#id-propagation-compliance`, `ai_core/graphs/upload_ingestion_graph.py`, `ai_core/graph/schemas.py`, `ai_core/contracts/scope.py`; acceptance: one canonical pattern, documented in code/doc pointers, so tenant/trace/run IDs are not "implicit knowledge")

## Layering / boundaries

- [ ] Graph package split: business vs technical graphs (e.g. `ai_core/graphs/business/` vs `ai_core/graphs/technical/`)
- [ ] Enforce import direction (business -> technical) via a lightweight repo test (no new dependencies)
- [ ] Capability-first rule: graphs call explicit capabilities (`ai_core/nodes/`, `ai_core/tools/`, domain services) instead of embedding ad-hoc logic

## Externalization readiness (later)

- [ ] Contract drift: migrate `ai_core/rag/schemas.py:Chunk` off `@dataclass` (pointers: `docs/audit/contract_divergence_report.md#1-chunk-schema-non-compliance`, `ai_core/rag/schemas.py`, `ai_core/rag/vector_client.py` (mutating `chunk.meta`), `ai_core/api.py`, `ai_core/tasks.py`; acceptance: a Pydantic `ChunkV2` (frozen) exists, call sites migrated, legacy kept only behind explicit adapter until removed)
- [ ] Technical graph interface: versioned Pydantic I/O (avoid implicit state dicts) for externalization readiness
- [ ] Graph executor boundary: local vs celery vs remote execution (business graphs call executor; planned location `ai_core/graph/execution/`)
- [ ] Graph registry: once a real GraphExecutor exists, extend `ai_core/graph/registry.py` to support factories + versioning + optional request/response model metadata

## Domain boundary cleanup (later)

- [ ] Framework analysis graph: separate orchestration vs persistence boundary (`ai_core/graphs/framework_analysis_graph.py`, `documents/framework_models.py`)

## Observability / operations

- [ ] Review queue + retry handling across `agents`/`ingestion` (`llm_worker/tasks.py`, `ai_core/tasks.py`, `common/celery.py`)

## Agent navigation (docs-only drift)

- [ ] Fix stale docs links and naming drift around `docs/architektur/` vs `docs/architecture/` (pointers: `docs/audit/cleanup_report.md#link-fixes-required`, `docs/litellm/admin-gui.md`, `docs/development/onboarding.md`, `docs/documents/contracts-reference.md`, `docs/rag/overview.md`; acceptance: no references to missing/renamed paths; LLM navigation lands on the canonical docs/code)

## Hygiene (lowest priority)

- [ ] Documentation hygiene: remove encoding artifacts in docs/strings (purely textual)
