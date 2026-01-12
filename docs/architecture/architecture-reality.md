# Architecture reality (code-backed inventory)

This document is a “what exists in code” snapshot. It lists the concrete code paths that currently implement the 4-layer model described in `docs/architecture/4-layer-firm-hierarchy.md`.

## Layer 1 — UI / HTTP boundaries

- UI templates + workbench pages: `theme/` (e.g. `theme/views.py`, `theme/templates/`)
- Main URL routing: `noesis2/urls.py`
- AI Core REST endpoints: `ai_core/views.py` and `ai_core/urls.py`, `ai_core/urls_v1.py`
- Worker polling/submit API: `llm_worker/views.py` and `llm_worker/urls.py`

## Layer 2 — Business context (Cases)

- Case model identifier used by APIs: `cases/models.py:Case.external_id`
- Case resolver: `cases/services.py:resolve_case`
- Case API endpoints: `cases/api.py`
- Case lifecycle events used by agent tasks: `cases/integration.py` (e.g. used from `llm_worker/tasks.py`)

### Business-heavy graph example (domain logic)

The repository currently contains at least one graph that implements domain/business logic (beyond technical orchestration) while still living under `ai_core/graphs/`:

- Framework analysis (agreement structure, versioning, persistence into domain models): `ai_core/graphs/business/framework_analysis_graph.py` (persists `documents/framework_models.py:FrameworkProfile` / `FrameworkDocument`)

## Layer 3 — Orchestration (Graphs / Domain services)

### Graph protocol + execution

- Graph runner protocol + checkpointer: `ai_core/graph/core.py`
- Graph meta normalization (`scope_context` + `tool_context`): `ai_core/graph/schemas.py:normalize_meta`
- Web execution orchestration (sync + worker offload): `ai_core/services/__init__.py:execute_graph`

### Graph construction pattern (LangGraph)

The repository uses “build_*” factory functions for many graph modules. Preferred pattern is to keep graph construction as a factory (new instance / compiled graph per call) instead of module-level singleton graphs, to support per-run configuration and future remote execution.

Factory examples:

- `ai_core/graphs/technical/universal_ingestion_graph.py:build_universal_ingestion_graph`
- `ai_core/graphs/web_acquisition_graph.py:build_web_acquisition_graph`
- `ai_core/graphs/technical/collection_search.py:build_compiled_graph`
- `ai_core/graphs/technical/retrieval_augmented_generation.py:build_graph`

### Graph implementations present in the repo

**Layer 2 - Business Graphs** (`ai_core/graphs/business/`):

- `framework_analysis_graph.py` (Domain logic: Agreement structure, versioning, persistence into domain models)

**Layer 3 - Technical Graphs** (`ai_core/graphs/technical/`):

- `collection_search.py` (includes HITL decision structures)
- `retrieval_augmented_generation.py`
- `universal_ingestion_graph.py`
- `document_service.py`
- `cost_tracking.py`
- `info_intake.py`

Additional technical graph modules in `ai_core/graphs/`:

- `web_acquisition_graph.py`

### Document orchestration boundary

- Document lifecycle + collection operations: `documents/domain_service.py:DocumentDomainService`
- Facades used by workers/graphs: `documents/service_facade.py`
- Lifecycle state machine: `documents/lifecycle.py`

## Layer 4 — Workers (Celery + I/O)

### Queues and tasks (code)

- Agents worker task: `llm_worker/tasks.py:run_graph` (`queue="agents-high"` default, `agents-low` for background)
- Ingestion graph task: `ai_core/tasks.py:run_ingestion_graph` (`queue="ingestion"`)

### Queue wiring (local compose)

- `docker-compose.yml` and `docker-compose.dev.yml` define a unified worker with explicit queues:
  - `worker`: `-Q agents-high,agents-low,crawler,celery,ingestion,ingestion-bulk,dead_letter,rag_delete`

### Context propagation (tasks)

- Task base + scope helpers: `common/celery.py` (`ScopedTask`, `with_scope_apply_async`)
- Canonical header constants used for propagation: `common/constants.py`

## HITL (Human-in-the-loop) code locations

- HITL node + decision payloads: `ai_core/graphs/technical/collection_search.py`
- HITL references in graph documentation: `ai_core/graph/README.md` and `ai_core/graphs/technical/collection_search_README.md`
- UI references mentioning HITL queue: `theme/views.py` (search for “HITL”)

## Roadmap pointer (non-code planning)

If you maintain a planning document for closing gaps or aligning naming, keep it in `roadmap/` (e.g. `roadmap/architecture-consolidation-2025.md`) and treat it as a plan/hypothesis, not as a runtime contract.
