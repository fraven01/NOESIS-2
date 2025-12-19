# Architecture Consolidation Roadmap 2025

**Status**: Draft  
**Owner**: Architecture Team  
**Goal**: Align architecture documentation with code and structure the graph layers for future externalization of worker-heavy execution.

## Scope and status

This file is a planning artifact; it is not a runtime contract. Runtime behavior and contracts are defined in code.

## References (code-backed)

- 4-layer lens: `docs/architecture/4-layer-firm-hierarchy.md`
- Current inventory snapshot: `docs/architecture/architecture-reality.md`
- LLM entry contract / code navigation: `AGENTS.md`
- Backlog index: `roadmap/README.md` and `roadmap/backlog.md`

## Workstreams (no timeboxing)

### A) Documentation alignment

- [x] Keep `docs/architecture/architecture-reality.md` as the canonical “what exists in code” inventory (layer -> code paths -> entry points).
- [x] Keep `docs/architecture/4-layer-firm-hierarchy.md` as the stable mental model and ensure it links to the inventory snapshot.
- [x] Add `docs/architecture/layer-contracts.md` (explanatory) with pointers to concrete boundaries:
  - Layer 1 ↔ Layer 2: HTTP/API boundaries (OpenAPI + `ai_core/views.py`, `noesis2/urls.py`)
  - Layer 2 ↔ Layer 3: graph meta + execution (`ai_core/graph/schemas.py`, `ai_core/graph/core.py`, `ai_core/services/__init__.py`)
  - Layer 3 ↔ Layer 4: task boundaries (`llm_worker/tasks.py`, `ai_core/tasks.py`, `common/celery.py`)
- [ ] ADRs (only if needed): cases/workflows, graph orchestration, sync vs worker-offload boundaries (link to concrete code paths).

### B) Graph layering (business vs technical)

Target idea:

- Technical graphs = reusable “capabilities” that orchestrate workers/I/O and remain domain-agnostic.
- Business graphs = case/workflow-aware orchestration that calls technical graphs and encodes business sequencing/HITL.
- UI/Views call business graphs by default; a developer workbench may call technical graphs directly.

Planned work:

- [x] Introduce package structure split for graphs (e.g. `ai_core/graphs/technical/` and `ai_core/graphs/business/`) without introducing Django-app overhead.
- [x] Define import direction expectations (business -> technical; technical should not depend on business).
- [ ] Enforce import direction later via a simple repo test (no new dependencies): fail builds if `ai_core/graphs/technical/**` imports `ai_core/graphs/business/**` (planning only; see also `roadmap/backlog.md`).
- [x] Identify current candidates:
  - Business-heavy graph: `ai_core/graphs/framework_analysis_graph.py`
  - Technical graphs: ingestion/retrieval/search graphs under `ai_core/graphs/` (see inventory doc)
- [x] Workbench exception policy (decision): technical-graph triggering from UI/views is `DEBUG`-only (explicitly guard the workbench endpoints).

### C) Externalization readiness (scale-out path)

Goal: allow moving worker-heavy execution out of the Django monolith later without rewriting business orchestration.

- [x] Establish a stable cross-graph interface for technical graphs (versioned Pydantic input/output; avoid implicit state-dict APIs).
- [ ] Add a “graph executor” boundary (local/in-process vs Celery vs remote service) so business graphs call an executor API instead of importing execution mechanism directly.
- [ ] Decide/anchor location for the execution layer: place the executor under `ai_core/graph/execution/` (next to `ai_core/graph/schemas.py`, `ai_core/graph/core.py`), so “graph meta normalization” and “graph execution” stay in the same module family.
- [ ] Define a minimal API surface (planning sketch): `executor.run(name, input, context) -> output` (sync) plus an optional async/remote submission API (`executor.submit(...) -> handle/task_id`), with `trace_id` propagation preserved.
- [ ] Reuse/extend the existing in-memory graph registry (`ai_core/graph/registry.py`) as the canonical `graph_name -> runner/factory` mapping (and later add versioning + request/response model metadata).
- [ ] Standardize context propagation across boundaries (`tenant_id`, `trace_id`, exactly one runtime ID, plus `workflow_id`/`case_id` when applicable) using existing context models.
- [ ] Observability: require end-to-end correlation (`trace_id`) across local/celery/remote execution.

### D) Boundary cleanup candidate: framework analysis persistence

Motivation: `ai_core/graphs/framework_analysis_graph.py` currently mixes domain/business decisions with direct ORM persistence (`FrameworkProfile`, `FrameworkDocument`, `DocumentCollection`) under `django.db.transaction`.

- [x] Extract persistence/versioning into a dedicated service boundary (e.g. in `documents/`) and call it from `ai_core/graphs/framework_analysis_graph.py` instead of direct ORM writes.
- [x] Keep the graph focused on orchestration (LLM calls, retrieval nodes, assembling the analysis output) and delegate side effects (DB writes, version toggling) to the service boundary.
- [x] Add/adjust tests to cover the service boundary and the graph integration path.

### F) “Real LangGraph” alignment + capability-first graphs

Motivation: today `ai_core/graphs/` contains a mix of implementations:

- “Real” LangGraph graphs (`langgraph.graph.StateGraph`): e.g. `ai_core/graphs/upload_ingestion_graph.py`, `ai_core/graphs/external_knowledge_graph.py`, `ai_core/graphs/collection_search.py`
- “LangGraph-inspired” / custom orchestrators: e.g. `ai_core/graphs/crawler_ingestion_graph.py`, `ai_core/graphs/framework_analysis_graph.py`, `ai_core/graphs/retrieval_augmented_generation.py`

Goal: converge on a single, capability-first pattern where:

- Technical graphs are true LangGraph graphs (or wrap a compiled LangGraph graph behind the existing `GraphRunner.run(state, meta)` boundary).
- Graph nodes call explicit technical capabilities (nodes/tools/services) instead of embedding ad-hoc logic.
- Placeholder/teaser logic (“halluzinierter Quatsch”, TODO-only flows) is removed or replaced by real capability calls.

Planned work:

- [ ] Create a graph inventory table (per file in `ai_core/graphs/`) that classifies each graph as `langgraph` vs `custom`, and records its inputs/outputs and key capability dependencies.
- [ ] Standardize LangGraph construction: prefer factory functions (e.g. `build_graph()` / `build_*_graph()`) that return a new graph instance/compiled graph, avoid module-level singleton graphs.
- [ ] Define a canonical “technical capability” surface (existing candidates: `ai_core/nodes/`, `ai_core/tools/`, and service boundaries like `documents/domain_service.py`) and require graphs to call those surfaces rather than implementing one-off logic inline.
- [ ] For each custom graph, define the target LangGraph structure (state schema + nodes + edges) and implement it behind the existing execution entry points.
- [ ] Remove/replace placeholder logic (TODO-only branches, comment-driven behavior) in graphs and docs; keep only code-backed behavior and explicit stubs with a ticket/backlog reference.

## LLM-driven execution (process note)

- The intent is to describe each roadmap item with enough code pointers and acceptance criteria that it can be executed by an LLM agent with minimal human clarification.

### G) Semantics vNext: `case_id` / `workflow_id` (keep `case_id` required)

Decision intent:

- Keep `case_id` in the system and treat it as the stable “business container” for a long-running endeavor (e.g. introduction/negotiation of a specific software with the Betriebsrat).
- Keep the current behavior that graph execution expects a `case_id` (today `ai_core/graph/schemas.py:normalize_meta` rejects requests without `case_id`).

Pragmatic dev convention (pre-MVP):

- Use a reserved/default case identifier for ad-hoc/dev flows: `case_id="local"`.
- For “everything else / misc” during development, use a single collecting case per tenant (e.g. also `local`, or a second reserved external_id like `misc`). Choose one and keep it consistent across tooling.

Where this shows up today (code-backed pointers):

- HTTP/graph meta normalization requires `case_id`: `ai_core/graph/schemas.py:normalize_meta`
- Some endpoints already default `case_id` to `local` when missing: `llm_worker/views.py`
- Case identity in the DB is `Case.external_id`: `cases/models.py`, `cases/api.py`

Proposed semantics split (to avoid overloading `workflow_id`):

- `case_id`: “project/procedure” container (software X introduction).
- `workflow_id`: a repeatable sub-process within a case (intake, analysis, negotiation draft, review, rollout check).
- “Where the project is”: use case state fields (`status`/`phase`) rather than encoding lifecycle into `workflow_id` (see `cases/*`).

Implementation candidates (planning, not current behavior):

- Ensure the reserved dev case exists per tenant on bootstrap (create `Case(external_id="local", ...)`).
- Allow `case_id="local"` in `DEBUG` and staging-like environments for ad-hoc flows; keep production strict (block reserved dev case IDs unless explicitly enabled).
- Implement the allow/deny policy at meta normalization (`ai_core/graph/schemas.py:normalize_meta`) so it applies consistently to HTTP + worker execution.

### H) Documentation hygiene: encoding artifacts (mojibake)

- [ ] Remove “ƒ…”/mojibake artifacts in docs/strings introduced by encoding issues; keep changes purely textual (no semantic edits).
