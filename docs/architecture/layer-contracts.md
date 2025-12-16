# Layer Contracts (The "Firm Hierarchy" Laws)

This document defines the strict API contracts and allowed call patterns between the 4 layers of the NOESIS-2 architecture.

**Reference**: `docs/architecture/4-layer-firm-hierarchy.md` (Mental Model)

---

## The 4 Layers

1. **Layer 1 (Customer/UI)**: `theme/`, `noesis2/`, `ai_core/views.py`
2. **Layer 2 (Business/Case)**: `cases/`, `ai_core/graphs/business/`
3. **Layer 3 (Technical/Capability)**: `ai_core/graphs/technical/`, `ai_core/services/`, `documents/`
4. **Layer 4 (Worker/Execution)**: `llm_worker/`, `ai_core/tasks.py`, `common/celery.py`

---

## 1. Import & Call Direction Rules

### ✅ Allowed (Downward)

* **L1 -> L2**: UI calls Case APIs to get context (e.g., `theme` views reading `Case` models).
* **L1 -> L3**: UI triggers Graph execution via `ai_core/services` (orchestrator).
* **L2 -> L3**: Business Graphs (`ai_core/graphs/business/`) call Technical Graphs (`ai_core/graphs/technical/`) or Capability Services.
* **L3 -> L4**: Orchestrator (`ai_core/services`) dispatches Tasks to Celery Workers.

### ❌ Forbidden (Upward/Circular)

* **L3 -> L2**: Technical Graphs MUST NOT import `cases/` models directly. They should receive IDs (`case_id`) and Context strings, but not depend on Business Logic classes.
* **L4 -> L1**: Workers MUST NOT depend on Views or UI logic.

---

## 2. API Boundaries

### Layer 1 ↔ Layer 2 (Context)

* **Contract**: Django Models (`Case`), Services (`ResolveCase`).
* **Pass**: `case_id`, `tenant_id`.

### Layer 2/1 ↔ Layer 3 (Graph Execution)

* **Contract**: `GraphRunner.run(state, meta)`.
* **Input**: `state` (Dict), `meta` (Standardized Meta Dict from `ai_core/graph/schemas.py`).
* **Output**: `(final_state, result)`.

### Layer 3 ↔ Layer 4 (Async Offload)

* **Contract**: Celery Tasks (`run_graph`, `run_ingestion`).
* **Payload**: Serializable (JSON/Pickle) versions of `state` and `meta`.
* **Context**: `ScopeContext` must be propagated via Celery Headers (`common/celery.py`).

---

## 3. Data Persistence Ownership

* **Business Data** (Agreements, Proposals): Owned by **Layer 2** (`cases/`, `documents/framework_models.py`).
* **Technical Data** (Embeddings, Ingestion State): Owned by **Layer 3** (`documents/`).
* **Business Graphs (`L2`)** vs **Technical Graphs (`L3`)**:
  * **Business Graphs** define *what* happens for a user flow and *may* write final business results (via Service calls, not direct ORM if possible).
  * **Technical Graphs** are pure capabilities (Search, Ingest) and write only their internal technical state (Vector DB, Blob Store).
