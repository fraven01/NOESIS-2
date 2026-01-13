# RAG Tools Architecture & Logic Leakage Analysis

## Context

This analysis compares the implementation of `rag-tools/` Views (UI Layer) against the underlying Business Graphs (Logic Layer).

**Findings Summary:**
The Views are not just "Synchronous Orchestrators" (as identified in the initial pass), but are actively **performing business logic** that belongs in the Graphs. This violates the "Thick Graph, Thin View" architecture.

---

## 1. Logic Leakage Analysis (View vs. Graph)

### 1.1 RAG Chat: History Management

**Status:** 🔴 **Leakage Detected**

- **Graph (`retrieval_augmented_generation.py`)**:
  - Defines `chat_history` in State.
  - Uses history for `standalone_question` generation.
  - **Missing:** Does *not* append the current turn (Question/Answer) to the history. Does *not* manage truncation.
- **View (`views_chat.py`)**:
  - Loads history from Checkpointer.
  - Runs Graph.
  - **Leak Logic:** `append_history(user)`, `append_history(assistant)`, `trim_history(limit)`.
  - Manually saves back to Checkpointer.
- **Problem**: Possible race conditions, view needs to know about Checkpointer internals, inconsistent history handling if accessed via API vs View.
- **Target Architecture**:
  - Graph's final node (or a dedicated `memory` node) should append Q/A to `chat_history` in State.
  - Graph config should define `history_limit`.
  - Checkpointer automatically persists State (including new history) at end of run.
  - View only passes `question` and `thread_id`.

### 1.2 Collection Search: Auto-Ingestion

**Status:** 🔴 **Broken Pattern / Leakage**

- **Graph (`collection_search.py`)**:
  - Has `auto_ingest` boolean in Input.
  - Has `auto_ingest_node` which returns `{"ingestion": {"status": "skipped"}}` (Placeholder!).
  - Has `build_plan_node` which *selects* URLs for ingestion but does not *execute*.
  - Has `optionally_delegate_node` which acts as a placeholder.
- **View (`views_web_search.py`)**:
  - `web_search()` runs the graph (Sync).
  - `web_search_ingest_selected()` is a separate endpoint that calls `CrawlerManager`.
- **Problem**: `auto_ingest=True` in Graph Input does nothing effectively. The UI forces a "Search -> User Review -> Ingest" flow (or separate calls).
- **Target Architecture**:
  - `CollectionSearchGraph` should have a real `delegate` node that calls `UniversalIngestionGraph` (or `CrawlerManager`) for the selected URLs if `auto_ingest=True`.
  - View becomes "Fire and Forget" for Auto-Ingest scenarios.

### 1.3 Web Acquisition: Select Best & Ingest

**Status:** 🟡 **Missing Capability**

- **Graph (`web_acquisition_graph.py`)**:
  - Has `select_node` (`select_search_candidates`).
  - Logic is "Acquire Only" (returns results).
  - No Ingestion capability.
- **View (`views_web_search.py`)**:
  - Handles "Ingest Selected" via separate endpoint.
- **Gap**: There is no atomic "Find X and Ingest it" operation callable via API/Graph.
- **Target Architecture**:
  - Compose: `WebAcquisitionGraph` -> `UniversalIngestionGraph`.
  - Or: `CollectionSearchGraph` (which is meant to be the Orchestrator) should use `WebAcquisitionGraph` as a *Tool/Sub-graph*.
  - Currently `CollectionSearch` uses `WebSearchWorker` (Tool) directly, duplicating `WebAcquisition` logic.

### 1.4 Crawler: Submit Logic

**Status:** 🔴 **Bypassed Layer**

- **View (`views_ingestion.py`)**:
  - Calls `run_crawler_runner` directly.
- **Service (`crawler/manager.py`)**:
  - `CrawlerManager` exists to orchestrate this (Dispatch, Persistence, Task Queueing).
- **Problem**: View bypasses the Manager domain service.

---

## 2. Refactoring Recommendations (Deep)

### P1: "Smart RAG Graph" (Memory Management)

Move history management into the Graph.

- **Refactor `compose_node`**: Return `chat_history` update operation (append Q+A).
- **Refactor Graph State**: Use reducer for `chat_history` (e.g., `operator.add` or internal logic) if using state-channels, or simple list append if simple state.
- **View**: Remove `load_history`, `append`, `trim`, `save`. Just `execute_graph(input={question})`.

### P1: "Autonomic Collection Search" (Delegation)

Implement the missing `delegate` logic in `CollectionSearchGraph`.

- **Implement `optionally_delegate_node`**:
  - If `plan.execution_mode == 'acquire_and_ingest'`:
  - Dispatch to `CrawlerManager` (or call `UniversalIngestionGraph` sub-graph).
  - Return `ingestion_task_ids`.

### P2: "Composition Over Duplication"

`CollectionSearchGraph` currently manually implements Search Strategy, Execution, and Ranking.

- **Long Term**: `CollectionSearchGraph` should just orchestrate:
    1. Call `StrategyGraph` (Plan)
    2. Call `WebAcquisitionGraph` (Execute)
    3. Call `UniversalIngestionGraph` (Ingest - optional)
- Currently, `CollectionSearchGraph` is a "God Graph" doing too much low-level work (embedding interaction, worker timeouts, etc.).

---

## 3. Revised Roadmap Strategy

The previous "Quick Wins" (Context, Async Wrappers) are valid stability fixes. However, true architectural alignment requires:

1. **Phase 1 (Stabilize)**: Quick Wins + Async (Stop blocking threads).
2. **Phase 2 (Thick Graph)**: Move History to RAG Graph. Implement Delegation in Collection Search.
3. **Phase 3 (Clean View)**: Views become pure Input-Mapping -> Graph-Invoke -> Template-Render pipelines. No business logic.

## 4. Immediate Action Items (Additions to Quick Wins)

- [ ] **RAG Graph History**: Add `chat_history` append logic to `retrieval_augmented_generation.py`.
- [ ] **Collection Search Delegation**: Implement `optionally_delegate_node` to trigger Crawler.

This analysis replaces the previous "Surface Level" finding.
