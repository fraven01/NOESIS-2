# ID Usage Overview

This document provides an overview of how IDs are defined, generated, and used across the system, specifically focusing on Workers and Graphs. It highlights observed inconsistencies and potential compliance issues with `docs/architecture/id-semantics.md`.

## ID Definitions (from `id-semantics.md`)

| ID | Scope | Source |
|---|---|---|
| `tenant_id` | Permanent per tenant | Header, Token, Worker Scope |
| `case_id` | Business Case | Header, Dispatcher |
| `workflow_id` | Business Workflow | Dispatcher (stable per workflow type) |
| `run_id` | Single Graph Execution | Graph Dispatcher or Worker |
| `ingestion_run_id` | Single Ingestion Job | Ingestion Entry Point |
| `trace_id` | Request/Flow | Web Layer / API |

**Key Rules:**

- Mutual exclusion: `run_id` XOR `ingestion_run_id`.
- `ingestion_run_id` is used for ingestion flows instead of `run_id`.

## Usage by Component

### Workers

#### Crawler Worker (`crawler/worker.py`)

- **IDs Handled:** `tenant_id`, `case_id`, `trace_id`, `crawl_id`, `document_id`.
- **Behavior:**
  - `tenant_id` is mandatory.
  - `case_id` and `trace_id` are optional and passed through.
  - **Missing IDs:** Does **not** generate or handle `ingestion_run_id` or `run_id`.
  - **Delegation:** Delegates processing to `ingestion_task` (likely `CrawlerIngestionGraph`), passing only `tenant_id`, `case_id`, `trace_id` in the payload/meta.

#### LLM Worker (`llm_worker/tasks.py`)

- **IDs Handled:** `tenant_id`, `case_id`, `trace_id`.
- **Behavior:**
  - Accepts these IDs as arguments for scoping.
  - Passes `state` and `meta` to the graph runner.
  - Does not explicitly enforce `run_id` or `workflow_id` at the task level, relying on the payload `meta`.

### Graphs

#### Crawler Ingestion Graph (`ai_core/graphs/crawler_ingestion_graph.py`)

- **IDs Handled:** `tenant_id`, `case_id`, `workflow_id`, `ingestion_run_id`, `trace_id`.
- **Behavior:**
  - **Generation:** Generates `ingestion_run_id` and `trace_id` if they are missing in the input state.
  - **Context:** Creates a `ScopeContext` using the (potentially generated) `ingestion_run_id`.
  - **Compliance:** Complies with `id-semantics.md` by generating the ID at the entry point (if the graph is considered the entry point relative to the worker).

#### Upload Ingestion Graph (`ai_core/graphs/upload_ingestion_graph.py`)

- **IDs Handled:** `tenant_id`, `workflow_id`, `case_id`, `trace_id`, `ingestion_run_id`.
- **Behavior:**
  - **Requirement:** Explicitly requires `ingestion_run_id` in the input payload (`_node_accept_upload`).
  - **Inconsistency:** Unlike Crawler Graph, it does *not* generate it if missing; it errors/fails if not provided (implied by `_require_str`).

#### Hybrid Search & Score Graph (`llm_worker/graphs/hybrid_search_and_score.py`)

- **IDs Handled:** `tenant_id`, `case_id`, `trace_id`.
- **Potential Issue:**
  - Instantiates `ToolContext` in `_retrieve_rag_context` **without** providing `run_id` or `ingestion_run_id`.
  - `ToolContext` validation requires exactly one of these IDs.
  - **Risk:** This code path likely fails runtime validation if `ToolContext` validation is active.

#### Collection Search Graph (`ai_core/graphs/collection_search.py`)

- **IDs Handled:** `tenant_id`, `workflow_id`, `case_id`, `run_id`, `ingestion_run_id`.
- **Behavior:**
  - Uses `ids` object (likely `ScopeContext`) to access IDs.
  - Handles both `run_id` and `ingestion_run_id` (mutual exclusion logic present).

## Observed Inconsistencies & Issues

1. **Ingestion ID Generation Strategy:**
    - `CrawlerWorker` does not generate `ingestion_run_id`; `CrawlerIngestionGraph` generates it.
    - `UploadIngestionGraph` requires `ingestion_run_id` from the caller.
    - **Recommendation:** Standardize where `ingestion_run_id` is generated. If the Worker is the entry point, it should probably generate it to ensure log correlation from the very beginning of the task.

2. **Missing Runtime IDs in Hybrid Search:**
    - `HybridSearchAndScoreGraph` seems to be missing `run_id` propagation to `ToolContext`. This violates the contract that Tools require a runtime ID.

3. **Worker ID Awareness:**
    - Workers are generally "loose" with IDs, often treating them as optional metadata, whereas Graphs and Tools have stricter requirements (`ScopeContext`, `ToolContext`).

## Next Steps

1. **Fix `HybridSearchAndScoreGraph`**: Ensure `run_id` is passed or generated and propagated to `ToolContext`.
2. **Standardize Ingestion ID**: Decide if `CrawlerWorker` should generate `ingestion_run_id` (aligning with "Entry Point" definition) or if the Graph generation is sufficient.
3. **Verify Upload Flow**: Ensure the caller of `UploadIngestionGraph` is correctly generating `ingestion_run_id`.
