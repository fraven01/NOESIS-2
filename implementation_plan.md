# Implementation Plan - ID Fixes and Harmonization

## Goal

Fix a critical runtime bug in `HybridSearchAndScoreGraph` where `ToolContext` is instantiated without a `run_id` or `ingestion_run_id`, and harmonize the generation of `ingestion_run_id` across the system.

## User Review Required
>
> [!IMPORTANT]
> This plan modifies `HybridSearchAndScoreGraph` to strictly require `run_id` or `ingestion_run_id` in its input metadata. This might affect callers if they are not currently providing these IDs.

## Proposed Changes

### LLM Worker

#### [MODIFY] [hybrid_search_and_score.py](file:///f:/NOESIS-2/NOESIS-2/llm_worker/graphs/hybrid_search_and_score.py)

- Update `_retrieve_rag_context` to extract `run_id` and `ingestion_run_id` from `meta`.
- Pass these IDs to the `ToolContext` constructor, ensuring the XOR rule is respected (prefer `run_id` if both are present, or handle as per `ScopeContext` logic).
- Ensure `run_id` is generated if missing, to guarantee `ToolContext` validity.

### AI Core

#### [MODIFY] [crawler_ingestion_graph.py](file:///f:/NOESIS-2/NOESIS-2/ai_core/graphs/crawler_ingestion_graph.py)

- Standardize `ingestion_run_id` generation to use `uuid4()` if not provided, consistent with other parts of the system.

## Verification Plan

### Automated Tests

- Run `test_hybrid_graph.py` to ensure the fix works and doesn't introduce regressions.
- Run `test_crawler_ingestion_graph.py` (if available) or relevant ingestion tests to verify the harmonization.

### Manual Verification

- None required for this backend logic fix.
