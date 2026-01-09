# RAG Phase 2: Retrieval Upgrade (Short Spec)

Status: Draft
Date: 2026-01-09
Scope: Advanced retrieval in `ai_core/graphs/technical/retrieval_augmented_generation.py`

## Goal

Improve answer quality by adding query transformation, optional re-ranking, and
confidence-based retrieval loops without changing core contracts (ToolContext,
Graph IO, BusinessContext).

## Inputs and Constraints

- Graph entry: `ai_core/graphs/technical/retrieval_augmented_generation.py`
- Retrieval node: `ai_core/nodes/retrieve.py`
- Existing strategy + re-rank logic lives in `ai_core/graphs/technical/collection_search.py`
- Hybrid parameters allowed: `ai_core/nodes/_hybrid_params.py`
- No new IDs or meta keys without confirmation (per AGENTS.md stop conditions).

## Options

### Option A: Reuse collection_search components

Reuse the LLM strategy generator and hybrid scoring adapter already used in
`collection_search`:

- Strategy: `_llm_strategy_generator` and `SearchStrategy` logic.
- Re-rank: `llm_worker.graphs.hybrid_search_and_score` adapter already in
  `CollectionSearchAdapter.build_graph`.

Pros:
- Minimal new code, proven in collection search.
- Same scoring semantics as other workflows.

Cons:
- Some logic is shaped around web search and candidates, not vector chunks.
- Adapter layer may need conversion between chunk matches and scoring inputs.

### Option B: Extract shared RAG retrieval modules

Create shared modules (extracted from `collection_search`) and import into both
graphs:

- `ai_core/rag/strategy.py` for query transformation
- `ai_core/rag/rerank.py` for scoring adapters

Pros:
- Clear separation of responsibilities, reuse across graphs.
- Easier to test in isolation.

Cons:
- New modules may require new identifiers or contracts.
- More refactor overhead.

## Proposed Recommendation (Short-Term)

Start with Option A (reuse) to validate impact. Keep any adapters local to the
RAG graph to avoid new public contracts. If reuse stabilizes, extract in a
later refactor.

## Target Flow (Phase 2)

1) Transform Query
   - Input: user question
   - Output: list of 3-5 queries
2) Retrieve (per query)
   - Use existing hybrid retrieval with configurable parameters.
3) Re-rank
   - Score chunks using `hybrid_search_and_score` or a cross-encoder adapter.
4) Confidence Gate
   - If low confidence, expand queries or relax retrieval params and retry.
5) Compose
   - Pass top-ranked snippets to compose as today.

## Implementation Sketch (Option A)

- Add a `transform_query` step in `retrieval_augmented_generation`:
  - Local adapter wraps `collection_search` strategy generator.
  - Produces 3-5 query variants.
- Add a `rerank` step:
  - Map retrieved chunks to `SearchCandidate` payload for scoring.
  - Use `llm_worker.graphs.hybrid_search_and_score` adapter as in
    `collection_search` for consistent scoring behavior.
- Add a `confidence` step:
  - Heuristic based on top score delta, average score, or total hits.
  - If below threshold, widen retrieval (lower min_sim or increase top_k).

## Risks / Open Questions

- Re-rank inputs: scoring expects metadata similar to search candidates.
- Confidence heuristics: decide minimal telemetry for decisions.
- Performance: multiple retrievals per query variant may increase latency.
- Tracing: ensure `ToolContext` stays the sole ID source.

## Acceptance Criteria

- RAG graph can generate multiple query variants and returns higher-quality
  retrieval sets without breaking existing Graph IO.
- Rerank step is optional and can be toggled off for baseline comparison.
- Confidence loop does not introduce new IDs or meta keys.
- Existing `retrieve` and `compose` nodes remain usable without change.
