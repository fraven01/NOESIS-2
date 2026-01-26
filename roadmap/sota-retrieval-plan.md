# SOTA Retrieval Plan (Pre-MVP)

Status: draft
Scope: SOTA Retrieval Architecture (Phase 1-3)
Assumption: Pre-MVP, DB reset allowed, breaking changes permitted with backlog traceability.

## Goals

- Improve retrieval quality with passage-first output and structure-aware reranking.
- Enable hybrid candidate generation (dense + lexical) with fusion.
- Establish an evidence graph to support adjacency/structure signals.

## Non-Goals (for now)

- Production-grade backward compatibility.
- ColBERT or late-interaction retrieval (explicitly deferred).

## Breaking Changes (Pre-MVP Allowed)

- Contract updates in retrieval/rerank internal payloads are allowed.
- If graph/tool schemas or meta fields change, add a backlog item with code pointers + acceptance criteria.

## Observability Requirements (All Phases)

- Emit spans or structured logs per stage with latency, counts, and source distribution.
- Record feature values for rerank and plan decisions for query planner.
- Track fallback paths explicitly (e.g. lexical only, dense only, missing metadata).

## Phase 1: Foundation (P0)

### R1.1 Evidence Graph Data Model

Deliverable:
- `ai_core/rag/evidence_graph.py`
- EvidenceGraph with nodes + edges built from retrieved chunks + metadata.

Key objects:
- Node: chunk_id, section_path, parent_ref, score, rank, doc_id.
- Edges: adjacent_to, parent_of, child_of.

APIs:
- `get_adjacent(chunk_id, *, max_hops=1)`
- `get_parent(chunk_id)`
- `get_subgraph(chunk_ids, *, max_hops=1)`

Acceptance:
- Unit tests for graph construction and traversal.
- Works with current chunk metadata shape (parent_ref, section_path, doc identifiers).
Observability:
- Log graph node/edge counts and build latency.

### R1.2 Passage Assembly

Deliverable:
- `ai_core/rag/passage_assembly.py`
- Passage dataclass: id, text, chunk_ids, section_path, score.

Rules:
- Anchor on highest-scoring chunk.
- Merge adjacent chunks if section_path compatible.
- Enforce token limit (configurable).

Integration:
- `ai_core/nodes/retrieve.py` passage assembly step.

Acceptance:
- Unit tests: respects section boundaries, token cap, anchor logic.
Observability:
- Log passage count, avg passage length, and merge rate.

### R1.3 Structure-Aware Rerank Features

Deliverable:
- `ai_core/rag/rerank_features.py`
- RerankFeatures dataclass + extractor.

Features:
- parent_relevance, section_match, confidence, adjacency_bonus, doc_type_match.

Integration:
- `ai_core/rag/rerank.py` feature extraction + telemetry.

Acceptance:
- Features computed for a retrieved set + evidence graph.
- Telemetry logs feature values.
- Configurable weights per quality_mode.
Observability:
- Emit feature stats (min/mean/max) and per-request weighting.

## Phase 2: Hybrid Candidates (P1)

### R2.1 Lexical Search (pg_trgm)

Deliverable:
- `ai_core/rag/lexical_search.py`
- Integrate into `ai_core/rag/query_builder.py`.

Acceptance:
- Lexical results return same Chunk format as dense.
- Similarity threshold configurable.
- Perf: <500ms for 100k chunks (target).
Observability:
- Log candidate count, latency, and threshold applied.

### R2.2 RRF Fusion

Deliverable:
- `ai_core/rag/hybrid_fusion.py`
- Integrate into `ai_core/nodes/retrieve.py`.

Acceptance:
- Parallel dense + lexical execution.
- RRF with configurable weights.
- Telemetry captures source distribution.
Observability:
- Log fusion latency, candidate count, and source mix.

### R2.3 Query Planner

Deliverable:
- `ai_core/rag/query_planner.py`
- Integrate into `ai_core/graphs/technical/rag_retrieval.py`.

Acceptance:
- QueryPlan + QueryConstraints models.
- Rule-based planner (fast).
- Optional LLM planner (feature-flagged).
Observability:
- Log chosen doc_type, expansions, constraints, and planner path (rule vs LLM).

## Phase 3: Advanced (P2)

### R3.1 Late-Interaction Retrieval

- Deferred. Keep lexical as interim.

### R3.2 Cross-Document Evidence Linking

- Add citation/reference edges to EvidenceGraph during ingestion.
- Retrieval can traverse references.
Feature flags:
- `RAG_REFERENCE_EXPANSION=1` enable reference expansion in retrieval.
- `RAG_REFERENCE_EXPANSION_LIMIT` max reference IDs to expand (default 5).
- `RAG_REFERENCE_EXPANSION_TOP_K` per-reference retrieve cap (default 3).
Observability:
- `retrieval_meta.reference_expansion` includes counts, latency, and reference IDs.

### R3.3 Adaptive Weight Learning

- Collect feedback, update rerank weights periodically.
Feature flags:
- `RAG_FEEDBACK_ENABLED=1` collect implicit feedback events.
- `RAG_RERANK_WEIGHT_MODE=learned` use learned weights when available.
- `RAG_RERANK_WEIGHT_ALPHA` blend learned vs static weights (default 0.5).
- `RAG_RERANK_FEEDBACK_WINDOW_DAYS` lookback window (default 7 days).
Observability:
- Feedback events stored in `RagFeedbackEvent`; learned weights stored in `RagRerankWeight`.

## Recommended Execution Order

1. R1.1 EvidenceGraph
2. R1.2 Passage Assembly
3. R1.3 Rerank Features
4. R2.1 Lexical Search
5. R2.2 RRF Fusion
6. R2.3 Query Planner

## Test Strategy

- Unit tests for EvidenceGraph, Passage Assembly, Rerank Features.
- Integration tests for retrieval pipeline on a small fixture dataset.
- Perf checks for lexical + fusion.
