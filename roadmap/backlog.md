# Backlog (prioritized)

This is a curated list of open work items for vibe-coding execution.
Top-to-bottom order within each section is priority order.
Prefer linking each item to concrete code paths (and optionally to an issue).

## Next up (highest leverage)




## Agentic AI-First System (target state)

### Pre AI-first blockers



## Code Quality & Architecture Cleanup (Pre-MVP Refactoring)



### Docs/Test touchpoints (checklist)

### P1 - High Value Cleanups (Low-Medium Effort)



- [ ] **Collection Search schema version tolerance (plan evolution)**:
  - **Details:** `roadmap/collection-search-review.md`
  - **Pointers:** `ai_core/graphs/technical/collection_search.py:CollectionSearchGraphRequest`, `ai_core/contracts/plans/*`
  - **Acceptance:** Graph accepts minor-compatible schema updates without failing hard; strict rejection for incompatible major changes; tests cover minor version acceptance and major version rejection



- [ ] **Activate AgenticChunker LLM Boundary Detection**: Implement `_detect_boundaries_llm()` in `ai_core/rag/chunking/agentic_chunker.py:354-379` (~25 LOC). Infrastructure complete (prompt template, Pydantic models, rate limiter, fallback logic). Needed to improve fallback quality for long documents (`token_count > max_tokens`). Details: `ai_core/rag/chunking/README.md#agentic-chunking`.

- [ ] **Simplify Adaptive Chunking Toggles (deferred post-MVP)**: Keep only two flags and one "new" path. Deferred because current flexibility benefits future use cases. Review after production experience shows which flags are unused. (pointers: `ai_core/rag/chunking/hybrid_chunker.py`, `ai_core/rag/chunking/late_chunker.py`, `ai_core/rag/chunking/README.md`)

### P2 - Long-term Improvements (High Effort)

- [ ] **Recursive Chunking for Long Documents**: Replace truncation fallback (`text[:2048]` in `late_chunker.py:1002`) with recursive chunking. When `token_count > max_tokens`, split into sections and chunk each section independently with proper boundary detection instead of hard truncation. (pointers: `ai_core/rag/chunking/late_chunker.py:984-1023`)

- [ ] **Late Chunker Section Fallback Quality**: When `late_chunker_document_too_long` triggers section-based fallback, resulting chunks can be very low quality (single headings like "Konzernbetriebsvereinbarung" with coherence=60, completeness=40). Consider: (1) minimum chunk content threshold, (2) merge adjacent tiny sections, (3) apply LLM boundary detection per section.
  - **Pointers:** `ai_core/rag/chunking/late_chunker.py:_chunk_by_sections`, `late_chunker.py:_chunk_by_sections_adaptive`
  - **Acceptance:** Section fallback produces chunks with minimum semantic content; single-heading chunks merged with following content; quality scores improve for fallback path

- [ ] **Robust Sentence Tokenizer**: Current splitting is fragile (`text.split(".")`) – breaks on abbreviations ("Dr."), decimals ("3.14"), URLs. Extract shared tokenizer to `ai_core/rag/chunking/utils.py` using regex with negative lookahead or `nltk.sent_tokenize`. (pointers: `agentic_chunker.py:347`, `late_chunker.py` sentence splitting)

## SOTA Developer RAG Chat (Pre-MVP)

**Roadmap**: [rag-chat-sota.md](rag-chat-sota.md)
**Total Effort**: ~3-4 Sprints (Medium-High Complexity)



## SOTA Retrieval Architecture

**Roadmap**: [metadata-aware-retrieval.md](metadata-aware-retrieval.md)
**Total Effort**: ~4-5 Sprints
**Vision**: Metadata-first, Passage-first, Hybrid, Multi-stage

**Principles**:
1. **Candidates nicht nur Dense**: Dense + Lexical + Late-Interaction (RRF Fusion)
2. **Reranking ist strukturbewusst**: Parent/Section/Confidence/Adjacency als Features
3. **Output sind Passagen**: Merged adjacent chunks mit Section-Boundaries
4. **Query wird geplant**: Doc-Type Routing + Query Expansion + Constraints
5. **Evidence Graph**: Parent/Child/Adjacency als Graph, Reranking über Subgraphen

### Phase 1: Foundation (P0)

- [ ] **SOTA-R1.1: Evidence Graph Data Model**: Represent chunk relationships (parent_of, child_of, adjacent_to) as traversable in-memory graph built from retrieved chunks + metadata.
  - **Pointers:** `ai_core/rag/evidence_graph.py` (new), `ai_core/rag/ingestion_contracts.py:69-95`
  - **Acceptance:** EvidenceGraph with nodes + edges; traversal methods (get_adjacent, get_parent, get_subgraph); unit tests for construction + traversal
  - **Effort:** M (1 Sprint)

- [ ] **SOTA-R1.2: Passage Assembly**: Merge adjacent chunks into coherent passages respecting section boundaries and token limits.
  - **Pointers:** `ai_core/rag/passage_assembly.py` (new), `ai_core/nodes/retrieve.py:543-600`
  - **Acceptance:** Passage dataclass; assembly respects section boundaries; token limit enforced; anchor on highest-scoring chunk
  - **Effort:** M (1 Sprint)

- [ ] **SOTA-R1.3: Structure-Aware Rerank Features**: Extract rerank features from metadata + graph (parent_relevance, section_match, confidence, adjacency_bonus, doc_type_match).
  - **Pointers:** `ai_core/rag/rerank_features.py` (new), `ai_core/rag/rerank.py:152-241`
  - **Acceptance:** RerankFeatures dataclass; feature extraction; configurable weights per quality_mode; telemetry logs feature values
  - **Effort:** M (1 Sprint)

### Phase 2: Hybrid Candidates (P1)

- [ ] **SOTA-R2.1: Lexical Search Integration (BM25)**: Add pg_trgm-based lexical search alongside dense retrieval for exact term matching.
  - **Pointers:** `ai_core/rag/lexical_search.py` (new), `ai_core/rag/query_builder.py`
  - **Acceptance:** Lexical search with pg_trgm; same Chunk format as dense; configurable similarity threshold; <500ms for 100k chunks
  - **Effort:** S (0.5 Sprint)

- [ ] **SOTA-R2.2: Hybrid Candidate Fusion (RRF)**: Merge dense + lexical candidates using Reciprocal Rank Fusion with parallel execution.
  - **Pointers:** `ai_core/rag/hybrid_fusion.py` (new), `ai_core/nodes/retrieve.py`
  - **Acceptance:** RRF implementation; parallel dense + lexical; configurable weights; telemetry captures source distribution
  - **Effort:** S (0.5 Sprint)

- [ ] **SOTA-R2.3: Query Planner**: Analyze query to determine doc_type routing, query expansion, and constraints (must_include, date_range, collections).
  - **Pointers:** `ai_core/rag/query_planner.py` (new), `ai_core/graphs/technical/rag_retrieval.py`
  - **Acceptance:** QueryPlan + QueryConstraints models; rule-based planner (fast); optional LLM planner; expansion templates per doc_type
  - **Effort:** M (1 Sprint)

### Phase 3: Advanced (P2)

- [ ] **SOTA-R3.1: Late-Interaction Retrieval (ColBERT-style)**: Token-level matching for precision on exact terms. Deferred - use lexical as interim.
  - **Acceptance:** ColBERT model hosting; token-level index; significant infra investment
  - **Effort:** L (2+ Sprints) - DEFERRED

- [ ] **SOTA-R3.2: Cross-Document Evidence Linking**: Detect citations/references during ingestion, add "references" edges to Evidence Graph.
  - **Pointers:** `ai_core/rag/evidence_graph.py`, ingestion pipeline
  - **Acceptance:** Citation detection; reference edges in graph; retrieval follows edges
  - **Effort:** M (1 Sprint)

- [ ] **SOTA-R3.3: Adaptive Weight Learning**: Learn optimal rerank weights from implicit feedback (clicks, answer sources).
  - **Acceptance:** Feedback collection; periodic weight updates; A/B testing support
  - **Effort:** M (1 Sprint)

## Observability Cleanup

### P0 - Critical

- [x] **OpenTelemetry SpanKind=None KeyError**: Telemetry export fails with `KeyError: None` in `_encode_span` when a span is created without explicit `SpanKind`. Traces for affected tasks are not exported to collector (Jaeger/Honeycomb).
  - **Pointers:** OpenTelemetry SDK `_encode_span` → `_SPAN_KIND_MAP[sdk_span.kind]`
  - **Acceptance:** All spans have explicit `SpanKind` set; no `KeyError` in telemetry export; traces successfully exported for all tasks

### Hygiene

- [ ] **Tenant Mapping Log-Spam**: Repeated warnings during Legacy-Tenant-ID conversion (e.g., `"2"` → deterministic UUID). Known migration state but causes excessive log volume.
  - **Pointers:** `ai_core/services/document_upload.py` (tenant_id mapping), Celery worker logs
  - **Acceptance:** Either (1) reduce log level to DEBUG for known legacy mappings, or (2) cache mapped tenant IDs to log only first occurrence per session, or (3) complete migration to UUID-only tenant IDs

- [ ] **Extract Chunker Utils**: Deduplicate shared logic between `LateChunker` and `AgenticChunker` into `ai_core/rag/chunking/utils.py`: sentence splitting, chunk ID generation (`_build_adaptive_chunk_id`), token counting. (pointers: `late_chunker.py`, `agentic_chunker.py`)

## Semantics / IDs

## Layering / boundaries

- [ ] **Explicit ToolContext in HybridScoreExecutor Protocol**: Replace `tenant_context: Mapping[str, Any]` with explicit `tool_context: ToolContext` parameter in `HybridScoreExecutor.run()`. Current workaround embeds `tool_context` in `tenant_context` dict for `tool_context_from_meta()` reconstruction. Clean solution: extend Protocol signature, update `_HybridExecutorAdapter`, remove dict-embedding hack.
  - **Pointers:** `ai_core/graphs/technical/collection_search.py:HybridScoreExecutor`, `ai_core/graphs/technical/collection_search.py:hybrid_score_node`, `ai_core/graphs/technical/collection_search.py:_HybridExecutorAdapter`, `llm_worker/graphs/hybrid_search_and_score.py`
  - **Acceptance:** `HybridScoreExecutor.run()` accepts `tool_context: ToolContext`; `tenant_context` removed or reduced to minimal tenant-specific fields; `_HybridExecutorAdapter` passes `tool_context` directly to sub-graph meta; no dict-embedding workaround needed

## Externalization readiness (later)

- [ ] Graph registry: add versioning + request/response model metadata (note: factory support already exists via `LazyGraphFactory`)


## Observability / operations


## Agent navigation (docs-only drift)

## Hygiene (lowest priority)

- [ ] Documentation hygiene: remove encoding artifacts in docs/strings (purely textual)
