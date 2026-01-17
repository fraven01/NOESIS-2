# Backlog (prioritized)

This is a curated list of open work items for vibe-coding execution.
Top-to-bottom order within each section is priority order.
Prefer linking each item to concrete code paths (and optionally to an issue).

## Next up (highest leverage)
- [ ] Review later: framework analysis graph convergence (`ai_core/graphs/business/framework_analysis_graph.py`)

## Agentic AI-First System (target state)

### Pre AI-first blockers

- [x] **Workflow-execution-aware checkpointing**:
  - **Pointers:** `ai_core/graph/core.py:FileCheckpointer._path`, `ai_core/graph/core.py:GraphContext`, `common/object_store_defaults.py:sanitize_identifier`, `ai_core/graph/README.md`
  - **Acceptance:** Checkpoint paths are derived from workflow execution scope (tenant_id + workflow_id + run_id, or plan_key when available) instead of case_id; no "None" paths or case-less collisions; missing workflow_id/run_id raises a clear error; ThreadAwareCheckpointer still prefers thread_id; workflow_execution terminology used in docs; tests updated in `ai_core/tests/test_checkpointer_file.py` to cover unique paths per run_id, missing run_id errors, and thread_id path precedence

- [x] **Docs alignment for plan_key + workflow_execution**:
  - **Pointers:** `ai_core/graph/README.md`, `docs/roadmap/UTP.md`, `roadmap/agentic-ai-first-strategy.md`
  - **Acceptance:** plan_key (derived, not minted) replaces plan_id in planning docs; workflow_execution terminology used consistently; no references to execution_case_id or required case_id for graph meta

- [x] **Blueprint + ImplementationPlan contracts (Phase 1)**:
  - **Pointers:** `ai_core/contracts/plans/` (new package), `ai_core/contracts/plans/plan.py`, `ai_core/contracts/plans/blueprint.py`, `ai_core/contracts/plans/evidence.py`, `ai_core/graphs/technical/collection_search.py:ImplementationPlan`
  - **Acceptance:** Pydantic models for Blueprint, ImplementationPlan, Slot, Task, Gate, Deviation, Evidence, and Confidence; schema_version on plan models ("v0" or semver); Evidence uses ref_type/ref_id (url | repo_doc | repo_chunk | object_store | confluence | screenshot); optional slot_type or json_schema_ref; JSON schema export; deterministic plan_key derivation helper (UUIDv5 or hash) over plan scope tuple (derived, not minted) with canonicalization (ordered scope tuple, normalized gremium_identifier, choose framework_profile_id or framework_profile_version); round-trip serialization tests; tests forbid execution_case_id, minted plan_id, or any additional UUID fields; references updated in `roadmap/agentic-ai-first-strategy.md`; no new IDs introduced

- [x] **Vertical slice: Plan-driven Collection Search**:
  - **Pointers:** `ai_core/graphs/technical/collection_search.py:build_plan_node`, `ai_core/graphs/technical/collection_search.py:HitlDecision`, `ai_core/services/collection_search/hitl.py`, `ai_core/services/collection_search/strategy.py`, `ai_core/services/collection_search/scoring.py`, `ai_core/graphs/technical/universal_ingestion_graph.py`
  - **Acceptance:** `build_plan_node` emits the new plan schema with slots/tasks/gates/deviations/evidence using workflow_execution terminology; HITL decisions update slot values and gate outcomes (slot completion); `execute_plan` consumes task outputs and records evidence; graph output links evidence to ingested artifacts; tests updated in `ai_core/tests/graphs/test_collection_search_graph.py`

- [ ] **Plan persistence and retrieval (vertical slice)**:
  - **Pointers:** `ai_core/graph/core.py:FileCheckpointer`, `ai_core/graph/state.py:PersistedGraphState`, `common/object_store_defaults.py`
  - **Acceptance:** Plan and evidence persist as part of the graph state envelope; deterministic plan_key resolves to the latest persisted state for the workflow execution (derived, not minted); retrieval helper returns validated plan + evidence; tests cover round-trip persistence; no new IDs introduced; no new DB tables in the slice

## Code Quality & Architecture Cleanup (Pre-MVP Refactoring)

### P0 - Critical Quick Wins (High Impact, Medium-High Effort)

- [ ] **Collection Search timeouts and stall protection**:
  - **Details:** `roadmap/collection-search-review.md`
  - **Pointers:** `ai_core/llm/client.py:391`, `ai_core/llm/client.py:738`, `ai_core/graphs/technical/collection_search.py:561`, `ai_core/graphs/technical/collection_search.py:650`, `llm_worker/graphs/hybrid_search_and_score.py:1116`, `llm_worker/graphs/score_results.py:353`
  - **Acceptance:** LLM client adds explicit connect/read timeouts (sync + streaming); parallel search uses a total timeout with partial results; hybrid score timeout handling is reachable and logged; tests updated to cover timeout behavior; see `roadmap/collection-search-review.md`

- [ ] **Collection Search boundary contract cleanup (V1-V8)**:
  - **Details:** `roadmap/collection-search-review.md`
  - **Pointers:** `ai_core/graphs/technical/collection_search.py:140`, `ai_core/graphs/technical/collection_search.py:480`, `ai_core/graphs/technical/collection_search.py:636`, `ai_core/graphs/technical/collection_search.py:837`, `ai_core/graphs/technical/collection_search.py:867`, `llm_worker/graphs/hybrid_search_and_score.py:626`, `llm_worker/graphs/hybrid_search_and_score.py:1389`, `llm_worker/graphs/score_results.py:353`
  - **Acceptance:** Graph internals pass typed models across nodes (no dict round-trips); search output uses typed structures at boundaries; dead branch removed; redundant model_dump removed; config/control/meta shape simplified; hardcoded jurisdiction/purpose removed; tests updated to enforce contracts; see `roadmap/collection-search-review.md`

### Docs/Test touchpoints (checklist)

### P1 - High Value Cleanups (Low-Medium Effort)

- [ ] **Collection Search strategy quality improvements**: add structured JSON output with examples and richer fallback queries.
  - **Details:** `roadmap/collection-search-review.md`
  - **Pointers:** `ai_core/services/collection_search/strategy.py:139`, `ai_core/services/collection_search/strategy.py:226`
  - **Acceptance:** Strategy prompt includes JSON-only schema + few-shot example; fallback strategy uses non-trivial query variants; tests cover schema parsing and fallback behavior

- [ ] **Collection Search adaptive embedding weights**: adjust embedding vs heuristic weights by quality_mode or context.
  - **Details:** `roadmap/collection-search-review.md`
  - **Pointers:** `ai_core/graphs/technical/collection_search.py:738`
  - **Acceptance:** Weight selection is driven by quality_mode (or explicit profile mapping) and recorded in telemetry; default remains unchanged when not configured

- [ ] **Collection Search hard graph timeout (worker-safe)**: add a whole-graph timeout without signal.alarm (Windows-safe).
  - **Details:** `roadmap/collection-search-review.md`
  - **Pointers:** `ai_core/graphs/technical/collection_search.py`, `llm_worker/tasks.py:run_graph`, `ai_core/services/__init__.py`
  - **Acceptance:** Graph execution enforces a hard cap via worker limits or explicit timeout wrapper; timeout produces a deterministic error payload; no signal.alarm usage

- [ ] **Activate AgenticChunker LLM Boundary Detection**: Implement `_detect_boundaries_llm()` in `ai_core/rag/chunking/agentic_chunker.py:354-379` (~25 LOC). Infrastructure complete (prompt template, Pydantic models, rate limiter, fallback logic). Needed to improve fallback quality for long documents (`token_count > max_tokens`). Details: `ai_core/rag/chunking/README.md#agentic-chunking`.

- [ ] **Simplify Adaptive Chunking Toggles (deferred post-MVP)**: Keep only two flags and one "new" path. Deferred because current flexibility benefits future use cases. Review after production experience shows which flags are unused. (pointers: `ai_core/rag/chunking/hybrid_chunker.py`, `ai_core/rag/chunking/late_chunker.py`, `ai_core/rag/chunking/README.md`)

### P2 - Long-term Improvements (High Effort)

- [ ] **Recursive Chunking for Long Documents**: Replace truncation fallback (`text[:2048]` in `late_chunker.py:1002`) with recursive chunking. When `token_count > max_tokens`, split into sections and chunk each section independently with proper boundary detection instead of hard truncation. (pointers: `ai_core/rag/chunking/late_chunker.py:984-1023`)

- [ ] **Robust Sentence Tokenizer**: Current splitting is fragile (`text.split(".")`) – breaks on abbreviations ("Dr."), decimals ("3.14"), URLs. Extract shared tokenizer to `ai_core/rag/chunking/utils.py` using regex with negative lookahead or `nltk.sent_tokenize`. (pointers: `agentic_chunker.py:347`, `late_chunker.py` sentence splitting)

### Observability Cleanup


### Hygiene

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
