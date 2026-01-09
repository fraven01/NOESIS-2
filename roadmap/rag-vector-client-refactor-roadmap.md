# RAG vector_client refactor roadmap (corrected)

Status: Phase A complete; Phase B complete; Phase C complete.
Date: 2026-01-07
Scope: ai_core/rag/vector_client.py and direct helpers.

## Current facts (as of 2026-01-07)
- File size: 5,629 lines, 251,871 bytes.
- Key anchors: _normalise_document_identity at L438, hybrid_search at L1572,
  _compute_document_embedding at L3781, _find_near_duplicate at L3828,
  _ensure_documents at L4142.

## Corrections to prior analysis
- Document identity duplication exists in _ensure_documents primary path
  (L4454-4616) and UniqueViolation retry path (L4713-4886).
- Debug logging duplication occurs at L4457-4587 and L4716-4847
  (events DEBUG.2..DEBUG.6). Keep event names as-is if centralized.
- Collection id parsing is not uniform:
  - strict: _ensure_documents (L4161-4171) raises ValueError("invalid_collection_id")
  - lenient: _replace_chunks (L5452-5464) swallows errors and sets None
  - additional conversions: L5017 and L5054
  Any shared helper must preserve strict vs lenient behavior.
- SQL templates in hybrid_search are already localized to nested builders
  (L1962-2005); extraction to a new module is optional and lower priority.
- Row-shape validation occurs in two spots (L2856, L2875), not three.

## Roadmap (phased, pre-MVP, breaking ok)
### Phase A: Low-risk DRY + utilities (no behavior changes)
1) Centralize debug logging for _ensure_documents (preserve event names and JSON encoding). — Done
   - Pointers: L4457-4587, L4716-4847
   - Acceptance: same event strings (including ".retry"), same fields, no new exceptions.

2) Deduplicate existing-document processing in _ensure_documents. — Done
   - Pointers: L4454-4616, L4713-4886
   - Acceptance: one shared helper; retry path still logs
     "Skipping unchanged document during upsert" and keeps same actions.

3) Add collection id parser with strict/lenient modes. — Done
   - Pointers: L4161-4171 (strict), L5452-5464 (lenient), L5017/L5054 (other conversions)
   - Acceptance: strict call sites still raise "invalid_collection_id";
     lenient call sites still fall back to None.

4) Extract hashing helpers to ai_core/rag/hashing.py. — Done
   - Pointers: L153-217, L5325-5373; import users: ai_core/rag/delta.py
   - Acceptance: unchanged hashes and payload normalization; tests for hash helpers pass.

5) Extract vector helpers to ai_core/rag/vector_utils.py. — Done
   - Pointers: L770-880, L5575-5590; import users: ai_core/rag/embedding_cache.py
   - Acceptance: identical vector normalization/coercion behavior; format errors unchanged.

### Phase B: Medium risk extractions (behavior-preserving)
6) Extract metadata handling to ai_core/rag/metadata_handler.py. — Done
   - Pointers: L306-438, L4885-4955
   - Acceptance: metadata normalization and parent_nodes derivation unchanged;
     fallback import to parents.limit_parent_payload still works.

7) Extract near-duplicate logic to ai_core/rag/deduplication.py. — Done
   - Pointers: dataclasses at L89-150; methods at L3781-4140
   - Acceptance: same SQL, thresholds, and operator gating;
     helper constructed with explicit dependencies from PgVectorClient.

### Phase C: High risk hybrid_search split
8) Create a data-flow map of hybrid_search and define explicit inputs/outputs for phases. ƒ?" Done
   - Pointers: L1572-3780
   - Acceptance: written state diagram and variable list; no code change yet.

9) Split into vector_search.py, lexical_search.py, score_fusion.py. ƒ?" Done
   - Acceptance: hybrid_search becomes orchestrator; outputs identical for same inputs;
     tests updated.

#### Phase C Step 8: hybrid_search data-flow map (written)

**Inputs (API params)**
- `query`, `tenant_id`, `case_id`, `top_k`, `filters`
- `alpha`, `min_sim`, `vec_limit`, `lex_limit`, `trgm_limit`, `trgm_threshold`
- `max_candidates`, `visibility`, `visibility_override_allowed`
- `collection_id`, `workflow_id` (passthrough compatibility)

**Inputs (settings/context)**
- Settings: `RAG_HYBRID_ALPHA`, `RAG_MIN_SIM`, `RAG_TRGM_LIMIT`,
  `RAG_DISTANCE_SCORE_MODE`, `RAG_MAX_CANDIDATES`, `RAG_INDEX_KIND`,
  `RAG_HNSW_EF_SEARCH`, `RAG_IVF_PROBES`, `RAG_LEXICAL_MODE`,
  `RAG_HYDE_ENABLED`
- Context: `get_log_context()` (trace_id, key_alias), `Visibility` enum

**Derived values / normalization**
- `top_k` clamped to `[1..10]`
- `visibility_mode` (`ACTIVE` default unless override allowed)
- `tenant_uuid`, `tenant` string
- `scope_plan = _prepare_scope_filters(...)`
- `normalized_filters`, `metadata_filters`, `case_value`, `collection_ids_filter`,
  `has_single_collection_filter`, `single_collection_value`, `collection_ids_count`
- `effective_collection_filter` + legacy `doc_class` routing (when enabled)
- `alpha_value`, `min_sim_value`, `trgm_limit_value` (clamped)
- `max_candidates_value`, `vec_limit_value`, `lex_limit_value`
- `query_norm`, `query_db_norm`
- `query_vec` (formatted vector or None) and `query_embedding_empty` flag

**State diagram (high-level)**
```
Inputs
  -> normalize visibility + scope filters + limits
  -> embed query (HyDE optional) -> format vector
  -> build WHERE clauses + params
  -> _operation():
       - vector query (optional) -> vector_rows
       - lexical query (primary + fallback) -> lexical_rows
       - optional deleted-visibility counts
  -> compile candidates (vector + lexical)
  -> fuse scores (RRF + alpha weighting)
  -> strict filter (tenant/case) + metadata contract
  -> apply min_sim cutoff
  -> build HybridSearchResult + metadata
```

**Key internal phases / outputs**
1) **Vector query phase**
   - Inputs: `query_vec`, `where_sql`, `where_params`, `vec_limit_value`
   - Output: `vector_rows` (rows with distance column)
2) **Lexical query phase**
   - Inputs: `query_db_norm`, `where_sql`, `lex_limit_value`, `trgm_limit_value`
   - Output: `lexical_rows` (primary or fallback), `lexical_query_variant`
3) **Candidate compilation**
   - Inputs: `vector_rows`, `lexical_rows`
   - Output: `candidates` dict keyed by chunk_id with vscore/lscore
4) **Fusion**
   - Inputs: `candidates`, `alpha_value`, rank weights
   - Output: fused scores, ordered results
5) **Strict filtering + contract**
   - Inputs: `tenant`, `case_value`, `normalized_filters`
   - Output: filtered `Chunk` list with `meta` enriched
6) **Cutoff + finalization**
   - Inputs: `min_sim_value`, allow_below_cutoff flags
   - Output: `HybridSearchResult` with counts + diagnostics

**Outputs**
- `HybridSearchResult` with:
  - `chunks` (sorted by fused score)
  - `vector_candidates`, `lexical_candidates`, `fused_candidates`
  - `duration_ms`, `alpha`, `min_sim`, `vec_limit`, `lex_limit`
  - `below_cutoff`, `returned_after_cutoff`
  - `query_embedding_empty`, `applied_trgm_limit`, `fallback_limit_used`
  - `visibility`, `deleted_matches_blocked`, `cached_total_candidates`
  - `scores` (per-result scores, optional)

## Guardrails
- Do not change ScopeContext/BusinessContext semantics.
- Preserve logging event names and identifiers.
- Keep strict vs lenient collection id behavior.
