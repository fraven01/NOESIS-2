# Retrieval Contracts (Code Map)

## Source of Truth

- ai_core/nodes/retrieve.py
- ai_core/nodes/_hybrid_params.py
- ai_core/rag/filter_spec.py
- ai_core/rag/visibility.py

## RetrieveInput (ai_core/nodes/retrieve.py:RetrieveInput)

fields:
- query: str = ""
- filters: FilterSpec | None
- process: str | None
- doc_class: str | None
- visibility: str | None
- hybrid: Mapping[str, Any] | None (required at runtime)
- top_k: int | None (override for hybrid.top_k)

notes:
- Business IDs (collection_id, workflow_id, document_id, document_version_id) are read from ToolContext.business.
- visibility_override_allowed is read from ToolContext.visibility_override_allowed.
- filters is a FilterSpec (not raw mapping); mapping input is normalized into FilterSpec.
- hybrid is required; missing or non-mapping => InputError.

## FilterSpec (ai_core/rag/filter_spec.py:FilterSpec)

fields:
- tenant_id: str | None
- case_id: str | None
- collection_id: str | None
- collection_ids: list[str] | None
- document_id: str | None
- document_version_id: str | None
- is_latest: bool | None
- metadata: dict[str, Any]

behavior:
- unknown keys are moved into metadata
- build_filter_spec merges explicit context values with raw filters

## Hybrid Parameters (ai_core/nodes/_hybrid_params.py)

allowed keys:
- alpha (float, 0..1)
- min_sim (float, 0..1)
- top_k (int, min=1, max=RAG.TOPK_MAX)
- vec_limit (int, min=1)
- lex_limit (int, min=1)
- trgm_limit (float | None, 0..1)
- max_candidates (int | None, >= top_k, vec_limit, lex_limit)
- diversify_strength (float, 0..1)

defaults:
- alpha: RAG.HYBRID_ALPHA_DEFAULT (0.7)
- min_sim: RAG.MIN_SIM_DEFAULT (0.15)
- top_k: RAG.TOPK_DEFAULT (5)
- diversify_strength: RAG.DIVERSIFY_STRENGTH_DEFAULT (0.3)

notes:
- override_top_k uses RetrieveInput.top_k
- diversified ordering is currently bypassed (pre-MVP accuracy)

## RetrieveMeta (ai_core/nodes/retrieve.py:RetrieveMeta)

fields:
- routing: profile, vector_space_id, process, doc_class, collection_id, workflow_id
- took_ms
- alpha
- min_sim
- top_k_effective
- matches_returned
- max_candidates_effective
- vector_candidates
- lexical_candidates
- deleted_matches_blocked
- visibility_effective
- diversify_strength
- lexical_index_used

## RetrieveOutput (ai_core/nodes/retrieve.py:RetrieveOutput)

matches:
- id, text, score, source, hash
- citation (optional)
- meta (optional, includes document_id, chunk_id, parent_ids, parents, neighbor, etc.)

behavior:
- parent context fetched when meta.parent_ids present
- neighbor chunks appended with meta.neighbor=true
- NotFoundError raised when no matches and router is not tenant-scoped

## Visibility

- enum: active | all | deleted (ai_core/rag/visibility.py)
- normalize + guard uses visibility_override_allowed from ToolContext
