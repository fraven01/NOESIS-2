# Capability-First TODOs

This list captures ad-hoc logic still embedded inside graphs. The goal is to
move these into explicit capabilities (nodes/tools/domain services) and have
graphs orchestrate only.

## Graph: `ai_core/graphs/business/framework_analysis_graph.py`

- Extract `normalize_gremium_identifier` into a shared capability (node/tool/helper).
- Extract `extract_toc_from_chunks` into a retrieval/toc capability.
- Move LLM prompt parsing + JSON extraction into a reusable node/tool.
- Isolate component validation heuristics into a capability with tests.
- Move profile persistence adapter to a domain service capability (wrapper).

## Graph: `ai_core/graphs/technical/collection_search.py`

- Move `_calculate_generic_heuristics` + `_cosine_similarity` into a scoring capability.
- Move `_coerce_query_list`, `_extract_strategy_payload`, `_fallback_strategy` into a strategy capability.
- Move `_llm_strategy_generator` into a dedicated tool/node (LLM strategy tool).
- Isolate HITL payload building into a capability (adapter for UI/workbench).
- Isolate auto-ingest selection logic into a capability (planner/selector).

## Graph: `ai_core/graphs/technical/universal_ingestion_graph.py`

- Move `_blocked_domain` + selection logic into a search selection capability.
- Move search-result → `NormalizedDocument` conversion into a capability.
- Consider moving the document-processing graph DI root into a service factory.

