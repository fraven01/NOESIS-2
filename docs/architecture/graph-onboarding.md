# Graph onboarding (code-backed pointers)

This note collects pointers to the existing graph execution surfaces and the ID/validation utilities used by graphs and workers in this repository.

## Code entry points

- Graph runner protocol + checkpointing: `ai_core/graph/core.py`
- Graph meta normalization (builds `scope_context` + `tool_context`): `ai_core/graph/schemas.py`
- AI Core graph execution orchestration: `ai_core/services/__init__.py:execute_graph`
- Ingestion task entrypoint (Celery): `ai_core/tasks/graph_tasks.py:run_ingestion_graph`
- ID normalization helpers: `ai_core/ids/` and `common/constants.py`

## Graph construction pattern (LangGraph)

Preferred build pattern for LangGraph-based graphs: expose factory functions that return a fresh graph instance / compiled graph, and avoid module-level singleton instances.

Examples of factory functions in the repo:

- `ai_core/graphs/technical/universal_ingestion_graph.py:build_universal_ingestion_graph`
- `ai_core/graphs/technical/collection_search.py:build_compiled_graph`
- `ai_core/graphs/business/framework_analysis_graph.py:build_graph`
- `ai_core/graphs/technical/retrieval_augmented_generation.py:build_graph`

## Existing graphs as examples

Graph implementations live in `ai_core/graphs/`. Repository examples include:

- `ai_core/graphs/technical/collection_search.py`
- `ai_core/graphs/technical/retrieval_augmented_generation.py`
- `ai_core/graphs/technical/universal_ingestion_graph.py`
- `ai_core/graphs/business/framework_analysis_graph.py`
- `ai_core/graphs/web_acquisition_graph.py`

## Observability hooks used in code

Tracing and spans are emitted via helpers in:

- `ai_core/infra/observability.py`
- `ai_core/infra/observability` usage patterns throughout `ai_core/services/__init__.py` and ingestion/task code
