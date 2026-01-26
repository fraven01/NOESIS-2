# L-Track Implementation Plan (Agent-Friendly Typing)

Date: 2026-01-23
Owner: TBD
Source: roadmap/L-Track_Deep_Review_2026-01-23.md

Goal
- Achieve L-Track targets: 95% graph state typing, 100% node return typing, 100% Celery kwargs typing, AX-score 95% average.
- Keep changes within existing contracts unless explicitly approved per AGENTS.md stop conditions.

Scope Assumptions
- This plan is a doc-only plan. No runtime changes are made here.
- Breaking contract changes require user confirmation and a backlog entry per AGENTS.md.

Phasing Overview (L-Track)
1) Doc and model clarity (AX-score first)
2) Celery TaskContext typing
3) Node-return typing and graph state typing
4) Retrieval and RAG typing
5) Contract hardening + test migration

Phase 1: AX-Score lift (Week 1)
Objective: remove the largest agent-readability gaps without changing runtime semantics.
Work items
- Document ChunkMeta fields (ai_core/rag/ingestion_contracts.py:69).
- Document CrawlerIngestionPayload fields (ai_core/rag/ingestion_contracts.py:109).
- Document RetrieveMeta fields (ai_core/nodes/retrieve.py:100).
- Document ComposeOutput fields (ai_core/nodes/compose.py:38).
Acceptance criteria
- All listed models have Field descriptions and any existing Literal types are retained.
- No schema shape change (descriptions only).
Notes
- This phase should be safe with no breaking behavior.

Phase 2: TaskContext typing for Celery (Week 2)
Objective: eliminate untyped Celery kwargs while preserving existing runtime behavior.
Work items
- Define TaskContext, TaskScopeContext, TaskContextMetadata models (new module).
- Use TaskContext in common/celery.py for meta extraction and validation (common/celery.py:270+).
- Replace raw dict access (meta.get("key_alias"), args[0] fallback) with typed reads (common/celery.py:315+, 354+).
Acceptance criteria
- Task meta is validated via Pydantic in common/celery.py.
- All fields formerly passed via kwargs are mapped into TaskContextMetadata.
- Tests updated to pass strict typing.
Risk
- This likely changes runtime semantics (args[0] fallback, raw dict access).
- Requires user confirmation and backlog entry before implementation.

Phase 3: Node return typing for ingestion graphs (Weeks 3-4)
Objective: eliminate dict[str, Any] in node outputs and graph state.
Work items
- Define TypedDict outputs for key nodes (ValidateInputNodeOutput, DeduplicationNodeOutput, PersistNodeOutput, ProcessNodeOutput).
- Apply TypedDicts in universal_ingestion_graph.py and related nodes (ai_core/graphs/technical/universal_ingestion_graph.py:110+).
- Identify and type graph state keys used across nodes (ai_core/graphs/technical/universal_ingestion_graph.py:135+).
Acceptance criteria
- Node outputs are typed with TypedDicts.
- Graph state uses typed structures for tool_context and document fields.
Risk
- May introduce new typed identifiers or keys.
- Requires confirmation if new IDs/meta keys are added.

Phase 4: Retrieval/RAG typing (Weeks 4-5)
Objective: remove high-risk untyped mappings in retrieval and RAG graphs.
Work items
- Replace MutableMapping usage in retrieval_augmented_generation.py with typed model or TypedDict (ai_core/graphs/technical/retrieval_augmented_generation.py:120+).
- Introduce FilterSpec (typed filters) and migrate call sites (ai_core/nodes/retrieve.py:16+, ai_core/rag/vector_store usage).
- Convert HybridParameters dataclass to Pydantic if required (ai_core/nodes/_hybrid_params.py:12+).
Acceptance criteria
- No use of dict[str, Any] in retrieval state where typing is feasible.
- Filters are validated by FilterSpec where used.
Risk
- FilterSpec introduction may be a contract change.
- Requires confirmation and backlog entry before implementation.

Phase 5: Contract hardening and tests (Weeks 5-6)
Objective: align tests with strict typing and remove legacy fallbacks.
Work items
- Update tests listed in the report to use TaskContext and typed meta (ai_core/tests/, common/tests/).
- Remove args[0] fallback, enforce tool_context_from_meta (common/celery.py:300+).
- Validate session_scope tuple elements (common/celery.py:430+).
Acceptance criteria
- All tests pass without dict fallbacks or raw meta access.
- No lingering untyped kwargs in Celery code path.
Risk
- Breaking change risk is high; requires user confirmation and backlog entry.

Dependencies and Order
- Phase 1 can start immediately.
- Phase 2 should precede Phase 5 (tests depend on typed TaskContext).
- Phase 3 and Phase 4 can run in parallel after Phase 2 is planned.

Decision Points (need explicit confirmation)
- Removing args[0] fallback in common/celery.py.
- Enforcing tool_context_from_meta strictness.
- Introducing FilterSpec and other new models that change contract shapes.

Code Pointers (from code review)
- common/celery.py: ContextTask._gather_context fallback to args[0]/args[1] and meta.get("key_alias") bypass.
- ai_core/rag/ingestion_contracts.py: ChunkMeta and CrawlerIngestionPayload lack per-field descriptions.
- ai_core/nodes/retrieve.py: RetrieveMeta lacks per-field descriptions.
- ai_core/nodes/compose.py: ComposeOutput lacks per-field descriptions.
- ai_core/graphs/technical/universal_ingestion_graph.py: Node returns use dict[str, Any].
- ai_core/graphs/technical/retrieval_augmented_generation.py: state/meta use MutableMapping[str, Any].
- ai_core/nodes/_hybrid_params.py: HybridParameters is a dataclass, not Pydantic.

Suggested Backlog Items (for roadmap/backlog.md)
1) L-Track: TaskContext models and Celery meta validation
2) L-Track: Node return TypedDicts for universal ingestion graph
3) L-Track: Retrieval/RAG state typing and FilterSpec
4) L-Track: Remove legacy meta fallbacks and strict test updates

Metrics Tracking
- Maintain a small checklist per phase to update:
  - graph state typing %
  - node return typing %
  - Celery kwargs typing %
  - AX-score average
