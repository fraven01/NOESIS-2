# Agentic Migration Roadmap (Review-Informed)

Phase 1: Agent Runtime Foundations

Purpose
- Stand up a central AgentRuntime with typed AgentState so control and stop semantics move out of LangGraph wiring.

Work Items
- Title: AgentState & AgentGoal contracts
  Intent: Define Pydantic models for agent goals, run metadata, slots/evidence bindings, and status enums to replace plain dict state.
  Relevant files / modules: `ai_core/agent/state.py` (new), `ai_core/contracts/plans/*`, `ai_core/graph/state.py`.
  What is removed, replaced, or introduced: Introduce typed AgentState/AgentGoal, deprecate dict-based state hand-offs between graphs and services.
  Acceptance criteria (testable): AgentState serializes to/from JSON with deterministic keys; state hashing/plan_key derivation matches PlanScope; unit tests cover validation and immutability.
  Cross-References: Review 1 (no first-class AgentState), Review 3 (deterministic state shape).
  Notes: Keep PlanScope/ImplementationPlan as the source for plan/evidence fields; no new ID categories introduced.
- Title: AgentRuntime skeleton & run lifecycle
  Intent: Build an AgentRuntime that owns goal registration, planning hook points, stop/replan semantics, and return statuses distinct from graph completion.
  Relevant files / modules: `ai_core/agent/runtime.py` (new), `ai_core/graph/execution/runner.py`, `ai_core/graph/bootstrap.py`.
  What is removed, replaced, or introduced: Introduce start/resume/cancel API with run records; remove reliance on LangGraph completion to signal success.
  Acceptance criteria (testable): Dummy capability can be executed via runtime with start/resume/cancel; run records persist status separately from capability output; tests assert stop conditions and resumability.
  Cross-References: Review 1 (control flow compile-time, success conflated with completion, async not tracked).
  Notes: Runtime should accept a capability descriptor (below) instead of a GraphRunner directly to avoid implicit globals.
- Title: Capability descriptor & registry bridge
  Intent: Define capability metadata (io_spec, cost/risk flags, sync/async, interruptibility) and bridge existing GraphIOSpec-based graphs into a capability registry.
  Relevant files / modules: `ai_core/agent/capabilities.py` (new), `ai_core/graph/io.py`, `ai_core/graph/registry.py`, `ai_core/graph/bootstrap.py`.
  What is removed, replaced, or introduced: Introduce capability registry entries for `web_acquisition` and `universal_ingestion`; mark others for decomposition; remove direct module_runner registration for agentic graphs.
  Acceptance criteria (testable): Capability registry lists bounded graphs with io_spec; runtime can resolve capability by name and execute via registry adapter; tests cover discovery and execution of `web_acquisition` and `crawler.ingestion`.
  Cross-References: Review 2 (capability-ready graphs), Review 1 (routing decisions centralized).
  Notes: Keep registry as source of truth; do not create parallel naming schemes.

Phase 2: Context & Control-Plane Hardening

Purpose
- Eliminate fatal context propagation issues and prepare async tracking/interrupts so agent decisions are deterministic and auditable.

Work Items
- Title: Canonical AgentEnvelope & context parser
  Intent: Replace per-call scope/business reconstruction with a typed AgentEnvelope that wraps ToolContext plus runtime metadata for agent runs.
  Relevant files / modules: `ai_core/graph/schemas.py`, `ai_core/tool_contracts/base.py`, `ai_core/tasks/graph_tasks.py`, `ai_core/services/` (graph/task entrypoints), `ai_core/views*.py`.
  What is removed, replaced, or introduced: Introduce AgentEnvelope parser; forbid getattr/dict fallbacks for IDs; ensure tool_context is parsed once and stored; remove nested dict merges in ingestion/task helpers.
  Acceptance criteria (testable): All graph/task entrypoints reject runs without a valid envelope; tests assert no getattr/meta fallback paths remain; chaos tests confirm IDs do not leak across runs.
  Cross-References: Review 3 (fatal context propagation, ID leakage), Review 1 (implicit execution context).
  Notes: Align with Option A contracts (scope vs business separation); no new headers introduced.
- Title: Dependency-injected routers and workers
  Intent: Remove global mutable routers/worker singletons and inject dependencies via runtime/capability descriptors.
  Relevant files / modules: `ai_core/rag/vector_store.py:get_default_router`, `ai_core/rag/semantic_cache.py`, `ai_core/rag/strategy.py`, `ai_core/tools/shared_workers.py`.
  What is removed, replaced, or introduced: Replace module-level router caches with factory/provider objects passed through AgentRuntime; make caches immutable per run/tenant.
  Acceptance criteria (testable): Capability execution uses injected router/worker handles; parallel runs for different tenants do not share mutable state; tests simulate two tenants to ensure isolation.
  Cross-References: Review 1 (global mutable routers), Review 3 (hidden dependencies).
  Notes: Evaluate perf impact of per-run construction; allow optional LRU if immutable.
- Title: Async run tracking, cancellation, and HITL interrupts
  Intent: Add run handles and cancellation/interrupt hooks so async work can be tracked and stopped without blocking worker threads.
  Relevant files / modules: `ai_core/graph/execution/celery.py`, `ai_core/tasks/graph_tasks.py`, `llm_worker/tasks.py`, `ai_core/graphs/business/framework_analysis/graph.py` (node timeouts), `ai_core/graphs/technical/retrieval_augmented_generation.py` (HITL flow).
  What is removed, replaced, or introduced: Introduce AgentRunHandle with status and cancel token; replace blocking HITL waits with interrupt/resume; propagate cancel tokens into capability execution.
  Acceptance criteria (testable): A submitted run can be polled and cancelled; cancellation prevents further node execution; HITL waits free the worker and resume with stored state; tests cover cancel and resume paths.
  Cross-References: Review 1 (HITL blocking, async not cancellable), Review 3 (deterministic replay).
  Notes: Coordinate with Celery task soft-timeouts to avoid duplicate cancellation semantics.

Phase 3: Capability Layer Consolidation

Purpose
- Bound the capability surface, harden persistence, and stop treating graph completion as task success.

Work Items
- Title: Capability classification & registry update
  Intent: Declare which graphs are retained as bounded capabilities (UniversalIngestionGraph, WebAcquisitionGraph) and which are slated for decomposition (RetrievalAugmentedGenerationGraph, FrameworkAnalysisStateGraph, CollectionSearchAdapter).
  Relevant files / modules: `ai_core/graph/bootstrap.py`, `ai_core/graph/registry.py`, `ai_core/graphs/README.md`.
  What is removed, replaced, or introduced: Tag registry entries with capability type; add wrappers for bounded graphs; remove legacy adapter registration that hides agentic behavior.
  Acceptance criteria (testable): Registry exposes capability type for each entry; bounded graphs run unchanged via runtime adapter; decomposed graphs emit warnings directing callers to AgentRuntime path.
  Cross-References: Review 2 (graph capability assessment), Review 1 (hidden agentic behavior inside graphs).
  Notes: Keep registry as single source for graph names to satisfy AGENTS contract.
- Title: Checkpointer upgrade for AgentState and decision log
  Intent: Persist AgentState plus decision/evidence logs separately from capability output; stop using graph completion as success signal.
  Relevant files / modules: `ai_core/graph/core.py`, `ai_core/graph/state.py`, `common/object_store_defaults.py`.
  What is removed, replaced, or introduced: Add status/result fields to persisted payload; store decision log (plan/evidence/routing/HITL events); decouple plan_key storage from graph_version.
  Acceptance criteria (testable): Restart/resume reloads AgentState and decision log; failing capability preserves state and marks run as failed without overwriting evidence; unit tests cover corrupted/partial checkpoint recovery.
  Cross-References: Review 1 (task success conflated with completion), Review 3 (deterministic replay).
  Notes: Preserve existing plan/evidence serialization shape for compatibility.
- Title: Boundary schema enforcement for capability graphs
  Intent: Ensure every StateGraph declares explicit input/output schemas and aligns with GraphIOSpec (covers backlog LG-1).
  Relevant files / modules: `ai_core/graphs/web_acquisition_graph.py`, `ai_core/graphs/technical/universal_ingestion_graph.py`, `ai_core/graphs/technical/retrieval_augmented_generation.py`, `ai_core/graphs/technical/collection_search.py`, `ai_core/graphs/business/framework_analysis/graph.py`, `documents/processing_graph.py`.
  What is removed, replaced, or introduced: Add/verify input_schema/output_schema on StateGraph init; reject implicit full-state defaults; align io_spec attachments.
  Acceptance criteria (testable): All listed graphs compile with explicit schemas; contract tests assert schema_id/schema_version validation; LG-1 acceptance satisfied.
  Cross-References: Review 1 (compile-time wiring without boundaries), Backlog LG-1.
  Notes: Coordinate with existing tests in `ai_core/tests/test_tool_signature_guardrails.py`.

Phase 4: Collection Search Decomposition

Purpose
- Move planning, HITL, and ingestion delegation out of the collection_search graph into AgentRuntime.

Work Items
- Title: Search/ranking capability extraction
  Intent: Refactor `collection_search` into a bounded capability that returns ranked candidates/telemetry; remove plan/HITL logic from the graph.
  Relevant files / modules: `ai_core/graphs/technical/collection_search.py` (remove hitl/build_plan/trigger_ingestion nodes), `ai_core/graphs/technical/collection_search_README.md`, `ai_core/services/collection_search/hitl.py`.
  What is removed, replaced, or introduced: Graph outputs candidates + scoring metrics only; plan assembly moves to runtime using `ai_core/contracts/plans`; HITL prompts emitted via runtime interrupt channel.
  Acceptance criteria (testable): Capability output excludes plan/HITL fields; runtime builds ImplementationPlan with evidence based on capability telemetry; tests verify ingestion is not triggered inside graph and HITL interrupts resume correctly.
  Cross-References: Review 2 (hidden agentic behavior), Review 1 (HITL blocking).
  Notes: Keep ingestion top-k/score thresholds configurable via runtime, not graph constants.
- Title: Collection Search AgentState & ingestion orchestration
  Intent: Add AgentState schema and runtime policy for retries, auto-ingest, and HITL decisions.
  Relevant files / modules: `ai_core/agent/state.py` (collection search state), `ai_core/services/crawler_runner.py`, `ai_core/tasks/ingestion_tasks.py`, `ai_core/graph/state.py`.
  What is removed, replaced, or introduced: Runtime triggers ingestion via capability call to UniversalIngestionGraph; adds explicit statuses (planning, waiting_hitl, ingesting, cancelled); records ingestion task handles in AgentState.
  Acceptance criteria (testable): End-to-end run can pause for HITL without holding worker, resume with decision, and trigger ingestion via runtime; cancellation stops ingestion dispatch; tests cover success, HITL, and cancel paths.
  Cross-References: Review 1 (async tracking, HITL), Review 2 (graph decomposition).
  Notes: Align acceptance with existing HITL gateway pattern in `ai_core/services/collection_search/hitl.py`.

Phase 5: RAG Graph Decomposition

Purpose
- Remove agentic logic from RetrievalAugmentedGenerationGraph and centralize planning/retries/HITL in AgentRuntime.

Work Items
- Title: Runtime-managed retrieval planning and retries
  Intent: Extract query planning, variant generation, retry/backoff, and confidence scoring from the RAG graph into runtime strategies.
  Relevant files / modules: `ai_core/graphs/technical/retrieval_augmented_generation.py`, `ai_core/rag/strategy.py`, `ai_core/rag/semantic_cache.py`, `ai_core/rag/metrics.py`.
  What is removed, replaced, or introduced: Graph executes a provided retrieval plan (queries, limits, rerank policy) without internal loop/retry; runtime owns retry policies and fallback selection.
  Acceptance criteria (testable): Capability run is deterministic given a plan; runtime tests cover replan after failure without modifying graph; internal graph no longer mutates plan mid-run.
  Cross-References: Review 1 (runtime replanning blocked), Review 2 (hidden agentic behavior).
  Notes: Preserve evidence logging but move confidence gating to runtime.
- Title: HITL and gating as runtime interrupts
  Intent: Replace in-graph blocking HITL and threshold gates with runtime-managed interrupts/resume.
  Relevant files / modules: `ai_core/graphs/technical/retrieval_augmented_generation.py` (HITL/guardrail nodes), `theme/views.py` (if applicable).
  What is removed, replaced, or introduced: Graph emits a HITL required flag and context; runtime queues HITL requests and resumes with user decision; no worker waits on UI.
  Acceptance criteria (testable): RAG run can pause for HITL without holding thread; resume consumes stored AgentState and user input; tests assert worker threads are free during wait.
  Cross-References: Review 1 (HITL blocking), Review 3 (context determinism during HITL).
  Notes: Ensure compose/answer nodes accept injected decision/results instead of reading globals.
- Title: RAG AgentState & evidence bindings
  Intent: Add typed state capturing question, query variants, retrieved chunks, evidence/confidence, and answer drafts linked to ImplementationPlan.Evidence.
  Relevant files / modules: `ai_core/agent/state.py`, `ai_core/graph/state.py`, `ai_core/contracts/plans/evidence.py`.
  What is removed, replaced, or introduced: Replace ad-hoc state keys (`chat_history`, retries, guardrail flags) with typed fields; tie evidence refs to chunk/document IDs.
  Acceptance criteria (testable): Persisted state includes evidence with confidence labels; replay produces same gating decisions; unit tests validate schema.
  Cross-References: Review 3 (deterministic replay and ID leakage).
  Notes: Keep semantic cache hooks but route through runtime-provided cache handle.

Phase 6: Framework Analysis Decomposition

Purpose
- Remove agentic control from FrameworkAnalysisStateGraph and hand orchestration to AgentRuntime.

Work Items
- Title: Split detection/assembly into capabilities orchestrated by runtime
  Intent: Break FrameworkAnalysisStateGraph into capabilities (init/fetch, detect_type, locate_components, assemble_profile) executed under runtime routing.
  Relevant files / modules: `ai_core/graphs/business/framework_analysis/graph.py`, `ai_core/graphs/business/framework_analysis/nodes.py`, `ai_core/services/framework_analysis_capabilities.py`.
  What is removed, replaced, or introduced: Remove monolithic graph orchestration and adapter; add capability wrappers with io_spec; runtime controls branching/early exit policies.
  Acceptance criteria (testable): Runtime can run individual capabilities and re-order/skip based on AgentState; graph wrapper no longer holds control flow; tests cover resume after partial completion.
  Cross-References: Review 2 (hidden agentic behavior), Review 1 (compile-time control flow).
  Notes: Keep retrieval service integration but inject via capability descriptor.
- Title: Runtime-owned timeouts and cancellation
  Intent: Move per-node ThreadPoolExecutor timeouts and error handling out of the graph into runtime policies.
  Relevant files / modules: `ai_core/graphs/business/framework_analysis/graph.py`, `ai_core/infra/observability.py`.
  What is removed, replaced, or introduced: Remove in-graph timeout wrappers; add runtime-level timeout/cancel tokens propagated to capabilities; normalize error payloads.
  Acceptance criteria (testable): Cancel/timeout from runtime halts capability execution; no ThreadPoolExecutor blocks remain; tests cover timeout-induced HITL/escalation paths.
  Cross-References: Review 1 (async tracking, stop semantics).
  Notes: Ensure existing observability events stay compatible.
