# Agentic Rebuild and Deprecate Roadmap (Runtime Contract First)

Current State Snapshot (entry points + execution path)
- Workbench entry points: `theme/urls.py` routes to view handlers re-exported in `theme/views.py` from `theme/views_rag_tools.py`, `theme/views_web_search.py`, `theme/views_chat.py`, `theme/views_ingestion.py`, `theme/views_framework.py`. Workbench context is centralized in `theme/helpers/context.py::prepare_workbench_context`, and the WebSocket workbench uses `theme/consumers.py`.
- Default graph execution paths:
  - Most workbench graph submissions use `theme/helpers/tasks.py::submit_business_graph` -> `ai_core/tasks/graph_tasks.py::run_business_graph` -> `ai_core/graph/registry.get` -> GraphRunner.run/invoke. Registry entries are in `ai_core/graph/bootstrap.py` (`collection_search`, `web_acquisition`, `framework_analysis`, `rag.default`, etc).
  - RAG chat uses `ai_core/services/rag_query.py::RagQueryService.execute` -> `ai_core/graphs/technical/retrieval_augmented_generation.run` (direct call, bypasses registry).
  - Framework analysis uses `ai_core/graphs/business/framework_analysis_graph.build_graph().run`.
  - Workbench crawler/ingestion uses `theme/views_ingestion.py` -> `CrawlerManager.dispatch_crawl_request` or `ai_core/services/handle_document_upload`, which route into `ai_core/tasks/graph_tasks.py::run_ingestion_graph`.

Contract anchor (non-UI)
- AgentRuntime + Capability IO specs + AgentState/Run records + golden harness artifacts are the primary contract. Workbench UI and endpoints are optional clients and are not contractual anchors.

Guardrails (hard constraints)
- ToolContext is the only ID source at runtime: AgentRuntime.start/resume accept only ToolContext instance + typed RuntimeConfig (no dict payloads); IDs may only come from `tool_context.<id>` and `agent_state.identity.<id>`; provide two ToolContext construction paths (workbench factory and dev runner fixtures/minimal DB snapshot); implement guardrail lint/test with allowlist + explicit suppression mechanism (no blanket disables).
- No new ID categories: Allowed technical identifiers are `run_id` and `event_id`; do not add new tenant/user/document ID categories, headers, or business identifier semantics.
- Bounded capability graphs (LangGraph-only): All new capabilities are LangGraph StateGraphs with explicit `input_schema`/`output_schema` and GraphIOSpec binding; single-pass only (no internal retries, no confidence gating, no adaptive query expansion, no semantic branching beyond explicit error exits); planning/replanning/retry logic belongs only to AgentRuntime policies.
- DecisionLog + StopDecision are contractual: every run terminates via explicit StopDecision (status, reason, evidence_refs) recorded in the decision log; resume reloads AgentState + decision log and continues deterministically; do not conflate graph completion with success.
- Storage separation: graph checkpoint stores graph state; AgentRunRecord stores lifecycle/status; harness artifacts store evaluation outputs (stable JSON schema); never merge these into one persisted blob.
- Cancellation semantics: early phases allow best-effort cancellation (prevent downstream execution, mark run cancelled); do not claim killing running Celery tasks until implemented.
- Harness diff is artifact-based: no text-equality requirements; enforce citation coverage, claim_to_citation completeness, ungrounded claim rate threshold, stable artifact schema, and time/token budget regression limits; baseline updates require explicit justification.
- Legacy graphs are frozen: no new behavior in legacy orchestration graphs (only deprecation markers/logging); new capabilities live under `ai_core/agent/capabilities/` and are registered via the registry bridge.
- Primary clients are Dev Runner and RagQueryService; workbench is optional and never a parity criterion.
- RuntimeConfig is typed and ID-free: only model routing labels, budgets, and runtime toggles/flags; no tenant/user identifiers.
- AgentRuntime flow execution uses a Flow Registry (named flows) with modular per-flow modules and typed I/O; avoid monolithic if/else dispatch in the runtime core.
- Bounded capability error exits are technical only; semantic fallbacks (low confidence, broaden, retry, replan, alternative strategies) live exclusively in AgentRuntime policies.
- Dev runner ToolContext is explicit and minimal: constructed from fixtures checked into the repo (and minimal DB access if strictly necessary); no implicit reconstruction, nested dict merges, or magical DB lookups.

Phase 1: AgentRuntime Foundations

Purpose
Define the central AgentRuntime with typed AgentState and explicit decision logging so control, stop semantics, and status are owned outside LangGraph wiring.

Work Items
- Title: AgentState, AgentRunRecord, and decision log contracts
  Intent: Introduce typed state and run record models for agentic flows, including decision log entries (plan/replan/stop/HITL) with deterministic serialization.
  Files or modules to inspect/modify: `ai_core/contracts/plans/`, `ai_core/agent/state.py` (new), `ai_core/agent/run_records.py` (new), `ai_core/graph/state.py`, `ai_core/graph/core.py`.
  Introduced components or removals: New AgentState/AgentRunRecord models and decision log schema; no changes to existing ToolContext schema.
  Acceptance criteria (testable): AgentState and AgentRunRecord round-trip with JSON serialization; decision log entries persist with deterministic ordering; tests confirm no implicit dict merges.
  Cross reference to review finding: Review 1 (no AgentState, success conflated with graph completion), Review 3 (deterministic state shape).
  Notes and allowed deviations: Reuse existing plan/evidence contracts; do not introduce new IDs or headers without explicit confirmation.

- Title: AgentRuntime lifecycle and run handles
  Intent: Implement runtime lifecycle APIs (start, resume, cancel, status) and explicit run handles that decouple task success from graph completion.
  Files or modules to inspect/modify: `ai_core/agent/runtime.py` (new), `ai_core/graph/core.py`, `ai_core/tasks/graph_tasks.py`, `llm_worker/tasks.py`.
  Introduced components or removals: AgentRuntime with run lifecycle and run handle identifiers; runtime status storage separate from graph results.
  Acceptance criteria (testable): A dummy capability can be executed, paused, resumed, and cancelled (best-effort cancellation); runtime flow selection goes through a Flow Registry with named flows and typed I/O; run status persists independently of capability output; introduce a StopDecision record (status, reason, evidence_refs) as part of AgentRunRecord; every runtime flow terminates via explicit StopDecision (not by graph completion); tests assert status transitions and StopDecision presence.
  Cross reference to review finding: Review 1 (compile-time control flow, async tracking/cancel, success equals completion).
  Notes and allowed deviations: Keep graph names sourced from `ai_core/graph/registry.py`; no new graph naming scheme.

Phase 2: Context and Determinism Hardening

Purpose
Harden AgentRuntime start/resume and eliminate implicit context reconstruction so runtime artifacts are deterministic and auditable.

Work Items
- Title: AgentRuntime entrypoint hardening (start/resume) and strict ToolContext origin
  Intent: Harden AgentRuntime.start/resume to accept only ToolContext plus minimal runtime config, and build AgentEnvelope inside the runtime with no implicit ID reconstruction.
  Files or modules to inspect/modify: `ai_core/agent/runtime.py` (new), `ai_core/agent/envelope.py` (new), `ai_core/tool_contracts/base.py`, `ai_core/graph/schemas.py`, `ai_core/agent/runtime_config.py` (new).
  Introduced components or removals: AgentEnvelope model and parser; remove getattr and nested dict fallbacks when constructing ToolContext for runtime flows.
  Acceptance criteria (testable): AgentRuntime.start/resume reject runs without ToolContext instance; RuntimeConfig is typed and contains no IDs; no node, service, or task reads IDs from dicts (only ToolContext and AgentState are sources); add a guardrail lint/test with allowlist + explicit suppression mechanism that fails on patterns like `state.get("tenant")`, `payload["tenant_id"]`, `getattr(x, "tenant_id")` in parsing paths, or `context["scope"]["tenant_id"]`, while allowlisting `tool_context.<id>` and `agent_state.identity.<id>`; IDs do not leak across concurrent runs.
  Cross reference to review finding: Review 3 (ID leaks, implicit context reconstruction, hidden dependencies).
  Notes and allowed deviations: ToolContext construction paths are limited to the workbench context factory (`theme/helpers/context.py::prepare_workbench_context`) and dev runner fixtures/minimal DB snapshot; runtime accepts ToolContext plus typed RuntimeConfig only; do not add new headers/meta keys without explicit confirmation.

- Title: Dependency injection for routers and caches
  Intent: Remove global mutable routers/workers by injecting dependencies through runtime capability descriptors.
  Files or modules to inspect/modify: `ai_core/rag/vector_store.py`, `ai_core/rag/vector_client.py`, `ai_core/rag/semantic_cache.py`, `ai_core/rag/strategy.py`, `ai_core/tools/shared_workers.py`.
  Introduced components or removals: Provider/factory interfaces owned by AgentRuntime; remove module-level singleton mutation for routing.
  Acceptance criteria (testable): Parallel runs for different tenants do not share mutable router state; tests simulate tenant isolation with concurrent runs.
  Cross reference to review finding: Review 1 (global mutable routers), Review 3 (hidden dependencies).
  Notes and allowed deviations: Allow immutable LRU caches if scoped per runtime instance.

- Title: Async run tracking and HITL interrupts
  Intent: Add cancellable run tracking and non-blocking HITL interrupts to prevent worker blocking.
  Files or modules to inspect/modify: `ai_core/agent/runtime.py`, `ai_core/tasks/graph_tasks.py`, `llm_worker/tasks.py`, `ai_core/services/` (HITL entrypoints).
  Introduced components or removals: Run handle with cancel token; interrupt/resume hooks; remove blocking waits in agentic flows.
  Acceptance criteria (testable): HITL waits do not block worker threads; cancellation halts downstream capability execution; tests cover cancel and resume paths; early phases can simulate HITL via resume inputs without UI changes.
  Cross reference to review finding: Review 1 (HITL blocking, async tracking), Review 3 (deterministic replay).
  Notes and allowed deviations: Align with Celery soft timeouts to avoid double cancellation logic.

Phase 3: Capability Registry and Bounded Graphs

Purpose
Limit LangGraph usage to deterministic, bounded capabilities and register them with clear IO specs for AgentRuntime.

Notes (bounded definition)
A bounded capability graph must be a LangGraph StateGraph with explicit `input_schema`/`output_schema` and GraphIOSpec binding. It must not contain internal retry loops, dynamic routing decisions, confidence-based branching, or adaptive query expansion. Conditional edges are permitted only for explicit error handling.

Work Items
- Title: Capability descriptor and registry bridge
  Intent: Define capability metadata (io_spec, sync/async, interruptibility) and bridge existing graph registry entries.
  Files or modules to inspect/modify: `ai_core/agent/capabilities.py` (new), `ai_core/graph/io.py`, `ai_core/graph/registry.py`, `ai_core/graph/bootstrap.py`.
  Introduced components or removals: Capability registry that references existing graph names; legacy graphs tagged as deprecated and excluded from default runtime resolution.
  Acceptance criteria (testable): Runtime can resolve and execute `web_acquisition` and `crawler.ingestion` as bounded capabilities; capability metadata lists io_spec and interruptibility; capabilities are registered via registry bridge without modifying legacy graph behavior.
  Cross reference to review finding: Review 2 (capability-ready graphs), Review 1 (agentic behavior hidden in graphs).
  Notes and allowed deviations: No new graph names; keep registry as the single source of truth.

- Title: New bounded retrieval/search capability (replacement for CollectionSearchAdapter)
  Intent: Implement a deterministic retrieval/search capability that returns ranked candidates and telemetry only.
  Files or modules to inspect/modify: `ai_core/graphs/technical/collection_search.py` (new bounded graph or new module), `ai_core/graph/io.py`, `ai_core/graph/bootstrap.py`.
  Introduced components or removals: New bounded capability graph; legacy collection_search orchestration graph stays frozen and deprecated.
  Acceptance criteria (testable): Capability output excludes planning/HITL/ingestion fields; io_spec validation at boundary; bounded definition enforced (no retries, no confidence gating, no adaptive expansion); tests confirm deterministic output given input.
  Cross reference to review finding: Review 2 (CollectionSearchAdapter unusable), Review 1 (compile-time control flow).
  Notes and allowed deviations: Do not refactor legacy graph beyond deprecation markers.

- Title: RetrievalCapability (bounded)
  Intent: Provide a bounded retrieval capability that accepts a retrieval_plan, filters, and budgets, and returns candidates, citations, and telemetry.
  Files or modules to inspect/modify: `ai_core/agent/capabilities/rag_retrieve.py` (new), `ai_core/graph/io.py`, `ai_core/graph/bootstrap.py`.
  Introduced components or removals: New RetrievalCapability module; legacy retrieval_augmented_generation graph remains frozen and deprecated.
  Acceptance criteria (testable): Input shape is retrieval_plan + filters + budgets; output shape is candidates + citations + telemetry; bounded definition enforced (no retries, no confidence gating, no adaptive expansion); io_spec validation passes.
  Cross reference to review finding: Review 2 (agentic smell in RAG graph), Review 1 (replanning/stop semantics).
  Notes and allowed deviations: Preserve model routing via labels from `MODEL_ROUTING.yaml`; do not refactor legacy RAG graph.

- Title: ComposeAnswerCapability (bounded)
  Intent: Provide a bounded answer composition capability that accepts a question plus selected evidence, and returns a draft plus claim-to-citation map.
  Files or modules to inspect/modify: `ai_core/agent/capabilities/rag_compose.py` (new), `ai_core/graph/io.py`, `ai_core/graph/bootstrap.py`.
  Introduced components or removals: New ComposeAnswerCapability module; legacy retrieval_augmented_generation graph remains frozen and deprecated.
  Acceptance criteria (testable): Input shape is question + selected evidence; output shape is draft + claim_to_citation map; bounded definition enforced (no retries, no confidence gating, no adaptive expansion); io_spec validation passes.
  Cross reference to review finding: Review 2 (agentic smell in RAG graph), Review 1 (replanning/stop semantics).
  Notes and allowed deviations: Preserve model routing via labels from `MODEL_ROUTING.yaml`; do not refactor legacy RAG graph.

Phase 4: AgentRuntime Flows + Dev Runner + Golden Harness

Purpose
Rebuild agentic behavior in AgentRuntime using bounded capabilities, with a dev runner and harness as the primary validation surface.

Work Items
- Title: Minimal dev runner for AgentRuntime flows
  Intent: Provide a minimal CLI runner that executes AgentRuntime flows without UI dependencies and supports resume inputs for simulated HITL.
  Files or modules to inspect/modify: `scripts/agent_run.py` (new), `ai_core/agent/runtime.py`, `ai_core/agent/state.py`, `ai_core/agent/runtime_config.py` (new).
  Introduced components or removals: Dev runner entrypoint that accepts ToolContext and runtime config, supports resume inputs, and emits run artifacts.
  Acceptance criteria (testable): `scripts/agent_run.py` can start and resume a run; HITL can be simulated via resume inputs; ToolContext construction path is fixtures/minimal DB snapshot (non-UI) with explicit inputs (no implicit reconstruction, no nested dict merges, no magical DB lookups); run artifacts are emitted with stable structure.
  Cross reference to review finding: Review 1 (HITL blocking, async tracking), Review 3 (determinism).
  Notes and allowed deviations: No UI requirement; the runner is the primary integration point until optional clients are wired.

- Title: Golden query set and diff harness for runtime artifacts
  Intent: Create a minimal golden dataset and a diff tool that can run both legacy and runtime paths for RAG chat and retrieval.
  Files or modules to inspect/modify: `ai_core/rag/quality/` (reuse patterns), `ai_core/tests/` (new eval tests), `scripts/` (new harness entrypoint).
  Introduced components or removals: Golden query fixtures and diff report output (text or JSON); no production impact.
  Acceptance criteria (testable): Harness runs in CI/test mode and can target both legacy and AgentRuntime executors; each run emits a JSON artifact with a stable output contract (`run_id`, inputs hash, retrieval candidates list with chunk ids and scores, citations list, final answer text, claim_to_citation map, decision log events); diffing compares these artifacts and enforces criteria for citation coverage (equal or better), claim_to_citation completeness, ungrounded claim rate below threshold, stable artifact schema, and time/token budget regression limits; golden harness baselines are versioned and frozen per migration step, and baseline changes require an explicit justification note in harness config.
  Cross reference to review finding: Review 1 (success != completion), Review 3 (determinism breaks).
  Notes and allowed deviations: Keep harness minimal and dev-only; no new public API surfaces.

- Title: Collection search runtime flow (planning, HITL, ingestion)
  Intent: Move planning, HITL, and auto-ingestion to AgentRuntime using bounded retrieval/search capability plus UniversalIngestion capability.
  Files or modules to inspect/modify: `ai_core/agent/runtime.py`, `ai_core/agent/state.py`, `ai_core/services/collection_search/`, `ai_core/graphs/technical/universal_ingestion_graph.py`.
  Introduced components or removals: New runtime flow and AgentState; legacy collection_search orchestration graph removed from default execution path.
  Acceptance criteria (testable): Runtime flow supports pause/resume for HITL, supports best-effort cancellation, and triggers ingestion via capability; decision log and run artifacts record each transition; StopDecision is recorded at termination.
  Cross reference to review finding: Review 1 (HITL blocking, async tracking), Review 2 (collection search agentic smell).
  Notes and allowed deviations: Keep ingestion thresholds/config in runtime policy, not graph constants.

- Title: RAG runtime flow (planning, retries, confidence gating)
  Intent: Implement RAG planning, retries, and gating in runtime; capabilities only retrieve and compose answers.
  Files or modules to inspect/modify: `ai_core/agent/runtime.py`, `ai_core/agent/state.py`, `ai_core/services/rag_query.py`, `ai_core/rag/strategy.py`.
  Introduced components or removals: Runtime RAG flow; legacy retrieval_augmented_generation graph remains available only behind legacy toggle.
  Acceptance criteria (testable): Runtime flow produces deterministic results for fixed plan; retries and replan decisions are logged in decision log; StopDecision recorded at termination; harness artifacts validate citation coverage, claim_to_citation completeness, and ungrounded claim thresholds.
  Cross reference to review finding: Review 1 (replanning blocked, success conflation), Review 2 (RAG graph agentic smell), Review 3 (determinism).
  Notes and allowed deviations: Preserve existing retrieval filters and model routing labels.

- Title: Framework analysis runtime flow
  Intent: Orchestrate framework analysis steps via runtime-owned control flow rather than graph compile-time edges.
  Files or modules to inspect/modify: `ai_core/agent/runtime.py`, `ai_core/services/framework_analysis_*` (new), `ai_core/graphs/business/framework_analysis_graph.py`.
  Introduced components or removals: Runtime flow with bounded sub-capabilities; legacy framework analysis graph deprecated.
  Acceptance criteria (testable): Runtime can reorder/skip steps based on AgentState; best-effort cancellation stops downstream execution; decision log captures branch decisions and timeouts; StopDecision recorded at termination.
  Cross reference to review finding: Review 1 (compile-time control flow), Review 2 (framework analysis agentic smell).
  Notes and allowed deviations: Keep existing output schema for compatibility until optional clients are wired.

Phase 5: Optional Clients and HITL Integration (Thin Adapter Only)

Purpose
Wire optional clients (including the workbench) via a thin adapter, only when needed for HITL interaction.

Work Items
- Title: Theme to runtime adapter (thin, optional)
  Intent: Provide a minimal adapter that calls AgentRuntime.start/resume using ToolContext provided by the workbench context factory, without broad view refactors.
  Files or modules to inspect/modify: `theme/helpers/runtime_adapter.py` (new), `ai_core/agent/runtime.py`.
  Introduced components or removals: Adapter wrapper for workbench calls; no broad changes to `theme/views_*`.
  Acceptance criteria (testable): Adapter can be invoked from tests with a ToolContext and typed RuntimeConfig; workbench changes are limited to thin adapter wiring when needed; no workbench parity criteria.
  Cross reference to review finding: Review 1 (control flow centralized), Review 3 (context determinism).
  Notes and allowed deviations: Workbench remains optional; do not use it as a contract anchor.

- Title: RagQueryService routing through runtime flow (legacy behind toggle)
  Intent: Route `ai_core/services/rag_query.py` through the runtime flow, keeping the legacy path behind an explicit toggle.
  Files or modules to inspect/modify: `ai_core/services/rag_query.py`, `ai_core/agent/runtime.py`.
  Introduced components or removals: Runtime entrypoint for RAG queries; legacy execution remains available only via toggle for comparison.
  Acceptance criteria (testable): RagQueryService produces runtime artifacts (AgentState, decision log, harness JSON) and respects the legacy toggle; uses typed RuntimeConfig with no IDs; no UI parity requirement.
  Cross reference to review finding: Review 2 (agentic smell in RAG graph), Review 1 (control flow).
  Notes and allowed deviations: Keep adapter thin; prefer dev runner and harness as primary validation.

Phase 6: Deprecation and Default Registry Cleanup

Purpose
Remove legacy orchestration graphs from default runtime resolution and keep only bounded capabilities for AgentRuntime.

Work Items
- Title: Deprecate legacy orchestration graphs and remove default wiring
  Intent: Mark legacy orchestration graphs as deprecated and remove them from default registry wiring used by the runtime.
  Files or modules to inspect/modify: `ai_core/graph/bootstrap.py`, `ai_core/graph/README.md`, `docs/development/rag-tools-workbench.md`.
  Introduced components or removals: Deprecated graph markers and logging; default registry excludes legacy orchestration graphs; legacy toggle kept only for comparison period.
  Acceptance criteria (testable): Default runtime path never resolves legacy orchestration graphs; legacy toggle remains only until the golden harness is green in CI for N runs (defined in harness config, >= 20) and regression criteria (citation coverage, claim_to_citation completeness, ungrounded claim threshold, time/token budgets) are not violated; deprecation is enforced by runtime artifact checks, not UI parity.
  Cross reference to review finding: Review 2 (agentic smell in legacy graphs), Review 1 (control flow compile-time).
  Notes and allowed deviations: If a breaking contract change is required, add a backlog item with code pointers and acceptance criteria before proceeding.
