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

Identity, Authz, Users (current state)
- Cases system: event-sourced lifecycle (phases derived from events), CaseEvent links to ingestion/search transitions, CaseMembership for user-case grants, CaseAuthz policies by tenant account type.
- User management: Django user + profile (role/account_type/expiry), invitations (token onboarding), auth via session/basic with policy enforcement, multi-tenancy via django-tenants.
- Identity in ToolContext: `user_id` for HTTP requests, `service_id` for S2S tasks; mutually exclusive per hop.
- Case constraint: `(tenant, external_id)` unique; case_id lives in `ToolContext.business.case_id`.
- Gaps: no REST CRUD for users, email not unique, invitation email stub; case lifecycle static; no parent/child case relations.
Note: We can full-break and reset the DB during migration.

Security and compliance guarantees (target)
| Component | Responsibility | Security guarantee |
|----------|----------------|--------------------|
| ToolContext | Carries user + tenant identity | Identity pinning (immutable identity per hop) |
| Execution scopes | Define allowed action radius | Prevents case/tenant leakage |
| DI (Phase 2.3) | Isolates routers/caches | Prevents in-memory data leaks |
| DecisionLog | Records all actions | Auditability and compliance trace |

Capability Inventory (current state)

This section documents the existing functional building blocks and their intended role in the new platform.

Legacy Orchestration Graphs (phase‑out, keep frozen)
| Name | Registry Key | Bounded | io_spec | Role |
|------|--------------|---------|---------|------|
| CollectionSearch | `collection_search` | No | No | Legacy orchestration (replace with unified RetrievalCapability + runtime flow) |
| RAG | `rag.default` | No | No | Legacy orchestration (replace with Retrieval + Compose capabilities) |
| RagRetrieval | `rag_retrieval` | No | Yes | Legacy retrieval graph (replace with RetrievalCapability) |
| FrameworkAnalysis | `framework_analysis` | No | Yes | PoC business graph; lowest priority, can be replaced later |

Current Bounded Capabilities (keep)
| Name | Registry Key | Bounded | io_spec |
|------|--------------|---------|---------|
| UniversalIngestion | `crawler.ingestion` | Yes | Yes |
| WebAcquisition | `web_acquisition` | Yes | Yes |
Note: WebAcquisition is a bounded acquisition capability target; keep and register via capability bridge.

Worker Capabilities (not in registry, used internally)
| Name | Module | Bounded | Singleton Issue | Status |
|------|--------|---------|-----------------|--------|
| HybridSearchAndScore | `llm_worker/graphs/hybrid_search_and_score.py` | Partial | Yes (`GRAPH` module-level) | Legacy/worker-internal; must not be depended on by AgentRuntime capabilities; singleton removal in Phase 3 |
| ScoreResults | `llm_worker/graphs/score_results.py` | Yes | No | Capability target (bounded) |

Domain Capabilities (non-graph services)
| Capability | Module | Function | Status |
|------------|--------|----------|--------|
| CaseLifecycle | `cases/lifecycle.py` | Apply lifecycle transitions, resolve case state | Active |
| CaseEvents | `cases/integration.py` / `cases/services.py` | Emit + record case lifecycle events | Active |

Target Platform Capabilities (atomic, to be built)
| Capability | Role | Notes |
|-----------|------|-------|
| RetrievalCapability | Retrieval + ranking from plan | Replaces RAG/RagRetrieval graphs |
| ComposeAnswerCapability | Answer drafting + claim mapping | Replaces RAG compose |
| EvidenceMappingCapability | Claim→citation extraction | Supports groundedness checks |
| DocAcquisitionCapability | Connectors (Confluence/SharePoint/Slack/etc.) | Plan‑driven, no UI dependency |
| IngestionCapability | UniversalIngestion wrapper | Stable “black box” ingestion |
| CaseResolutionCapability | Resolve case/workflow constraints | Optional future (after PoC) |

Nodes (`ai_core/nodes/`) - Atomic LLM operations
| Node | Input Contract | Output Contract | Function |
|------|----------------|-----------------|----------|
| retrieve | `RetrieveInput(query, filters, hybrid, top_k)` | `RetrieveOutput(matches, routing, telemetry)` | Hybrid Vector Search |
| compose | `ComposeInput(question, snippets)` | `ComposeOutput(answer, reasoning, used_sources)` | Answer Generation |
| extract | `ExtractInput(text)` | `ExtractOutput(items)` | Fact Extraction |
| classify | `ClassifyInput(text)` | `ClassifyOutput(classification)` | Co-determination Classification |
| assess | `AssessInput(...)` | `AssessOutput(...)` | Assessment |
| needs | `NeedsInput(...)` | `NeedsOutput(...)` | Needs Analysis |
| draft_blocks | `DraftBlocksInput(...)` | `DraftBlocksOutput(...)` | Block Drafting |

RAG Infrastructure (`ai_core/rag/`) - Services used by capabilities
| Service | Module | Function | DI Required (P2.3) |
|---------|--------|----------|-------------------|
| VectorStoreRouter | `vector_store.py` | Multi-tenant vector routing | Yes (`get_default_router()`) |
| VectorClient | `vector_client.py` | pgvector access | No |
| EmbeddingCache | `embedding_cache.py` | Embedding caching | Yes (module-level) |
| SemanticCache | `semantic_cache.py` | Query result caching | Yes (module-level) |
| ProfileResolver | `profile_resolver.py` | Embedding profile resolution | No |
| HybridFusion | `hybrid_fusion.py` | RRF score fusion | No |
| Reranker | `rerank.py` | Cross-encoder reranking | No |
| QueryPlanner | `query_planner.py` | Query variant generation | No |
| LexicalSearch | `lexical_search.py` | BM25/trigram search | No |
| EvidenceGraph | `evidence_graph.py` | Claim-citation mapping | No |
| ContextualEnrichment | `contextual_enrichment.py` | Chunk enrichment | No |

Chunking (`ai_core/rag/chunking/`)
| Chunker | Type | Bounded |
|---------|------|---------|
| HybridChunker | Semantic + Size | Yes |
| AgenticChunker | LLM-based | Partial (LLM call) |
| LateChunker | Late-interaction | Yes |

High-Level Services (`ai_core/services/`) - Current entrypoints
| Service | Module | Uses | Status |
|---------|--------|------|--------|
| RagQueryService | `rag_query.py` | RAG graph directly | Legacy → P5.2 |
| FrameworkAnalysisService | `framework_analysis.py` | FrameworkAnalysis graph | Legacy → P4.5 |
| CollectionSearchService | `collection_search/` | CollectionSearch graph + HITL | Legacy → P4.3 |
| DocumentUploadService | `document_upload.py` | UniversalIngestion | Active |
| CrawlerRunner | `crawler_runner.py` | UniversalIngestion | Active |

Capability → Node/Service Mapping (target state)
- P3.3 RetrievalCapability (unified): uses `nodes.retrieve`, `rag.hybrid_fusion`, `rag.rerank`; DI: VectorStoreRouter, EmbeddingCache, Reranker; intent-driven via RetrievalPlan (e.g., `COLLECTION_BROWSE`, `RAG_QUERY`)
- P3.4 ComposeAnswerCapability: uses `nodes.compose`, `rag.evidence_graph`; DI: LLM-Client
- P3.x EvidenceMappingCapability: uses `rag.evidence_graph`; DI: LLM-Client (if needed for extraction)
- P3.x DocAcquisitionCapability: uses connector adapters (Confluence/SharePoint/Slack); DI: Connector clients, Auth provider
- P3.6 DocumentStorageCapability: uses `documents.repository`, `documents.lifecycle_store`, `ingestion.start_ingestion_run`; DI: DocumentsRepository, DocumentLifecycleStore

Document Repository Infrastructure (`documents/`)
| Component | Module | Function | DI Required (P2.3) |
|-----------|--------|----------|-------------------|
| DocumentsRepository | `repository.py` | CRUD for NormalizedDocument | Yes (factory in `ai_core/services/repository.py`) |
| DocumentLifecycleStore | `repository.py` | State transitions (ACTIVE/RETIRED/DELETED) | Yes (module-level default) |
| PersistentDocumentLifecycleStore | `repository.py` | DB-backed lifecycle + ingestion run tracking | Yes |
| Storage | `storage.py` | Blob storage abstraction (ObjectStore, Local, S3) | Yes |
| DomainService | `domain_service.py` | High-level document operations | No (uses injected repository) |

Document Repository Methods (Contract Surface)
| Method | Input Contract | Output Contract | Function |
|--------|----------------|-----------------|----------|
| `upsert` | `NormalizedDocument, workflow_id, scope, audit_meta` | `NormalizedDocument` | Create/update document |
| `get` | `tenant_id, document_id, version?, prefer_latest?, include_retired?` | `Optional[NormalizedDocument]` | Fetch document by ID |
| `list_by_collection` | `tenant_id, collection_id, limit, cursor, latest_only?` | `Tuple[List[DocumentRef], cursor?]` | List documents in collection |
| `delete` | `tenant_id, document_id, workflow_id?, hard?` | `bool` | Soft/hard delete |
| `add_asset` | `Asset, workflow_id?` | `Asset` | Attach asset to document |
| `get_asset` | `tenant_id, asset_id, workflow_id?` | `Optional[Asset]` | Fetch asset |
| `delete_asset` | `tenant_id, asset_id, workflow_id?, hard?` | `bool` | Delete asset |

Document Lifecycle States
| State | Transitions To | Description |
|-------|---------------|-------------|
| `active` | `active`, `retired`, `deleted` | Document is visible and indexed |
| `retired` | `active`, `retired`, `deleted` | Soft-deleted, excluded from search |
| `deleted` | `deleted` | Terminal state (hard delete) |

Ingestion Run Tracking (`IngestionRunRecord`)
| Method | Module | Function |
|--------|--------|----------|
| `record_ingestion_run_queued` | `repository.py` | Create queued run record |
| `mark_ingestion_run_running` | `repository.py` | Mark run as started |
| `mark_ingestion_run_completed` | `repository.py` | Mark run as succeeded/failed |
| `get_ingestion_run` | `repository.py` | Get run status by tenant+case |
| `start_ingestion_run` | `ai_core/services/ingestion.py` | Orchestrate full ingestion flow |

Ingestion Run Status Flow
```
queued → running → succeeded
                 ↘ failed
```

Crawler Infrastructure (`crawler/`, `ai_core/services/crawler_*.py`, `ai_core/contracts/crawler_runner.py`)
| Component | Module | Function | DI Required (P2.3) |
|-----------|--------|----------|-------------------|
| CrawlerManager | `crawler/manager.py` | Dispatch crawl tasks to workers | Yes (L3 Manager) |
| CrawlerRunRequest | `ai_core/schemas.py` | Request schema with modes, origins, limits | No (Pydantic) |
| CrawlerOriginConfig | `ai_core/schemas.py` | Per-URL configuration (provider, tags, limits) | No (Pydantic) |
| CrawlerGraphState | `ai_core/contracts/crawler_runner.py` | Full graph state for execution | No (dataclass) |
| CrawlerControlState | `ai_core/contracts/crawler_runner.py` | Control flags (snapshot, fetch, dry_run) | No (dataclass) |
| CrawlerStateBuilder | `ai_core/services/crawler_state_builder.py` | Build graph state from request | Yes |
| CrawlerRunner | `ai_core/services/crawler_runner.py` | Coordinate crawler execution | Yes |
| crawl_url_task | `crawler/tasks.py` | Celery task for URL crawling | No (Celery task) |

Crawler Run Modes
| Mode | Fetch | Store | Process | Use Case |
|------|-------|-------|---------|----------|
| `live` | Yes | Yes | Yes | Full crawl + ingestion |
| `manual` | Yes | Yes | HITL | Requires human review |
| `store_only` | No | Yes | Yes | Store provided content |
| `fetch_only` | Yes | No | No | Fetch without storage |

Crawler Control Flags
| Flag | Type | Description |
|------|------|-------------|
| `snapshot` | bool | Enable page snapshot |
| `snapshot_label` | str | Label for snapshot |
| `fetch` | bool | Enable HTTP fetch |
| `shadow_mode` | bool | Log only, no mutations |
| `dry_run` | bool | Validate without execution |
| `review` | enum | `required`, `approved`, `rejected` |
| `force_retire` | bool | Force retire existing document |
| `recompute_delta` | bool | Recompute document delta |

Crawler → Ingestion Flow
```
CrawlerManager.dispatch_crawl_request()
    ↓
crawl_url_task (Celery)
    ↓
UniversalIngestionGraph
    ↓
IngestionRunRecord (queued → running → succeeded/failed)
```

WebSearch/WebAcquisition Infrastructure (`ai_core/tools/web_search.py`, `ai_core/graphs/web_acquisition_graph.py`)
| Component | Module | Function | DI Required (P2.3) |
|-----------|--------|----------|-------------------|
| WebSearchWorker | `tools/web_search.py` | Execute web search queries | Yes (`get_web_search_worker()`) |
| WebSearchInput | `tools/web_search.py` | Query input model | No (Pydantic) |
| WebSearchResponse | `tools/web_search.py` | Results + outcome | No (Pydantic) |
| GoogleSearchAdapter | `tools/search_adapters/google.py` | Google Custom Search API | Yes (API keys) |
| WebAcquisitionGraph | `graphs/web_acquisition_graph.py` | Bounded graph for web search | No (already bounded) |

WebSearch Contract Surface
| Method | Input Contract | Output Contract | Function |
|--------|----------------|-----------------|----------|
| `WebSearchWorker.run` | `query: str, context: ToolContext` | `WebSearchResponse` | Execute search |
| `WebAcquisitionGraph` | `WebAcquisitionGraphInput` | `WebAcquisitionGraphOutput` | Bounded search capability |

WebAcquisition Graph I/O (already bounded with io_spec)
```python
WebAcquisitionGraphInput:
  query: str
  search_config: dict | None
  preselected_results: list | None
  tool_context: ToolContext

WebAcquisitionGraphOutput:
  search_results: list[{title, url, snippet}]
  selected_result: dict | None
  decision: "acquired" | "error" | "no_results"
  error: str | None
  telemetry: dict
```

Capability Distinction: Crawler vs. WebSearch
```
┌─────────────────────────────────────────────────────────────────────────┐
│                           AGENT RUNTIME                                  │
│                                                                          │
│  ┌─────────────────────┐              ┌─────────────────────┐           │
│  │     WebSearch       │              │      Crawler        │           │
│  │     (P3.8)          │              │      (P3.7)         │           │
│  │                     │              │                     │           │
│  │  Query → Results    │──────────────│  URL → Document     │           │
│  │  (Discover URLs)    │  discovered  │  (Fetch & Ingest)   │           │
│  │                     │    URLs      │                     │           │
│  └─────────────────────┘              └─────────────────────┘           │
│           ↓                                      ↓                       │
│   WebSearchResponse                     IngestionRunRecord               │
│   (URLs + Snippets)                     (Document persisted)             │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

Guardrails (hard constraints)
- ToolContext is the only ID source at runtime: AgentRuntime.start/resume accept only ToolContext instance + typed RuntimeConfig (no dict payloads); IDs may only come from `tool_context.<id>` and `agent_state.identity.<id>`; provide two ToolContext construction paths (workbench factory and dev runner fixtures/minimal DB snapshot); implement guardrail lint/test with allowlist + explicit suppression mechanism (no blanket disables).
- No new ID categories: Allowed technical identifiers are `run_id` and `event_id`; do not add new tenant/user/document ID categories, headers, or business identifier semantics.
- Bounded capability graphs (LangGraph-only): All new capabilities are LangGraph StateGraphs with explicit `input_schema`/`output_schema` and GraphIOSpec binding; single-pass only (no internal retries, no confidence gating, no adaptive query expansion, no semantic branching beyond explicit error exits); planning/replanning/retry logic belongs only to AgentRuntime policies.
- Plan+Validate adjustments are schema bound only: defaulting, clamping, normalization, and rejection. Any strategic plan modification, retries, replan decisions, or query expansion must occur only in AgentRuntime policies, never inside bounded capabilities.
- DecisionLog + StopDecision are contractual: every run terminates via explicit StopDecision (status, reason, evidence_refs) recorded in the decision log; resume reloads AgentState + decision log and continues deterministically; do not conflate graph completion with success.
- Execution scopes are explicit in RuntimeConfig: `CASE`, `TENANT`, `SYSTEM`. CASE requires `case_id` + `workflow_id`; TENANT requires `ToolContext.scope.tenant_id` and forbids `workflow_id`; SYSTEM forbids `case_id` and permits `tenant_id` only for read-only observability correlation. Case lifecycle emissions are permitted only in CASE scope; attempts outside CASE scope must fail fast and be logged. SYSTEM scope hard-fails on any mutation in documents, cases, vector stores, crawler, and ingestion services. `execution_scope` is provided explicitly at runtime start/resume; no env/default-based inference.
- Storage separation: graph checkpoint stores graph state; AgentRunRecord stores lifecycle/status; harness artifacts store evaluation outputs (stable JSON schema); never merge these into one persisted blob.
- Cancellation semantics: early phases allow best-effort cancellation (prevent downstream execution, mark run cancelled); do not claim killing running Celery tasks until implemented.
- Harness diff is artifact-based: no text-equality requirements; enforce citation coverage, claim_to_citation completeness, ungrounded claim rate threshold, stable artifact schema, and time/token budget regression limits; baseline updates require explicit justification.
- Reset-first development: CI and local dev must be valid on a clean DB; the only allowed persistence assumption is deterministic seed fixtures.
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
- Title: Capability inventory snapshot + doc alignment matrix (code-backed)
  Intent: Produce a code-backed inventory of current capabilities/graphs/services and a doc alignment matrix that marks drift vs code and required updates.
  Files or modules to inspect/modify: `ai_core/graph/registry.py`, `ai_core/graph/bootstrap.py`, `ai_core/services/`, `ai_core/agent/capabilities/` (new), `docs/architecture/architecture-reality.md`, `roadmap/dependency_analysis.md` (or new `roadmap/doc_alignment.md`).
  Introduced components or removals: Inventory snapshot section in docs; doc alignment matrix with code pointers and fix plan.
  Acceptance criteria (testable): Inventory lists only code-registered or directly-used capabilities; each doc claim about behavior has a code pointer or is marked drift; matrix includes fix owner/phase/fix-by date.
  Cross reference to review finding: Review 3 (hidden dependencies, implicit reconstruction), Review 1 (compile-time control flow confusion).
  Notes and allowed deviations: Code is source of truth; docs only explain.

- Title: Identity and authz contract alignment (post-reset)
  Intent: Define the identity/authz contract and update docs to match code, with DB reset allowed.
  Files or modules to inspect/modify: `cases/models.py`, `cases/lifecycle.py`, `cases/authz.py`, `cases/services.py`, `users/models.py`, `profiles/models.py`, `profiles/authentication.py`, `profiles/policies.py`, `ai_core/contracts/scope.py`.
  Introduced components or removals: Contract summary for identity + authz; decision on email uniqueness and invitation behavior; explicit note that DB reset is acceptable.
  Acceptance criteria (testable): ToolContext identity rules are explicit (user_id vs service_id mutually exclusive); case_id remains in BusinessContext; case authz uses CaseMembership/roles per account_type; account_type expiry enforced at auth layer; authorization/policy tests assume a clean DB and create required actors/resources via fixtures/seed; gaps (no REST CRUD, invitation email stub) are explicitly tracked with owner/phase/fix-by date.
  Cross reference to review finding: Review 3 (hidden dependencies), Review 1 (contract confusion).
  Notes and allowed deviations: This is documentation + contract clarity only; behavior follows code; DB reset is permitted.

- Title: Contract manifest (runtime surface versions)
  Intent: Provide a single canonical manifest that versions all runtime contracts (state, decision log, harness artifacts, capabilities, flows).
  Files or modules to inspect/modify: `ai_core/agent/contracts/manifest.yaml` (new), `docs/development/` (short note).
  Introduced components or removals: Contract manifest with schema/version map; CI gate for version bumps on schema changes.
  Acceptance criteria (testable): Manifest includes `agent_state_schema_version`, `decision_log_schema_version`, `harness_artifact_schema_version`, `tool_context_hash_version`, `capability_iospec_versions` (map), and `flow_versions` (map); CI gate requires a version bump and migration note when any schema changes.
  Cross reference to review finding: Review 3 (determinism/traceability), Review 1 (contract drift).
  Notes and allowed deviations: Use semver; no new ID categories. DB reset is an accepted operational assumption during migration: no schema backfills or data migrations required; only deterministic seed scripts are supported.
  Example manifest shape (illustrative):
  ```yaml
  agent_state_schema_version: "1.0.0"
  decision_log_schema_version: "1.0.0"
  harness_artifact_schema_version: "0.1.0"
  tool_context_hash_version: "1.0.0"
  capability_iospec_versions:
    rag.retrieve: "0.1.0"
    rag.compose: "0.1.0"
    web.search: "0.1.0"
  flow_versions:
    rag.query: "0.1.0"
    collection.search: "0.1.0"
  ```

- Title: Harness artifact schema v0 (early freeze)
  Intent: Define and validate a minimal harness artifact schema before flows exist to enforce harness-first development.
  Files or modules to inspect/modify: `ai_core/agent/harness/schema.py` (new) or `ai_core/rag/quality/`, `ai_core/tests/` (schema validator), `scripts/` (baseline policy hook).
  Introduced components or removals: Artifact schema v0 + validator; baseline update policy activated.
  Acceptance criteria (testable): Schema v0 validates artifacts with required fields (`run_id`, `inputs_hash`, `decision_log`, `stop_decision`) and optional fields for retrieval/answer/citations; baseline update policy is enforced in tests/CI even before runtime flows are implemented.
  Cross reference to review finding: Review 1 (success != completion), Review 3 (determinism).
  Notes and allowed deviations: Optional fields may be empty/absent until capabilities land.

- Title: Flow contracts (per-flow I/O + required capabilities)
  Intent: Define explicit contracts per flow to keep runtime orchestration modular and testable.
  Files or modules to inspect/modify: `ai_core/agent/flows/<flow_name>/contract.py` (new), `ai_core/agent/flows/registry.py` (new or existing).
  Introduced components or removals: Flow contract models (Input/Output), `required_capabilities`, `supported_scopes`, and flow versioning.
  Acceptance criteria (testable): Every flow has a contract file with Input/Output models, `required_capabilities`, `supported_scopes`, and a version; CI enforces version bumps on contract changes.
  Cross reference to review finding: Review 1 (compile-time control flow), Review 3 (determinism/traceability).
  Notes and allowed deviations: Flow contracts must remain ID-free; ToolContext remains the only ID source.
  Example flow contract skeleton (illustrative):
  ```python
  # ai_core/agent/flows/rag_query/contract.py
  class RagQueryInput(BaseModel):
      question: str
      retrieval_plan: RetrievalPlan | None = None

  class RagQueryOutput(BaseModel):
      answer: str
      citations: list[Citation]
      claim_to_citation: dict[str, list[str]]

  flow_version = "0.1.0"
  required_capabilities = ["rag.retrieve", "rag.compose", "rag.evidence"]
  supported_scopes = {"CASE", "TENANT"}
  ```

- Title: AgentState, AgentRunRecord, and decision log contracts
  Intent: Introduce typed state and run record models for agentic flows, including decision log entries (plan/replan/stop/HITL) with deterministic serialization.
  Files or modules to inspect/modify: `ai_core/contracts/plans/`, `ai_core/agent/state.py` (new), `ai_core/agent/run_records.py` (new), `ai_core/graph/state.py`, `ai_core/graph/core.py`.
  Introduced components or removals: New AgentState/AgentRunRecord models and decision log schema; no changes to existing ToolContext schema.
  Acceptance criteria (testable): AgentState and AgentRunRecord round-trip with JSON serialization; StopDecision is stored once as `AgentRunRecord.terminal_decision`; DecisionLog emits a `stop` event that references/embeds the same structure and a validator enforces equality; decision log entries persist with deterministic ordering; tests confirm no implicit dict merges.
  Cross reference to review finding: Review 1 (no AgentState, success conflated with graph completion), Review 3 (deterministic state shape).
  Notes and allowed deviations: Reuse existing plan/evidence contracts; do not introduce new IDs or headers without explicit confirmation.

- Title: AgentRuntime lifecycle and run handles
  Intent: Implement runtime lifecycle APIs (start, resume, cancel, status) and explicit run handles that decouple task success from graph completion.
  Files or modules to inspect/modify: `ai_core/agent/runtime.py` (new), `ai_core/graph/core.py`, `ai_core/tasks/graph_tasks.py`, `llm_worker/tasks.py`.
  Introduced components or removals: AgentRuntime with run lifecycle and run handle identifiers; runtime status storage separate from graph results.
  Acceptance criteria (testable): A dummy capability can be executed, paused, resumed, and cancelled (best-effort cancellation); runtime flow selection goes through a Flow Registry with named flows and typed I/O; run status persists independently of capability output; introduce a StopDecision record (status, reason, evidence_refs) as part of AgentRunRecord; every runtime flow terminates via explicit StopDecision (not by graph completion); tests assert status transitions and StopDecision presence.
  Cross reference to review finding: Review 1 (compile-time control flow, async tracking/cancel, success equals completion).
  Notes and allowed deviations: Keep legacy registry graph keys sourced from `ai_core/graph/registry.py`; no new legacy registry graph keys. Capability identifiers are separate and live under `ai_core/agent/capabilities/*`.

Phase 2: Context and Determinism Hardening

Purpose
Harden AgentRuntime start/resume and eliminate implicit context reconstruction so runtime artifacts are deterministic and auditable.

Work Items
- Title: Observability contract for runtime vs capability (decision log + spans)
  Intent: Define the minimal observability contract for AgentRuntime and bounded capabilities (decision log fields, StopDecision logging, required span attributes).
  Files or modules to inspect/modify: `ai_core/infra/observability.py`, `ai_core/agent/runtime.py`, `ai_core/agent/run_records.py`, `docs/development/` (new short note or existing).
  Introduced components or removals: Observability checklist embedded in roadmap or docs; no behavioral changes yet.
  Acceptance criteria (testable): Contract enumerates required decision log fields and span attributes; clarifies where `run_id`/`event_id` are emitted; aligns with guardrails (no new ID categories); `tool_context_hash` is derived only from ToolContext `canonical_json()` allowlist fields and must not depend on observability-only fields (`ts`, `event_id`, trace/span IDs).
  Decision log entries include `run_id`, `event_id`, `ts` (timestamp), `kind`, `status`, `reason`, `tool_context_hash`, and `evidence_refs`. Spans include `run_id`, `capability_name`, `graph_name`, plus scope-appropriate IDs (`tenant_id` for TENANT/CASE; `case_id` + `workflow_id` for CASE).
  Minimal required fields (DecisionLog event):
    - `run_id`
    - `event_id` (technical, unique)
    - `ts` (timestamp, ISO-8601 or epoch)
    - `kind` (aka `event_type`, e.g., `plan`, `capability_start`, `capability_end`, `stop`)
    - `status` (e.g., `started`, `succeeded`, `failed`, `cancelled`)
    - `reason` (required for `stop`)
    - `tool_context_hash` (derived, stable hash of ToolContext)
    - `evidence_refs` (required for `stop`, may be empty list)
  Required when applicable:
    - `capability_name` (when event relates to a capability)
    - `graph_name` (when a graph executes inside a capability)
    - `flow_name`
    - `trace_id`
    - `tenant_id`, `case_id`, `workflow_id` (from ToolContext; required for non-system runs)
  Minimal required span attributes (AgentRuntime + capabilities):
    - `run_id`
    - `tenant.id`, `case.id`, `workflow.id` (non-system runs)
    - `capability.name` (capability spans only)
    - `graph.name` (when a graph is involved)
    - `status`
  Recommended for correlation:
    - `event_id`, `trace_id`, `flow.name`
  Execution scope visibility:
    - `execution_scope` is recorded on decision log entries and spans
    - `tenant_id` appears on spans for all TENANT/CASE runs; `case_id` and `workflow_id` appear only for CASE runs
  Cross reference to review finding: Review 1 (async tracking), Review 3 (deterministic replay/observability).
  Notes and allowed deviations: Keep contract minimal; enforce later in runtime/harness tests.

- Title: Entrypoint quarantine for direct Graph.run calls (legacy paths)
  Intent: Make direct GraphRunner.run/invoke usage outside the registry explicit, logged, and prevented from spreading.
  Files or modules to inspect/modify: `ai_core/services/rag_query.py`, `ai_core/services/framework_analysis.py`, `ai_core/graphs/technical/*`, `ai_core/graphs/business/*`, `ai_core/graph/core.py` (logging hook).
  Introduced components or removals: DecisionLog marker for direct graph calls; lint/AST rule or explicit inventory list of direct entrypoints.
  Acceptance criteria (testable): Every direct GraphRunner.run/invoke outside the registry emits a DecisionLog event with `legacy_direct_call=true` and `entrypoint_name`; lint/AST rule fails on new direct graph.run usage under `ai_core/services/**` unless allowlisted; maintain a small inventory list of direct entrypoints (file paths) to prevent silent additions; allowlist entries must include owner + `removal_by` date, and CI warns (or fails) when the date is past due.
  Cross reference to review finding: Review 1 (control flow centralization), Review 3 (determinism).
  Notes and allowed deviations: Temporary guard until Phase 5 routes services through AgentRuntime; no new legacy entrypoints without explicit allowlist.

- Title: AgentRuntime entrypoint hardening (start/resume) and strict ToolContext origin
  Intent: Harden AgentRuntime.start/resume to accept only ToolContext plus minimal runtime config, and build AgentEnvelope inside the runtime with no implicit ID reconstruction.
  Files or modules to inspect/modify: `ai_core/agent/runtime.py` (new), `ai_core/agent/envelope.py` (new), `ai_core/tool_contracts/base.py`, `ai_core/graph/schemas.py`, `ai_core/agent/runtime_config.py` (new).
  Introduced components or removals: AgentEnvelope model and parser; remove getattr and nested dict fallbacks when constructing ToolContext for runtime flows.
  Acceptance criteria (testable): AgentRuntime.start/resume reject runs without ToolContext instance; RuntimeConfig is typed and contains no IDs; ToolContext exposes `canonical_json()` with deterministic ordering and a stable hash (`sha256(canonical_json)`); `canonical_json()` is built from an explicit allowlist of stable fields (documented as `tool_context_hash_version`) and excludes volatile/observability fields (headers, user agent, locale, timestamps, trace_id, invocation_id, request/session ids, auth scheme, ip); any new ToolContext field is excluded from `canonical_json()` until explicitly added to the allowlist and `tool_context_hash_version` is bumped; tests confirm semantically equivalent ToolContexts produce the same hash (including different headers/locale/etc.) and that `tool_context_hash` does not vary with decision-log fields; no node, service, or task reads IDs from dicts (only ToolContext and AgentState are sources); add a guardrail lint/test with allowlist + explicit suppression mechanism that fails on patterns like `state.get("tenant")`, `payload["tenant_id"]`, `getattr(x, "tenant_id")` in parsing paths, or `context["scope"]["tenant_id"]`, while allowlisting `tool_context.<id>` and `agent_state.identity.<id>`; guardrail lint also forbids `.dict()`/`.model_dump()` on ToolContext/Scope/Identity as an ID source (except explicit allowlisted serialization for API output), forbids `kwargs.get("tenant_id")`/payload merges for IDs, and forbids `os.environ` access for IDs in runtime/capability code; IDs do not leak across concurrent runs.
  Cross reference to review finding: Review 3 (ID leaks, implicit context reconstruction, hidden dependencies).
  Notes and allowed deviations: ToolContext construction paths are limited to the workbench context factory (`theme/helpers/context.py::prepare_workbench_context`) and dev runner fixtures/minimal DB snapshot; runtime accepts ToolContext plus typed RuntimeConfig only; do not add new headers/meta keys without explicit confirmation.

- Title: Execution scope enforcement + fixed workflow taxonomy (dev/migration policy)
  Intent: Enforce explicit execution scopes (CASE/TENANT/SYSTEM) and a controlled workflow_id set for case-scoped runs during development and migration.
  Files or modules to inspect/modify: `theme/helpers/context.py::prepare_workbench_context`, dev runner fixtures under `scripts/` or `ai_core/tests/fixtures/`, `ai_core/agent/runtime.py` (validation only), ToolContext guardrail lint/tests.
  Introduced components or removals: RuntimeConfig `execution_scope` validation; canonical dev case fixture (e.g., `DEV_CASE_DEFAULT`) for CASE runs; workflow taxonomy for CASE scope only.
  Acceptance criteria (testable):
    - CASE scope without `case_id` or `workflow_id` fails fast.
    - TENANT scope requires `ToolContext.scope.tenant_id` and fails fast if `workflow_id` is provided; CASE lifecycle emissions are forbidden and hard-fail.
    - SYSTEM scope with `case_id` fails fast; SYSTEM scope hard-fails on any call to `DocumentsRepository.upsert`, `DocumentsRepository.delete`, `DocumentsRepository.add_asset`, `DocumentsRepository.delete_asset`, `CaseEvents.emit`, `CaseLifecycle.apply`, `IngestionRunRecord.record_*`, vector store writes, or ingestion/crawler dispatch.
    - Any attempt to emit CaseEvent, apply lifecycle definitions, write case-linked ingestion run records, or run collection-search lifecycle transitions outside CASE scope hard-fails and is logged as a policy violation.
    - All mutating services call a central `ScopePolicy.guard_mutation(action, tool_context, runtime_config, details)`; tests cover guard behavior plus representative call sites.
    - Maintain a mutation-sink inventory (repo methods, ORM save/delete, vector upserts, Celery dispatch, external connector writes).
    - CI/test gate: every mutation sink either calls `guard_mutation` or is explicitly allowlisted with rationale; SYSTEM-scope tests assert zero mutation-sink calls via call-tracking (not only that the guard exists).
    - workflow_id is restricted to `{INITIAL, ONGOING, END}` for CASE scope unless explicitly suppressed.
    - Suppression is explicit and scoped: per test or per capability module via an allowlist (e.g., `@allow_workflow_id("MIGRATION")` or test-only config); never via a global disable or runtime/env flag.
    - Transition-only allowlist value `LEGACY` is permitted solely for legacy entrypoints; any `LEGACY` usage must surface as a policy-violation in the harness and carry an explicit removal date in config.
    - All CaseEvent emissions go through a single gateway (e.g., `cases/services.py::emit_case_event(...)`), which enforces `execution_scope == CASE` and `case_id` presence; direct CaseEvent ORM writes are forbidden and flagged by lint/guardrail.
    - logs for CASE flows include `tenant_id`, `case_id`, `workflow_id`, `run_id`, `execution_scope`.
    - dev runner injects the dev case by default for CASE scope unless explicitly overridden.
    - no worker or graph fabricates or mutates `workflow_id`.
  Cross reference to review finding: Review 3 (ID leaks, implicit context reconstruction), Review 1 (non-deterministic observability).
  Notes and allowed deviations: This is a development and migration constraint; taxonomy expansion or relaxation requires an explicit roadmap item; this is validation on ToolContext usage and does not change ToolContext schema.

- Title: Dependency injection for routers and caches (read-path only)
  Intent: Remove global mutable routers/workers from read paths by injecting dependencies through runtime capability descriptors.
  Files or modules to inspect/modify: `ai_core/rag/vector_store.py`, `ai_core/rag/vector_client.py`, `ai_core/rag/semantic_cache.py`, `ai_core/rag/strategy.py`, `ai_core/tools/shared_workers.py`.
  Introduced components or removals: Provider/factory interfaces owned by AgentRuntime; remove module-level singleton mutation for routing (read path).
  Acceptance criteria (testable): Read-path access no longer uses module-level singletons; deterministic construction verified in tests; immutable caches are scoped per runtime instance; lint/AST rule forbids `get_default_*`, `GLOBAL_*`, or `module_level_cache` usage in `ai_core/agent/**` and `ai_core/agent/capabilities/**`; AgentRuntime provides a single DependencyProvider/container interface and capabilities receive dependencies only from this provider (no direct construction inside capabilities).
  Cross reference to review finding: Review 1 (global mutable routers), Review 3 (hidden dependencies).
  Notes and allowed deviations: Write-path isolation handled in the next work item; allow immutable LRU caches if scoped per runtime instance. HybridSearchAndScore remains legacy/worker-internal and must not be depended on by AgentRuntime capabilities in Phase 2.

- Title: Dependency injection for routers and caches (write-path isolation + concurrency tests)
  Intent: Isolate write paths and add concurrency tests to guarantee tenant separation under load.
  Files or modules to inspect/modify: `ai_core/rag/vector_store.py`, `ai_core/rag/vector_client.py`, `ai_core/rag/semantic_cache.py`, `ai_core/rag/strategy.py`, `ai_core/tools/shared_workers.py`, `ai_core/tests/` (new concurrency tests).
  Introduced components or removals: Write-path DI boundaries + concurrency test harness.
  Acceptance criteria (testable): Parallel runs for different tenants do not share mutable router or cache state under concurrent writes; concurrency tests cover write-path isolation and assert a tenant-specific marker (for example, router instance id) is recorded in spans or DecisionLog per run; HybridSearchAndScore singleton removal remains a Phase 3 worker item.
  Cross reference to review finding: Review 1 (global mutable routers), Review 3 (hidden dependencies).
  Notes and allowed deviations: Keep tests deterministic; avoid flaky timing-based assertions.

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
A bounded capability graph must be a LangGraph StateGraph with explicit `input_schema`/`output_schema` and GraphIOSpec binding. It must not contain internal retry loops, dynamic routing decisions, confidence-based branching, or adaptive query expansion. Conditional edges are permitted only for explicit error handling. Plan+Validate “adjustments” are limited to deterministic schema-bound normalization (clamp/default/reject/normalize) and must not perform strategic plan changes.

Work Items
- Title: Capability registry bridge + io_spec enforcement
  Intent: Define capability metadata and enforce io_spec binding for bounded capabilities through the registry bridge.
  Files or modules to inspect/modify: `ai_core/agent/capabilities.py` (new), `ai_core/graph/io.py`, `ai_core/graph/registry.py`, `ai_core/graph/bootstrap.py`.
  Introduced components or removals: Capability registry that references existing graph names; legacy graphs tagged as deprecated and excluded from default runtime resolution.
  Acceptance criteria (testable): Runtime resolves capabilities by name; each bounded capability registers io_spec; capability registry emits a machine-readable map of `capability_name -> (graph_registry_key | python_entrypoint)` and versions it in a manifest; legacy graphs are not modified beyond deprecation markers.
  Cross reference to review finding: Review 2 (capability-ready graphs), Review 1 (agentic behavior hidden in graphs).
  Notes and allowed deviations: No new legacy registry graph keys; keep registry as the single source of truth. Capability names are not graph registry keys; capabilities may wrap existing graphs or define new bounded graphs under the capability namespace.

- Title: Worker capability alignment (ScoreResults + HybridSearchAndScore)
  Intent: Treat ScoreResults as a bounded capability and remove singleton reliance in HybridSearchAndScore.
  Files or modules to inspect/modify: `llm_worker/graphs/score_results.py`, `llm_worker/graphs/hybrid_search_and_score.py`, `llm_worker/tasks.py`.
  Introduced components or removals: Bounded wrapper + io_spec for ScoreResults; decision on refactor/replace for HybridSearchAndScore.
  Acceptance criteria (testable): ScoreResults exposes a bounded io_spec contract; HybridSearchAndScore no longer relies on module-level singleton or is replaced; worker tasks consume ToolContext only.
  Cross reference to review finding: Review 1 (global mutable state), Review 3 (hidden dependencies).
  Notes and allowed deviations: Keep worker graph usage internal; do not expand public API surface.

- Title: Bounded capability enforcement (linter + decorator)
  Intent: Mechanically enforce bounded-capability rules to prevent retries, policy knobs, or semantic branching inside capabilities.
  Files or modules to inspect/modify: `ai_core/agent/capabilities/` (decorator/marker), `ai_core/tests/` (bounded linter tests), `scripts/` (optional lint entrypoint).
  Introduced components or removals: `@bounded_capability` marker; bounded linter that scans capability code for retry/attempt/backoff loops or policy knobs; tests for conditional edges restricted to error exits.
  Acceptance criteria (testable): Every capability is annotated with `@bounded_capability`; linter fails on retry loops or confidence/policy knob usage inside capabilities; import blocklist is enforced (for example `tenacity`, `backoff`) or an explicit allowlist is used; blocked function-name patterns (retry, replan, confidence, expand, fallback) are flagged within capability modules; linter disallows calls to known retry helpers and disallows loops/recursion that re-invoke the capability's main node; LangGraph edge check ensures conditional edges only go to error_exit/fail_exit nodes, disallows back-edges to mainline, and enforces "no edge may target any node already executed in the same run" (except explicit error_exit); running any bounded capability in SYSTEM scope must either (a) fail fast with policy violation, or (b) be read-only with zero mutations (enforced via ScopePolicy call tracking/mocks); CI runs the bounded linter.
  Cross reference to review finding: Review 1 (agentic behavior in graphs), Review 2 (agentic smell), Review 3 (determinism).
  Notes and allowed deviations: Linter can be heuristic (AST-based); keep false positives low with an explicit allowlist if needed.

- Title: Core plan contracts (Retrieval/Compose/WebSearch/DocAcquisition/Ingestion/DocumentStorage)
  Intent: Define the minimal set of plan models used by AgentRuntime policies to drive atomic capabilities.
  Files or modules to inspect/modify: `ai_core/agent/plans/` (new), `ai_core/contracts/plans/`, `ai_core/agent/runtime.py`.
  Introduced components or removals: Plan models with stable io_spec-facing fields (no new ID categories).
  Acceptance criteria (testable): Plans validate and serialize deterministically; plans contain intent/hints only and no IDs; RuntimeConfig and flow inputs are ID-free; RetrievalPlan includes intent values such as `COLLECTION_BROWSE` and `RAG_QUERY`; plan inputs are referenced by capabilities via io_spec; capability outputs may include evidence/document refs but must not include plan mutation fields (for example `plan_changed` or `strategy_selected`) or query-expansion directives.
  Cross reference to review finding: Review 1 (compile-time control flow), Review 3 (determinism).
  Notes and allowed deviations: Plans must not embed tenant/user identifiers; ToolContext remains the only ID source.

- Title: RAG atomic capabilities (Retrieval, Compose, EvidenceMapping)
  Intent: Build bounded LangGraph StateGraph capabilities for retrieval, answer composition, and evidence mapping.
  Files or modules to inspect/modify: `ai_core/agent/capabilities/rag_retrieve.py` (new), `ai_core/agent/capabilities/rag_compose.py` (new), `ai_core/agent/capabilities/rag_evidence.py` (new), `ai_core/graph/io.py`.
  Introduced components or removals: Atomic RAG capabilities with explicit io_spec; legacy RAG graphs remain frozen.
  Acceptance criteria (testable): Bounded definition enforced (no retries, no confidence gating, no adaptive expansion); io_spec validation passes; outputs include candidates/citations, answer draft, and claim_to_citation map.
  Cross reference to review finding: Review 2 (agentic smell in RAG graph), Review 1 (replanning/stop semantics).
  Notes and allowed deviations: Preserve model routing via labels from `MODEL_ROUTING.yaml`; do not refactor legacy RAG graphs.

- Title: Acquisition + ingestion capabilities (WebSearch/DocAcquisition/Crawler/Ingestion/DocumentStorage)
  Intent: Build bounded capabilities for acquisition and ingestion using the Plan+Validate pattern and guardrail validation.
  Files or modules to inspect/modify: `ai_core/agent/capabilities/web_search.py` (new), `ai_core/agent/capabilities/doc_acquisition.py` (new), `ai_core/agent/capabilities/crawler.py` (new), `ai_core/agent/capabilities/ingestion.py` (new), `ai_core/agent/capabilities/document_storage.py` (new), `ai_core/tools/web_search.py`, `crawler/manager.py`, `documents/repository.py`, `ai_core/rag/guardrails.py`.
  Introduced components or removals: Bounded acquisition/ingestion capabilities with applied config outputs and adjustment logs.
  Acceptance criteria (testable): Plan+Validate pattern enforced; adjustments are deterministic schema-bound normalization only (clamp/default/reject/normalize), never strategic replans; guardrail adjustments are recorded; outputs include required `adjustments: list[AdjustmentLogEntry]` where `AdjustmentLogEntry = {field_path, action(defaulted|clamped|normalized|rejected), before, after, reason, policy_ref}`; bounded definition enforced (technical error exits only); outputs include applied config + adjustments; no IDs are derived from dicts.
  Cross reference to review finding: Review 1 (global mutable state, control flow), Review 3 (hidden dependencies, determinism).
  Notes and allowed deviations: Hints are advisory; strategic retry/replan decisions remain in AgentRuntime; async dispatch returns task IDs immediately.

Phase 4: AgentRuntime Flows + Dev Runner + Golden Harness

Purpose
Rebuild agentic behavior in AgentRuntime using bounded capabilities, with a dev runner and harness as the primary validation surface.

Phase 4 dependency mapping (capabilities → flows)
- Collection search flow depends on Phase 3:
  - Core plan contracts
  - RetrievalCapability (`ai_core/agent/capabilities/rag_retrieve.py`) with intent `COLLECTION_BROWSE`
  - WebSearchCapability (`ai_core/agent/capabilities/web_search.py`) for discovery
  - CrawlerCapability (`ai_core/agent/capabilities/crawler.py`) for fetch/ingest
  - IngestionCapability (`ai_core/agent/capabilities/ingestion.py`)
  - DocumentStorageCapability (`ai_core/agent/capabilities/document_storage.py`) when persisting
- RAG flow depends on Phase 3:
  - Core plan contracts
  - RetrievalCapability (`ai_core/agent/capabilities/rag_retrieve.py`)
  - ComposeAnswerCapability (`ai_core/agent/capabilities/rag_compose.py`)
  - EvidenceMappingCapability (`ai_core/agent/capabilities/rag_evidence.py`)
- Framework analysis flow depends on Phase 3:
  - Capability registry bridge + io_spec enforcement
  - (Optional) DocAcquisitionCapability (`ai_core/agent/capabilities/doc_acquisition.py`) if sourcing from external systems
- Dev runner + harness depend on Phase 3:
  - Core plan contracts
  - Capability registry bridge + io_spec enforcement
  - RAG atomic capabilities for artifact validation

Work Items
- Title: Test migration plan (legacy freeze vs rewrite) + harness CI gate
  Intent: Classify legacy tests as freeze/remove/rewrite and define CI gates based on harness artifacts rather than UI parity.
  Files or modules to inspect/modify: `ai_core/tests/`, `theme/tests/`, `roadmap/dependency_analysis.md` (or new `roadmap/test_migration_plan.md`), CI config if documented.
  Introduced components or removals: Test migration matrix; harness-based CI gate definition.
  Acceptance criteria (testable): Each legacy test suite is marked freeze/remove/rewrite; new runtime tests are enumerated; CI gate references harness artifact diffs and thresholds.
  Cross reference to review finding: Review 1 (success != completion), Review 3 (determinism).
  Notes and allowed deviations: Keep legacy tests until harness gate is green; no UI parity requirement.

- Title: Minimal dev runner for AgentRuntime flows
  Intent: Provide a minimal CLI runner that executes AgentRuntime flows without UI dependencies and supports resume inputs for simulated HITL.
  Files or modules to inspect/modify: `scripts/agent_run.py` (new), `ai_core/agent/runtime.py`, `ai_core/agent/state.py`, `ai_core/agent/runtime_config.py` (new).
  Introduced components or removals: Dev runner entrypoint that accepts ToolContext and runtime config, supports resume inputs, and emits run artifacts.
  Acceptance criteria (testable): `scripts/agent_run.py` can start and resume a run; HITL can be simulated via resume inputs; ToolContext construction path is fixtures/minimal DB snapshot (non-UI) with explicit inputs (no implicit reconstruction, no nested dict merges, no magical DB lookups); dev runner bootstraps a clean DB by running idempotent seed scripts that create default fixtures after reset (`DEV_TENANT_DEFAULT`, `DEV_USER_ADMIN`, `DEV_CASE_DEFAULT`, and worker service_id or equivalent); no reliance on pre-existing data; dev case is injected by default unless overridden; run artifacts are emitted with stable structure.
  Cross reference to review finding: Review 1 (HITL blocking, async tracking), Review 3 (determinism).
  Notes and allowed deviations: No UI requirement; the runner is the primary integration point until optional clients are wired.

- Title: Golden query set and diff harness for runtime artifacts
  Intent: Create a minimal golden dataset and a diff tool that can run both legacy and runtime paths for RAG chat and retrieval.
  Files or modules to inspect/modify: `ai_core/rag/quality/` (reuse patterns), `ai_core/tests/` (new eval tests), `scripts/` (new harness entrypoint).
  Introduced components or removals: Golden query fixtures and diff report output (text or JSON); no production impact.
  Acceptance criteria (testable): Harness runs in CI/test mode and can target both legacy and AgentRuntime executors; each run emits a JSON artifact with a stable output contract (`run_id`, inputs hash, retrieval candidates list with chunk ids and scores, citations list, final answer text, claim_to_citation map, decision log events); diffing compares these artifacts and enforces criteria for citation coverage (equal or better), claim_to_citation completeness, ungrounded claim rate below threshold, stable artifact schema, and time/token budget regression limits; golden harness baselines are versioned and frozen per migration step, and baseline changes require an explicit justification note in harness config; after DB reset, baselines may be regenerated but still require explicit justification (no silent updates).
  Cross reference to review finding: Review 1 (success != completion), Review 3 (determinism breaks).
  Notes and allowed deviations: Keep harness minimal and dev-only; no new public API surfaces.

- Title: Collection search runtime flow (planning, HITL, ingestion)
  Intent: Move planning, HITL, and auto-ingestion to AgentRuntime using bounded retrieval/search capability plus UniversalIngestion capability.
  Files or modules to inspect/modify: `ai_core/agent/runtime.py`, `ai_core/agent/state.py`, `ai_core/services/collection_search/`, `ai_core/graphs/technical/universal_ingestion_graph.py`.
  Introduced components or removals: New runtime flow and AgentState; legacy collection_search orchestration graph removed from default execution path.
  Acceptance criteria (testable): Runtime flow supports pause/resume for HITL, supports best-effort cancellation, and triggers ingestion via capability; decision log and run artifacts record each transition; StopDecision is recorded at termination; flow uses Phase 3 Core plan contracts + Acquisition/Ingestion capabilities + unified RetrievalCapability (intent `COLLECTION_BROWSE`) only (no legacy graph calls).
  Cross reference to review finding: Review 1 (HITL blocking, async tracking), Review 2 (collection search agentic smell).
  Notes and allowed deviations: Keep ingestion thresholds/config in runtime policy, not graph constants.

- Title: RAG runtime flow (planning, retries, confidence gating)
  Intent: Implement RAG planning, retries, and gating in runtime; capabilities only retrieve and compose answers.
  Files or modules to inspect/modify: `ai_core/agent/runtime.py`, `ai_core/agent/state.py`, `ai_core/services/rag_query.py`, `ai_core/rag/strategy.py`.
  Introduced components or removals: Runtime RAG flow; legacy retrieval_augmented_generation graph remains available only behind legacy toggle.
  Acceptance criteria (testable): Runtime flow produces deterministic results for fixed plan; retries and replan decisions are logged in decision log; StopDecision recorded at termination; flow uses Phase 3 Core plan contracts + RAG atomic capabilities only (RetrievalCapability intent `RAG_QUERY`); harness artifacts validate citation coverage, claim_to_citation completeness, and ungrounded claim thresholds.
  Cross reference to review finding: Review 1 (replanning blocked, success conflation), Review 2 (RAG graph agentic smell), Review 3 (determinism).
  Notes and allowed deviations: Preserve existing retrieval filters and model routing labels.

- Title: Framework analysis runtime flow
  Intent: Orchestrate framework analysis steps via runtime-owned control flow rather than graph compile-time edges.
  Files or modules to inspect/modify: `ai_core/agent/runtime.py`, `ai_core/services/framework_analysis_*` (new), `ai_core/graphs/business/framework_analysis_graph.py`.
  Introduced components or removals: Runtime flow with bounded sub-capabilities; legacy framework analysis graph deprecated.
  Acceptance criteria (testable): Runtime can reorder/skip steps based on AgentState; best-effort cancellation stops downstream execution; decision log captures branch decisions and timeouts; StopDecision recorded at termination; flow uses Phase 3 capability registry + bounded capabilities only (legacy graph remains frozen).
  Cross reference to review finding: Review 1 (compile-time control flow), Review 2 (framework analysis agentic smell).
  Notes and allowed deviations: Keep existing output schema for compatibility until optional clients are wired.

Phase 5: Optional Clients and HITL Integration (Thin Adapter Only)

Purpose
Wire optional clients (including the workbench) via a thin adapter, only when needed for HITL interaction.

Work Items
- Title: Deprecate legacy rag-tools workbench and introduce new dev workbench URL
  Intent: Deprecate the existing `/rag-tools/` workbench and add a new dev-only workbench under a new URL for AgentRuntime flows.
  Files or modules to inspect/modify: `theme/urls.py`, `theme/views_*`, `docs/development/rag-tools-workbench.md`.
  Introduced components or removals: Legacy rag-tools endpoints marked deprecated; new dev workbench routes are optional clients; no parity requirement.
  Acceptance criteria (testable): Old rag-tools routes are clearly marked deprecated; new dev workbench routes target AgentRuntime flows via thin adapters; no workbench parity criteria are used for acceptance.
  Cross reference to review finding: Review 1 (control flow centralized), Review 3 (context determinism).
  Notes and allowed deviations: Workbench remains optional; prioritize dev runner and harness as primary validation surfaces.

- Title: Theme to runtime adapter (thin, optional)
  Intent: Provide a minimal adapter that calls AgentRuntime.start/resume using ToolContext provided by the workbench context factory, without broad view refactors.
  Files or modules to inspect/modify: `theme/helpers/runtime_adapter.py` (new), `ai_core/agent/runtime.py`.
  Introduced components or removals: Adapter wrapper for workbench calls; no broad changes to `theme/views_*`.
  Acceptance criteria (testable): Adapter can be invoked from tests with a ToolContext and typed RuntimeConfig; workbench changes are limited to thin adapter wiring when needed; no workbench parity criteria.
  Cross reference to review finding: Review 1 (control flow centralized), Review 3 (context determinism).
  Notes and allowed deviations: Workbench remains optional; do not use it as a contract anchor.

- Title: RagQueryService routing through runtime flow (legacy behind toggle)
  Intent: Route `ai_core/services/rag_query.py` through the runtime flow, keeping the legacy path behind an explicit toggle and adding an explicit compare mode.
  Files or modules to inspect/modify: `ai_core/services/rag_query.py`, `ai_core/agent/runtime.py`.
  Introduced components or removals: Runtime entrypoint for RAG queries; legacy execution remains available only via toggle for comparison.
  Acceptance criteria (testable): RagQueryService supports `mode="runtime"|"legacy"|"compare"`; compare mode emits two artifacts plus a diff summary (time/token/citation metrics); runtime mode produces runtime artifacts (AgentState, decision log, harness JSON) and respects the legacy toggle; legacy toggle must include a `removal_by` date in config and be surfaced in logs; uses typed RuntimeConfig with no IDs; no UI parity requirement.
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

Review checklist (vibe-coding risk gates)
- Hash determinism: Is `canonical_json()` an explicit allowlist, and do tests show different headers/user-agent/locale produce the same hash?
- DI real (not cosmetic): Any `get_default_*()` usage in `ai_core/agent/**` or `ai_core/agent/capabilities/**`? Any global mutable router/cache shared across tenants?
- Bounded linter strict: Import blocklist enforced (tenacity/backoff)? Graph edge check prevents back-edges? Function-name blocklist catches retry/replan/confidence/expand/fallback helpers?
- ScopePolicy coverage: Is there a mutation-sink inventory? Do SYSTEM-scope tests assert zero mutation-sink calls (not just guard presence)?
CI gates (make the checklist executable)
- `check_tool_context_hash_determinism`
- `check_no_get_default_in_agent_layer`
- `check_bounded_capabilities_lint`
- `check_scopepolicy_mutation_sinks_covered`

Appendix: First 8 PR slices (suggested)
- Contract manifest file + validator test
- Harness schema v0 + validator test
- DecisionLog schema + enum validator
- ToolContext canonical_json + hash tests
- Flow registry skeleton + one dummy flow contract
- AgentState/RunRecord models + roundtrip tests
- ScopePolicy guard + 2 representative call sites
- Minimal dev runner that executes a dummy flow and emits an artifact
