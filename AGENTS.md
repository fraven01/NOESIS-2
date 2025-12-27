# NOESIS-2 - LLM Entry Contract (`AGENTS.md`)

## Source of truth & documentation hierarchy

- Source of truth is code (Python/TypeScript). Documentation explains and references code; it does not define behavior.
- When documentation and code diverge, follow code and update documentation to match.
- For any file change, follow the most specific `AGENTS.md` in the directory tree (deepest one wins).

## Agent stop conditions (ask before changing)

If a change would introduce new runtime semantics (new identifiers, new contracts, or boundary breaks), STOP and ask for confirmation (and/or add a backlog item in `roadmap/backlog.md` with code pointers).

- New or changed IDs / meta keys / headers: anything affecting `ScopeContext` (`ai_core/contracts/scope.py`), `ToolContext` (`ai_core/tool_contracts/base.py`), graph meta (`ai_core/graph/schemas.py:normalize_meta`), or header constants (`common/constants.py`).
- Contract changes: Pydantic input/output models or JSON schema shape for tools/graphs (e.g. `ai_core/tool_contracts/base.py`, `ai_core/graph/schemas.py`, `ai_core/nodes/`, `ai_core/tools/`).
- Architecture boundary deviations: examples include UI/views triggering technical graphs outside a dev/workbench exception, business-heavy graphs writing via ORM instead of a service boundary, or violating intended import direction (business -> technical allowed; reverse forbidden per roadmap).

Proceed without asking only when the change stays within existing enforced contracts (bugfixes, refactors, tests, doc alignment that does not introduce new runtime fields/semantics).

## Working agreements (pre-MVP)

- Stop-Conditions are binding: new IDs, new meta keys, contract changes, or boundary breaks must never happen silently and must be explicitly confirmed (this replaces ADRs for now).
- There is exactly one planning anchor: `roadmap/backlog.md` is the only queue for "what's next" (no parallel decision locations).
- Graph names have a single origin: `ai_core/graph/registry.py` is the source of truth; no versioning/lifecycle process until a real GraphExecutor exists.
- Pre-MVP / no production data: breaking changes are allowed (including schema/contract changes). Prefer removing compatibility layers over carrying duplicates; if a reset is required for correctness, a reset is acceptable.
- Traceability rule: if a confirmed change breaks an existing contract (schemas/IDs/meta/boundaries), always add a corresponding item to `roadmap/backlog.md` with concrete code pointers and acceptance criteria.
- Agents-first workflow: keep roadmap/backlog items LLM-executable (code pointers + acceptance criteria) and treat docs as explanations, not alternative sources of runtime truth.

## Primary code locations (navigation)

- Graph execution protocol & checkpointing: `ai_core/graph/core.py`
- Graph request meta normalization: `ai_core/graph/schemas.py`
- Graph implementations: `ai_core/graphs/`
- Graph nodes/capabilities: `ai_core/nodes/`
- Tool contract envelopes (`ToolContext`, `ToolResult`, `ToolError`): `ai_core/tool_contracts/base.py`
- Tool error identifiers: `ai_core/tools/errors.py`
- RAG implementation: `ai_core/rag/`
- Document ingestion, lifecycle, collections: `documents/`
- UI templates/static assets (HTMX/Tailwind): `theme/`

## Architecture docs (explanatory)

- 4-layer lens: `docs/architecture/4-layer-firm-hierarchy.md`
- Code-backed inventory snapshot: `docs/architecture/architecture-reality.md`

## Canonical identifiers (IDs) and their sources

### HTTP header names (canonical)

Canonical header constants live in `common/constants.py`:

- `X-Tenant-ID`, `X-Tenant-Schema`, `X-Case-ID`, `X-Trace-ID`, `X-Workflow-ID`, `X-Collection-ID`, `X-Key-Alias`, `Idempotency-Key`

### Request -> scope context

The canonical scope model is `ai_core/contracts/scope.py:ScopeContext`.

ScopeContext is built from requests in:

- Django `HttpRequest` (User Request Hop): `ai_core/ids/http_scope.py:normalize_request`
- Celery Tasks (S2S Hop): `ai_core/ids/http_scope.py:normalize_task_context`
- Generic objects (incl. DRF request): `ai_core/graph/schemas.py:_build_scope_context`

As implemented in `ScopeContext` + normalizers:

- `tenant_id` exists for every scope.
- `trace_id` is normalized/coerced; when missing it is generated.
- At least one runtime identifier required: `run_id` and/or `ingestion_run_id` (may co-exist when workflow triggers ingestion).
- `case_id` is optional at HTTP request level, validated for format when present (pattern in `ai_core/ids/headers.py`). Required for graph executions (enforced in `normalize_meta`, see below). `ToolContext` itself does not enforce `case_id` - individual tools may require it based on their business logic.

#### Identity IDs (Pre-MVP ID Contract)

Identity is tracked via `user_id` and `service_id` which are **mutually exclusive**:

| Hop Type | `user_id` | `service_id` | Example |
|----------|-----------|--------------|---------|
| User Request Hop | REQUIRED (when auth) | ABSENT | HTTP API call with JWT |
| S2S Hop | ABSENT | REQUIRED | Celery task, internal graph call |
| Public Endpoint | ABSENT | ABSENT | Health check, public docs |

- `normalize_request()` extracts `user_id` from Django auth and sets `service_id=None`.
- `normalize_task_context()` requires `service_id` and sets `user_id=None`.
- `initiated_by_user_id` (who triggered the flow) is for `audit_meta` only, not `ScopeContext`.

#### Audit Meta (Entity Persistence)

For entity persistence, use `ai_core/contracts/audit_meta.py:audit_meta_from_scope`:

```python
audit_meta = audit_meta_from_scope(scope, created_by_user_id=scope.user_id)
```

Keys in `audit_meta`:
- `trace_id`, `invocation_id`: Correlation
- `created_by_user_id`: Entity owner (set once, immutable)
- `initiated_by_user_id`: Who triggered the flow (causal tracking)
- `last_hop_service_id`: Last service that wrote (from `scope.service_id`)

### Graph request meta (`meta`) for AI Core graphs

`ai_core/graph/schemas.py:normalize_meta` produces the canonical graph meta dictionary and attaches:

- `scope_context`: serialized `ScopeContext`
- `tool_context`: serialized `ToolContext` built from scope

`normalize_meta` rejects requests without `case_id` (`ai_core/graph/schemas.py:normalize_meta`).

### Tool context contract

Canonical tool envelope models live in `ai_core/tool_contracts/base.py`.

- `ToolContext` is immutable (`ConfigDict(frozen=True)`).
- At least one runtime ID required: `run_id` and/or `ingestion_run_id` (`ai_core/tool_contracts/base.py:ToolContext.check_run_ids`).
- Identity validation: `user_id` and `service_id` are mutually exclusive (`ai_core/tool_contracts/base.py:ToolContext.check_identity`).
- `case_id` is optional in `ToolContext` (no validation enforced at the ToolContext level).
- `ai_core/tool_contracts/__init__.py` re-exports the canonical `ToolContext` from `ai_core/tool_contracts/base.py`.
- Build from scope: `scope.to_tool_context()` or `tool_context_from_scope(scope)`.

### Deprecated identifier key

`request_id` is treated as deprecated alias for `trace_id` in `ai_core/ids/contracts.py:normalize_trace_id`.

## Graph execution & state persistence

- Graph interface: `ai_core/graph/core.py:GraphRunner` exposes `run(state: dict, meta: dict) -> (state, result)`.
- Graph execution context: `ai_core/graph/core.py:GraphContext`.
- File-backed checkpoint location: `common/object_store_defaults.py:BASE_PATH` (`.ai_core_store/`) + `ai_core/graph/core.py:FileCheckpointer._path` (`{tenant}/{case}/state.json`).
- Transition payload shape used by graphs: `ai_core/graphs/transition_contracts.py:StandardTransitionResult` and `ai_core/graphs/transition_contracts.py:GraphTransition`.

## Tool error identifiers

Deterministic error type identifiers are defined in `ai_core/tools/errors.py:ToolErrorType`:

- `RATE_LIMIT`, `TIMEOUT`, `UPSTREAM`, `VALIDATION`, `RETRYABLE`, `FATAL`

## Documents: lifecycle + dev endpoints

- Lifecycle states and allowed transitions: `documents/lifecycle.py` (`pending`, `ingesting`, `embedded`, `active`, `failed`, `deleted`).
- Domain service boundary for document + collection operations: `documents/domain_service.py:DocumentDomainService`.
- Service facades used by ingestion/deletion dispatchers: `documents/service_facade.py`.
- Dev-only document/collection endpoints are guarded by `settings.DEBUG` (`documents/dev_api.py:_require_debug`) and wired in `noesis2/urls.py`.
- Health endpoint for lifecycle checks: `noesis2/urls.py` -> `api/health/document-lifecycle/`.

## Worker queue (ingestion)

- Ingestion Celery queue name is `"ingestion"` (e.g. `ai_core/tasks.py:run_ingestion_graph` uses `@shared_task(..., queue="ingestion", name="ai_core.tasks.run_ingestion_graph")`).

## Worker queue (agents)

- Graph execution can be proxied via the `agents` Celery queue by scheduling `llm_worker.tasks.run_graph` (`llm_worker/tasks.py:run_graph`, enqueued from `ai_core/services/__init__.py` with `queue="agents"`).

## Local commands (pointers)

- Repo scripts and common entrypoints: `package.json`, `Makefile`, `scripts/`

## Vibe coding commands (Windows/Docker)

- Start full stack: `npm run win:dev:stack`
- Init (migrate/bootstrap/RAG schema): `npm run dev:init`
- Smoke checks: `npm run win:dev:check`
- Django manage.py (in Docker): `npm run win:dev:manage -- <manage.py args...>` (e.g. `npm run win:dev:manage -- check`)
- Lint/format (host Python): `npm run lint`, `npm run lint:fix`, `npm run format`

## Tests (Docker)

**All Python tests**:
- Linux: `npm run test:py` (vollständig inkl. `@pytest.mark.slow`)
- Windows: `npm run win:test:py`

**Einzelne Tests**:
- Linux: `npm run test:py:single -- path/to/test.py`
- Windows: `npm run win:test:py:single -- path/to/test.py`

**Beispiele**:
```bash
# Einzelne Testdatei
npm run win:test:py:single -- ai_core/tests/graphs/test_universal_ingestion_graph.py

# Spezifische Testfunktion
npm run win:test:py:single -- ai_core/tests/graphs/test_universal_ingestion_graph.py::test_function_name

# Testklasse und -methode
npm run win:test:py:single -- ai_core/tests/graphs/test_universal_ingestion_graph.py::TestClass::test_method
```

**Weitere Test-Varianten**:
- Fast (ohne `@pytest.mark.slow`): `npm run test:py:fast` / `npm run win:test:py:fast`
- Unit (ohne DB): `npm run test:py:unit` / `npm run win:test:py:unit`
- Coverage: `npm run test:py:cov` / `npm run win:test:py:cov`
- Clean install: `npm run test:py:clean` / `npm run win:test:py:clean`
