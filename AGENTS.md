# NOESIS-2 - LLM Entry Contract (`AGENTS.md`)

## Source of truth & documentation hierarchy

- Source of truth is code (Python/TypeScript). Documentation explains and references code; it does not define behavior.
- When documentation and code diverge, follow code and update documentation to match.
- For any file change, follow the most specific `AGENTS.md` in the directory tree (deepest one wins).

## Agent stop conditions (ask before changing)

If a change would introduce new runtime semantics (new identifiers, new contracts, or boundary breaks), STOP and ask for confirmation (and/or add a backlog item in `roadmap/backlog.md` with code pointers).

- New or changed IDs / meta keys / headers: anything affecting `ScopeContext` (`ai_core/contracts/scope.py`), `BusinessContext` (`ai_core/contracts/business.py`), `ToolContext` (`ai_core/tool_contracts/base.py`), graph meta (`ai_core/graph/schemas.py:normalize_meta`), or header constants (`common/constants.py`).
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

**Infrastructure Headers (ScopeContext)**:

- `X-Tenant-ID`, `X-Tenant-Schema`, `X-Trace-ID`, `X-Invocation-ID`, `Idempotency-Key`

**Business Domain Headers (BusinessContext)**:

- `X-Case-ID`, `X-Collection-ID`, `X-Workflow-ID`, `X-Document-ID`, `X-Document-Version-ID`

**Other Headers**:

- `X-Key-Alias`, `X-Retry-Attempt`

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

**BREAKING CHANGE (Option A)**: `case_id` is NO LONGER in `ScopeContext`. It was moved to `BusinessContext` (see section below).

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

### Business Context (Domain Identifiers)

**BREAKING CHANGE (Option A - Strict Separation)**: Business domain IDs were moved from `ScopeContext` to a separate `BusinessContext` model in Phase 1 of the Option A migration.

The canonical business context model is `ai_core/contracts/business.py:BusinessContext`.

#### Golden Rule (Strict Separation)

**Infrastructure (WHO/WHEN)** → `ScopeContext`
**Business Domain (WHAT)** → `BusinessContext`

```
ScopeContext: tenant_id, trace_id, invocation_id, user_id, service_id,
              run_id, ingestion_run_id, tenant_schema, idempotency_key, timestamp

BusinessContext: case_id, collection_id, workflow_id, document_id, document_version_id
```

#### BusinessContext Fields (All Optional)

All business domain IDs are **optional** in `BusinessContext`. Individual graphs validate their required fields.

| Field | Type | HTTP Header | Django META Key | Description |
|-------|------|-------------|-----------------|-------------|
| `case_id` | `str \| None` | `X-Case-ID` | `HTTP_X_CASE_ID` | Legal case identifier |
| `collection_id` | `str \| None` | `X-Collection-ID` | `HTTP_X_COLLECTION_ID` | Document collection ID |
| `workflow_id` | `str \| None` | `X-Workflow-ID` | `HTTP_X_WORKFLOW_ID` | Workflow identifier |
| `document_id` | `str \| None` | `X-Document-ID` | `HTTP_X_DOCUMENT_ID` | Document ID |
| `document_version_id` | `str \| None` | `X-Document-Version-ID` | `HTTP_X_DOCUMENT_VERSION_ID` | Document version ID |

#### HTTP Header → BusinessContext Flow

BusinessContext is extracted from HTTP headers in `ai_core/graph/schemas.py:normalize_meta`:

```python
# Extract business context IDs from request headers (all optional)
case_id = _coalesce(request, X_CASE_ID_HEADER, META_CASE_ID_KEY)
workflow_id = _coalesce(request, X_WORKFLOW_ID_HEADER, META_WORKFLOW_ID_KEY)
collection_id = _coalesce(request, X_COLLECTION_ID_HEADER, META_COLLECTION_ID_KEY)
document_id = _coalesce(request, X_DOCUMENT_ID_HEADER, META_DOCUMENT_ID_KEY)
document_version_id = _coalesce(request, X_DOCUMENT_VERSION_ID_HEADER, META_DOCUMENT_VERSION_ID_KEY)

# Build BusinessContext (all fields optional per Option A)
business = BusinessContext(
    case_id=case_id,
    workflow_id=workflow_id,
    collection_id=collection_id,
    document_id=document_id,
    document_version_id=document_version_id,
)

# Attach to ToolContext
tool_context = scope.to_tool_context(business=business, metadata=context_metadata)
```

#### Accessing BusinessContext in Tools

Tools receive `ToolContext` which contains both `scope` and `business`:

```python
from ai_core.tool_contracts import ToolContext, ToolOutput
from ai_core.contracts import BusinessContext, ScopeContext

def run(context: ToolContext, input: MyToolInput) -> ToolOutput[MyToolInput, MyToolOutput]:
    # Infrastructure IDs from ScopeContext
    tenant_id = context.scope.tenant_id
    trace_id = context.scope.trace_id

    # Business IDs from BusinessContext
    case_id = context.business.case_id
    collection_id = context.business.collection_id
    document_id = context.business.document_id

    # Backward compatibility (deprecated)
    case_id_deprecated = context.case_id  # delegates to context.business.case_id
```

#### Graph-Specific Validation

**BREAKING CHANGE (Option A)**: `normalize_meta` does NOT enforce `case_id` globally. Individual graphs validate their required fields.

Example from Framework Analysis Graph:

```python
# In ai_core/graphs/framework_analysis_graph.py
def validate_business_context(context: ToolContext) -> None:
    """Validate Framework Analysis requires case_id and document_id."""
    if not context.business.case_id:
        raise InputError(
            message="Framework Analysis requires case_id",
            error_code="FRAMEWORK_MISSING_CASE_ID"
        )
    if not context.business.document_id:
        raise InputError(
            message="Framework Analysis requires document_id",
            error_code="FRAMEWORK_MISSING_DOCUMENT_ID"
        )
```

#### Migration from Old Code (Pre-Option A)

**Old code (before Option A)**:

```python
# ❌ BROKEN: case_id no longer in ScopeContext
case_id = scope.case_id

# ❌ BROKEN: Tool inputs no longer have business IDs
class RetrieveInput(BaseModel):
    query: str
    collection_id: str  # ❌ Removed
```

**New code (after Option A)**:

```python
# ✅ CORRECT: case_id in BusinessContext
case_id = context.business.case_id

# ✅ CORRECT: Tool inputs have only functional params
class RetrieveInput(BaseModel):
    query: str
    # collection_id read from context.business.collection_id
```

#### Complete Example: New Tool with BusinessContext

```python
from pydantic import BaseModel, Field
from ai_core.tool_contracts import ToolContext, ToolOutput
from ai_core.tools.errors import InputError

# Input: Functional parameters only (no IDs)
class MyToolInput(BaseModel):
    query: str = Field(..., description="Search query")
    limit: int = Field(default=10, ge=1, le=100)

# Output: Business results
class MyToolOutput(BaseModel):
    results: list[str]
    count: int

def run(context: ToolContext, input: MyToolInput) -> ToolOutput[MyToolInput, MyToolOutput]:
    """Example tool following Golden Rule."""

    # Validate required business context
    if not context.business.case_id:
        raise InputError(
            message="MyTool requires case_id",
            error_code="MYTOOL_MISSING_CASE_ID"
        )

    # Read IDs from context (not input!)
    case_id = context.business.case_id
    collection_id = context.business.collection_id  # optional
    tenant_id = context.scope.tenant_id

    # Business logic using context IDs
    results = perform_search(
        query=input.query,
        case_id=case_id,
        collection_id=collection_id,
        tenant_id=tenant_id,
        limit=input.limit
    )

    return ToolOutput(
        input=input,
        output=MyToolOutput(results=results, count=len(results))
    )
```

### Graph request meta (`meta`) for AI Core graphs

`ai_core/graph/schemas.py:normalize_meta` produces the canonical graph meta dictionary and attaches:

- `scope_context`: serialized `ScopeContext`
- `business_context`: serialized `BusinessContext` (all fields optional)
- `tool_context`: serialized `ToolContext` built from scope + business

**BREAKING CHANGE (Option A)**: `normalize_meta` does NOT enforce `case_id` globally. All BusinessContext fields are optional. Individual graphs validate their required business IDs.

### Tool context contract

Canonical tool envelope models live in `ai_core/tool_contracts/base.py`.

- `ToolContext` is immutable (`ConfigDict(frozen=True)`).
- **Compositional structure (Option A)**: `ToolContext` contains `scope: ScopeContext` + `business: BusinessContext` + runtime metadata.
- At least one runtime ID required: `run_id` and/or `ingestion_run_id` (`ai_core/tool_contracts/base.py:ToolContext.check_run_ids`).
- Identity validation: `user_id` and `service_id` are mutually exclusive (`ai_core/tool_contracts/base.py:ToolContext.check_identity`).
- **Business IDs**: All business domain IDs (`case_id`, `collection_id`, etc.) are in `context.business`, not `context.scope`.
- **Backward compatibility**: Deprecated `@property` accessors delegate to `scope.X` or `business.X` for compatibility.
- `ai_core/tool_contracts/__init__.py` re-exports the canonical `ToolContext` from `ai_core/tool_contracts/base.py`.
- Build from scope + business: `scope.to_tool_context(business=business_context)` or `tool_context_from_scope(scope, business=business_context)`.

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

## Documents: User Integration & Collaboration

- **User Attribution**: `Document.created_by` / `Document.updated_by` (driven by `AuditMeta`).
- **Authorization**: `documents/authz.py:DocumentAuthzService` (checks owner > permission > case > role).
- **Activity Tracking**: `documents/activity_service.py` logs to `DocumentActivity`.
- **Collaboration**: `DocumentComment`, `DocumentMention`, `UserDocumentFavorite` (Phase 4a).
- **Notifications**: In-app (`DocumentNotification`) and external (`NotificationEvent` / `NotificationDelivery`) (Phase 4b).

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
