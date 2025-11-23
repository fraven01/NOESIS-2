# LLM Contract Readiness Report

## Repository Map
- **Runtime services**: Django web, Celery workers (agents queue) and ingestion worker per Architekturübersicht.【F:docs/architektur/overview.md†L5-L22】
- **AI Core**: Graph orchestrations (`ai_core/graphs`), shared RAG nodes and capabilities, middleware, and tool contracts used across workers.【F:ai_core/graphs/README.md†L1-L49】【F:AGENTS.md†L90-L143】
- **Domain apps**: `cases` (case lifecycle and events), `documents`/`crawler` (ingestion), `customers` (tenancy), `profiles`/`organizations` (account data), and supporting `common`/`noesis2` Django setup (discovered via repo layout).
- **Cross-cutting concerns**: Tenancy headers/contexts, trace propagation, LangGraph workflow IDs, ingestion runs, and RAG collections per case domain overview.【F:docs/domain/cases.md†L9-L74】

## Existing Conventions
- **Glossar & Pflichtfelder**: `tenant_id`, `trace_id`, `invocation_id`, exactly one of `run_id` or `ingestion_run_id`, optional `case_id`, `workflow_id`, idempotency key; `trace_id` replaces `request_id`.【F:AGENTS.md†L70-L177】
- **HTTP headers**: `X-Tenant-ID` required; `X-Case-ID`, `Idempotency-Key` optional; middleware populates trace/span IDs and echoes metadata.【F:AGENTS.md†L75-L176】【F:ai_core/middleware/context.py†L18-L98】
- **Tool layer**: Pydantic `ToolContext` enforces trace/tenant/invocation plus exactly one runtime ID, with optional workflow/case/collection/document identifiers and locale/budget fields.【F:ai_core/tool_contracts/base.py†L25-L80】
- **Domain alignment**: `case_id` corresponds to `Case.external_id` and drives lifecycle events and ingestion linkage.【F:docs/domain/cases.md†L9-L74】

## LLM Friction Points
- **Middleware vs. tool context divergence**: HTTP middleware returns `response_meta` with `trace_id`, `tenant_id`, `case_id`, but omits `workflow_id`, `run_id`, or `invocation_id` required by `ToolContext`; header normalization also exposes `key_alias`/client IP fields not reflected in tool contracts.【F:ai_core/middleware/context.py†L45-L99】【F:ai_core/tool_contracts/base.py†L25-L80】 This shape mismatch forces LLMs to hand-roll bridging code.
- **Distributed invariants**: Requirements for `case_id` resolution and lifecycle enforcement live in domain docs but are not encapsulated in helpers used by middleware or tools, leading to potential drift in how `Case.external_id` is validated and tagged.【F:docs/domain/cases.md†L35-L74】
- **Redundant ID validators**: `ai_core.ids.contracts.normalize_trace_id` and ad-hoc header parsing in middleware overlap with Pydantic validation in `ToolContext`, without shared aliases or type hints to prevent `run_id`/`ingestion_run_id` swaps.【F:ai_core/middleware/context.py†L104-L144】【F:ai_core/tool_contracts/base.py†L54-L80】【F:ai_core/ids/contracts.py†L1-L70】
- **Inconsistent entry points**: Graph runners and workers consume loosely typed `dict` contexts (see `GraphNode` signatures) rather than a contract that carries the Glossar fields, making discovery for LLMs harder.【F:ai_core/graphs/README.md†L5-L49】

## Proposed Central Contracts & Helpers
- **ScopeContext (unified)**: Frozen Pydantic/BaseModel bridging HTTP and tool layers with aliases for headers, containing `tenant_id`, `trace_id`, `invocation_id`, `run_id` XOR `ingestion_run_id`, optional `case_id`, `workflow_id`, `collection_id`, `document_id`, `document_version_id`, `locale`, `idempotency_key`, and timestamp. Enforce Glossar invariants once.
- **Type aliases**: `TenantId = str`, `TraceId = str`, `InvocationId = UUID`, `RunId = str`, `IngestionRunId = str`, `WorkflowId = str`, `CaseId = str`, `CollectionId = str`, `DocumentId = str`, `DocumentVersionId = str` to prevent swaps and improve IDE/searchability.
- **Golden-path factories**:
  - `ScopeContext.from_http(request)` → extracts headers/query/body using shared normalization helpers.
  - `ScopeContext.for_tool(context: ToolContext)` → narrows/extends tool metadata for downstream nodes.
  - `start_ingestion_run(scope: ScopeContext, payload)` and `trigger_case_event(scope: ScopeContext, event_type, payload)` helpers that stamp mandatory IDs and forward to existing ingestion/case services.
- **Shared normalization module**: Reuse `normalize_trace_id` plus header parsers (`X-Tenant-ID`, `X-Case-ID`, `Idempotency-Key`, `traceparent`) to avoid duplication between middleware and tool bootstrap.

## Pre-MVP Change Plan
1. **Introduce central contracts module**: Add `ai_core/contracts/scope.py` defining `ScopeContext` and ID aliases; include conversion helpers from HTTP/tool contexts using existing validators (`normalize_trace_id`) and Glossar constraints.
2. **Refactor middleware to emit ScopeContext**: Update `RequestContextMiddleware` to build `ScopeContext` and attach to request (and response headers) while keeping current behavior for backward compatibility. Replace bespoke normalization with shared helper calls.
3. **Tool bootstrap helper**: Provide `ai_core/tool_contracts/helpers.py` with `ScopeContext.from_tool_context` to align tool invocations with graph/state consumers and reduce per-tool boilerplate.
4. **Graph entry adoption**: Update representative graphs (`ai_core/graphs` README samples and one graph runner) to accept `ScopeContext` instead of raw dicts; add docstrings to guide LLMs.
5. **Case lifecycle guardrails**: Add helper in `cases` (e.g., `cases/services/case_scope.py`) to resolve `case_id` ↔ `Case.external_id`, ensuring middleware, ingestion, and tools reuse the same check.
6. **Documentation & schema breadcrumbs**: Document the canonical contracts in `docs/agents/tool-contracts.md` and cross-link from `AGENTS.md` Glossar plus new README snippet to improve discoverability.

## Code Sketches
```python
# ai_core/contracts/scope.py
from pydantic import BaseModel, Field, model_validator
from uuid import UUID

TenantId = str
TraceId = str
InvocationId = UUID
RunId = str
IngestionRunId = str
WorkflowId = str
CaseId = str
CollectionId = str
DocumentId = str
DocumentVersionId = str

class ScopeContext(BaseModel):
    tenant_id: TenantId
    trace_id: TraceId
    invocation_id: InvocationId | None = None
    run_id: RunId | None = None
    ingestion_run_id: IngestionRunId | None = None
    workflow_id: WorkflowId | None = None
    case_id: CaseId | None = None
    collection_id: CollectionId | None = None
    document_id: DocumentId | None = None
    document_version_id: DocumentVersionId | None = None
    idempotency_key: str | None = None
    locale: str | None = None
    now_iso: datetime

    @model_validator(mode="after")
    def ensure_single_run(cls, values):
        if bool(values.run_id) == bool(values.ingestion_run_id):
            raise ValueError("Provide exactly one of run_id or ingestion_run_id")
        return values

    @classmethod
    def from_http(cls, request: HttpRequest) -> "ScopeContext":
        # delegates to shared header normalization helpers
        ...

    @classmethod
    def from_tool_context(cls, ctx: ToolContext) -> "ScopeContext":
        return cls(**ctx.model_dump())
```

```python
# ai_core/tool_contracts/helpers.py
from ai_core.contracts.scope import ScopeContext
from ai_core.tool_contracts.base import ToolContext

def scope_from_tool(ctx: ToolContext) -> ScopeContext:
    return ScopeContext.from_tool_context(ctx)
```

```python
# cases/services/case_scope.py
from cases.models import Case
from ai_core.contracts.scope import ScopeContext

def resolve_case(scope: ScopeContext) -> Case | None:
    if not scope.case_id:
        return None
    return Case.objects.filter(tenant__schema_name=scope.tenant_id, external_id=scope.case_id).first()
```
