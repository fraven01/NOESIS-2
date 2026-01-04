# ID Usage Plan: Graphs & Workers

This document defines how IDs should be consistently used across all graphs and worker tasks in the codebase.

## Overview

The Pre-MVP ID Contract establishes clear patterns for:
1. **Identity IDs**: Who is executing (user_id vs service_id)
2. **Runtime IDs**: What execution context (run_id, ingestion_run_id)
3. **Correlation IDs**: Traceability (trace_id, invocation_id)
4. **Audit Meta**: Entity persistence tracking

---

## Service ID Registry

Each service/worker should use a consistent `service_id`:

| Service | service_id | Queue | Entry Point |
|---------|------------|-------|-------------|
| Ingestion Worker | `celery-ingestion-worker` | `ingestion` | `ai_core/tasks.py:run_ingestion_graph` |
| Agents Worker | `celery-agents-worker` | `agents-high` (default), `agents-low` (background) | `llm_worker/tasks.py:run_graph` |
| Crawler Worker | `crawler-worker` | `ingestion` | `crawler/worker.py` |
| Document Processing | `document-processor` | `ingestion` | (embedded in graphs) |

---

## Graph Entry Points

### 1. HTTP → Graph (User Request Hop)

```python
# In ai_core/views.py or similar
from ai_core.ids.http_scope import normalize_request

def graph_view(request):
    scope = normalize_request(request)  # user_id from auth, service_id=None

    # Dispatch to Celery (becomes S2S Hop)
    run_graph.delay(
        scope.tenant_id,
        scope.case_id,
        graph_name="my_graph",
        trace_id=scope.trace_id,
        initiated_by_user_id=scope.user_id,  # For audit_meta
    )
```

### 2. Celery Task → Graph (S2S Hop)

```python
# In ai_core/tasks.py
from ai_core.ids.http_scope import normalize_task_context
from ai_core.contracts.audit_meta import audit_meta_from_scope

@shared_task(queue="ingestion", name="ai_core.tasks.run_ingestion_graph")
def run_ingestion_graph(
    tenant_id: str,
    case_id: str,
    document_ids: list[str],
    *,
    trace_id: str | None = None,
    initiated_by_user_id: str | None = None,
    **kwargs,
):
    scope = normalize_task_context(
        tenant_id=tenant_id,
        case_id=case_id,
        service_id="celery-ingestion-worker",  # REQUIRED
        trace_id=trace_id,  # Inherited from parent
        # invocation_id generated fresh (new hop)
    )

    # For entity persistence
    audit_meta = audit_meta_from_scope(
        scope,
        initiated_by_user_id=initiated_by_user_id,
    )

    # Execute graph with scope
    result = execute_graph(graph_name, scope=scope, ...)
```

---

## Graph Node Patterns

### Pattern 1: Reading Data (No Persistence)

```python
def my_read_node(state: dict) -> tuple[GraphTransition, bool]:
    scope = state["scope_context"]

    # Use scope for filtering/authorization
    documents = Document.objects.filter(
        tenant_id=scope.tenant_id,
        case_id=scope.case_id,
    )

    return GraphTransition(decision="fetched", ...), True
```

### Pattern 2: Creating Entities

```python
from ai_core.contracts.audit_meta import audit_meta_from_scope

def my_create_node(state: dict) -> tuple[GraphTransition, bool]:
    scope = state["scope_context"]

    # Build audit_meta for persistence
    audit_meta = audit_meta_from_scope(
        scope,
        created_by_user_id=state.get("initiated_by_user_id"),
    )

    # Create entity with audit tracking
    entity = MyEntity.objects.create(
        tenant_id=scope.tenant_id,
        case_id=scope.case_id,
        audit_meta=audit_meta.model_dump(),
        ...
    )

    return GraphTransition(decision="created", ...), True
```

### Pattern 3: Workflow Triggering Ingestion

When a workflow graph triggers ingestion, both `run_id` and `ingestion_run_id` co-exist:

```python
def trigger_ingestion_node(state: dict) -> tuple[GraphTransition, bool]:
    scope = state["scope_context"]

    # Current workflow's run_id stays
    workflow_run_id = scope.run_id

    # Generate new ingestion_run_id for the triggered ingestion
    import uuid
    ingestion_run_id = uuid.uuid4().hex

    # Dispatch ingestion task
    run_ingestion_graph.delay(
        tenant_id=scope.tenant_id,
        case_id=scope.case_id,
        document_ids=state["urls_to_ingest"],
        trace_id=scope.trace_id,  # Keep correlation
        workflow_run_id=workflow_run_id,
        ingestion_run_id=ingestion_run_id,
    )

    return GraphTransition(decision="ingestion_triggered", ...), True
```

---

## Worker Task Patterns

### Ingestion Worker

```python
# ai_core/tasks.py

@shared_task(queue="ingestion", name="ai_core.tasks.run_ingestion_graph")
def run_ingestion_graph(
    tenant_id: str,
    case_id: str,
    document_ids: list[str],
    embedding_profile: str,
    *,
    run_id: str | None = None,
    ingestion_run_id: str | None = None,
    trace_id: str | None = None,
    idempotency_key: str | None = None,
    initiated_by_user_id: str | None = None,
    **kwargs,
):
    scope = normalize_task_context(
        tenant_id=tenant_id,
        case_id=case_id,
        service_id="celery-ingestion-worker",
        trace_id=trace_id,
        run_id=run_id,
        ingestion_run_id=ingestion_run_id or uuid.uuid4().hex,
        idempotency_key=idempotency_key,
    )

    # Execute with proper scope
    ...
```

Idempotency and cache guards (Redis-only):
- `chunk`: cache key derived from `tenant_id` + `content_hash` (+ `embedding_profile` when provided), TTL 1h.
- `embed`: cache key derived from `tenant_id` + `chunks_path` + `embedding_profile`, TTL 24h, plus dedupe key `task:dedupe:{task_name}:{idempotency_key}`.
- `upsert`: dedupe key derived from `tenant_id` + `vector_space_id` + `content_hash` + `embedding_profile`, TTL 24h.
- Cache hits emit structured logs and Langfuse events for tracing.

### Crawler Worker

```python
# crawler/worker.py

def process_crawl_job(job: CrawlJob):
    scope = normalize_task_context(
        tenant_id=job.tenant_id,
        case_id=job.case_id or "crawler-system",  # System task may not have case
        service_id="crawler-worker",
        trace_id=job.trace_id,
        ingestion_run_id=uuid.uuid4().hex,
    )

    # Process with scope
    ...
```

---

## Migration Checklist

### Phase 1: Core Services (DONE)

- [x] `ScopeContext` updated with identity fields
- [x] `ToolContext` synchronized
- [x] `normalize_request()` extracts user_id
- [x] `normalize_task_context()` requires service_id
- [x] `audit_meta_from_scope()` implemented

### Phase 2: Task Entry Points

For each Celery task in `ai_core/tasks.py`:

- [ ] Add `service_id` parameter to `normalize_task_context()`
- [ ] Add `initiated_by_user_id` parameter for audit tracking
- [ ] Build `audit_meta` for entity creation
- [ ] Log service_id in structured logs

### Phase 3: Graph Implementations

For each graph in `ai_core/graphs/`:

- [ ] Verify scope propagation through nodes
- [ ] Add `audit_meta` to entity creation nodes
- [ ] Document service_id expectations

### Phase 4: Entity Models

For each entity that needs traceability:

- [ ] Add `audit_meta` JSONField (if not present)
- [ ] Migrate existing entities with default audit_meta
- [ ] Add query helpers for trace correlation

---

## Observability Integration

### Structured Logging

All tasks should log identity context:

```python
import structlog

log = structlog.get_logger()

log.info(
    "task_started",
    tenant_id=scope.tenant_id,
    trace_id=scope.trace_id,
    invocation_id=scope.invocation_id,
    service_id=scope.service_id,
    run_id=scope.run_id,
    ingestion_run_id=scope.ingestion_run_id,
)
```

### Langfuse Spans

Ensure spans include identity tags:

```python
from langfuse.decorators import observe

@observe(name="my_graph_node")
def my_node(state: dict):
    scope = state["scope_context"]
    # Langfuse will pick up structlog context
    ...
```

---

## Testing Guidelines

### Unit Tests for Tasks

```python
def test_task_uses_correct_service_id(monkeypatch):
    captured_scope = None

    def capture_scope(*args, scope=None, **kwargs):
        nonlocal captured_scope
        captured_scope = scope

    monkeypatch.setattr("ai_core.graphs.universal_ingestion_graph.run", capture_scope)

    run_ingestion_graph(
        tenant_id="test",
        case_id="case-1",
        document_ids=["doc-1"],
        embedding_profile="standard",
    )

    assert captured_scope.service_id == "celery-ingestion-worker"
    assert captured_scope.user_id is None
```

### Integration Tests for Graphs

```python
def test_graph_propagates_identity():
    scope = ScopeContext(
        tenant_id="test",
        trace_id="trace-1",
        invocation_id="inv-1",
        case_id="case-1",
        run_id="run-1",
        service_id="test-service",
    )

    result = run_my_graph(initial_state, scope=scope)

    # Verify identity propagated through all nodes
    assert result["audit_meta"]["last_hop_service_id"] == "test-service"
```

---

## References

- **ID Contract One-Pager**: `docs/architecture/id-contract-one-pager.md`
- **ID Semantics**: `docs/architecture/id-semantics.md`
- **Review Checklist**: `docs/architecture/id-contract-review-checklist.md`
- **Contracts Package**: `ai_core/contracts/`
