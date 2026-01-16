# Technical Document CRUD Graph - Spezifikation

## Executive Summary

**Problem:** Inkonsistente Orchestrierung von Dokument-Operationen in `ai_core`:
- Upload/Delete laufen über Service-Funktionen oder direkte Celery Tasks
- Nur Ingestion nutzt einen strukturierten Technical Graph
- Fehlende Observability, State-Management, und einheitliche Error-Handling für non-ingestion Operationen

**Lösung:** Ein zentraler **Technical Document Service Graph**, der alle CRUD-Operationen orchestriert.

**Status:** Backlog (nicht implementiert)

---

## Architektur

### Scope

Der `document_service_graph` orchestriert folgende Operationen:

| Operation | Aktuell | Ziel |
|-----------|---------|------|
| **CREATE** | `ai_core/services/document_upload.py:handle_document_upload()` → Celery Task | Graph mit Knoten: `validate_input` → `persist_metadata` → `delegate_ingestion` |
| **DELETE** | `ai_core/rag/hard_delete.py:hard_delete()` (Celery Task) | Graph mit Knoten: `validate_input` → `authorize` → `soft_delete_db` → `delete_embeddings` → `finalize` |
| **UPDATE** | Nicht existierend (Re-Ingestion über Upload) | Graph mit Knoten: `validate_input` → `check_version` → `delegate_re_ingestion` |
| **READ** | Direkter Repository-Call (bleibt so) | Kein Graph nötig (zu simpel) |

### Repository Refactor Markers

CRUD persistence is already marked for this refactor in repository docstrings:
- `documents/repository.py` (`DocumentsRepository.delete`, `delete_asset`)
- `documents/repository.py` (`InMemoryDocumentsRepository.delete`, `delete_asset`)
- `ai_core/adapters/db_documents_repository.py` (`delete`, `delete_asset`)

### Contracts

#### Input

```python
# ai_core/graphs/technical/document_service_graph.py

from enum import Enum
from pydantic import BaseModel, Field
from ai_core.tool_contracts.base import ToolContext

class DocumentOperation(str, Enum):
    CREATE = "create"   # Upload + Initial Persist
    UPDATE = "update"   # Re-ingestion / Metadata Update
    DELETE = "delete"   # Soft Delete + Hard Delete with Embeddings

class CreatePayload(BaseModel):
    file_bytes: bytes
    filename: str
    content_type: str
    metadata: dict[str, object] | None = None

class DeletePayload(BaseModel):
    document_ids: list[str] = Field(min_length=1)
    reason: str
    ticket_ref: str | None = None
    hard_delete: bool = True  # Delete embeddings + DB

class UpdatePayload(BaseModel):
    document_id: str
    metadata_updates: dict[str, object] | None = None
    force_re_ingestion: bool = False

class DocumentServiceInput(BaseModel):
    schema_id: str = "noesis.graphs.document_service"
    schema_version: str = "1.0.0"
    operation: DocumentOperation
    context: dict  # Serialized ToolContext
    create_payload: CreatePayload | None = None
    delete_payload: DeletePayload | None = None
    update_payload: UpdatePayload | None = None
```

#### Output

```python
class DocumentServiceOutput(BaseModel):
    schema_id: str = "noesis.graphs.document_service"
    schema_version: str = "1.0.0"
    operation: DocumentOperation
    decision: str  # "success", "partial", "failed"
    reason: str | None = None

    # Operation-specific results
    document_id: str | None = None
    document_ids_deleted: list[str] | None = None
    chunks_deleted: int | None = None
    ingestion_run_id: str | None = None

    # Telemetry
    telemetry: dict = Field(default_factory=dict)
```

---

## DELETE Operation (Priority 1)

### Motivation

Aktuell ist DELETE ein direkter Celery Task (`ai_core/rag/hard_delete.py:hard_delete()`):
- Keine Graph-Transitions (schwer zu tracken in Langfuse)
- Keine Zwischenzustände (z.B. "Deleting embeddings...")
- Kein strukturiertes Error-Handling per Knoten

### Knoten-Struktur

```
DELETE Operation:
1. validate_input     → Parse ToolContext, validate document_ids, reason
2. authorize_delete   → Check if user/service can delete (Admin, Service Key)
3. soft_delete_db     → Mark documents as deleted in DB (Document.deleted_at)
4. delete_embeddings  → Remove from vector store (pgvector via vector_client)
5. delete_metadata    → Remove from object store (optional cleanup)
6. finalize           → Emit telemetry, transitions, return output
```

### Transitions

```python
GraphTransition(
    decision="authorized",
    reason="Service key 'ops-admin' authorized for hard delete",
    severity="info",
    attributes={"actor_mode": "service_key", "operator_label": "ops-admin"}
)

GraphTransition(
    decision="embeddings_deleted",
    reason="Deleted 142 chunks for 3 documents",
    severity="info",
    attributes={"documents": 3, "chunks_deleted": 142, "vector_space": "global"}
)

GraphTransition(
    decision="completed",
    reason="Hard delete completed successfully",
    severity="info",
    attributes={"documents_deleted": 3, "total_duration_ms": 1250}
)
```

### Code Pointers (DELETE)

**Before (Current):**
- Entry: `ai_core/views.py:2014-2101` (`RagHardDeleteAdminView.post()`)
- Task: `ai_core/rag/hard_delete.py:164-291` (`hard_delete()` Celery task)
- Authorization: `ai_core/rag/hard_delete.py:50-97` (`_resolve_actor()`)
- Vector deletion: `ai_core/rag/hard_delete.py:140-157` (`_dispatch_delete_to_reaper()`)

**After (Target):**
- Entry: `ai_core/views.py:RagHardDeleteAdminView.post()` → dispatch to `document_service_graph`
- Graph: `ai_core/graphs/technical/document_service_graph.py:build_delete_graph()`
- Nodes: `ai_core/nodes/document_service/{validate_input,authorize_delete,soft_delete_db,delete_embeddings,finalize}.py`

---

## CREATE Operation (Priority 2)

### Motivation

Aktuell ist CREATE eine Service-Funktion (`ai_core/services/document_upload.py:handle_document_upload()`):
- 337 Zeilen monolithische Funktion
- Kein State-Management (was passiert bei Teilfehlern?)
- Schwer zu testen (Mock gesamte Funktion oder nichts)

### Knoten-Struktur

```
CREATE Operation:
1. validate_input         → Parse ToolContext, validate file, metadata
2. ensure_collection      → Create/fetch manual collection if needed
3. persist_metadata       → Write metadata to object store
4. dispatch_ingestion     → Delegate to universal_ingestion_graph (Celery)
5. finalize               → Return response with document_id, ingestion_run_id
```

### Delegation zu Ingestion

CREATE nutzt `universal_ingestion_graph` als Sub-Graph:
```python
# In persist_metadata node:
state["ingestion_payload"] = {
    "document_id": document_uuid,
    "file_bytes": file_bytes,
    "metadata": metadata,
}

# In dispatch_ingestion node:
from documents.tasks import upload_document_task
signature = upload_document_task.s(
    file_bytes=state["ingestion_payload"]["file_bytes"],
    ...
)
with_scope_apply_async(signature, scope_context, task_id=ingestion_run_id)
```

**Wichtig:** CREATE erstellt nur die Metadata-Hülle. Die tatsächliche Verarbeitung (Chunking, Embedding) bleibt in `universal_ingestion_graph`.

### Code Pointers (CREATE)

**Before (Current):**
- Entry: `ai_core/views.py:1767-1840` (`RagUploadView.post()`)
- Service: `ai_core/services/document_upload.py:39-337` (`handle_document_upload()`)
- Celery dispatch: `ai_core/services/document_upload.py:299-325`

**After (Target):**
- Entry: `ai_core/views.py:RagUploadView.post()` → dispatch to `document_service_graph`
- Graph: `ai_core/graphs/technical/document_service_graph.py:build_create_graph()`
- Nodes: `ai_core/nodes/document_service/{validate_input,ensure_collection,persist_metadata,dispatch_ingestion,finalize}.py`

---

## UPDATE Operation (Priority 3, Future)

### Motivation

Aktuell gibt es keinen dedizierten Update-Pfad:
- Metadata-Updates erfordern kompletten Re-Upload
- Kein Versionierungs-Konzept für Dokumente

### Knoten-Struktur (Entwurf)

```
UPDATE Operation:
1. validate_input           → Parse ToolContext, validate document_id, updates
2. check_version            → Fetch current document version
3. apply_metadata_updates   → Update DB fields (if metadata_only)
4. delegate_re_ingestion    → Trigger universal_ingestion_graph (if force_re_ingestion)
5. finalize                 → Return updated document_id, version
```

**Status:** Nicht priorisiert (Pre-MVP). Wird nur dokumentiert für zukünftige Erweiterung.

---

## Observability

### Graph-Transitions

Jeder Knoten emittiert strukturierte Transitions:

```python
# In authorize_delete node:
emit_transition(
    GraphTransition(
        decision="authorized",
        reason=f"User '{operator_label}' authorized for hard delete",
        severity="info",
        attributes={
            "actor_mode": actor_mode,  # "user_admin", "org_admin", "service_key"
            "operator_label": operator_label,
            "document_count": len(document_ids),
        }
    )
)
```

### Langfuse Spans

Automatisch via `@observe_span`:

```
document.delete.validate_input
document.delete.authorize
document.delete.soft_delete_db
document.delete.delete_embeddings
document.delete.finalize
```

### Strukturierte Logs

```python
logger.info(
    "document.delete.completed",
    extra={
        "tenant_id": context.scope.tenant_id,
        "trace_id": context.scope.trace_id,
        "documents_deleted": len(document_ids),
        "chunks_deleted": chunks_deleted,
        "duration_ms": duration,
    }
)
```

---

## Migration-Strategie

### Phase 1: DELETE (Breaking Change acceptable in Pre-MVP)

**Schritt 1:** Implementiere `document_service_graph` mit DELETE-Operation
- Erstelle Graph-Builder
- Implementiere DELETE-Knoten
- Tests für DELETE-Pfad

**Schritt 2:** Migrate `RagHardDeleteAdminView`
- Ändere `RagHardDeleteAdminView.post()` → dispatch zu Graph
- Entferne direkten Call zu `hard_delete()` Task
- Behalte `hard_delete()` Task als DEPRECATED (für laufende Jobs)

**Schritt 3:** Cleanup (nach grace period)
- Entferne `ai_core/rag/hard_delete.py:hard_delete()` Task
- Update Celery queue config

### Phase 2: CREATE (Breaking Change acceptable in Pre-MVP)

**Schritt 1:** Implementiere CREATE-Operation im Graph
- Knoten für CREATE-Pfad
- Tests für CREATE-Pfad

**Schritt 2:** Migrate `RagUploadView`
- Ändere `RagUploadView.post()` → dispatch zu Graph
- Entferne `ai_core/services/document_upload.py:handle_document_upload()` Call
- Behalte `handle_document_upload()` als DEPRECATED

**Schritt 3:** Cleanup
- Entferne `ai_core/services/document_upload.py:handle_document_upload()`

### Phase 3: UPDATE (Future, nicht priorisiert)

Wird nur implementiert wenn Business-Need besteht.

---

## Testing-Strategie

### Unit-Tests pro Knoten

```python
# ai_core/tests/nodes/document_service/test_authorize_delete.py

def test_authorize_delete_service_key():
    """Service key with valid internal key authorizes delete."""
    context = build_test_context(service_id="ops-service")
    actor = {"internal_key": "ops-admin", "label": "Ops Team"}
    state = {"document_ids": ["doc-1", "doc-2"], "actor": actor}

    transition, continue_flag = authorize_delete(state, context)

    assert transition.decision == "authorized"
    assert continue_flag is True
    assert state["operator_label"] == "Ops Team"
    assert state["actor_mode"] == "service_key"

def test_authorize_delete_non_admin_user():
    """Non-admin user is rejected."""
    context = build_test_context(user_id="user-123")
    actor = {"user_id": "user-123", "label": "Normal User"}
    state = {"document_ids": ["doc-1"], "actor": actor}

    with pytest.raises(HardDeleteAuthorisationError):
        authorize_delete(state, context)
```

### Integration-Tests pro Operation

```python
# ai_core/tests/graphs/test_document_service_graph.py

def test_delete_operation_end_to_end(fake_vector_client, fake_repo):
    """DELETE operation with service key succeeds end-to-end."""
    graph = build_document_service_graph()
    context = build_test_context(service_id="ops-service")

    input_data = DocumentServiceInput(
        operation=DocumentOperation.DELETE,
        context=context.model_dump(mode="json"),
        delete_payload=DeletePayload(
            document_ids=["doc-1", "doc-2"],
            reason="Test cleanup",
            ticket_ref="TICK-123",
        ),
    )

    # Prepare fake data
    fake_repo.documents = [
        FakeDocument(id="doc-1", tenant_id="tenant-1"),
        FakeDocument(id="doc-2", tenant_id="tenant-1"),
    ]
    fake_vector_client.chunks = [
        FakeChunk(document_id="doc-1"),
        FakeChunk(document_id="doc-1"),
        FakeChunk(document_id="doc-2"),
    ]

    # Execute graph
    result = graph.invoke(input_data)

    # Assertions
    assert result.decision == "success"
    assert result.chunks_deleted == 3
    assert len(fake_repo.soft_deleted) == 2
    assert len(fake_vector_client.deleted_chunks) == 3
```

### Contract-Tests

```python
# ai_core/tests/graphs/test_document_service_contracts.py

def test_delete_input_contract():
    """DELETE input validates required fields."""
    with pytest.raises(ValidationError):
        DocumentServiceInput(
            operation=DocumentOperation.DELETE,
            context={},
            # Missing delete_payload
        )

    with pytest.raises(ValidationError):
        DocumentServiceInput(
            operation=DocumentOperation.DELETE,
            context={},
            delete_payload=DeletePayload(
                document_ids=[],  # Empty list not allowed
                reason="Test",
            ),
        )
```

---

## Acceptance-Kriterien

### DELETE Operation

- [ ] `document_service_graph` erstellt mit DELETE-Knoten
- [ ] DELETE-Knoten validiert: `validate_input`, `authorize_delete`, `soft_delete_db`, `delete_embeddings`, `finalize`
- [ ] `RagHardDeleteAdminView.post()` dispatcht zu Graph statt direktem Task-Call
- [ ] Graph emittiert strukturierte Transitions für jeden Knoten
- [ ] Langfuse-Spans für DELETE-Operation sichtbar
- [ ] Tests:
  - Unit-Tests für jeden DELETE-Knoten
  - Integration-Test für DELETE end-to-end
  - Contract-Tests für DELETE Input/Output
- [ ] Legacy `hard_delete()` Task als DEPRECATED markiert (oder entfernt nach grace period)

### CREATE Operation

- [ ] CREATE-Knoten validiert: `validate_input`, `ensure_collection`, `persist_metadata`, `dispatch_ingestion`, `finalize`
- [ ] `RagUploadView.post()` dispatcht zu Graph statt Service-Funktion
- [ ] CREATE delegiert korrekt zu `universal_ingestion_graph` (Celery Task)
- [ ] Graph emittiert strukturierte Transitions
- [ ] Tests:
  - Unit-Tests für jeden CREATE-Knoten
  - Integration-Test für CREATE end-to-end
- [ ] Legacy `handle_document_upload()` entfernt

### Observability

- [ ] Alle Operationen emittieren Langfuse-Spans mit korrektem Naming (`document.<operation>.<node>`)
- [ ] Strukturierte Logs enthalten `tenant_id`, `trace_id`, `operation`, `decision`
- [ ] Error-Transitions mit `severity="error"` bei Fehlern

### Documentation

- [ ] `ai_core/graphs/README.md` aktualisiert mit `document_service_graph`
- [ ] `docs/architecture/overview.md` erwähnt Technical Document CRUD Graph
- [ ] Diese Spezifikation als "implemented" markiert

---

## Nicht-Ziele (Explicitly Out of Scope)

- **READ-Operationen:** Bleiben direkter Repository-Call (zu simpel für Graph)
- **Bulk-Operations:** Keine spezielle Batch-Unterstützung (loop über Single-Operations)
- **Versionierung:** Kein Versions-Konzept für Dokumente (Future)
- **Rollback:** Keine Transactional Rollback-Logik (Soft-Delete erlaubt manuelle Wiederherstellung)

---

## Offene Fragen

- **Q1:** Soll DELETE auch Soft-Delete unterstützen (ohne Embedding-Deletion)?
  - **A:** Ja, via `delete_payload.hard_delete` Flag (default: `True`)

- **Q2:** Soll CREATE synchron warten auf Ingestion-Completion?
  - **A:** Nein, bleibt asynchron (wie aktuell). Graph returned sofort mit `ingestion_run_id`.

- **Q3:** Welche Permissions für DELETE?
  - **A:** Wie aktuell: Service Key (RAG_INTERNAL_KEYS) ODER Tenant Admin ODER Org Admin

---

## Code-Pointer Übersicht

### DELETE Migration

| Komponente | Before | After |
|------------|--------|-------|
| View Entry | `ai_core/views.py:2014-2101` | `ai_core/views.py:RagHardDeleteAdminView` → Graph dispatch |
| Task | `ai_core/rag/hard_delete.py:164-291` | **ENTFERNT** (Logik in Graph-Knoten) |
| Authorization | `ai_core/rag/hard_delete.py:50-97` | `ai_core/nodes/document_service/authorize_delete.py` |
| Vector Deletion | `ai_core/rag/hard_delete.py:140-157` | `ai_core/nodes/document_service/delete_embeddings.py` |
| Graph | **N/A** | `ai_core/graphs/technical/document_service_graph.py` |

### CREATE Migration

| Komponente | Before | After |
|------------|--------|-------|
| View Entry | `ai_core/views.py:1767-1840` | `ai_core/views.py:RagUploadView` → Graph dispatch |
| Service | `ai_core/services/document_upload.py:39-337` | **ENTFERNT** (Logik in Graph-Knoten) |
| Collection Ensure | `ai_core/services/document_upload.py:136-159` | `ai_core/nodes/document_service/ensure_collection.py` |
| Metadata Persist | `ai_core/services/document_upload.py:252-296` | `ai_core/nodes/document_service/persist_metadata.py` |
| Celery Dispatch | `ai_core/services/document_upload.py:299-325` | `ai_core/nodes/document_service/dispatch_ingestion.py` |
| Graph | **N/A** | `ai_core/graphs/technical/document_service_graph.py` |

---

## Referenzen

- **Architectural Analysis:** `docs/audit/architecture-anti-patterns-2025-12-31.md`
- **Graph Patterns:** `ai_core/graphs/README.md`
- **Universal Ingestion:** `ai_core/graphs/technical/universal_ingestion_graph.py`
- **Tool Contracts:** `ai_core/tool_contracts/base.py`
