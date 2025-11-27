# Unified Document Lifecycle Architecture Plan

**Status**: Pre-MVP Implementation Plan
**Version**: 1.0
**Datum**: 2025-01-27
**Breaking Changes**: ✅ Erlaubt (Pre-MVP, keine Production-Daten)
**DB Reset**: ✅ Erforderlich

---

## Executive Summary

**Problem**: Fragmentierte Document Ingestion über `service_facade`, `crawler_runner` und `crawler_ingestion_graph` mit inkonsistenter Vector Store Synchronisation und fehlendem zentralem Lifecycle Tracking.

**Lösung**: Zentralisierung über `DocumentDomainService` mit einheitlicher API, durchgängigem Lifecycle State Management und garantierter Vector Store Konsistenz.

**Scope**: Pre-MVP Architektur-Konsolidierung ohne Legacy-Ballast.

---

## Track 1: Architektur-Prinzipien

### Kernprinzipien

1. **Single Source of Truth**
   - `DocumentDomainService` ist die einzige Autorität für Document Operations
   - Alle Ingestion-Pfade (Crawler, Manual, API) nutzen denselben Entry Point
   - Kein direkter DB-Zugriff außerhalb des Domain Service

2. **Lifecycle State Management**
   - Jedes Document hat einen expliziten `lifecycle_state`
   - State Transitions werden in separater Historie getrackt
   - States orientieren sich an realen Pipeline-Phasen

3. **Vector Store Synchronität**
   - Domain Service garantiert Sync zwischen Django DB und pgvector
   - Dispatcher-Pattern für transaktionssichere Enqueuing
   - Keine orphaned vectors oder missing collections

4. **Collection Management**
   - Collections werden zentral über Domain Service verwaltet
   - Idempotente `ensure_collection` Operation
   - Lifecycle-Tracking auch für Collections (optional, nicht MVP-kritisch)

5. **Testability First**
   - Domain Service ist ohne DB/Vector Store testbar (Dependency Injection)
   - Klare Boundaries für Integration Tests
   - Dev APIs für manuelles Testing

---

## Track 2: Konkrete Implementierung

### Etappe 1: Foundation (Lifecycle State + Domain Service Core)

**Ziel**: Lifecycle State am Document verankern, Domain Service minimal erweitern.

#### 1.1 Lifecycle State Enum (MVP-Scope)

**Datei**: `documents/lifecycle.py` (neu)

```python
from enum import Enum

class DocumentLifecycleState(str, Enum):
    """Simplified lifecycle states for MVP."""
    PENDING = "pending"          # Initial state after creation
    INGESTING = "ingesting"      # Chunking + Embedding in progress
    EMBEDDED = "embedded"        # Embeddings created, upsert pending
    ACTIVE = "active"            # Ready for retrieval
    FAILED = "failed"            # Processing error occurred
    DELETED = "deleted"          # Soft-deleted

# State Transitions (validiert in Domain Service)
VALID_TRANSITIONS = {
    DocumentLifecycleState.PENDING: {INGESTING, FAILED, DELETED},
    DocumentLifecycleState.INGESTING: {EMBEDDED, FAILED, DELETED},
    DocumentLifecycleState.EMBEDDED: {ACTIVE, FAILED, DELETED},
    DocumentLifecycleState.ACTIVE: {INGESTING, DELETED},  # Re-ingestion
    DocumentLifecycleState.FAILED: {PENDING, DELETED},    # Retry
    DocumentLifecycleState.DELETED: set(),                # Terminal
}
```

**Begründung**: 6 States decken alle MVP-Flows ab, keine Collection-States (nicht kritisch).

#### 1.2 Model Changes

**Datei**: `documents/models.py`

```python
# BREAKING: Add lifecycle_state to Document
class Document(models.Model):
    # ... existing fields ...

    # NEW: Lifecycle tracking
    lifecycle_state = models.CharField(
        max_length=32,
        default="pending",
        db_index=True,
    )
    lifecycle_updated_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        indexes = [
            # ... existing ...
            models.Index(
                fields=["tenant", "lifecycle_state"],
                name="doc_tenant_state_idx"
            ),
        ]

# OPTIONAL: Lifecycle history tracking (später, nicht Etappe 1)
# class DocumentLifecycleHistory(models.Model):
#     document = models.ForeignKey(Document, ...)
#     from_state = models.CharField(...)
#     to_state = models.CharField(...)
#     changed_at = models.DateTimeField(auto_now_add=True)
#     reason = models.TextField(blank=True)
```

**Migration**:
```bash
python manage.py makemigrations documents --name add_document_lifecycle_state
python manage.py migrate
```

**Pre-MVP**: Kein Data Migration nötig (DB wird resettet).

#### 1.3 Domain Service API (Kern)

**Datei**: `documents/domain_service.py` (erweitern)

```python
class DocumentDomainService:
    """BREAKING: Simplified API with lifecycle management."""

    # --- EXISTING (keep as-is) ---
    def __init__(
        self,
        *,
        ingestion_dispatcher: IngestionDispatcher | None = None,
        deletion_dispatcher: DeletionDispatcher | None = None,
        vector_store: object | None = None,
        allow_missing_ingestion_dispatcher_for_tests: bool = False,
    ) -> None:
        # ... existing implementation ...

    # --- NEW: Lifecycle Management ---
    def update_lifecycle_state(
        self,
        *,
        document: Document,
        new_state: DocumentLifecycleState,
        reason: str | None = None,
        validate_transition: bool = True,
    ) -> None:
        """Update document lifecycle state with optional validation."""
        from .lifecycle import VALID_TRANSITIONS

        if validate_transition:
            current = DocumentLifecycleState(document.lifecycle_state)
            if new_state not in VALID_TRANSITIONS.get(current, set()):
                raise ValueError(
                    f"Invalid transition: {current} -> {new_state}"
                )

        document.lifecycle_state = new_state.value
        document.lifecycle_updated_at = timezone.now()
        document.save(update_fields=["lifecycle_state", "lifecycle_updated_at"])

        logger.info(
            "document_lifecycle_updated",
            extra={
                "document_id": str(document.id),
                "tenant_id": str(document.tenant_id),
                "new_state": new_state.value,
                "reason": reason,
            },
        )

    # --- MODIFIED: ingest_document mit lifecycle ---
    def ingest_document(
        self,
        *,
        tenant: Tenant,
        source: str,
        content_hash: str,
        metadata: Mapping[str, object] | None = None,
        collections: Iterable[str | UUID | DocumentCollection] = (),
        embedding_profile: str | None = None,
        scope: str | None = None,
        dispatcher: IngestionDispatcher | None = None,
        document_id: UUID | None = None,
        initial_lifecycle_state: str = "pending",  # NEW
        allow_missing_ingestion_dispatcher_for_tests: bool | None = None,
    ) -> PersistedDocumentIngest:
        """Persist or update document with lifecycle tracking."""
        # ... existing implementation ...

        # NEW: Set initial lifecycle state
        with transaction.atomic():
            document, created = Document.objects.update_or_create(
                tenant=tenant,
                source=source,
                hash=content_hash,
                defaults={
                    "metadata": metadata_payload,
                    "lifecycle_state": initial_lifecycle_state,  # NEW
                    "lifecycle_updated_at": timezone.now(),      # NEW
                    **({"id": document_id} if document_id is not None else {}),
                },
            )
            # ... rest of existing implementation ...
```

**Änderungen**:
- `update_lifecycle_state()`: Zentrale Methode für State Updates
- `ingest_document()`: Bekommt `initial_lifecycle_state` Parameter (default: "pending")
- Keine überladene API mit 10+ Parametern

---

### Etappe 2: Service Facade Konsolidierung

**Ziel**: `service_facade` konsequent über Domain Service führen, alte DB-Zugriffe entfernen.

#### 2.1 Refactor service_facade.py

**Datei**: `documents/service_facade.py`

**BREAKING Changes**:
```python
# REMOVE: _DELETE_OUTBOX, _COLLECTION_OUTBOX (verwenden Domain Service statt)

def ingest_document(
    scope: ScopeContext,
    *,
    meta: Mapping[str, Any],
    chunks_path: str,
    embedding_state: Mapping[str, Any] | None = None,
    dispatcher: IngestionDispatcher | None = None,
) -> MutableMapping[str, Any]:
    """Simplified facade routing through domain service."""

    # Resolve tenant
    tenant = TenantContext.resolve_identifier(
        meta.get("tenant_id") or scope.tenant_id,
        allow_pk=True
    )

    # Prepare collections
    collection_identifier = meta.get("collection_id")
    collections: tuple[str, ...] = (str(collection_identifier),) if collection_identifier else ()

    # Build metadata
    metadata = dict(meta)
    metadata["chunks_path"] = chunks_path

    # Domain Service Call
    service = DocumentDomainService(ingestion_dispatcher=dispatcher)

    with schema_context(scope.tenant_schema) if scope.tenant_schema else nullcontext():
        result = service.ingest_document(
            tenant=tenant,
            source=str(meta.get("source") or "unknown"),
            content_hash=str(meta.get("hash") or meta.get("content_hash")),
            metadata=metadata,
            collections=collections,
            embedding_profile=meta.get("embedding_profile"),
            scope=meta.get("scope"),
            dispatcher=dispatcher,
            document_id=UUID(str(meta["document_id"])) if meta.get("document_id") else None,
            initial_lifecycle_state="ingesting",  # NEW: Set state explicitly
        )

    return {
        "status": "queued",
        "document_id": str(result.document.id),
        "collection_ids": [str(cid) for cid in result.collection_ids],
        "lifecycle_state": result.document.lifecycle_state,  # NEW
        "trace_id": scope.trace_id,
        "ingestion_run_id": scope.ingestion_run_id,
    }

def delete_document(
    scope: ScopeContext,
    document_ids: Sequence[object],
    *,
    reason: str,
    ticket_ref: str,
    actor: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Simplified delete routing through domain service."""

    # ... existing validation ...

    # Use Domain Service instead of inline logic
    from ai_core.rag.vector_client import get_default_client

    service = DocumentDomainService(
        vector_store=get_default_client(),
        deletion_dispatcher=lambda payload: _DELETE_OUTBOX.append(payload),  # Keep for now
    )

    with schema_context(scope.tenant_schema) if scope.tenant_schema else nullcontext():
        documents = Document.objects.filter(id__in=normalized_ids, tenant_id=tenant_uuid)
        for document in documents:
            service.delete_document(document, reason=reason)

    # ... rest of existing implementation ...
```

**Änderungen**:
- Alle DB-Operationen gehen durch `DocumentDomainService`
- Lifecycle State wird explizit gesetzt (`"ingesting"` bei Ingestion)
- Outbox-Pattern bleibt vorerst (Migration in Etappe 4)

---

### Etappe 3: Crawler Runner auf Bulk Ingestion

**Ziel**: Crawler ruft nicht mehr direkt Models an, sondern gibt Specs an Domain Service.

#### 3.1 Bulk Ingestion API

**Datei**: `documents/domain_service.py` (erweitern)

```python
@dataclass(frozen=True)
class DocumentIngestSpec:
    """Specification for bulk document ingestion."""
    source: str
    content_hash: str
    metadata: Mapping[str, object]
    collections: Sequence[str | UUID] = ()
    embedding_profile: str | None = None
    scope: str | None = None
    document_id: UUID | None = None

@dataclass(frozen=True)
class BulkIngestResult:
    """Result of bulk ingestion operation."""
    ingested: list[PersistedDocumentIngest]
    failed: list[tuple[DocumentIngestSpec, Exception]]
    total: int
    succeeded: int

class DocumentDomainService:
    # ... existing methods ...

    def bulk_ingest_documents(
        self,
        *,
        tenant: Tenant,
        documents: Sequence[DocumentIngestSpec],
        dispatcher: IngestionDispatcher | None = None,
    ) -> BulkIngestResult:
        """Bulk ingestion for crawler runs (optimized)."""

        ingested: list[PersistedDocumentIngest] = []
        failed: list[tuple[DocumentIngestSpec, Exception]] = []

        # Pre-fetch collections (optimization)
        all_collection_keys = {
            str(col)
            for spec in documents
            for col in spec.collections
        }
        collection_cache = {
            col.key: col
            for col in DocumentCollection.objects.filter(
                tenant=tenant,
                key__in=all_collection_keys
            )
        }

        for spec in documents:
            try:
                # Resolve collections from cache
                collection_instances = [
                    collection_cache.get(str(col))
                    or self.ensure_collection(
                        tenant=tenant,
                        key=str(col),
                        embedding_profile=spec.embedding_profile,
                        scope=spec.scope,
                    )
                    for col in spec.collections
                ]

                result = self.ingest_document(
                    tenant=tenant,
                    source=spec.source,
                    content_hash=spec.content_hash,
                    metadata=spec.metadata,
                    collections=collection_instances,
                    embedding_profile=spec.embedding_profile,
                    scope=spec.scope,
                    dispatcher=dispatcher,
                    document_id=spec.document_id,
                    initial_lifecycle_state="pending",
                )
                ingested.append(result)

            except Exception as exc:
                failed.append((spec, exc))
                logger.exception(
                    "bulk_ingest_document_failed",
                    extra={"source": spec.source, "hash": spec.content_hash},
                )

        return BulkIngestResult(
            ingested=ingested,
            failed=failed,
            total=len(documents),
            succeeded=len(ingested),
        )
```

#### 3.2 Crawler Runner Refactoring

**Datei**: `ai_core/services/crawler_runner.py`

```python
# REMOVE: _register_documents_for_builds (direkter DB-Zugriff)
# REMOVE: _ensure_collection_with_warning (fragmentierte Logic)

def run_crawler_runner(
    *,
    meta: dict[str, Any],
    request_model: CrawlerRunRequest,
    lifecycle_store: object | None,
    graph_factory: Callable[[], object] | None = None,
) -> CrawlerRunnerCoordinatorResult:
    """Refactored to use domain service bulk ingestion."""

    # ... existing setup ...

    tenant = _resolve_tenant(meta.get("tenant_id"))
    if not tenant:
        raise ValueError("tenant_not_resolved")

    # Build ingestion specs from crawler builds
    ingest_specs: list[DocumentIngestSpec] = []

    for build in state_builds:
        normalized = build.state.get("normalized_document_input")
        if not isinstance(normalized, Mapping):
            continue

        metadata = dict(normalized.get("meta") or {})

        blob_payload = normalized.get("blob") if isinstance(normalized, Mapping) else None
        checksum = blob_payload.get("sha256") if isinstance(blob_payload, Mapping) else None
        checksum = checksum or normalized.get("checksum")

        if not checksum:
            logger.warning("crawler_runner.missing_checksum", extra={"origin": build.origin})
            continue

        source = metadata.get("origin_uri") or build.origin
        collection_id = build.collection_id or normalized.get("ref", {}).get("collection_id")

        ingest_specs.append(
            DocumentIngestSpec(
                source=str(source),
                content_hash=str(checksum),
                metadata=metadata,
                collections=[str(collection_id)] if collection_id else [],
                embedding_profile=request_model.embedding_profile,
                scope=request_model.scope,
            )
        )

    # Bulk ingest via domain service
    from documents.domain_service import DocumentDomainService
    from ai_core.rag.vector_client import get_default_client

    service = DocumentDomainService(
        vector_store=get_default_client(),
        ingestion_dispatcher=lambda *_: None,  # No immediate dispatch
    )

    with schema_context(tenant.schema_name):
        bulk_result = service.bulk_ingest_documents(
            tenant=tenant,
            documents=ingest_specs,
        )

    # Update state_builds with ingested document IDs
    for i, ingested in enumerate(bulk_result.ingested):
        if i < len(state_builds):
            state_builds[i].document_id = str(ingested.document.id)
            state_builds[i].state["document_id"] = str(ingested.document.id)

    # ... rest of existing implementation (task submission) ...
```

**Änderungen**:
- Direkter DB-Zugriff entfernt
- Bulk Ingestion über `DocumentDomainService.bulk_ingest_documents()`
- Collection Pre-Fetching im Service (Performance-Optimierung)

---

### Etappe 4: Collection Service Auslagern

**Ziel**: Dedizierte `CollectionService` für Collection-spezifische Logik, alte Hilfsfunktionen ablösen.

#### 4.1 Collection Service

**Datei**: `documents/collection_service.py` (neu)

```python
from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID
import hashlib

if TYPE_CHECKING:
    from customers.models import Tenant
    from .models import DocumentCollection

from .domain_service import DocumentDomainService

class CollectionType:
    """Simple collection type constants (MVP)."""
    SYSTEM = "system"      # System-managed collections
    USER = "user"          # User-created collections

class CollectionService:
    """Dedicated service for collection lifecycle management."""

    def __init__(
        self,
        *,
        domain_service: DocumentDomainService | None = None,
        vector_client: object | None = None,
    ):
        from ai_core.rag.vector_client import get_default_client

        self._domain = domain_service or DocumentDomainService(
            vector_store=vector_client or get_default_client()
        )

    def ensure_collection(
        self,
        *,
        tenant: Tenant,
        key: str,
        name: str | None = None,
        embedding_profile: str | None = None,
        scope: str | None = None,
        collection_type: str = CollectionType.USER,
        **kwargs,
    ) -> DocumentCollection:
        """Unified collection ensure with type tracking."""

        metadata = kwargs.get("metadata") or {}
        metadata["collection_type"] = collection_type

        return self._domain.ensure_collection(
            tenant=tenant,
            key=key,
            name=name or key,
            embedding_profile=embedding_profile,
            scope=scope,
            metadata=metadata,
            **kwargs,
        )

    def ensure_manual_collection(
        self,
        tenant_id: str,
    ) -> str:
        """Specialized for manual uploads (deterministic UUID)."""
        from customers.tenant_context import TenantContext

        tenant = TenantContext.resolve_identifier(tenant_id, allow_pk=True)

        # Deterministic UUID from tenant_id
        slug = "manual-uploads"
        namespace = UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # RFC 4122 namespace
        collection_uuid = UUID(hashlib.sha1(f"{tenant_id}:{slug}".encode()).hexdigest()[:32])

        collection = self.ensure_collection(
            tenant=tenant,
            key=slug,
            name="Manual Uploads",
            collection_type=CollectionType.SYSTEM,
            collection_id=collection_uuid,
        )

        return str(collection.collection_id)
```

#### 4.2 Migration ai_core/rag/collections.py

**Datei**: `ai_core/rag/collections.py` (refactor)

```python
# DEPRECATED: Redirect to CollectionService

from documents.collection_service import CollectionService

def ensure_manual_collection(tenant_id: str) -> str:
    """DEPRECATED: Use CollectionService.ensure_manual_collection()."""
    import warnings
    warnings.warn(
        "ensure_manual_collection is deprecated, use CollectionService",
        DeprecationWarning,
        stacklevel=2,
    )

    service = CollectionService()
    return service.ensure_manual_collection(tenant_id)

# Keep MANUAL_COLLECTION_SLUG, MANUAL_COLLECTION_LABEL for backwards compat
MANUAL_COLLECTION_SLUG = "manual-uploads"
MANUAL_COLLECTION_LABEL = "Manual Uploads"
```

---

### Etappe 5: Dev APIs & Health Checks

**Ziel**: Dev Endpoints für manuelles Testing, Health Checks für Lifecycle-Monitoring.

#### 5.1 Dev API Endpoints

**Datei**: `documents/dev_api.py` (neu, nur DEBUG)

```python
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from django.conf import settings
from django_tenants.utils import schema_context

from customers.tenant_context import TenantContext
from .domain_service import DocumentDomainService, DocumentIngestSpec
from .collection_service import CollectionService
from ai_core.rag.vector_client import get_default_client

if not settings.DEBUG:
    raise RuntimeError("dev_api module should only be imported in DEBUG mode")

class DocumentDevViewSet(viewsets.ViewSet):
    """Dev-only endpoints for manual testing (DEBUG mode only)."""

    @action(detail=False, methods=['post'])
    def ingest(self, request, tenant_id):
        """POST /api/dev/documents/{tenant_id}/ingest

        Body:
        {
          "source": "test-source",
          "content_hash": "abc123",
          "metadata": {"title": "Test Doc"},
          "collections": ["collection-key-1"],
          "embedding_profile": "standard",
          "scope": "test"
        }
        """
        tenant = TenantContext.resolve_identifier(tenant_id, allow_pk=True)

        service = DocumentDomainService(
            vector_store=get_default_client(),
            allow_missing_ingestion_dispatcher_for_tests=True,
        )

        with schema_context(tenant.schema_name):
            result = service.ingest_document(
                tenant=tenant,
                source=request.data["source"],
                content_hash=request.data["content_hash"],
                metadata=request.data.get("metadata", {}),
                collections=request.data.get("collections", []),
                embedding_profile=request.data.get("embedding_profile"),
                scope=request.data.get("scope"),
                initial_lifecycle_state="pending",
            )

        return Response({
            "document_id": str(result.document.id),
            "collection_ids": [str(c) for c in result.collection_ids],
            "lifecycle_state": result.document.lifecycle_state,
            "hash": result.document.hash,
        }, status=status.HTTP_201_CREATED)

    @action(detail=False, methods=['get'])
    def get_document(self, request, tenant_id, document_id):
        """GET /api/dev/documents/{tenant_id}/{document_id}"""
        from .models import Document

        tenant = TenantContext.resolve_identifier(tenant_id, allow_pk=True)

        with schema_context(tenant.schema_name):
            doc = Document.objects.get(id=document_id, tenant=tenant)

        return Response({
            "id": str(doc.id),
            "source": doc.source,
            "hash": doc.hash,
            "lifecycle_state": doc.lifecycle_state,
            "lifecycle_updated_at": doc.lifecycle_updated_at,
            "metadata": doc.metadata,
            "created_at": doc.created_at,
        })

class CollectionDevViewSet(viewsets.ViewSet):
    """Dev-only collection endpoints."""

    @action(detail=False, methods=['post'])
    def ensure(self, request, tenant_id):
        """POST /api/dev/collections/{tenant_id}/ensure

        Body:
        {
          "key": "test-collection",
          "name": "Test Collection",
          "embedding_profile": "standard",
          "scope": "test"
        }
        """
        tenant = TenantContext.resolve_identifier(tenant_id, allow_pk=True)
        service = CollectionService()

        with schema_context(tenant.schema_name):
            collection = service.ensure_collection(
                tenant=tenant,
                key=request.data["key"],
                name=request.data.get("name"),
                embedding_profile=request.data.get("embedding_profile"),
                scope=request.data.get("scope"),
            )

        return Response({
            "id": str(collection.id),
            "collection_id": str(collection.collection_id),
            "key": collection.key,
            "name": collection.name,
        })
```

**Datei**: `documents/dev_urls.py` (neu)

```python
from django.urls import path
from .dev_api import DocumentDevViewSet, CollectionDevViewSet

urlpatterns = [
    path(
        '<str:tenant_id>/ingest',
        DocumentDevViewSet.as_view({'post': 'ingest'}),
        name='dev-document-ingest',
    ),
    path(
        '<str:tenant_id>/<uuid:document_id>',
        DocumentDevViewSet.as_view({'get': 'get_document'}),
        name='dev-document-get',
    ),
    path(
        'collections/<str:tenant_id>/ensure',
        CollectionDevViewSet.as_view({'post': 'ensure'}),
        name='dev-collection-ensure',
    ),
]
```

**Datei**: `noesis2/urls.py` (erweitern)

```python
# ... existing imports ...

urlpatterns = [
    # ... existing patterns ...
]

# Dev APIs (nur DEBUG)
if settings.DEBUG:
    urlpatterns += [
        path('api/dev/documents/', include('documents.dev_urls')),
    ]
```

#### 5.2 Health Check

**Datei**: `documents/health.py` (neu)

```python
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.utils import timezone
from django.db import connection

@api_view(['GET'])
def document_lifecycle_health(request):
    """GET /api/health/document-lifecycle

    Checks:
    - Domain service availability
    - Database connectivity
    - Lifecycle state consistency
    """

    checks = {}

    # Check 1: Database connectivity
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
        checks["database"] = {"status": "healthy"}
    except Exception as exc:
        checks["database"] = {"status": "unhealthy", "error": str(exc)}

    # Check 2: Document model availability
    try:
        from .models import Document
        count = Document.objects.count()
        checks["document_model"] = {"status": "healthy", "total_documents": count}
    except Exception as exc:
        checks["document_model"] = {"status": "unhealthy", "error": str(exc)}

    # Check 3: Lifecycle states distribution
    try:
        from .models import Document
        from django.db.models import Count

        distribution = dict(
            Document.objects.values("lifecycle_state")
            .annotate(count=Count("id"))
            .values_list("lifecycle_state", "count")
        )
        checks["lifecycle_distribution"] = {
            "status": "healthy",
            "states": distribution,
        }
    except Exception as exc:
        checks["lifecycle_distribution"] = {"status": "unhealthy", "error": str(exc)}

    all_healthy = all(c.get("status") == "healthy" for c in checks.values())

    return Response({
        "status": "healthy" if all_healthy else "degraded",
        "checks": checks,
        "timestamp": timezone.now().isoformat(),
    })
```

**URL Registration**:

```python
# noesis2/urls.py
urlpatterns += [
    path('api/health/document-lifecycle', document_lifecycle_health),
]
```

---

## Umsetzungs-Checkliste

### Etappe 1: Foundation (1-2 Tage)
- [ ] Lifecycle State Enum erstellen (`documents/lifecycle.py`)
- [ ] Model Migration für `Document.lifecycle_state` erstellen
- [ ] DB Reset durchführen (`npm run dev:reset`)
- [ ] Domain Service um `update_lifecycle_state()` erweitern
- [ ] Bestehende Tests anpassen (lifecycle state berücksichtigen)
- [ ] Unit Tests für Lifecycle Transitions

**Checkpoint**: Migration läuft sauber, Tests grün, Domain Service hat Lifecycle-API.

### Etappe 2: Service Facade (0.5-1 Tag)
- [ ] `service_facade.py` refactoren (kein direkter DB-Zugriff)
- [ ] `ingest_document()` nutzt Domain Service
- [ ] `delete_document()` nutzt Domain Service
- [ ] Integration Tests für Service Facade

**Checkpoint**: Service Facade ist nur noch dünne Schicht über Domain Service.

### Etappe 3: Crawler Integration (1-2 Tage)
- [ ] `DocumentIngestSpec` Dataclass erstellen
- [ ] `bulk_ingest_documents()` in Domain Service implementieren
- [ ] `crawler_runner.py` auf Bulk Ingestion umstellen
- [ ] `_register_documents_for_builds()` entfernen
- [ ] Integration Tests für Crawler Flow

**Checkpoint**: Crawler verwendet Bulk Ingestion, keine direkten Model-Calls.

### Etappe 4: Collection Service (0.5-1 Tag)
- [ ] `collection_service.py` erstellen
- [ ] `ensure_manual_collection()` migrieren
- [ ] `ai_core/rag/collections.py` auf deprecated setzen
- [ ] Alle Caller auf `CollectionService` umstellen
- [ ] Tests für Collection Service

**Checkpoint**: Alle Collection-Operationen über `CollectionService`.

### Etappe 5: Dev APIs & Health (0.5 Tag)
- [ ] `dev_api.py` erstellen (nur DEBUG)
- [ ] Dev URLs registrieren
- [ ] Health Check Endpoint implementieren
- [ ] Manual Testing Guide schreiben

**Checkpoint**: Dev APIs funktionieren, Health Check liefert sinnvolle Daten.

---

## Dev URLs für manuelles Testing

```bash
# Document Operations (nur DEBUG)
POST   http://localhost:8000/api/dev/documents/{tenant_id}/ingest
GET    http://localhost:8000/api/dev/documents/{tenant_id}/{document_id}

# Collection Operations (nur DEBUG)
POST   http://localhost:8000/api/dev/collections/{tenant_id}/ensure

# Health Check (immer verfügbar)
GET    http://localhost:8000/api/health/document-lifecycle
```

**Beispiel Requests**:

```bash
# 1. Collection erstellen
curl -X POST http://localhost:8000/api/dev/collections/TENANT_ID/ensure \
  -H "Content-Type: application/json" \
  -d '{
    "key": "test-collection",
    "name": "Test Collection",
    "embedding_profile": "standard"
  }'

# 2. Document ingesten
curl -X POST http://localhost:8000/api/dev/documents/TENANT_ID/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "source": "manual-test",
    "content_hash": "abc123",
    "metadata": {"title": "Test Doc"},
    "collections": ["test-collection"],
    "embedding_profile": "standard"
  }'

# 3. Document Status prüfen
curl http://localhost:8000/api/dev/documents/TENANT_ID/DOCUMENT_ID

# 4. Health Check
curl http://localhost:8000/api/health/document-lifecycle
```

---

## Testing Strategy

### Unit Tests

**Fokus**: Domain Service Logic, Lifecycle Transitions

```python
# tests/documents/test_domain_service_lifecycle.py

def test_update_lifecycle_state_validates_transitions(tenant):
    service = DocumentDomainService(allow_missing_ingestion_dispatcher_for_tests=True)

    doc = Document.objects.create(
        tenant=tenant,
        source="test",
        hash="abc",
        lifecycle_state="pending",
    )

    # Valid transition
    service.update_lifecycle_state(doc, DocumentLifecycleState.INGESTING)
    assert doc.lifecycle_state == "ingesting"

    # Invalid transition
    with pytest.raises(ValueError):
        service.update_lifecycle_state(doc, DocumentLifecycleState.PENDING)

def test_bulk_ingest_handles_partial_failures(tenant, vector_store_stub):
    service = DocumentDomainService(
        vector_store=vector_store_stub,
        allow_missing_ingestion_dispatcher_for_tests=True,
    )

    specs = [
        DocumentIngestSpec(source="s1", content_hash="h1", metadata={}),
        DocumentIngestSpec(source="s2", content_hash="invalid", metadata={}),  # Will fail
    ]

    result = service.bulk_ingest_documents(tenant=tenant, documents=specs)

    assert result.succeeded == 1
    assert len(result.failed) == 1
```

### Integration Tests

**Fokus**: End-to-End Flows (Crawler → Domain Service → Vector Store)

```python
# tests/integration/test_crawler_to_vector.py

@pytest.mark.django_db
def test_crawler_ingestion_sets_lifecycle_states(tenant, mock_vector_client):
    # Setup crawler request
    request = CrawlerRunRequest(
        origins=["https://example.com"],
        embedding_profile="standard",
    )

    # Run crawler
    result = run_crawler_runner(
        meta={"tenant_id": str(tenant.id)},
        request_model=request,
        lifecycle_store=None,
    )

    # Verify document was created with correct lifecycle state
    doc = Document.objects.get(tenant=tenant)
    assert doc.lifecycle_state == "pending"

    # Verify vector store was called
    assert len(mock_vector_client.ensure_collection_calls) > 0
```

---

## Architektur-Diagramme

### Document Ingestion Flow (Nach Migration)

```
┌─────────────┐
│   Crawler   │
│   Manual    │───┐
│   API       │   │
└─────────────┘   │
                  ▼
         ┌────────────────────┐
         │  Service Facade    │ (dünn)
         │  - ingest_document │
         └────────────────────┘
                  │
                  ▼
         ┌────────────────────────────┐
         │  DocumentDomainService     │
         │  - ingest_document         │
         │  - bulk_ingest_documents   │
         │  - update_lifecycle_state  │
         │  - ensure_collection       │
         └────────────────────────────┘
                  │
       ┌──────────┴──────────┐
       ▼                     ▼
┌─────────────┐      ┌──────────────┐
│  Django DB  │      │ Vector Store │
│  - Document │      │  - pgvector  │
│  - Collection      │  - Embeddings│
└─────────────┘      └──────────────┘
```

### Lifecycle State Machine (MVP)

```
    ┌─────────┐
    │ PENDING │◄────┐
    └────┬────┘     │
         │          │
         ▼          │
   ┌───────────┐   │
   │ INGESTING │   │
   └─────┬─────┘   │
         │          │
         ▼          │
   ┌──────────┐    │
   │ EMBEDDED │    │
   └─────┬────┘    │
         │          │
         ▼          │
    ┌────────┐     │
    │ ACTIVE │─────┘ (Re-ingestion)
    └────────┘
         │
         ▼
    ┌─────────┐
    │ DELETED │ (Terminal)
    └─────────┘

    (Alle States → FAILED möglich)
```

---

## Risiken & Mitigations

| Risiko | Wahrscheinlichkeit | Impact | Mitigation |
|--------|-------------------|--------|------------|
| Breaking Changes brechen Crawler Flow | Mittel | Hoch | Integration Tests vor Merge, schrittweise Migration |
| Vector Store Desync nach Migration | Niedrig | Hoch | Health Checks, Sync-Verification Script |
| Performance-Regression bei Bulk Ingestion | Niedrig | Mittel | Benchmark Tests, Collection Pre-Fetching |
| Dev APIs in Production deployed | Sehr niedrig | Hoch | Nur unter `settings.DEBUG`, Import-Guard |
| Zu viele Lifecycle States später nicht genutzt | Mittel | Niedrig | MVP-Scope (6 States), später erweitern |

---

## Success Criteria

### Funktional
- ✅ Alle Ingestion-Pfade (Crawler, Manual, API) nutzen `DocumentDomainService`
- ✅ Lifecycle State wird für jedes Document getrackt
- ✅ Collections werden zentral über `CollectionService` verwaltet
- ✅ Vector Store Sync läuft über Domain Service Dispatcher
- ✅ Keine direkten DB-Zugriffe außerhalb Domain Service

### Technisch
- ✅ Alle Tests grün (Unit + Integration)
- ✅ Health Check zeigt "healthy" Status
- ✅ Dev APIs funktionieren für manuelles Testing
- ✅ Keine Performance-Regression (< 10ms Overhead)

### Dokumentation
- ✅ Dieser Plan im `docs/` Verzeichnis
- ✅ Manual Testing Guide verfügbar
- ✅ Lifecycle State Machine dokumentiert
- ✅ `AGENTS.md` aktualisiert mit Referenz auf diesen Plan

---

## Nächste Schritte

1. **Review & Approval**: Plan mit Team reviewen, Feedback einarbeiten
2. **Kick-off Etappe 1**: Lifecycle State Model & Migration
3. **Checkpoint nach jeder Etappe**: Tests grün, Integration funktioniert
4. **Manual Testing**: Nach Etappe 5 mit Dev APIs
5. **Dokumentation finalisieren**: Nach vollständiger Migration

---

**Maintainer**: AI Platform Team
**Review Cycle**: Nach jeder Etappe
**Zuletzt aktualisiert**: 2025-01-27
