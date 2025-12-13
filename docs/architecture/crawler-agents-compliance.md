# Crawler Architecture vs AGENTS.md Compliance Check

## 🎯 Compliance Analysis

**Question**: Ist die vorgeschlagene Crawler-Architektur konform mit AGENTS.md?

**Answer**: ✅ **JA, aber mit wichtigen Anpassungen nötig!**

---

## 📋 AGENTS.md Key Requirements

### **1. Document Lifecycle (Unified Architecture)**

**AGENTS.md Lines 41-65:**

```
- DocumentDomainService: Zentrale Autorität für alle Document Operations
- Lifecycle States: 6 MVP-States (pending, ingesting, embedded, active, failed, deleted)
- Ingestion Flows: Alle Pfade (Crawler, Manual, API) nutzen einheitlichen Entry Point
```

**Current Proposal**: ❌ **NICHT konform**

- Ich schlage `repository.upsert()` direkt in Celery tasks vor
- ABER: AGENTS.md verlangt `DocumentDomainService` als zentrale Autorität!

**Fix**:

```python
# NICHT SO (mein Proposal):
repository = get_repository()
stored_doc = repository.upsert(doc)

# SONDERN SO (AGENTS.md konform):
domain_service = DocumentDomainService()
result = domain_service.ingest_document(
    tenant=tenant,
    source="crawler",
    content_hash=content_hash,
    metadata=metadata,
    collections=collections,
    embedding_profile=embedding_profile,
    dispatcher=dispatcher_fn,
)
```

---

### **2. API Contracts (AGENTS.md Lines 51-58)**

**Required**:

- ✅ `DocumentDomainService.ingest_document()` - Single document ingestion
- ✅ `DocumentDomainService.bulk_ingest_documents()` - Bulk ingestion (Crawler-optimiert)
- ✅ `DocumentDomainService.update_lifecycle_state()` - Lifecycle transitions

**Current Proposal**: ⚠️ **Teilweise konform**

- Ich nutze `repository.upsert()` statt `DocumentDomainService`
- Aber `bulk_ingest_documents` ist GENAU was wir brauchen für Crawler!

**Fix**: Nutze die vorhandenen Contracts!

---

### **3. Integration Points (AGENTS.md Lines 60-64)**

**AGENTS.md says**:

```
- Crawler: ai_core/services/crawler_runner.py → bulk_ingest_documents()
- Manual Upload: Frontend → Dev API → ingest_document()
```

**Current State**:

- ✅ `crawler_runner.py` existiert bereits
- ❌ Nutzt aber NICHT `bulk_ingest_documents()`
- ❌ Macht parallel registration stattdessen

**Fix**: Refactor `crawler_runner.py` to use `bulk_ingest_documents()`

---

### **4. Schichten & Verantwortlichkeiten (AGENTS.md Lines 86-106)**

**AGENTS.md Layer Model**:

```
Business/Orchestrierung → ai_core/graphs (LangGraph)
Capabilities → ai_core/nodes (wiederverwendbare Nodes)
Platform-Kernel → ai_core/llm, ai_core/infra, ai_core/middleware
```

**Current Proposal**: ⚠️ **Nicht explizit aligned**

- Ich schlage Worker/Fetch/Ingest Layer vor
- ABER: Passt nicht sauber ins AGENTS.md Modell

**Fix**: Map to AGENTS.md Layers:

```
HTTP Layer → Platform-Kernel (API Entry)
Coordinator → Business/Orchestrierung (Service Layer)
Fetch Worker → Capabilities (Reusable Component)
Ingest Task → Business/Orchestrierung (Celery Task)
Graph → Business/Orchestrierung (LangGraph)
```

---

### **5. Paketgrenzen (AGENTS.md Lines 129-135)**

**AGENTS.md Rules**:

```
services → shared (nur nach unten)
tools → services, shared
ai_core/graphs → tools, shared
Frontend ist getrennt (keine Rückimporte)
```

**Current Proposal**: ✅ **Konform**

- Coordinator in `ai_core/services` ✅
- Worker in `crawler` (shared-level) ✅
- Tasks in `crawler/tasks.py` ✅
- Graphs in `ai_core/graphs` ✅

---

### **6. Glossar & Feld-Matrix (AGENTS.md Lines 183-210)**

**Required Fields**:

- `tenant_id` (Pflicht)
- `trace_id` (Pflicht)
- `case_id` (Optional)
- `workflow_id` (Optional)
- `run_id` ODER `ingestion_run_id` (Pflicht - eine von)

**Current Proposal**: ❌ **NICHT explizit**

- Ich erwähne diese IDs nicht im Proposal

**Fix**: Alle Celery tasks MÜSSEN diese IDs mitführen!

---

## ✅ Corrected Architecture (AGENTS.md Compliant)

### **Layer 1: HTTP Entry (Platform-Kernel)**

**Location**: `ai_core/views.py`

```python
@require_POST
def fetch_for_review(request):
    """HITL Phase 1: Fetch URLs for preview."""
    # Parse request
    data = json.loads(request.body)
    
    # Prepare meta (AGENTS.md compliant)
    meta, error = _prepare_request(request)
    if error:
        return JsonResponse(error.data, status=error.status_code)
    
    # Create session
    session = ReviewSession.objects.create(
        tenant_id=meta["tenant_id"],
        case_id=meta.get("case_id"),
        urls=data["urls"],
    )
    
    # Queue Celery task
    task = fetch_urls_for_preview.delay(
        session_id=str(session.session_id),
        urls=data["urls"],
        tenant_id=meta["tenant_id"],
        case_id=meta.get("case_id"),
        trace_id=meta["trace_id"],  # AGENTS.md: trace_id is Pflicht!
    )
    
    return JsonResponse({
        "session_id": str(session.session_id),
        "task_id": task.id,
    })

@require_POST
def ingest_selected(request):
    """HITL Phase 2: Ingest selected URLs."""
    data = json.loads(request.body)
    meta, error = _prepare_request(request)
    if error:
        return JsonResponse(error.data, status=error.status_code)
    
    # Queue ingestion (with AGENTS.md fields!)
    task = ingest_selected_urls.delay(
        session_id=data["session_id"],
        selected_urls=data["selected_urls"],
        mode=data["mode"],
        tenant_id=meta["tenant_id"],
        case_id=meta.get("case_id"),
        trace_id=meta["trace_id"],
        workflow_id=data.get("workflow_id", "crawler"),
        collection_id=data.get("collection_id"),
        embedding_profile=data.get("embedding_profile"),
    )
    
    return JsonResponse({"task_id": task.id})
```

---

### **Layer 2: Celery Tasks (Business/Orchestrierung)**

**Location**: `crawler/tasks.py`

```python
from uuid import uuid4
from documents.domain_service import DocumentDomainService

@shared_task
def fetch_urls_for_preview(
    session_id: str,
    urls: list[str],
    tenant_id: str,
    case_id: str | None,
    trace_id: str,
) -> dict:
    """
    Fetch URLs for HITL preview.
    AGENTS.md compliant: Uses trace_id, tenant_id.
    """
    session = ReviewSession.objects.get(pk=session_id)
    
    for url in urls:
        try:
            # Fetch (no persistence!)
            worker = FetchWorker()
            result = worker.fetch_url(url)
            
            # Cache content
            cache_key = f"preview:{session_id}:{url}"
            redis_client.setex(cache_key, 3600, result.content_body)
            
            # Create preview
            FetchedPreview.objects.create(
                session_id=session_id,
                url=url,
                title=result.title,
                snippet=result.snippet,
                content_cache_key=cache_key,
                ...
            )
        except Exception as e:
            # Log failure
            FetchedPreview.objects.create(
                session_id=session_id,
                url=url,
                status="failed",
                error=str(e),
            )
    
    session.status = "completed"
    session.save()
    
    return {"session_id": session_id}


@shared_task
def ingest_selected_urls(
    session_id: str,
    selected_urls: list[str],
    mode: Literal["rag", "archive", "ephemeral"],
    tenant_id: str,
    case_id: str | None,
    trace_id: str,
    workflow_id: str,
    collection_id: str | None,
    embedding_profile: str | None,
) -> dict:
    """
    Ingest selected URLs using DocumentDomainService.
    AGENTS.md compliant: Uses domain service, not repository directly!
    """
    # AGENTS.md: Get tenant object (required by DocumentDomainService)
    tenant = Tenant.objects.get(schema_name=tenant_id)
    
    # AGENTS.md: Use DocumentDomainService as central authority!
    domain_service = DocumentDomainService(
        vector_store=get_default_client()
    )
    
    # Get previews
    previews = FetchedPreview.objects.filter(
        session_id=session_id,
        url__in=selected_urls,
        status="fetched"
    )
    
    results = {"completed": [], "failed": {}, "documents_created": []}
    
    for preview in previews:
        try:
            # Get cached content
            content_body = redis_client.get(preview.content_cache_key)
            if not content_body:
                raise ValueError("Cache expired")
            
            # Prepare metadata
            metadata = {
                "title": preview.title,
                "origin_uri": preview.url,
                "content_type": preview.content_type,
                "workflow_id": workflow_id,
                "case_id": case_id,
                "trace_id": trace_id,
                # AGENTS.md: external_ref structure
                "external_ref": {
                    "provider": "web",
                    "external_id": f"web::{preview.url}",
                },
            }
            
            # Hash content for idempotency
            content_hash = hashlib.sha256(content_body).hexdigest()
            
            # Get/create collection
            collections = []
            if collection_id:
                coll = domain_service.ensure_collection(
                    tenant=tenant,
                    collection_id=UUID(collection_id),
                    embedding_profile=embedding_profile,
                )
                collections.append(coll)
            
            # **AGENTS.md CONTRACT**: Use DocumentDomainService.ingest_document!
            ingest_result = domain_service.ingest_document(
                tenant=tenant,
                source="crawler",
                content_hash=content_hash,
                metadata=metadata,
                collections=collections,
                embedding_profile=embedding_profile if mode == "rag" else None,
                scope=None,
                dispatcher=lambda doc_id, coll_ids, profile, scope: (
                    # Dispatcher only fires if mode == "rag"
                    trigger_embedding_task(doc_id, profile)
                    if mode == "rag" else None
                ),
                initial_lifecycle_state=(
                    "pending" if mode == "rag" else "active"
                ),
            )
            
            results["completed"].append(preview.url)
            results["documents_created"].append(str(ingest_result.document.id))
            
        except Exception as e:
            results["failed"][preview.url] = str(e)
    
    # Cleanup cache
    for preview in previews:
        redis_client.delete(preview.content_cache_key)
    
    return results
```

---

### **Layer 3: FetchWorker (Capabilities)**

**Location**: `crawler/fetch_worker.py`

```python
@dataclass
class FetchResult:
    """Result of fetching a single URL."""
    url: str
    content_body: bytes
    content_type: str
    title: str | None
    snippet: str
    metadata: dict[str, Any]


class FetchWorker:
    """Fetch content without persistence (Capabilities layer)."""
    
    def fetch_url(self, url: str) -> FetchResult:
        """Fetch single URL and extract preview metadata."""
        # HTTP fetch
        fetcher = HttpFetcher()
        content = fetcher.fetch(url)
        
        # Parse
        parser = get_parser_for_content_type(content.content_type)
        parsed = parser.parse(content.body)
        
        return FetchResult(
            url=url,
            content_body=content.body,
            content_type=content.content_type,
            title=parsed.title or self._extract_title_from_url(url),
            snippet=parsed.primary_text[:500] if parsed.primary_text else "",
            metadata={
                "language": parsed.content_language,
                "size_bytes": len(content.body),
            }
        )
```

---

## 📊 Compliance Matrix

| AGENTS.md Requirement | Current Proposal | Corrected | Status |
|----------------------|------------------|-----------|--------|
| **DocumentDomainService as authority** | ❌ Uses repository.upsert() | ✅ Uses domain_service.ingest_document() | FIXED |
| **bulk_ingest_documents() for crawler** | ❌ Not used | ✅ Use for multi-URL | TODO |
| **Lifecycle States (6 MVP)** | ⚠️ Not explicit | ✅ Set initial_lifecycle_state | FIXED |
| **Dispatcher Pattern** | ⚠️ Direct queue | ✅ Callback in ingest_document() | FIXED |
| **trace_id, tenant_id (Pflicht)** | ❌ Not in proposal | ✅ All tasks have them | FIXED |
| **Layer Separation** | ⚠️ Custom layers | ✅ Mapped to AGENTS.md | FIXED |
| **Paketgrenzen** | ✅ Services → Shared | ✅ No changes | OK |

---

## 🎯 Recommendations

### **Option A: Full AGENTS.md Compliance** (Recommended)

**Changes**:

1. ✅ Use `DocumentDomainService` instead of direct repository
2. ✅ All tasks carry `tenant_id`, `trace_id`, `workflow_id`, etc.
3. ✅ Use dispatcher pattern for embedding trigger
4. ✅ Set lifecycle states explicitly
5. ✅ Consider `bulk_ingest_documents()` for multi-URL optimization

**Benefits**:

- ✅ Fully aligned with architecture
- ✅ Leverage existing infrastructure
- ✅ No drift from documented standards

**Timeline**: Same (2 weeks), but cleaner

---

### **Option B: Minimal Compliance** (Not Recommended)

**Changes**:

1. Keep repository.upsert() but wrap in service
2. Add required IDs

**Problems**:

- ⚠️ Bypasses domain service authority
- ⚠️ Future drift

---

## 🚀 Final Recommendation

**Use Option A** with these key changes to original proposal:

1. **Replace** `repository.upsert()` **with** `DocumentDomainService.ingest_document()`
2. **Add** all AGENTS.md required IDs (`trace_id`, `tenant_id`, etc.)
3. **Use** dispatcher pattern for embedding triggers
4. **Set** lifecycle states explicitly
5. **Consider** `bulk_ingest_documents()` optimization later

**Result**: Clean architecture that's **100% AGENTS.md compliant** 🎯
