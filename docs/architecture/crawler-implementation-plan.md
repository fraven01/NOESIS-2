# Crawler Redesign: Implementation Plan (Facade Pattern)

## 🎯 Strategy

**Keep the Facade, Rebuild the Engine**

1. ✅ `crawl_selected()` API bleibt unverändert (Facade)
2. ✅ Search & Ingest nutzt automatisch neue Mechanik (transparent upgrade)
3. ✅ Neue Crawler-Mechanik ist sauber (4-Layer, HITL-ready, AGENTS.md konform)

**User-Insight**:
> "Search & Ingest fetcht auch nur URLs - Crawler kann das genauso!"

---

## 📋 Implementation Phases

### **Phase 1: Build New Engine (Layer 3 & 4)** - 3 days

**Goal**: Neue Crawler-Mechanik implementieren (ohne alte zu brechen)

#### **Step 1.1: Layer 4 - FetchWorker** (1 day)

**Create**: `crawler/fetch_worker.py`

```python
"""Pure worker: Fetch content without persistence."""
from dataclasses import dataclass
from typing import Any
from crawler.http_fetcher import HttpFetcher
from documents.parsers import get_parser_for_content_type


@dataclass
class FetchResult:
    """Result of fetching a single URL."""
    url: str
    content_body: bytes
    content_type: str
    title: str | None
    snippet: str
    metadata: dict[str, Any]
    fetch_success: bool
    error: str | None = None


class FetchWorker:
    """Layer 4 Worker: Pure HTTP fetching + parsing."""
    
    def fetch_url(self, url: str, timeout: int = 10) -> FetchResult:
        """
        Fetch single URL and extract preview metadata.
        
        Layer 4 responsibility: Just fetch, no persistence!
        """
        try:
            # HTTP fetch (Infrastructure)
            fetcher = HttpFetcher()
            content = fetcher.fetch(url, timeout=timeout)
            
            # Parse (Infrastructure)
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
                    "fetch_timestamp": datetime.now(timezone.utc).isoformat(),
                },
                fetch_success=True,
                error=None,
            )
        except Exception as e:
            # Return failed result (don't raise - let coordinator decide)
            return FetchResult(
                url=url,
                content_body=b"",
                content_type="",
                title=None,
                snippet="",
                metadata={},
                fetch_success=False,
                error=str(e),
            )
    
    def _extract_title_from_url(self, url: str) -> str:
        """Fallback title from URL path."""
        from urllib.parse import urlparse
        parsed = urlparse(url)
        path = parsed.path.strip("/")
        if path:
            return path.split("/")[-1] or parsed.netloc
        return parsed.netloc
```

**Tests**: `crawler/tests/test_fetch_worker.py`

```python
def test_fetch_worker_success():
    worker = FetchWorker()
    result = worker.fetch_url("https://example.com")
    assert result.fetch_success
    assert result.content_body
    assert result.title

def test_fetch_worker_failure():
    worker = FetchWorker()
    result = worker.fetch_url("https://invalid-domain-xyz.com")
    assert not result.fetch_success
    assert result.error
```

---

#### **Step 1.2: Layer 4 - Celery Tasks** (1 day)

**Create**: `crawler/tasks.py`

```python
"""Celery tasks for crawler operations."""
from celery import shared_task
from documents.domain_service import DocumentDomainService
from crawler.fetch_worker import FetchWorker


@shared_task
def ingest_urls_task(
    urls: list[str],
    mode: str,  # "rag" | "archive" | "ephemeral"
    tenant_id: str,
    trace_id: str,
    case_id: str | None,
    workflow_id: str,
    collection_id: str | None,
    embedding_profile: str | None,
    session_id: str | None = None,  # For HITL (future)
) -> dict:
    """
    Celery task: Ingest URLs using DocumentDomainService.
    
    Layer 4 Task: Coordinates workers, calls Layer 3 services.
    
    AGENTS.md Compliant:
    - Uses DocumentDomainService (central authority)
    - Includes all required IDs
    - Dispatcher pattern for embedding
    """
    from customers.models import Tenant
    import hashlib
    from uuid import UUID
    
    # Get tenant
    tenant = Tenant.objects.get(schema_name=tenant_id)
    
    # Get domain service (Layer 3)
    domain_service = DocumentDomainService(
        vector_store=get_default_client()
    )
    
    results = {
        "completed": [],
        "failed": {},
        "documents_created": [],
    }
    
    for url in urls:
        try:
            # Layer 4: Fetch using worker
            worker = FetchWorker()
            fetch_result = worker.fetch_url(url)
            
            if not fetch_result.fetch_success:
                results["failed"][url] = fetch_result.error
                continue
            
            # Prepare metadata (AGENTS.md compliant)
            metadata = {
                "title": fetch_result.title,
                "origin_uri": url,
                "workflow_id": workflow_id,
                "case_id": case_id,
                "trace_id": trace_id,
                "content_type": fetch_result.content_type,
                "external_ref": {
                    "provider": "web",
                    "external_id": f"web::{url}",
                },
            }
            
            # Hash content for idempotency
            content_hash = hashlib.sha256(fetch_result.content_body).hexdigest()
            
            # Get/create collection
            collections = []
            if collection_id:
                coll = domain_service.ensure_collection(
                    tenant=tenant,
                    collection_id=UUID(collection_id),
                    embedding_profile=embedding_profile,
                )
                collections.append(coll)
            
            # **Layer 3: DocumentDomainService** (AGENTS.md!)
            ingest_result = domain_service.ingest_document(
                tenant=tenant,
                source="crawler",
                content_hash=content_hash,
                metadata=metadata,
                collections=collections,
                embedding_profile=embedding_profile if mode == "rag" else None,
                scope=None,
                dispatcher=lambda doc_id, coll_ids, profile, scope: (
                    trigger_embedding_task(doc_id, profile)
                    if mode == "rag" else None
                ),
                initial_lifecycle_state="pending" if mode == "rag" else "active",
            )
            
            results["completed"].append(url)
            results["documents_created"].append(str(ingest_result.document.id))
            
        except Exception as e:
            results["failed"][url] = str(e)
            logger.exception(f"Failed to ingest {url}")
    
    return results
```

**Tests**: `crawler/tests/test_tasks.py`

---

#### **Step 1.3: Layer 3 - CrawlerCoordinator** (1 day)

**Create**: `ai_core/services/crawler_coordinator.py`

```python
"""Layer 3: Technical Manager for crawler operations."""
from typing import Literal
from dataclasses import dataclass
from uuid import uuid4
from crawler.tasks import ingest_urls_task


@dataclass
class IngestResult:
    """Result from crawler coordinator."""
    task_ids: list[str]
    session_id: str | None = None
    status: str = "accepted"


class CrawlerCoordinator:
    """
    Layer 3: Technical Manager for crawler.
    
    Knows: "I need to queue fetch tasks, trigger ingestion"
    Does NOT know: Business workflows, case management
    """
    
    def ingest_direct(
        self,
        *,
        urls: list[str],
        mode: Literal["rag", "archive", "ephemeral"],
        tenant_id: str,
        trace_id: str,
        case_id: str | None = None,
        workflow_id: str = "crawler",
        collection_id: str | None = None,
        embedding_profile: str | None = None,
    ) -> IngestResult:
        """
        Direct ingestion (bypass HITL).
        
        Used by:
        - crawl_selected() facade
        - API direct calls
        
        Returns immediately with task_id (async processing).
        """
        # Queue Celery task (Layer 4)
        task = ingest_urls_task.delay(
            urls=urls,
            mode=mode,
            tenant_id=tenant_id,
            trace_id=trace_id,
            case_id=case_id,
            workflow_id=workflow_id,
            collection_id=collection_id,
            embedding_profile=embedding_profile,
            session_id=None,  # No HITL
        )
        
        return IngestResult(
            task_ids=[task.id],
            status="accepted",
        )
    
    def fetch_for_review(
        self,
        *,
        urls: list[str],
        tenant_id: str,
        trace_id: str,
        case_id: str | None = None,
    ) -> IngestResult:
        """
        HITL Phase 1: Fetch URLs for preview (FUTURE).
        
        Returns session_id for review.
        """
        # TODO: Implement HITL flow
        raise NotImplementedError("HITL not yet implemented")
```

**Tests**: `ai_core/services/tests/test_crawler_coordinator.py`

---

### **Phase 2: Rebuild Facade (crawl_selected)** - 1 day

**Goal**: `crawl_selected()` nutzt neue Mechanik, bleibt API-kompatibel

**Modify**: `ai_core/views.py:crawl_selected()`

```python
@require_POST
def crawl_selected(request):
    """
    Handle crawl selected requests from the RAG tools page.
    
    API Contract (UNCHANGED):
    - POST JSON: {"urls": [...], "mode": "live"|"archive", ...}
    - Returns: {"task_ids": [...], "status": "accepted"}
    
    Internal: Now uses new CrawlerCoordinator (Layer 3)
    """
    try:
        # Existing validation (UNCHANGED)
        from rest_framework.request import Request as DRFRequest
        drf_request = DRFRequest(request)
        meta, error = _prepare_request(drf_request)
        if error:
            return JsonResponse(error.data, status=error.status_code)
        
        # Parse request (UNCHANGED)
        try:
            data = json.loads(request.body)
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON"}, status=400)
        
        urls = data.get("urls")
        if not urls:
            return JsonResponse({"error": "URLs are required"}, status=400)
        
        # Map old mode to new mode
        old_mode = data.get("mode", "live")
        new_mode = "rag" if old_mode == "live" else "archive"
        
        collection_id = data.get("collection_id", "crawler-demo")
        workflow_id = data.get("workflow_id", "crawler-demo")
        
        # **NEW**: Use CrawlerCoordinator (Layer 3)
        from ai_core.services.crawler_coordinator import CrawlerCoordinator
        
        coordinator = CrawlerCoordinator()
        result = coordinator.ingest_direct(
            urls=urls,
            mode=new_mode,
            tenant_id=meta["tenant_id"],
            trace_id=meta["trace_id"],
            case_id=meta.get("case_id"),
            workflow_id=workflow_id,
            collection_id=collection_id,
            embedding_profile=data.get("embedding_profile"),
        )
        
        # Return same response format (UNCHANGED API)
        return JsonResponse({
            "task_ids": result.task_ids,
            "status": result.status,
            # ... other fields for compatibility
        })
        
    except Exception as e:
        logger.error(f"crawl_selected exception: {repr(e)}", exc_info=True)
        return JsonResponse({"error": str(e)}, status=500)
```

**Changes**:

- ✅ API contract UNCHANGED
- ✅ Internal: Uses `CrawlerCoordinator` instead of `run_crawler_runner`
- ✅ Maps `mode: "live"` → `"rag"`, `"archive"` → `"archive"`
- ✅ Search & Ingest automatically uses new flow!

---

### **Phase 3: Test & Verify** - 1 day

**Verification Checklist**:

1. **Unit Tests**:
   - [ ] `test_fetch_worker.py` - all pass
   - [ ] `test_tasks.py` - all pass
   - [ ] `test_crawler_coordinator.py` - all pass

2. **Integration Tests**:
   - [ ] `test_crawl_selected_api.py` - API unchanged
   - [ ] Mode mapping works (`live` → `rag`, `archive` → `archive`)

3. **Manual Testing** (`/rag-tools/`):
   - [ ] **Web Search Tab**:
     - [ ] Search returns results
     - [ ] "Ingest Selected" button works
     - [ ] Status panel shows progress
     - [ ] Documents appear in Document Explorer

   - [ ] **Crawler Tab**:
     - [ ] Manual URL entry works
     - [ ] "Start Crawl" triggers processing
     - [ ] Results show in status area

4. **Database Verification**:
   - [ ] Documents created with correct `external_ref` structure
   - [ ] Lifecycle state correct (`pending` for RAG, `active` for archive)
   - [ ] Collections assigned correctly
   - [ ] No orphaned documents

---

## 🎯 Success Criteria

**After Phase 2**:

- ✅ Search & Ingest uses new crawler (transparent upgrade)
- ✅ NO API changes
- ✅ NO UI changes
- ✅ Parallel registration GONE
- ✅ AGENTS.md compliant
- ✅ Ready for HITL (Phase 4 - future)

---

## 📊 Migration Path

### **Current State**

```
crawl_selected()
    ↓
run_crawler_runner()
    ↓
CrawlerWorker._register_document() ← Parallel registration ❌
    ↓
Graph processing (assumes doc exists)
```

### **New State** (After Phase 2)

```
crawl_selected()
    ↓
CrawlerCoordinator.ingest_direct()
    ↓
ingest_urls_task.delay()
    ↓
FetchWorker.fetch_url() + DocumentDomainService.ingest_document() ✅
    ↓
Single persistence point!
```

---

## 🚀 Timeline

| Phase | Duration | Start After | Output |
|-------|----------|-------------|--------|
| 1.1 FetchWorker | 1 day | Immediately | Pure fetch worker |
| 1.2 Celery Tasks | 1 day | Phase 1.1 | ingest_urls_task |
| 1.3 Coordinator | 1 day | Phase 1.2 | CrawlerCoordinator |
| 2. Facade | 1 day | Phase 1.3 | crawl_selected updated |
| 3. Test & Verify | 1 day | Phase 2 | All tests pass |

**Total**: 5 days (1 week)

---

## 🔧 What Gets Deleted (Later)

**After successful migration**:

- ❌ `crawler/worker.py:_register_document()` (parallel registration)
- ❌ `ai_core/services/crawler_runner.py:run_crawler_runner()` (replaced by coordinator)
- ❌ `ai_core/services/crawler_state_builder.py` (no longer needed)

**Lines saved**: ~300 lines of complex code removed!

---

## 💡 Future: Add HITL (Phase 4 - Optional)

**After Phase 3 stabilizes**, add HITL:

```python
# New endpoint
POST /api/crawler/fetch-for-review
    ↓
coordinator.fetch_for_review()
    ↓
Returns preview_session_id
    ↓
User reviews in UI
    ↓
POST /api/crawler/ingest-selected
    ↓
coordinator.ingest_selected(session_id, selected_urls, mode="rag")
```

**Benefits**:

- ✅ HITL as addon (not breaking change)
- ✅ Existing flow keeps working
- ✅ Users can opt-in to HITL

---

**Ready to start Phase 1.1 (FetchWorker)?** 🚀
