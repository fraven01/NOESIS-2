# Clean Crawler Architecture - Redesign Proposal

## Context

**Status**: Pre-MVP (Release in 6 months)  
**Permission**: Breaking changes allowed  
**Problem**: Current parallel registration is messy  
**Solution**: Complete redesign with 3 clear modes

---

## 🎯 User Requirements

### Input

- **Single URL** or **Multiple URLs**
- **Source**: Manual, Search, or Intelligent Search

### Output Modes

1. **Mode A: RAG** - Full ingestion pipeline
   - Fetch → Parse → Chunk → Embed → Vector DB
   - Persist to document repo
   - Full-text search enabled

2. **Mode B: Archive** - Repository only
   - Fetch → Parse → Store
   - No embedding, no vector storage
   - Searchable via full-text only

3. **Mode C: Ephemeral** - LLM context only
   - Fetch → Parse → Return text
   - No persistence
   - Used for real-time research/analysis

### Processing

- **Asynchronous** via Celery (multiple documents)
- **Idempotent** (same URL = same document)
- **Atomic** (all-or-nothing per document)

---

## 🏗️ Proposed Architecture

### API Layer

```python
# New endpoint: /api/v1/crawler/ingest
POST /api/v1/crawler/ingest
{
  "urls": ["https://example.com", "https://example.org"],
  "mode": "rag",  // "rag" | "archive" | "ephemeral"
  "collection_id": "uuid-here",  // optional
  "embedding_profile": "standard",  // optional
  "metadata": {  // optional
    "tags": ["research"],
    "source": "intelligent_search"
  }
}

Response:
{
  "task_id": "celery-task-id",
  "urls_queued": 2,
  "mode": "rag",
  "status": "processing"
}
```

---

### Celery Task Structure

```python
# New task: crawler.tasks.ingest_urls
@shared_task
def ingest_urls(
    urls: list[str],
    mode: Literal["rag", "archive", "ephemeral"],
    tenant_id: str,
    case_id: str | None = None,
    collection_id: str | None = None,
    embedding_profile: str = "standard",
    metadata: dict | None = None,
) -> dict:
    """
    Ingest multiple URLs with specified mode.
    
    Returns:
        {
            "completed": ["url1", "url2"],
            "failed": {"url3": "error_reason"},
            "documents_created": ["doc-id-1", "doc-id-2"],
        }
    """
    results = {
        "completed": [],
        "failed": {},
        "documents_created": [],
    }
    
    for url in urls:
        try:
            doc_id = _ingest_single_url(
                url=url,
                mode=mode,
                tenant_id=tenant_id,
                case_id=case_id,
                collection_id=collection_id,
                embedding_profile=embedding_profile,
                metadata=metadata,
            )
            results["completed"].append(url)
            if doc_id:
                results["documents_created"].append(doc_id)
        except Exception as e:
            results["failed"][url] = str(e)
            logger.exception(f"Failed to ingest {url}")
    
    return results
```

---

### Core Ingestion Logic

```python
def _ingest_single_url(
    url: str,
    mode: Literal["rag", "archive", "ephemeral"],
    tenant_id: str,
    **kwargs
) -> str | None:
    """
    Ingest a single URL based on mode.
    
    Returns document_id if persisted, None if ephemeral.
    """
    
    # 1. Fetch content
    fetcher = HttpFetcher()
    content = fetcher.fetch(url)
    
    # 2. Parse content
    parser = get_parser_for_content_type(content.content_type)
    parsed = parser.parse(content.body)
    
    # 3. Mode-specific processing
    if mode == "ephemeral":
        # Return text, don't persist
        return None  # Or return text directly?
    
    # 4. Create document (SINGLE point of persistence)
    document = NormalizedDocument(
        ref=DocumentRef(
            tenant_id=tenant_id,
            document_id=uuid4(),
            workflow_id=kwargs.get("case_id", "crawler"),
            collection_id=kwargs.get("collection_id"),
        ),
        meta=DocumentMeta(
            tenant_id=tenant_id,
            workflow_id=kwargs.get("case_id", "crawler"),
            title=parsed.title or _extract_title_from_url(url),
            origin_uri=url,
            external_ref={
                "provider": "web",
                "external_id": f"web::{url}",
            },
        ),
        blob=FileBlob(
            type="file",
            uri=_persist_to_storage(content.body),
            sha256=hashlib.sha256(content.body).hexdigest(),
            size=len(content.body),
            media_type=content.content_type,
        ),
        checksum=hashlib.sha256(content.body).hexdigest(),
        created_at=datetime.now(timezone.utc),
        source="crawler",
    )
    
    # 5. Persist to repository
    repository = get_documents_repository()
    stored_doc = repository.upsert(document, workflow_id=document.ref.workflow_id)
    
    # 6. Mode-specific post-processing
    if mode == "rag":
        # Trigger embedding pipeline
        trigger_embedding_task(
            document_id=stored_doc.ref.document_id,
            embedding_profile=kwargs.get("embedding_profile", "standard"),
        )
    elif mode == "archive":
        # Just persist, skip embedding
        pass
    
    return str(stored_doc.ref.document_id)
```

---

## 🔑 Key Design Decisions

### 1. Single Persistence Point ✅

- **Only** `repository.upsert()` persists documents
- No parallel registration
- No double-persistence
- Atomic: success or failure, no orphans

### 2. Mode-Driven Processing ✅

- Clear branching based on `mode` parameter
- Easy to add new modes later
- Each mode is independently testable

### 3. Celery for Async ✅

- One task handles multiple URLs
- Parallel processing within task (optional)
- Progress tracking via task metadata

### 4. Idempotency ✅

- URL + content_hash = unique document
- Duplicate URL = retrieve existing document
- Safe retries

### 5. Clean Separation ✅

```
Fetching   → HTTP layer
Parsing    → Parser layer
Persistence → Repository layer
Embedding   → Separate task (async)
```

---

## 📊 Comparison: Old vs New

| Aspect | Old (Messy) | New (Clean) |
|--------|-------------|-------------|
| **Persistence Points** | 2 (parallel + graph) | 1 (repository only) |
| **Modes** | Implicit | Explicit (3 modes) |
| **Async** | Celery after registration | Celery from start |
| **Idempotency** | Broken | Built-in |
| **Code Complexity** | High (2 paths) | Low (1 path) |
| **Failure Handling** | Orphans | Clean rollback |
| **Lines of Code** | ~500 | ~200 |

---

## 🚀 Implementation Plan

### Phase 1: New Task (2 days)

- Create `crawler/tasks.py:ingest_urls`
- Implement 3 modes
- Single persistence point
- Unit tests

### Phase 2: API Endpoint (1 day)

- Add `/api/v1/crawler/ingest`
- Validate input
- Queue Celery task
- Return task_id

### Phase 3: Migration (1 day)

- Deprecate old `CrawlerWorker`
- Update Search & Ingest to use new API
- Database migration (if needed)

### Phase 4: Cleanup (1 day)

- Remove parallel registration code
- Remove old `CrawlerIngestionGraph` (or refactor)
- Delete dead code

### Phase 5: Testing (1 day)

- E2E tests for all 3 modes
- Performance benchmarks
- Load testing

**Total**: 1 week (5 days)

---

## 🧪 Testing Strategy

### Mode A (RAG) Test

```python
def test_mode_rag_full_pipeline():
    response = client.post("/api/v1/crawler/ingest", {
        "urls": ["https://example.com"],
        "mode": "rag",
        "collection_id": "test-collection",
    })
    
    task_id = response.json()["task_id"]
    result = wait_for_task(task_id)
    
    # Verify document created
    doc_id = result["documents_created"][0]
    doc = repository.get(tenant_id, doc_id)
    assert doc is not None
    assert doc.meta.origin_uri == "https://example.com"
    
    # Verify embedding queued
    embeddings = get_embeddings_for_document(doc_id)
    assert len(embeddings) > 0
```

### Mode B (Archive) Test

```python
def test_mode_archive_skips_embedding():
    response = client.post("/api/v1/crawler/ingest", {
        "urls": ["https://example.com"],
        "mode": "archive",
    })
    
    result = wait_for_task(response.json()["task_id"])
    doc_id = result["documents_created"][0]
    
    # Verify document created
    doc = repository.get(tenant_id, doc_id)
    assert doc is not None
    
    # Verify NO embeddings
    embeddings = get_embeddings_for_document(doc_id)
    assert len(embeddings) == 0
```

### Mode C (Ephemeral) Test

```python
def test_mode_ephemeral_no_persistence():
    response = client.post("/api/v1/crawler/ingest", {
        "urls": ["https://example.com"],
        "mode": "ephemeral",
    })
    
    result = wait_for_task(response.json()["task_id"])
    
    # Verify NO documents created
    assert len(result["documents_created"]) == 0
    
    # Verify text returned (or how to access it?)
    # Maybe task result should include extracted text?
```

---

## 📝 Breaking Changes

### What Breaks

1. ❌ `CrawlerWorker._register_document()` - removed
2. ❌ `DocumentDomainService.ingest_document()` - maybe removed (check other uses)
3. ❌ Parallel registration flow
4. ❌ Old crawler task structure

### Migration Path

```python
# Old way:
CrawlerWorker().process(request)

# New way:
ingest_urls.delay(
    urls=[request.url],
    mode="rag",
    tenant_id=request.tenant_id,
    ...
)
```

---

## 🎯 Success Criteria

After implementation:

- ✅ 3 modes working (RAG, Archive, Ephemeral)
- ✅ Single persistence point
- ✅ No orphaned documents
- ✅ Idempotent ingestion
- ✅ ~300 lines of code removed
- ✅ All tests passing
- ✅ Search & Ingest integrated

---

## 🤔 Open Questions

1. **Mode C (Ephemeral)**: How should we return the extracted text?
   - Option A: Store in Celery result backend
   - Option B: Stream to caller via WebSocket
   - Option C: Return via polling endpoint

2. **Batch Size**: Should we limit URLs per task?
   - Recommendation: Max 100 URLs per task
   - Auto-split larger batches

3. **Priority**: Should different modes have different priorities?
   - Ephemeral = High (real-time)
   - RAG = Medium
   - Archive = Low

4. **Retry Logic**: How many retries for failed URLs?
   - Recommendation: 3 retries with exponential backoff

---

## 💡 Recommendation

**Start with Phase 1** (new task) alongside current system.

Once proven:

- ✅ Flip feature flag
- ✅ Migrate Search & Ingest
- ✅ Remove old code

**Timeline**: 1 week to production-ready

---

**Question for you**:

1. For Mode C (Ephemeral), how should we deliver the extracted text to the caller?
2. Any other modes you envision?
3. Should we start implementation now, or discuss design first?
