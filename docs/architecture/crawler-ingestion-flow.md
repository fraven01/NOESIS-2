# Crawler Document Ingestion Flow - Complete Documentation

## Overview

This document maps the **complete flow** of document ingestion via the crawler, highlighting the **parallel registration problem** (Drift #3).

---

## 🔄 High-Level Flow

```
HTTP Request
    ↓
┌─────────────────────────────────────────┐
│  CrawlerWorker.process()                │
│  (crawler/worker.py)                    │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  _compose_state()                       │
│  Prepares state for ingestion           │
└─────────────────────────────────────────┘
    ↓
    ├─ **PATH A: Parallel Registration** ❌
    │  (Lines 258-274 in crawler/worker.py)
    │  ↓
    │  _register_document()
    │  (Lines 646-700)
    │  ↓
    │  DocumentDomainService.ingest_document()
    │  (documents/domain_service.py:139-234)
    │  ↓
    │  ✅ **Document created in DB**
    │  ✅ Collections assigned
    │  ✅ Lifecycle state: "PENDING"
    │  ✅ Ingestion dispatcher queued
    │
    └─ **PATH B: Graph Processing** ✅
       (After PATH A completes)
       ↓
       Publish to Celery task
       ↓
       run_ingestion_graph()
       (ai_core/tasks.py)
       ↓
       CrawlerIngestionGraph.run()
       (ai_core/graphs/crawler_ingestion_graph.py)
       ↓
       _ensure_normalized_payload()
       ↓
       build_document_processing_graph()
       ↓
       Parse → Chunk → Embed
       ↓
       ❓ **Where does persistence happen?**
```

---

## 📍 PATH A: Parallel Registration (BEFORE Graph)

### Step 1: Worker Calls _register_document()

**File**: `crawler/worker.py:258-274`

```python
resolved_document_id = self._register_document(
    tenant_id=tenant_id,
    source=request.canonical_source,
    content_hash=payload_checksum,
    metadata=raw_meta,
    collection_identifier=raw_meta.get("collection_id"),
    embedding_profile=ingestion_overrides.get("embedding_profile"),
    scope=ingestion_overrides.get("scope"),
)
```

**Purpose**: Pre-register document in database
**Problem**: If graph fails later, document is orphaned

---

### Step 2: _register_document() Implementation

**File**: `crawler/worker.py:646-700`

```python
def _register_document(self, ...) -> str | None:
    tenant = self._resolve_tenant(tenant_id)
    service = self._get_domain_service()  # ← DocumentDomainService
    
    # Ensure collection exists
    collections: list[DocumentCollection] = []
    if collection_identifier is not None:
        ensured = self._ensure_collection_with_warning(...)
        collections.append(ensured)
    
    # **CRITICAL CALL**: Creates document in DB
    ingest_result = service.ingest_document(
        tenant=tenant,
        source=source,
        content_hash=content_hash,
        metadata=metadata,
        collections=collections,
        ...
    )
    
    return str(ingest_result.document.id)
```

**Side Effects**:

- ✅ Document row created in `documents_document` table
- ✅ Collection membership created
- ✅ Lifecycle state set to "PENDING"
- ✅ Ingestion dispatcher queued (via `transaction.on_commit`)

---

### Step 3: DocumentDomainService.ingest_document()

**File**: `documents/domain_service.py:139-234`

```python
def ingest_document(self, ...) -> PersistedDocumentIngest:
    with transaction.atomic():
        # **DATABASE WRITE**
        document, created = Document.objects.update_or_create(
            tenant=tenant,
            source=source,
            hash=content_hash,
            defaults={
                "metadata": metadata_payload,
                "lifecycle_state": lifecycle_state.value,  # "PENDING"
                "lifecycle_updated_at": lifecycle_timestamp,
            },
        )
        
        # Assign to collections
        for collection in collection_instances:
            DocumentCollectionMembership.objects.get_or_create(
                document=document,
                collection=collection,
            )
        
        # Queue ingestion dispatcher (after commit)
        if dispatcher_fn:
            transaction.on_commit(
                lambda: dispatcher_fn(
                    document.id,
                    collection_ids,
                    embedding_profile,
                    scope,
                )
            )
    
    return PersistedDocumentIngest(document=document, ...)
```

**Result**: Document EXISTS in DB at this point, even though graph hasn't run yet

---

## 📍 PATH B: Graph Processing (AFTER Parallel Registration)

### Step 4: Celery Task Queued

**File**: `ai_core/tasks.py`

```python
@shared_task
def run_ingestion_graph(state, meta):
    # Prepare working state
    working_state = _prepare_working_state(state, ingestion_ctx, trace_context)
    
    # Normalize raw document
    contract = NormalizedDocumentInputV1.from_raw(
        raw_reference=raw_reference,
        tenant_id=ingestion_ctx.tenant_id,
        ...
    )
    normalized_payload = normalize_from_raw(contract=contract)
    
    # **INVOKE GRAPH**
    result, meta = GRAPH.run(working_state, {})
    
    return result
```

---

### Step 5: CrawlerIngestionGraph.run()

**File**: `ai_core/graphs/crawler_ingestion_graph.py:317-494`

```python
def run(self, state, meta=None):
    # Ensure normalized payload exists
    normalized_payload = self._ensure_normalized_payload(state)
    
    # Build processing context
    context = DocumentProcessingContext.from_document(
        normalized_payload.document,
        case_id=case_id,
        trace_id=trace_id,
        ...
    )
    
    # **INVOKE DOCUMENT PROCESSING GRAPH**
    result_state = self._document_graph.invoke(
        DocumentProcessingState(
            document=normalized_payload.document,
            config=self._pipeline_config,
            context=context,
            storage=self._storage,
            run_until=run_until,
        )
    )
    
    # Handle transitions (lifecycle updates)
    # ... guardrails, delta decisions, etc.
    
    return working_state, result_meta
```

**Question**: Where is `self._document_graph` defined?

---

### Step 6: build_document_processing_graph()

**File**: `documents/processing_graph.py:207-xxx`

This is the **actual processing pipeline**:

- Parse (extract text from HTML/PDF/etc)
- Chunk (split into semantic chunks)
- Embed (create vector embeddings)

**Question**: Does this graph persist the final result? Let's check.

---

### Step 7: Graph Persistence - WHERE?

**Investigation Needed**:

1. Does `build_document_processing_graph` call `repository.upsert()`?
2. Does `CrawlerIngestionGraph` update the document after graph?
3. Is persistence implicit via lifecycle updates?

Let me check the graph construction:

---

## 🔍 Investigation: Where Does Graph Persist?

### Finding 1: Graph Initialization

**File**: `ai_core/graphs/crawler_ingestion_graph.py:126-176`

```python
def __init__(self, ...):
    components = require_document_components()
    
    # Build document processing graph
    self._document_graph = build_document_processing_graph(
        parser=self._parser_dispatcher,
        chunker=chunker,
        enricher=enricher,
        repository=repository,  # ← Repository is passed!
        ...
    )
```

**Key**: Repository IS passed to the graph!

---

### Finding 2: Repository in Processing Graph

The `build_document_processing_graph` receives a repository. This suggests **the graph DOES persist**.

But let me verify what the graph actually does with the repository.

---

## 🎯 The Parallel Registration Problem

### Current State

```
Timeline:
T0: HTTP Request received
T1: CrawlerWorker._register_document() → Document created in DB ✅
T2: Celery task queued
T3: Graph starts processing
T4: Graph parses document
T5: Graph chunks document
T6: Graph embeds chunks
T7: Graph completes

If graph fails at T5:
- Document exists in DB (created at T1) ✅
- But processing incomplete ❌
- Lifecycle state = "PENDING" ❌
- No chunks, no embeddings ❌
- Document is ORPHANED ❌
```

---

### Ideal State

```
Timeline:
T0: HTTP Request received
T1: CrawlerWorker prepares state (NO DB write)
T2: Celery task queued
T3: Graph starts processing
T4: Graph parses document
T5: Graph chunks document
T6: Graph CREATES document in DB ✅
T7: Graph embeds chunks
T8: Graph updates lifecycle ✅

If graph fails at T5:
- NO document in DB ✅
- Clean failure ✅
- Can retry safely ✅
```

---

## 📊 Comparison Table

| Aspect | Current (Parallel) | Ideal (Graph-Only) |
|--------|-------------------|--------------------|
| **Document Creation** | Before graph (T1) | In graph (T6) |
| **Failure Safety** | ❌ Orphaned docs | ✅ Clean rollback |
| **Idempotency** | ⚠️ Document exists but incomplete | ✅ Atomic |
| **Collection Assignment** | Twice (PATH A + dispatcher) | Once (in graph) |
| **Complexity** | High (2 paths) | Low (1 path) |
| **Lines of Code** | ~100 (parallel path) | ~50 (removed) |

---

## 🚨 Evidence of the Problem

### Red Flag 1: Double Persistence

```python
# PATH A (crawler/worker.py:258):
resolved_document_id = self._register_document(...)  # ← Creates document

# PATH B (graph):
self._document_graph.invoke(...)  # ← Also persists?
```

**This is the smoking gun!**

---

### Red Flag 2: Transaction Isolation

```python
# In DocumentDomainService.ingest_document():
with transaction.atomic():
    document, created = Document.objects.update_or_create(...)
    # Transaction commits HERE
    # ↓
    # Graph is queued AFTER commit
    # ↓
    # If graph fails, document already exists!
```

**Problem**: Document is committed BEFORE graph even starts.

---

### Red Flag 3: Lifecycle State Confusion

```python
# Parallel registration sets:
lifecycle_state = "PENDING"

# Graph completion should set:
lifecycle_state = "ACTIVE"

# But if graph fails:
lifecycle_state = "PENDING" (forever!)
```

**Problem**: No cleanup mechanism for stuck "PENDING" documents.

---

## 🎯 Next Steps (Phase 2+)

Now that we understand the flow, we can implement the solution:

1. **Phase 2**: Add rollback logic if graph fails
2. **Phase 3**: Make graph create document if not exists
3. **Phase 4**: Add feature flag to skip parallel registration
4. **Phase 5**: Deprecate parallel path
5. **Phase 6**: Remove parallel path
6. **Phase 7**: Optimize graph persistence

---

## 📝 Questions for Stakeholders

1. **Q**: Is `DocumentDomainService` used elsewhere besides crawler?
   **A**: Need to check usage across codebase

2. **Q**: Can we afford a breaking change (remove parallel path)?
   **A**: Yes, we're pre-MVP

3. **Q**: How many "PENDING" documents exist in production?
   **A**: Need to query DB: `SELECT COUNT(*) FROM documents_document WHERE lifecycle_state = 'PENDING'`

4. **Q**: Does the graph ACTUALLY persist, or does it delegate back?
   **A**: Need to trace `build_document_processing_graph` implementation

---

## 🔬 Further Investigation Needed

1. ✅ Find `DocumentDomainService` - DONE
2. ✅ Understand `ingest_document()` - DONE
3. ⏳ Trace `build_document_processing_graph` persistence
4. ⏳ Find where lifecycle state updates to "ACTIVE"
5. ⏳ Count orphaned documents in dev DB

---

**Date**: 2025-12-11  
**Status**: Flow documented, ready for Phase 2 implementation  
**Next**: Trace graph persistence logic to complete picture
