# Crawler Architecture Analysis - Layers & Responsibilities

## 🎯 Goal

Identify clean layer separation to avoid drift while enabling:

1. 3 Modes (RAG, Archive, Ephemeral)
2. HITL (Preview before ingest)
3. Single persistence point
4. No breaking changes to Search & Ingest

---

## 🏗️ NOESIS-2 Architecture: 4-Layer Firm Hierarchy

### The Firm Analogy

Think of NOESIS-2 as a **company with 4 hierarchical layers**:

```
┌────────────────────────────────────────────────────┐
│  LAYER 1: FRONTEND (Customer Interface)           │
│  What the customer sees and interacts with        │
└────────────────┬───────────────────────────────────┘
                 │
┌────────────────▼───────────────────────────────────┐
│  LAYER 2: BUSINESS MANAGERS (TODO - Mostly)       │
│  Business processes, workflows, cases              │
│  Example: /rag-tools/#framework                   │
└────────────────┬───────────────────────────────────┘
                 │
┌────────────────▼───────────────────────────────────┐
│  LAYER 3: TECHNICAL MANAGERS                       │
│  Know which workers to use for technical tasks    │
│  - LangGraph (ai_core/graphs)                     │
│  - Services (DocumentDomainService)               │
│  - Coordinators                                    │
└────────────────┬───────────────────────────────────┘
                 │
┌────────────────▼───────────────────────────────────┐
│  LAYER 4: WORKERS (The actual work)                │
│  - Celery tasks                                    │
│  - HTTP Fetcher, Parsers, Repository              │
│  - Infrastructure                                  │
└────────────────────────────────────────────────────┘
```

**Key Insight**: HITL (Human-In-The-Loop) must flow through ALL 4 layers!

---

## 📊 Current Architecture (Reverse-Engineered)

### Layer 1: Frontend (Customer Interface)

**Location**: `theme/templates/theme/rag_tools.html`  
**Responsibility**: UI the customer interacts with

```
Search & Ingest UI (/rag-tools/#search)
    ↓
User enters URLs
User clicks "Search and Ingest"
```

**Current Consumers**:

- Search & Ingest UI (`/rag-tools/#search`)
- Document Explorer
- Framework Analysis (`/rag-tools/#framework`) ← Business layer example

**Status**: ✅ Implemented

---

### Layer 2: Business Managers (TODO - Mostly)

**Responsibility**: Business process orchestration using workflows and cases

**Current State**:

- ⚠️ **Mostly missing!**
- ✅ Hinted at in `/rag-tools/#framework`
- ✅ IDs prepared (`case_id`, `workflow_id`)

**Future State** (not yet implemented):

```python
class WorkflowEngine:
    """Orchestrate business processes."""
    def start_workflow(self, workflow_type, case_id):
        # Step 1: Ingest documents → calls Technical Manager
        # Step 2: Analyze with LLM → calls Technical Manager
        # Step 3: HITL Review → wait for human approval
        # Step 4: Generate report → calls Technical Manager
```

**Examples** (to be built):

- Workflow designer UI
- Case management dashboard
- Business rule engine
- SLA tracking, audit trails

**Status**: ❌ TODO (prepare for it now with IDs)

---

### Layer 3: Technical Managers

**Responsibility**: Know which workers to use for technical tasks

**Current State**: ✅ **Partially Implemented**

**Components**:

#### **A) HTTP/View Entry Points**

**Location**: `ai_core/views.py:crawl_selected()`

```python
@require_POST
def crawl_selected(request):
    # Parse request
    # Validate with CrawlerRunRequest
    # Call run_crawler_runner() → Technical Manager
```

**Role**: Entry point that delegates to Technical Managers

---

#### **B) Coordinator/Service Layer**

**Location**: `ai_core/services/crawler_runner.py:run_crawler_runner()`

```python
run_crawler_runner(meta, request_model, ...)
├─ Build crawler states from URLs
├─ For each URL:
│  ├─ Fetch content
│  ├─ Run graph inline OR async
│  └─ Collect results
└─ Return results
```

**Problems**:

- ❌ Mixes synchronous and asynchronous modes
- ❌ No clear separation of fetch vs ingest
- ❌ Coordinator does too much

---

#### **C) LangGraph Orchestrations** (`ai_core/graphs/`)

**Location**: `ai_core/graphs/crawler_ingestion_graph.py`

```python
class CrawlerIngestionGraph:
    """
    Technical Manager for crawler ingestion.
    Knows: "I need to fetch, parse, chunk, embed"
    Does NOT know: "This is Case #12345 Legal Review"
    """
    def run(self, state, meta):
        # Orchestrates technical steps
        normalized = self._ensure_normalized_payload(state)
        result = self._document_graph.invoke(...)
        # Handle transitions, guardrails, etc.
```

**Knows**:

- Which workers to call (parser, chunker, embedder)
- Sequence of operations
- Technical error handling

**Does NOT know**:

- Business workflows
- Case management
- Business SLAs

---

#### **D) Domain Services**

**Location**: `documents/domain_service.py`

```python
class DocumentDomainService:
    """
    Technical Manager for document operations.
    AGENTS.md: Central authority for all document operations
    """
    def ingest_document(self, tenant, source, content_hash, ...):
        # Orchestrates: repository, lifecycle, dispatcher
        document = Document.objects.update_or_create(...)
        # Assign to collections
        # Queue embedding dispatcher
```

**Status**: ✅ Implemented (AGENTS.md compliant)

---

### Layer 4: Workers (The Actual Work)

**Responsibility**: Execute specific tasks, no orchestration

**Current State**: ✅ **Partially Implemented**

**Components**:

#### **A) CrawlerWorker** (Current - Has Problems)

**Location**: `crawler/worker.py:CrawlerWorker`

```python
CrawlerWorker.process(request)
├─ Fetch HTTP content
├─ _compose_state()
│  ├─ Extract metadata
│  ├─ Persist payload to storage
│  └─ **_register_document()** ← PARALLEL REGISTRATION! (Problem)
│
└─ Publish to Celery
```

**Problems**:

- ❌ Parallel registration (creates document before graph)
- ❌ Acts like Technical Manager (too much responsibility)
- ❌ Should be pure worker (just fetch)

---

#### **B) Proposed: FetchWorker** (Clean)

**Location**: `crawler/fetch_worker.py` (NEW - refactored from CrawlerWorker)

```python
class FetchWorker:
    """Pure worker: Fetch content WITHOUT persistence."""
    
    def fetch_url(self, url: str) -> FetchResult:
        """Fetch single URL, return content + metadata."""
        content = HttpFetcher().fetch(url)
        parser = get_parser(content.content_type)
        parsed = parser.parse(content.body)
        
        return FetchResult(
            url=url,
            content_body=content.body,
            content_type=content.content_type,
            title=parsed.title,
            snippet=parsed.primary_text[:500],
            metadata={...},
        )
```

**Responsibility**:

- ✅ Just fetch and parse
- ✅ No persistence
- ✅ Pure function
- ✅ Returns structured data

---

#### **C) Infrastructure Workers**

**Components**:

- **HTTP Fetcher**: `crawler/http_fetcher.py`
- **Parsers**: `documents/parsers.py` (HTML, PDF, Docx, etc.)
- **Repository**: `ai_core/adapters/db_documents_repository.py`
- **Object Storage**: S3/GCS adapters
- **Vector Store**: pgvector queries

**Responsibility**:

- Execute specific infrastructure tasks
- No business logic
- Stateless when possible

---

## 🔄 Flow Comparison: Current vs Proposed

### *Current Flow (Messy - Mixed Layers)**

```
Search & Ingest UI (Layer 1)
    ↓
crawl_selected view (Layer 3 entry)
    ↓
run_crawler_runner (Layer 3)
    ├─ build_crawler_state
    ├─ CrawlerWorker.process() (Layer 4 but acts like Layer 3!)
    │  ├─ Fetch
    │  ├─ **_register_document()** ← Document created! (Wrong layer!)
    │  └─ Publish to Celery
    └─ OR run graph inline
        ↓
    CrawlerIngestionGraph.run() (Layer 3 Graph)
        ├─ Assumes document exists
        └─ Process
```

**Problems**:

- Document created in Layer 4 (Worker) - should be Layer 3 (Technical Manager)!
- No Layer 2 (Business) involvement
- No HITL support
- Mixed sync/async

---

### **Proposed Flow (Clean - 4-Layer HITL)**

```
┌─────────────────────────────────────────────────────┐
│ LAYER 1: Frontend                                   │
│ /rag-tools/#search                                  │
│ User enters URLs → "Fetch for Review"               │
└────────────┬────────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────────┐
│ LAYER 2: Business Manager (Future)                  │
│ WorkflowEngine.start("Web Research", case_id)      │
│ - Step 1: Fetch for preview                        │
└────────────┬────────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────────┐
│ LAYER 3: Technical Manager                         │
│ CrawlerCoordinator.fetch_for_review()              │
│ - Queues: fetch_urls_task.delay()                  │
└────────────┬────────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────────┐
│ LAYER 4: Workers                                    │
│ Celery: fetch_urls_task                            │
│ - FetchWorker.fetch_url() → HTTP                   │
│ - Parser.parse() → Extract                         │
│ - Redis.cache() → Store preview                    │
└────────────┬────────────────────────────────────────┘
             │
             ↓
👤 HUMAN REVIEWS (UI shows previews) ← HITL!
             │
User selects → "Ingest Selected"
             │
┌────────────▼────────────────────────────────────────┐
│ LAYER 2: Business Manager (Future)                  │
│ WorkflowEngine.continue(case_id)                   │
│ - Step 2: Ingest selected                          │
└────────────┬────────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────────┐
│ LAYER 3: Technical Manager                         │
│ DocumentDomainService.ingest_document()            │
│ - Persistence happens HERE (correct layer!)        │
│ - Dispatcher triggers embedding                    │
└────────────┬────────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────────┐
│ LAYER 4: Workers                                    │
│ - Repository.upsert() → DB                         │
│ - Embedder.embed() → Vectors                       │
└─────────────────────────────────────────────────────┘
```

**Benefits**:

- ✅ Clear layer separation
- ✅ HITL flows through all layers
- ✅ Persistence in Layer 3 (Technical Manager)
- ✅ Workers just work (Layer 4)
- ✅ Prepared for Layer 2 (Business) with IDs

---

## 🏗️ Proposed Clean Architecture

### Principle: Separation of Concerns

```
┌─────────────────────────────────────────────────────┐
│ HTTP/View Layer (Django)                            │
│ - Request validation                                │
│ - Mode detection                                    │
│ - Response formatting                               │
└─────────────────┬───────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────┐
│ Service/Coordinator Layer                           │
│ - Orchestrate flow based on mode                   │
│ - HITL session management                           │
│ - Task queuing                                      │
└─────────────────┬───────────────────────────────────┘
                  │
        ┌─────────┴──────────┐
        │                    │
┌───────▼──────┐   ┌─────────▼────────┐
│ Fetch Layer  │   │  Ingest Layer    │
│ (Worker)     │   │  (Celery Task)   │
└───────┬──────┘   └─────────┬────────┘
        │                    │
        └─────────┬──────────┘
                  │
        ┌─────────▼──────────┐
        │  Graph Layer       │
        │  (Processing)      │
        └────────────────────┘
```

---

## 📦 Layer Responsibilities (Clean Design)

### **Layer 1: HTTP Entry Points**

**Location**: `ai_core/views.py`

**New Endpoints**:

```python
# HITL Flow
POST /api/v1/crawler/fetch-for-review
GET  /api/v1/crawler/review-session/{id}
POST /api/v1/crawler/ingest-selected

# Direct Flow (bypass HITL)
POST /api/v1/crawler/ingest-direct
```

**Responsibility**:

- ✅ Request validation
- ✅ Mode extraction (RAG/Archive/Ephemeral)
- ✅ Dispatch to service layer
- ❌ NO business logic
- ❌ NO graph invocation

---

### **Layer 2: Service/Coordinator**

**Location**: `ai_core/services/crawler_coordinator.py` (NEW)

**Responsibilities**:

```python
class CrawlerCoordinator:
    """Orchestrates crawler operations based on mode."""
    
    def fetch_for_review(self, urls, ...) -> ReviewSession:
        """Phase 1 of HITL: Fetch and cache preview."""
        task = fetch_urls_forpreview.delay(urls, ...)
        return ReviewSession(task_id=task.id, ...)
    
    def ingest_selected(self, session_id, selected_urls, mode, ...) -> IngestResult:
        """Phase 2 of HITL: Ingest selected URLs with mode."""
        task = ingest_urls.delay(session_id, selected_urls, mode, ...)
        return IngestResult(task_id=task.id, ...)
    
    def ingest_direct(self, urls, mode, ...) -> IngestResult:
        """Direct ingestion (bypass HITL)."""
        task = ingest_urls.delay(None, urls, mode, ...)
        return IngestResult(task_id=task.id, ...)
```

**Key Decision**:

- ✅ Coordinator doesn't fetch/process
- ✅ Coordinator only queues Celery tasks
- ✅ Mode-driven dispatch

---

### **Layer 3: Fetch Layer (Worker)**

**Location**: `crawler/fetch_worker.py` (REFACTORED from `worker.py`)

**Responsibilities**:

```python
class FetchWorker:
    """Fetch content WITHOUT persistence."""
    
    def fetch_url(self, url: str) -> FetchResult:
        """Fetch single URL, return content + metadata."""
        content = HttpFetcher().fetch(url)
        parser = get_parser(content.content_type)
        parsed = parser.parse(content.body)
        
        return FetchResult(
            url=url,
            content_body=content.body,
            content_type=content.content_type,
            title=parsed.title,
            snippet=parsed.primary_text[:500],
            metadata={...},
        )
```

**Key Changes**:

- ✅ NO document registration
- ✅ NO persistence
- ✅ Pure fetching + parsing
- ✅ Returns structured data

**Used By**:

- HITL preview task
- Direct ingest task (via cache)

---

### **Layer 4: Ingest Layer (Celery Tasks)**

**Location**: `crawler/tasks.py` (NEW)

**Responsibilities**:

```python
from documents.domain_service import DocumentDomainService

@shared_task
def ingest_urls(
    session_id: str | None,  # None = direct ingest
    urls: list[str],
    mode: Literal["rag", "archive", "ephemeral"],
    tenant_id: str,
    trace_id: str,  # AGENTS.md: Required!
    case_id: str | None,
    workflow_id: str,
    collection_id: str | None,
    embedding_profile: str | None,
    **kwargs
) -> dict:
    """
    Ingest URLs with specified mode.
    
    AGENTS.md Compliant:
    - Uses DocumentDomainService (central authority)
    - Includes required IDs (tenant_id, trace_id)
    - Dispatcher pattern for embedding trigger
    - Lifecycle states set explicitly
    """
    # AGENTS.md: Get tenant object
    tenant = Tenant.objects.get(schema_name=tenant_id)
    
    # AGENTS.md: Use DocumentDomainService as central authority!
    domain_service = DocumentDomainService(vector_store=get_default_client())
    
    results = {"completed": [], "failed": {}, "documents_created": []}
    
    for url in urls:
        try:
            # Get content (from cache or fetch)
            if session_id:
                content = get_cached_content(session_id, url)
            else:
                fetch_result = FetchWorker().fetch_url(url)
                content = fetch_result.content_body
            
            # Prepare metadata
            metadata = {
                "title": fetch_result.title,
                "origin_uri": url,
                "workflow_id": workflow_id,
                "case_id": case_id,
                "trace_id": trace_id,  # AGENTS.md: Pflicht!
                "external_ref": {  # AGENTS.md: Correct structure
                    "provider": "web",
                    "external_id": f"web::{url}",
                },
            }
            
            # Hash content for idempotency
            content_hash = hashlib.sha256(content).hexdigest()
            
            # Get/create collections
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
                    # Dispatcher fires embedding task only if mode == "rag"
                    trigger_embedding_task(doc_id, profile)
                    if mode == "rag" else None
                ),
                initial_lifecycle_state=(
                    "pending" if mode == "rag" else "active"
                ),
            )
            
            results["completed"].append(url)
            results["documents_created"].append(str(ingest_result.document.id))
            
        except Exception as e:
            results["failed"][url] = str(e)
    
    return results
```

**Key Decisions** (AGENTS.md Compliant):

- ✅ Uses `DocumentDomainService` (not repository directly)
- ✅ Carries all required IDs (`tenant_id`, `trace_id`, `workflow_id`, `case_id`)
- ✅ Dispatcher pattern for embedding triggers
- ✅ Lifecycle states set explicitly (`pending` for RAG, `active` for Archive)
- ✅ Single persistence point through domain service
- ✅ Atomic (transaction or fail)

---

### **Layer 5: Graph Layer**

**Location**: `ai_core/graphs/crawler_ingestion_graph.py`

**Responsibilities**:

```python
class CrawlerIngestionGraph:
    """Process document through pipeline."""
    
    def run(self, document: NormalizedDocument, mode: str):
        """
        Process document based on mode.
        Document ALREADY persisted by ingest layer.
        """
        # Parse → Chunk → Embed (based on mode)
        if mode == "rag":
            parsed = self._parser.parse(document)
            chunks = self._chunker.chunk(parsed)
            # Embeddings handled by separate task
        elif mode == "archive":
            # Just parse for full-text search
            parsed = self._parser.parse(document)
        elif mode == "ephemeral":
            # No-op, document shouldn't even be here
            pass
```

**Key Changes**:

- ✅ Document already exists
- ✅ Graph focuses on PROCESSING
- ✅ NO persistence in graph
- ✅ Mode-aware

---

## 🔄 Flow Comparison

### **Old Flow (Current - Messy)**

```
Search & Ingest UI
    ↓
crawl_selected view
    ↓
run_crawler_runner
    ├─ build_crawler_state
    ├─ CrawlerWorker.process()
    │  ├─ Fetch
    │  ├─ **_register_document()** ← Document created!
    │  └─ Publish to Celery
    └─ OR run graph inline
        ↓
    CrawlerIngestionGraph.run()
        ├─ Assumes document exists
        └─ Process
```

**Problems**:

- Document created BEFORE graph
- No mode support
- No HITL
- Mixed sync/async

---

### **New Flow (Clean - HITL)**

```
Search & Ingest UI
    ↓
POST /crawler/fetch-for-review
    ↓
CrawlerCoordinator.fetch_for_review()
    ↓
Celery: fetch_urls_for_preview
    ├─ FetchWorker.fetch_url() (no persist)
    ├─ Extract preview
    └─ Cache in Redis
    ↓
GET /crawler/review-session/{id}
    ↓
UI shows previews
    ↓
User selects
    ↓
POST /crawler/ingest-selected
    ↓
CrawlerCoordinator.ingest_selected()
    ↓
Celery: ingest_urls (mode="rag")
    ├─ Get cached content
    ├─ Parse
    ├─ **repository.upsert()** ← Single persistence!
    └─ trigger_embedding_task() (mode="rag")
```

**Benefits**:

- ✅ HITL support
- ✅ Mode-driven
- ✅ Single persistence
- ✅ Clean separation

---

### **New Flow (Clean - Direct)**

```
POST /crawler/ingest-direct {"urls": [...], "mode": "archive"}
    ↓
CrawlerCoordinator.ingest_direct()
    ↓
Celery: ingest_urls (mode="archive")
    ├─ FetchWorker.fetch_url()
    ├─ Parse
    ├─ **repository.upsert()** ← Single persistence!
    └─ Skip embedding (mode="archive")
```

**Benefits**:

- ✅ Bypass HITL for automation
- ✅ Still mode-aware
- ✅ Same persistence logic

---

## 🔧 Migration Strategy

### Phase 1: Create New Layers (NO Breaking Changes)

1. **New**: `crawler/fetch_worker.py` (refactored CrawlerWorker)
   - Extract fetch logic
   - Remove `_register_document()`
   - Pure function

2. **New**: `crawler/tasks.py`
   - `ingest_urls` task
   - Mode-driven logic
   - Single persistence

3. **New**: `ai_core/services/crawler_coordinator.py`
   - Orchestration logic
   - HITL support

4. **New**: Endpoints in `ai_core/views.py`
   - `/crawler/fetch-for-review`
   - `/crawler/ingest-selected`
   - `/crawler/ingest-direct`

**Status**: Old code still works, new code available

---

### Phase 2: Migrate Search & Ingest

**Current**:

```python
# theme/views.py
response = crawl_selected(crawl_request)
```

**New**:

```python
# theme/views.py
coordinator = CrawlerCoordinator()
result = coordinator.fetch_for_review(urls, ...)
# ... user reviews ...
result = coordinator.ingest_selected(session_id, selected_urls, mode="rag")
```

**Impact**: Search & Ingest gets HITL for free!

---

### Phase 3: Deprecate Old Code

1. Add deprecation warning to `crawl_selected`
2. Add deprecation warning to `CrawlerWorker._register_document`
3. Monitor usage

---

### Phase 4: Remove Old Code

1. Delete `crawl_selected` (if unused)
2. Delete parallel registration
3. Clean up

---

## 📊 Comparison Table

| Aspect | Current (Messy) | Proposed (Clean - AGENTS.md Compliant) |
|--------|-----------------|----------------------------------------|
| **Layers** | 5 (mixed) | 5 (separated) |
| **Persistence Points** | 2 (parallel + ?) | 1 (DocumentDomainService) |
| **Central Authority** | None | DocumentDomainService ✅ |
| **Required IDs** | Partial | All (tenant_id, trace_id, etc.) ✅ |
| **Mode Support** | ❌ | ✅ 3 modes |
| **HITL** | ❌ | ✅ Built-in |
| **Async** | Mixed | Clear (Celery) |
| **Lifecycle States** | Implicit | Explicit (6 MVP states) ✅ |
| **Dispatcher Pattern** | ❌ | ✅ Callback-based |
| **Search & Ingest Impact** | - | Upgrade path |
| **Breaking Changes** | - | None (phased) |
| **Lines of Code** | ~1000 | ~600 |
| **AGENTS.md Compliance** | ❌ | ✅ 100% |

---

## 🎯 Recommendations

### Option A: Full Redesign (Recommended)

**Pros**:

- ✅ Clean architecture
- ✅ Mode support
- ✅ HITL built-in
- ✅ Single persistence
- ✅ No drift

**Cons**:

- Migration effort (1-2 weeks)

**Timeline**:

- Phase 1: 3 days (new code)
- Phase 2: 2 days (migrate Search & Ingest)
- Phase 3-4: 3 days (deprecate & remove)

---

### Option B: Minimal Fix (Drift #3 only)

**Pros**:

- ✅ Quick (1 week)
- ✅ Fixes parallel registration

**Cons**:

- ❌ No mode support
- ❌ No HITL
- ❌ Still messy

**Not recommended** given pre-MVP freedom.

---

## 🤔 Open Questions

1. **Search & Ingest HITL**: Should we enable HITL for Search & Ingest immediately, or later?
   - Recommendation: Later (Phase 2), but enable in Phase 1

2. **Backwards Compatibility**: Should we keep `crawl_selected` during transition?
   - Recommendation: Yes, with deprecation warning

3. **Mode Default**: What's the default mode if not specified?
   - Recommendation: `"rag"` (current behavior)

4. **Ephemeral Mode**: Should ephemeral mode even create a document?
   - Recommendation: No, return text in task result

---

## 🚀 Decision Required

**Question**: Which approach?

- **A**: Full redesign (clean architecture, 2 weeks)
- **B**: Minimal fix (just remove parallel registration, 1 week)
- **C**: Something else?

**My Recommendation**: **Option A** - we're pre-MVP, let's do it right!
