# Reality Check: /rag-tools/ Components - No Hallucinations

## 🎯 Objective

Verify what ACTUALLY exists in `/rag-tools/` and ensure our crawler redesign doesn't break it.

**Date**: 2025-12-11  
**Method**: Code inspection, no assumptions

---

## ✅ VERIFIED: What EXISTS in /rag-tools/

### Tab 1: Web Search (`#tab-search`)

**Location**: `theme/templates/theme/rag_tools.html:92-201`

**Features** (Line-by-line verified):

1. **Search Type Selection** (Lines 101-116):
   - ✅ External Knowledge (radio)
   - ✅ Collection Search (radio)

2. **Form Fields** (Lines 118-178):
   - ✅ Query (text input, required)
   - ✅ Purpose (text input, required for collection search)
   - ✅ Mode (select: `live` or `archive`)
   - ✅ Workflow ID (text input, default: `web-search`)
   - ✅ Collection ID (text input with datalist)
   - ✅ Quality Mode (select, hidden unless collection search)
   - ✅ Auto Ingest (checkbox, hidden unless collection search)

3. **Submit Button** (Line 181-184):
   - ✅ "Search" button
   - ✅ HTMX POST to `{% url 'web-search' %}`
   - ✅ Target: `#web-search-results`

4. **Result Display** (Lines 192-199):
   - ✅ `#web-search-results` div
   - ✅ Shows "Enter a query to see results here." placeholder

5. **Ingestion Status Panel** (Line 189):
   - ✅ `#ingestion-status-panel` div
   - ✅ Target for "Ingest Selected" results

**Backend Handler**: `theme/views.py:web_search_and_ingest()`

**Current Flow**:

```
User fills form → clicks "Search"
    ↓
HTMX POST to /web-search
    ↓
web_search_and_ingest() view
    ├─ If search_type == "external_knowledge":
    │  └─ Calls ExternalKnowledgeGraph
    │     └─ Returns search results with "Ingest Selected" button
    │
    └─ If search_type == "collection_search":
       └─ Calls CollectionSearchGraph
          └─ Returns search results with optional auto-ingest
```

**"Ingest Selected" Flow** (Verified in `theme/views.py:850-950`):

```
User clicks "Ingest Selected" on search results
    ↓
HTMX with selected URLs
    ↓
web_search_ingest_selected() view (Line 850+)
    ├─ Extract URLs from request
    ├─ Build crawl_payload:
    │  {
    │    "urls": [...],
    │    "workflow_id": "web-search-ingestion",
    │    "collection_id": "...",
    │    "mode": "live" or "archive",
    │  }
    ├─ Create synthetic HttpRequest
    └─ Calls crawl_selected(crawl_request) ← THIS IS THE INTEGRATION POINT!
       ↓
    Returns ingestion status to #ingestion-status-panel
```

**KEY FINDING**:

- ✅ Web Search → Ingest Selected calls `crawl_selected()`
- ✅ This is the integration point we MUST NOT BREAK!

---

### Tab 2: Crawler (`#tab-crawler`)

**Location**: `theme/templates/theme/rag_tools.html:203-269`

**Features** (Verified):

1. **Form Fields** (Lines 210-253):
   - ✅ Origin URL (url input)
   - ✅ Additional Origins (textarea, one per line)
   - ✅ Mode (select: `live` or `archive`)
   - ✅ Workflow ID (text input, default: `crawler-manual`)
   - ✅ Fetch Content (checkbox)
   - ✅ Dry Run (checkbox)
   - ✅ Shadow Mode (checkbox)

2. **Submit Button** (Lines 256-259):
   - ✅ "Start Crawl" button
   - ✅ HTMX POST to `{% url 'crawler-submit' %}`
   - ✅ Target: `#crawler-status-area`

3. **Status Display** (Lines 263-267):
   - ✅ `#crawler-status-area` div
   - ✅ Shows "Crawler status will appear here." placeholder

**Backend Handler**: `theme/views.py:crawler_submit_view()`

**Current Flow**:

```
User fills form → clicks "Start Crawl"
    ↓
HTMX POST to /crawler-submit
    ↓
crawler_submit_view()
    ├─ Parse origin_url and origin_urls
    ├─ Build CrawlerRunRequest payload
    └─ Calls run_crawler_runner() ← Different from Web Search!
       ↓
    Returns crawler status
```

**KEY FINDING**:

- ✅ Crawler Tab does NOT use `crawl_selected()`
- ✅ Uses `run_crawler_runner()` directly
- ⚠️ Different code path than Web Search!

---

### Tab 3: Ingestion (`#tab-ingestion`)

**Location**: `theme/templates/theme/rag_tools.html:271-305`

**Features** (Verified):

1. **Form Fields** (Lines 278-289):
   - ✅ Document IDs (textarea, JSON list or comma separated)
   - ✅ Embedding Profile (text input)

2. **Submit Button** (Lines 292-295):
   - ✅ "Run Ingestion" button
   - ✅ HTMX POST to `{% url 'ingestion-submit' %}`
   - ✅ Target: `#ingestion-response`

3. **Response Display** (Lines 299-303):
   - ✅ `#ingestion-response` div
   - ✅ Shows "Ingestion response will appear here." placeholder

**Backend Handler**: `theme/views.py:ingestion_submit_view()`

**Current Flow**:

```
User enters document IDs → clicks "Run Ingestion"
    ↓
HTMX POST to /ingestion-submit
    ↓
ingestion_submit_view()
    ├─ Parse document_ids
    ├─ Queue ingestion tasks
    └─ Returns task IDs
```

**KEY FINDING**:

- ✅ Completely independent from crawler
- ✅ Does NOT use crawler code paths

---

## 🔌 Integration Points (MUST NOT BREAK)

### **1. crawl_selected() - Critical Integration Point**

**Used By**:

- ✅ Web Search → "Ingest Selected" button
- ✅ Through `web_search_ingest_selected()`

**Location**: `ai_core/views.py:crawl_selected()`

**Contract** (Verified):

```python
@require_POST
def crawl_selected(request):
    # Expects JSON body:
    # {
    #   "urls": ["url1", "url2"],
    #   "workflow_id": "web-search-ingestion",
    #   "mode": "live" | "archive",
    #   "collection_id": "...",
    # }
    
    # Returns JSON:
    # {
    #   "task_ids": [...],
    #   "status": "accepted" | "completed",
    #   ...
    # }
```

**Current Implementation** (Verified in code):

```python
crawl_selected(request)
    ↓
_prepare_request(request) → meta
    ↓
json.loads(request.body) → data
    ↓
Build CrawlerRunRequest from data.urls
    ↓
run_crawler_runner(meta, request_model, lifecycle_store, graph_factory)
    ↓
Returns JsonResponse with task_ids
```

**⚠️ CRITICAL**: Any changes to crawler architecture must:

- ✅ Keep `crawl_selected()` API contract
- ✅ Accept same JSON body structure
- ✅ Return same JSON response structure
- ✅ Support `mode` parameter (live/archive)

---

### **2. run_crawler_runner() - Internal Integration**

**Used By**:

- ✅ `crawl_selected()` (Web Search path)
- ✅ `crawler_submit_view()` (Crawler Tab path)

**Location**: `ai_core/services/crawler_runner.py:run_crawler_runner()`

**Contract** (Verified):

```python
def run_crawler_runner(
    *,
    meta: dict[str, Any],
    request_model: CrawlerRunRequest,
    lifecycle_store: object | None,
    graph_factory: Callable[[], object] | None = None,
) -> CrawlerRunnerCoordinatorResult:
    # Returns:
    # CrawlerRunnerCoordinatorResult(
    #     payload={"task_ids": [...], ...},
    #     status_code=200 | 202,
    #     idempotency_key=...
    # )
```

**Current Flow** (Verified):

```python
run_crawler_runner(...)
    ↓
For each URL in request_model.origins:
    ├─ build_crawler_state() → state dict
    ├─ CrawlerWorker.process() ← Calls parallel registration!
    │  └─ _register_document() ← Creates document in DB
    └─ Publish to Celery OR run graph inline
```

**⚠️ PROBLEM**: This is where parallel registration happens!

---

## 🚨 What Our Redesign MUST Preserve

### **Backwards Compatibility Requirements**

1. **✅ crawl_selected() API**:
   - Same endpoint path
   - Same JSON contract
   - Same response structure
   - Support `mode` parameter

2. **✅ Web Search Integration**:
   - "Ingest Selected" button works
   - Results render correctly
   - Status panel updates

3. **✅ Crawler Tab**:
   - Form submission works
   - Status updates work
   - Can specify URLs manually

4. **✅ Mode Parameter**:
   - `mode: "live"` and `mode: "archive"` both work
   - Our new modes (rag/archive/ephemeral) should map correctly

---

## 🎯 Safe Migration Strategy

### **Option 1: Non-Breaking Refactor** (Recommended)

**Step 1**: Keep `crawl_selected()` as-is (facade)

```python
@require_POST
def crawl_selected(request):
    """
    UNCHANGED facade - maintains API contract.
    Internal implementation can change!
    """
    # ... existing validation ...
    
    # NEW: Delegate to new coordinator
    coordinator = CrawlerCoordinator()
    result = coordinator.ingest_direct(
        urls=urls,
        mode=mode,  # Map mode parameter
        tenant_id=meta["tenant_id"],
        workflow_id=workflow_id,
        collection_id=collection_id,
        ...
    )
    
    # Return same response format
    return JsonResponse({
        "task_ids": result.task_ids,
        "status": "accepted",
        ...
    })
```

**Step 2**: New coordinator handles logic

```python
class CrawlerCoordinator:
    def ingest_direct(self, urls, mode, ...):
        """
        NEW internal implementation.
        Maps old 'mode' to new modes:
        - "live" → "rag"
        - "archive" → "archive"
        """
        new_mode = "rag" if mode == "live" else "archive"
        task = ingest_urls.delay(
            session_id=None,  # No HITL for direct calls
            urls=urls,
            mode=new_mode,
            ...
        )
        return IngestResult(task_ids=[task.id])
```

**Benefits**:

- ✅ No API changes
- ✅ Web Search keeps working
- ✅ Cleaner internal architecture
- ✅ Can add HITL later without breaking anything

---

### **Option 2: Add HITL Alongside** (Future Enhancement)

**Keep existing flow**:

```
Web Search → "Ingest Selected" → crawl_selected() → works as before
```

**Add new HITL flow** (parallel, not replacing):

```
New UI → "Fetch for Review" → NEW endpoint → HITL flow
```

**Benefits**:

- ✅ Zero risk to existing features
- ✅ Can test HITL separately
- ✅ Gradual migration

---

## 📋 Verification Checklist

Before deploying any changes, verify:

### **Web Search Tab**

- [ ] Search returns results
- [ ] "Ingest Selected" button appears
- [ ] Clicking "Ingest Selected" triggers ingestion
- [ ] Status panel shows progress
- [ ] Mode parameter (live/archive) works

### **Crawler Tab**

- [ ] Can enter URLs manually
- [ ] "Start Crawl" triggers processing
- [ ] Status area shows results
- [ ] Mode parameter works

### **API Contracts**

- [ ] `crawl_selected()` accepts same JSON
- [ ] `crawl_selected()` returns same JSON structure
- [ ] `run_crawler_runner()` signature unchanged (or backwards compatible)

### **Error Cases**

- [ ] Invalid URLs handled gracefully
- [ ] Missing tenant_id shows error
- [ ] Empty URL list shows error

---

## 🎯 Conclusion

**What we VERIFIED**:

1. ✅ `/rag-tools/` has 3 tabs (Web Search, Crawler, Ingestion)
2. ✅ Web Search → "Ingest Selected" → calls `crawl_selected()`
3. ✅ `crawl_selected()` is the critical integration point
4. ✅ Current `mode` parameter: "live" or "archive"
5. ✅ Parallel registration happens in `run_crawler_runner()`

**What we MUST preserve**:

1. ✅ `crawl_selected()` API contract
2. ✅ Web Search → Ingest Selected flow
3. ✅ Mode parameter support
4. ✅ Response structure

**Safe approach**:

- ✅ Keep `crawl_selected()` as facade
- ✅ Refactor internal implementation
- ✅ Map old modes to new modes
- ✅ Add HITL as separate feature (later)

**NO DRIFT!** All existing `/rag-tools/` features will continue working! 🎯
