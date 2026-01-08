# Universal Ingestion Graph - Implementation Plan

**Branch**: `UniversalIngestionGraph`
**Status**: In Review
**Priority**: Critical fixes required before merge

This document provides a step-by-step implementation plan for addressing issues identified in the code review of the Universal Ingestion Graph refactoring.

---

## Phase 1: Critical Fixes (Blocking Merge)

### 1.1 Add invocation_id Validation in Crawler Runner

**File**: `ai_core/services/crawler_runner.py`
**Location**: Lines 209-227
**Priority**: 🔴 Critical

**Current Code**:
```python
# Validate mandatory IDs before graph invocation
tenant_id = scope_meta.get("tenant_id")
if not tenant_id:
    raise ValueError("tenant_id is mandatory for crawler ingestion")

case_id = scope_meta.get("case_id")
if not case_id:
    raise ValueError("case_id is mandatory for AI Core graph runs")

trace_id = scope_meta.get("trace_id")
if not trace_id:
    trace_id = str(uuid4())
    logger.warning("trace_id_missing_generated", ...)
```

**Required Changes**:
```python
# Validate mandatory IDs before graph invocation
required_ids = {
    "tenant_id": "tenant_id is mandatory for crawler ingestion",
    "case_id": "case_id is mandatory for AI Core graph runs",
    "trace_id": "trace_id is mandatory for correlation",
    "invocation_id": "invocation_id is mandatory per ID contract",
}

for field, error_msg in required_ids.items():
    if not scope_meta.get(field):
        raise ValueError(error_msg)

tenant_id = scope_meta["tenant_id"]
case_id = scope_meta["case_id"]
trace_id = scope_meta["trace_id"]
```

**Acceptance Criteria**:
- ✅ Raises `ValueError` if `invocation_id` is missing
- ✅ Raises `ValueError` if any required ID is missing
- ✅ No auto-generation fallbacks for required fields
- ✅ Test case verifies exception is raised

**Test**:
```python
# In ai_core/tests/test_crawler_runner.py
def test_crawler_runner_requires_invocation_id():
    meta = {
        "scope_context": {
            "tenant_id": "t1",
            "case_id": "c1",
            "trace_id": "tr1",
            # Missing invocation_id
        }
    }
    request = CrawlerRunRequest(...)

    with pytest.raises(ValueError, match="invocation_id is mandatory"):
        run_crawler_runner(meta=meta, request_model=request, lifecycle_store=None)
```

---

### 1.2 Add service_id Validation for S2S Hops

**File**: `ai_core/services/crawler_runner.py`
**Location**: After line 227
**Priority**: 🔴 Critical

**Required Changes**:
```python
# After validating required IDs, validate identity hop type
service_id = scope_meta.get("service_id")
user_id = scope_meta.get("user_id")

# Crawler runner is always an S2S hop (Celery task)
if not service_id:
    raise ValueError(
        "service_id is required for S2S hops (crawler ingestion). "
        "Expected value: 'crawler-worker' or 'celery-ingestion-worker'"
    )

if user_id:
    logger.warning(
        "user_id_present_in_s2s_hop",
        extra={
            "service_id": service_id,
            "user_id": user_id,
            "tenant_id": str(tenant_id),
            "trace_id": trace_id,
        }
    )
```

**Acceptance Criteria**:
- ✅ Raises `ValueError` if `service_id` is missing in crawler context
- ✅ Logs warning if both `user_id` and `service_id` present (invalid state)
- ✅ Test verifies S2S hop validation

**Test**:
```python
def test_crawler_runner_requires_service_id():
    meta = {
        "scope_context": {
            "tenant_id": "t1",
            "case_id": "c1",
            "trace_id": "tr1",
            "invocation_id": "inv1",
            # Missing service_id for S2S hop
        }
    }

    with pytest.raises(ValueError, match="service_id is required for S2S"):
        run_crawler_runner(meta=meta, request_model=request, lifecycle_store=None)
```

---

### 1.3 Remove trace_id Fallback in persist_node

**File**: `ai_core/graphs/technical/universal_ingestion_graph.py`
**Location**: Line 474
**Priority**: 🔴 Critical

**Current Code**:
```python
scope = ScopeContext(
    tenant_id=context["tenant_id"],
    trace_id=context["trace_id"],
    # Fallback to trace_id if invocation_id not explicit
    invocation_id=context.get("invocation_id") or context["trace_id"],
    ...
)
```

**Required Changes**:
```python
scope = ScopeContext(
    tenant_id=context["tenant_id"],
    trace_id=context["trace_id"],
    invocation_id=context["invocation_id"],  # No fallback - let it fail if missing
    case_id=context.get("case_id"),
    workflow_id=context.get("workflow_id"),
    run_id=context.get("run_id"),
    ingestion_run_id=context.get("ingestion_run_id"),
    collection_id=state.get("input", {}).get("collection_id"),
    user_id=context.get("user_id"),
    service_id=context.get("service_id"),
)
```

**Acceptance Criteria**:
- ✅ No fallback to `trace_id` when `invocation_id` missing
- ✅ Raises `KeyError` or `ValueError` if `invocation_id` not in context
- ✅ Test verifies error propagation

**Test**:
```python
def test_persist_node_requires_invocation_id():
    state = {
        "normalized_document": norm_doc,
        "context": {
            "tenant_id": "t1",
            "trace_id": "tr1",
            # Missing invocation_id
        },
        "input": {"collection_id": "col1"},
    }

    result = persist_node(state)
    assert "error" in result
    assert "invocation_id" in result["error"].lower()
```

---

### 1.4 Fix Idempotency Cache Collision Risk

**File**: `ai_core/services/crawler_runner.py`
**Location**: Lines 148-165
**Priority**: 🔴 Critical

**Current Code**:
```python
fingerprint_payload = {
    "tenant_id": str(tenant_id_for_fp),
    "case_id": scope_meta.get("case_id"),  # Optional - collision risk!
    "workflow_id": str(workflow_resolved),
    "collection_id": request_model.collection_id,
    "mode": request_model.mode,
    "origins": sorted([...]),
}
```

**Required Changes**:
```python
# Strategy 1: Require case_id in fingerprint (recommended)
fingerprint_payload = {
    "tenant_id": str(tenant_id_for_fp),
    "case_id": str(scope_meta["case_id"]),  # Required (already validated above)
    "workflow_id": str(workflow_resolved),
    "collection_id": request_model.collection_id,
    "mode": request_model.mode,
    "origins": sorted([...]),
}

# OR Strategy 2: Separate cache keys for with/without case_id
cache_namespace = "with_case" if scope_meta.get("case_id") else "no_case"
cache_key = f"{CACHE_PREFIX}{cache_namespace}:fp:{fingerprint}"
```

**Acceptance Criteria**:
- ✅ No collision between requests from different cases
- ✅ Idempotency works correctly for same request
- ✅ Test verifies different cases produce different fingerprints

**Test**:
```python
def test_idempotency_cache_case_isolation():
    # Same request, different cases
    request1 = CrawlerRunRequest(origins=[...])
    meta1 = {"scope_context": {"case_id": "case1", ...}}
    meta2 = {"scope_context": {"case_id": "case2", ...}}

    result1 = run_crawler_runner(meta=meta1, request_model=request1, ...)
    result2 = run_crawler_runner(meta=meta2, request_model=request1, ...)

    # Should NOT be idempotent (different cases)
    assert not result1.payload.get("idempotent")
    assert not result2.payload.get("idempotent")
```

---

### 1.5 Create Comprehensive Universal Ingestion Graph Tests

**File**: `ai_core/tests/graphs/test_universal_ingestion_graph.py` (NEW)
**Priority**: 🔴 Critical

**Test Coverage Required**:

```python
"""Comprehensive tests for UniversalIngestionGraph."""
import pytest
from ai_core.graphs.technical.universal_ingestion_graph import (
    build_universal_ingestion_graph,
    UniversalIngestionInput,
    UniversalIngestionState,
)

# ===== Upload Source Tests =====
def test_upload_source_ingest_only():
    """Test upload with inline blob."""
    graph = build_universal_ingestion_graph()

    input_payload = {
        "source": "upload",
        "mode": "ingest_only",
        "collection_id": "col-123",
        "upload_blob": {
            "type": "inline",
            "media_type": "application/pdf",
            "base64": "...",
            "sha256": "abc123...",
        },
        "metadata_obj": {"title": "Test Doc"},
        "normalized_document": None,
        "search_query": None,
        "search_config": None,
        "preselected_results": None,
    }

    context = {
        "tenant_id": "tenant-1",
        "trace_id": "trace-1",
        "invocation_id": "inv-1",
        "case_id": "case-1",
        "ingestion_run_id": "run-1",
    }

    result = graph.invoke({"input": input_payload, "context": context})

    assert result["output"]["decision"] == "ingested"
    assert result["output"]["document_id"] is not None
    assert "normalize" in result["output"]["transitions"]
    assert "persist" in result["output"]["transitions"]
    assert "process" in result["output"]["transitions"]


def test_upload_source_with_normalized_document():
    """Test upload with pre-normalized document (bypass blob building)."""
    # ... test normalized_document input path


# ===== Crawler Source Tests =====
def test_crawler_source_ingest_only():
    """Test crawler with pre-normalized document."""
    input_payload = {
        "source": "crawler",
        "mode": "ingest_only",
        "collection_id": "col-123",
        "normalized_document": {
            "ref": {...},
            "meta": {...},
            "blob": {...},
            "checksum": "...",
        },
        # Other fields null
    }
    # ... assert ingestion succeeds


# ===== Search Source Tests =====
def test_search_source_acquire_and_ingest():
    """Test search with query → acquisition → ingestion."""
    input_payload = {
        "source": "search",
        "mode": "acquire_and_ingest",
        "collection_id": "col-123",
        "search_query": "test query",
        "search_config": {
            "min_snippet_length": 40,
            "blocked_domains": [],
            "top_n": 5,
            "prefer_pdf": True,
        },
    }

    context = {
        ...,
        "runtime_worker": MockSearchWorker(),  # Mock search worker
    }

    result = graph.invoke({"input": input_payload, "context": context})

    assert result["output"]["decision"] == "ingested"
    assert "search" in result["output"]["transitions"]
    assert "select" in result["output"]["transitions"]
    assert "normalize" in result["output"]["transitions"]


def test_search_source_acquire_only():
    """Test search with acquire_only mode (no ingestion)."""
    input_payload = {
        "source": "search",
        "mode": "acquire_only",
        "search_query": "test query",
        ...
    }

    result = graph.invoke({"input": input_payload, "context": context})

    assert result["output"]["decision"] == "acquired"
    assert result["output"]["document_id"] is None  # Not persisted
    assert "search" in result["output"]["transitions"]
    assert "select" in result["output"]["transitions"]
    assert "normalize" not in result["output"]["transitions"]


def test_search_source_with_preselected_results():
    """Test search with preselected_results (bypass search worker)."""
    input_payload = {
        "source": "search",
        "mode": "acquire_and_ingest",
        "collection_id": "col-123",
        "search_query": None,  # Not required when preselected provided
        "preselected_results": [
            {"url": "https://example.com/doc1", "title": "Doc 1"},
            {"url": "https://example.com/doc2", "title": "Doc 2"},
        ],
    }

    result = graph.invoke({"input": input_payload, "context": context})

    assert result["output"]["decision"] == "ingested"
    # Should use preselected results directly, not call search worker


# ===== Error Handling Tests =====
def test_validation_error_propagates():
    """Test that validation errors are properly handled."""
    input_payload = {
        "source": "upload",
        "mode": "ingest_only",
        # Missing collection_id and upload_blob
    }

    result = graph.invoke({"input": input_payload, "context": context})

    assert result["output"]["decision"] == "error"
    assert "collection_id" in result["output"]["reason"].lower()


def test_missing_tenant_id():
    """Test error when tenant_id missing."""
    context = {
        "trace_id": "trace-1",
        # Missing tenant_id
    }

    result = graph.invoke({"input": input_payload, "context": context})

    assert result["output"]["decision"] == "error"
    assert "tenant_id" in result["output"]["reason"].lower()


def test_missing_invocation_id():
    """Test error when invocation_id missing."""
    context = {
        "tenant_id": "t1",
        "trace_id": "tr1",
        # Missing invocation_id - should fail in persist_node
    }

    result = graph.invoke({"input": input_payload, "context": context})

    assert result["output"]["decision"] == "error"


# ===== Feature Tests =====
def test_embedding_auto_enabled_for_search():
    """Test that enable_embedding=True is set for search results."""
    # ... verify pipeline_config.enable_embedding is True


def test_embedding_auto_enabled_for_crawler():
    """Test that enable_embedding=True is set for crawler."""
    # ... verify pipeline_config.enable_embedding is True


@pytest.mark.slow
def test_document_processing_graph_cached():
    """Test that processing graph is cached (singleton)."""
    from ai_core.graphs.technical.universal_ingestion_graph import (
        _get_cached_processing_graph,
        _CACHED_PROCESSING_GRAPH,
    )

    graph1 = _get_cached_processing_graph()
    graph2 = _get_cached_processing_graph()

    assert graph1 is graph2  # Same instance


# ===== Integration Tests =====
@pytest.mark.slow
def test_end_to_end_upload_ingestion(db, tenant_context):
    """Full integration test for upload ingestion."""
    # Uses real database, real services
    # Verifies document is persisted and embedded
    ...


@pytest.mark.slow
def test_end_to_end_crawler_ingestion(db, tenant_context):
    """Full integration test for crawler ingestion."""
    ...
```

**Acceptance Criteria**:
- ✅ All 3 sources tested (upload, crawler, search)
- ✅ All 3 modes tested (ingest_only, acquire_only, acquire_and_ingest)
- ✅ Error handling tests for missing required fields
- ✅ Feature tests for embedding auto-enable
- ✅ Integration tests with real DB (marked `@pytest.mark.slow`)
- ✅ >90% code coverage for universal_ingestion_graph.py

---

### 1.6 Add Crawler Runner Integration Tests

**File**: `ai_core/tests/test_crawler_runner.py` (NEW or UPDATE)
**Priority**: 🔴 Critical

**Test Coverage Required**:

```python
"""Integration tests for CrawlerRunner with Universal Ingestion Graph."""

def test_crawler_runner_single_origin():
    """Test crawler runner with single origin."""
    request = CrawlerRunRequest(
        origins=[
            CrawlerOrigin(
                url="https://example.com/doc.pdf",
                provider="https",
            )
        ],
        collection_id="col-123",
        mode="ingest_and_process",
    )

    meta = {
        "scope_context": {
            "tenant_id": "tenant-1",
            "trace_id": "trace-1",
            "invocation_id": "inv-1",
            "case_id": "case-1",
            "service_id": "crawler-worker",
            "ingestion_run_id": "run-1",
        }
    }

    result = run_crawler_runner(
        meta=meta,
        request_model=request,
        lifecycle_store=None,
    )

    assert result.status_code == 200
    assert not result.payload.get("idempotent")
    assert len(result.payload["origins"]) == 1
    assert result.payload["origins"][0]["document_id"] is not None


def test_crawler_runner_idempotency():
    """Test that duplicate requests are idempotent."""
    # Run same request twice
    result1 = run_crawler_runner(...)
    result2 = run_crawler_runner(...)  # Same fingerprint

    assert not result1.payload.get("idempotent")
    assert result2.payload.get("idempotent")
    assert result2.payload.get("skipped")


def test_crawler_runner_dry_run():
    """Test dry_run mode doesn't persist."""
    request = CrawlerRunRequest(dry_run=True, ...)
    result = run_crawler_runner(...)

    # Should complete but not persist to DB
    assert result.status_code == 200
    # Verify document not in DB


def test_crawler_runner_multiple_origins():
    """Test crawler with multiple origins."""
    request = CrawlerRunRequest(
        origins=[
            CrawlerOrigin(url="https://example.com/doc1.pdf"),
            CrawlerOrigin(url="https://example.com/doc2.pdf"),
        ],
        ...
    )

    result = run_crawler_runner(...)

    assert len(result.payload["origins"]) == 2
    assert len(result.payload["transitions"]) == 2


def test_crawler_runner_with_guardrail_failure():
    """Test guardrail rejection."""
    request = CrawlerRunRequest(
        origins=[CrawlerOrigin(url="https://huge-file.com/10GB.pdf")],
        max_document_bytes=1024 * 1024,  # 1 MB limit
    )

    result = run_crawler_runner(...)

    assert result.status_code == 413  # Request Entity Too Large
    assert "guardrail" in result.payload.get("code", "").lower()
```

**Acceptance Criteria**:
- ✅ Tests single and multiple origins
- ✅ Tests idempotency behavior
- ✅ Tests dry_run mode
- ✅ Tests guardrail failures
- ✅ Verifies ingestion_run_id propagation

---

### 1.7 Run ID Contract Checklist Verification

**File**: Execute tests from `docs/architecture/id-contract-review-checklist.md`
**Priority**: 🔴 Critical

**Commands to Run**:
```bash
# Core contracts
npm run win:test:py:unit -- ai_core/contracts/ ai_core/tool_contracts/ -v

# ID normalization
npm run win:test:py:unit -- ai_core/ids/tests/ -v

# Middleware
npm run win:test:py:unit -- ai_core/tests/test_request_context_middleware.py -v

# Tool context
npm run win:test:py:unit -- ai_core/tests/test_tool_context.py ai_core/tests/test_tool_context_adapter.py -v

# Full unit suite
npm run win:test:py:unit
```

**Acceptance Criteria**:
- ✅ All contract tests pass
- ✅ All ID normalization tests pass
- ✅ All middleware tests pass
- ✅ All tool context tests pass
- ✅ Zero test failures in unit suite

---

## Phase 2: High Priority Improvements (Before Merge)

### 2.1 Replace Magic Checksum with URL Hash

**File**: `ai_core/graphs/technical/universal_ingestion_graph.py`
**Location**: Line 425
**Priority**: 🟡 High

**Current Code**:
```python
norm_doc = NormalizedDocument(
    ref=ref,
    meta=meta,
    blob=blob,
    checksum="0" * 64,  # Magic string!
    source="other",
    created_at=datetime.now(timezone.utc),
)
```

**Required Changes**:
```python
import hashlib

# Calculate checksum from URL for search results
url = selected.get("url", "")
url_checksum = hashlib.sha256(url.encode("utf-8")).hexdigest()

norm_doc = NormalizedDocument(
    ref=ref,
    meta=meta,
    blob=blob,
    checksum=url_checksum,  # Deterministic hash from URL
    source="other",
    created_at=datetime.now(timezone.utc),
)
```

**Acceptance Criteria**:
- ✅ No magic strings for checksums
- ✅ Same URL produces same checksum (deterministic)
- ✅ Test verifies checksum is SHA256 of URL

---

### 2.2 Refactor normalize_document_node

**File**: `ai_core/graphs/technical/universal_ingestion_graph.py`
**Location**: Lines 353-455
**Priority**: 🟡 High

**Goal**: Break down 100+ line function into source-specific helpers

**New Structure**:
```python
def _normalize_from_upload(
    input_data: UniversalIngestionInput,
    context: dict[str, Any],
) -> NormalizedDocument:
    """Build NormalizedDocument from upload source."""
    # Extract logic from lines 371-372
    ...


def _normalize_from_crawler(
    input_data: UniversalIngestionInput,
) -> NormalizedDocument:
    """Build NormalizedDocument from crawler source."""
    # Extract logic for crawler (already normalized)
    raw_doc = input_data.get("normalized_document")
    if isinstance(raw_doc, dict):
        return NormalizedDocument.model_validate(raw_doc)
    return raw_doc


def _normalize_from_search(
    selected_result: dict[str, Any],
    collection_id: str,
    context: dict[str, Any],
) -> NormalizedDocument:
    """Build NormalizedDocument from search result."""
    # Extract logic from lines 375-428
    ...


def _ensure_embedding_enabled(
    norm_doc: NormalizedDocument,
    source: str,
) -> NormalizedDocument:
    """Force enable_embedding=True for search/crawler sources."""
    # Extract logic from lines 434-445
    ...


@observe_span(name="node.normalize")
def normalize_document_node(state: UniversalIngestionState) -> dict[str, Any]:
    """Normalize input into a NormalizedDocument object."""
    inp = state.get("input", {})
    source = inp.get("source")
    context = state.get("context", {})

    try:
        # Check explicit normalized input first
        raw_doc = inp.get("normalized_document")
        if raw_doc:
            norm_doc = _normalize_from_crawler(inp)
        elif source == "upload":
            norm_doc = _normalize_from_upload(inp, context)
        elif source == "search":
            selected = state.get("selected_result")
            if not selected:
                raise ValueError("Missing selected_result for search source")
            collection_id = inp.get("collection_id")
            if not collection_id:
                raise ValueError("Missing collection_id for search ingestion")
            norm_doc = _normalize_from_search(selected, collection_id, context)
        else:
            return {"error": "Could not verify normalized document"}

        # Ensure embedding enabled for all sources
        norm_doc = _ensure_embedding_enabled(norm_doc, source)

        return {"normalized_document": norm_doc}

    except ValidationError as ve:
        logger.error(f"Normalization failed: {ve}")
        return {"error": f"Normalization failed: {ve}"}
    except Exception as exc:
        logger.exception("Unexpected normalization error")
        return {"error": str(exc)}
```

**Acceptance Criteria**:
- ✅ Each helper function < 30 lines
- ✅ Clear separation of concerns
- ✅ All existing tests still pass
- ✅ Code coverage maintained or improved

---

### 2.3 Add Timeout Protection for Search Worker

**File**: `ai_core/graphs/technical/universal_ingestion_graph.py`
**Location**: Lines 242-280
**Priority**: 🟡 High

**Current Code**:
```python
response: WebSearchResponse = worker.run(query=query, context=telemetry_ctx)
```

**Required Changes**:
```python
from django.conf import settings

# Get timeout from settings with fallback
search_timeout = getattr(settings, "SEARCH_WORKER_TIMEOUT_SECONDS", 30)

try:
    # Wrap worker call with timeout
    import signal

    def timeout_handler(signum, frame):
        raise TimeoutError("Search worker exceeded timeout")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(search_timeout)

    try:
        response: WebSearchResponse = worker.run(query=query, context=telemetry_ctx)
    finally:
        signal.alarm(0)  # Cancel alarm

except TimeoutError:
    logger.warning(
        f"Search worker timeout after {search_timeout}s",
        extra={"query": query, "timeout": search_timeout}
    )
    return {"error": f"Search timeout after {search_timeout}s", "search_results": []}
```

**OR use threading.Timer**:
```python
import threading
from queue import Queue, Empty

def _run_search_with_timeout(worker, query, context, timeout):
    """Run search worker with timeout protection."""
    result_queue = Queue()

    def worker_thread():
        try:
            response = worker.run(query=query, context=context)
            result_queue.put(("success", response))
        except Exception as e:
            result_queue.put(("error", e))

    thread = threading.Thread(target=worker_thread, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        raise TimeoutError(f"Search worker exceeded {timeout}s timeout")

    try:
        status, result = result_queue.get_nowait()
        if status == "error":
            raise result
        return result
    except Empty:
        raise TimeoutError("Search worker produced no result")

# In search_node:
try:
    response = _run_search_with_timeout(
        worker,
        query,
        telemetry_ctx,
        timeout=search_timeout
    )
except TimeoutError as e:
    logger.warning(str(e), extra={"query": query})
    return {"error": str(e), "search_results": []}
```

**Acceptance Criteria**:
- ✅ Search worker times out after configured limit
- ✅ Timeout value configurable in Django settings
- ✅ Default timeout is 30 seconds
- ✅ Test verifies timeout behavior

---

### 2.4 Fix collection_id UUID Conversion

**File**: `ai_core/services/crawler_state_builder.py`
**Location**: Lines 295-298, 322-324
**Priority**: 🟡 High

**Current Code**:
```python
collection_uuid = _resolve_document_uuid(
    request_data.collection_id if request_data.collection_id else None
)

# Later used in ref:
normalized_document_input = NormalizedDocument(
    ref={
        "collection_id": collection_uuid,  # UUID object
        ...
    },
    ...
)
```

**Required Changes**:
```python
# collection_id should stay as string per ScopeContext contract
collection_id_str = request_data.collection_id

# In normalized_document_input:
normalized_document_input = NormalizedDocument(
    ref={
        "tenant_id": str(scope_meta.get("tenant_id")),
        "workflow_id": str(workflow_id),
        "document_id": document_uuid,  # UUID is OK here
        "collection_id": collection_id_str,  # String, not UUID
    },
    ...
)
```

**Acceptance Criteria**:
- ✅ `collection_id` remains string throughout
- ✅ No UUID conversion for `collection_id`
- ✅ Test verifies type is `str`, not `UUID`

---

### 2.5 Externalize Configuration

**File**: `ai_core/services/crawler_runner.py` and `ai_core/graphs/technical/universal_ingestion_graph.py`
**Priority**: 🟡 High

**Changes Required**:

**In crawler_runner.py**:
```python
# Current hardcoded values:
CACHE_PREFIX = "crawler_idempotency:"
CACHE_TTL = 3600  # 1 hour

# Change to:
from django.conf import settings

CACHE_PREFIX = getattr(settings, "CRAWLER_IDEMPOTENCY_CACHE_PREFIX", "crawler_idempotency:")
CACHE_TTL = getattr(settings, "CRAWLER_IDEMPOTENCY_CACHE_TTL", 3600)
```

**In universal_ingestion_graph.py**:
```python
# Add configuration for search
SEARCH_TIMEOUT = getattr(settings, "SEARCH_WORKER_TIMEOUT_SECONDS", 30)
SEARCH_MIN_SNIPPET_LENGTH = getattr(settings, "SEARCH_MIN_SNIPPET_LENGTH", 40)
SEARCH_TOP_N_DEFAULT = getattr(settings, "SEARCH_TOP_N_DEFAULT", 5)
```

**In Django settings** (`config/settings/base.py`):
```python
# Crawler Configuration
CRAWLER_IDEMPOTENCY_CACHE_PREFIX = env.str("CRAWLER_IDEMPOTENCY_CACHE_PREFIX", default="crawler_idempotency:")
CRAWLER_IDEMPOTENCY_CACHE_TTL = env.int("CRAWLER_IDEMPOTENCY_CACHE_TTL", default=3600)

# Search Configuration
SEARCH_WORKER_TIMEOUT_SECONDS = env.int("SEARCH_WORKER_TIMEOUT_SECONDS", default=30)
SEARCH_MIN_SNIPPET_LENGTH = env.int("SEARCH_MIN_SNIPPET_LENGTH", default=40)
SEARCH_TOP_N_DEFAULT = env.int("SEARCH_TOP_N_DEFAULT", default=5)
```

**Acceptance Criteria**:
- ✅ All magic numbers moved to settings
- ✅ Environment variables documented in `.env.example`
- ✅ Default values preserved
- ✅ Test with custom settings

---

### 2.6 Add Migration Documentation

**File**: `docs/architecture/universal-ingestion-migration-guide.md` (NEW)
**Priority**: 🟡 High

**Content Required**:

```markdown
# Migration Guide: Universal Ingestion Graph

## Overview

The Universal Ingestion Graph consolidates three previously separate ingestion flows:
- Upload ingestion
- Crawler ingestion
- Search ingestion

## Breaking Changes

### 1. collection_id Type Change
**Before**: `UUID` object
**After**: `str` (UUID string representation)

**Migration Required**: Convert UUID objects to strings
```python
# Old code
collection_id = uuid.UUID("550e8400-...")

# New code
collection_id = "550e8400-e29b-41d4-a716-446655440000"
```

### 2. invocation_id Now Mandatory
**Before**: Optional, would fallback to `trace_id`
**After**: Mandatory in all contexts

**Migration Required**: Always provide `invocation_id` in requests
```python
# Old code - would work with just trace_id
headers = {"X-Trace-ID": "trace-123"}

# New code - requires invocation_id
headers = {
    "X-Trace-ID": "trace-123",
    "X-Invocation-ID": "inv-456",  # Required!
}
```

### 3. service_id Required for S2S Hops
**Before**: Optional for Celery tasks
**After**: Mandatory for all S2S hops (crawler, ingestion workers)

**Migration Required**: Set `service_id` in task contexts
```python
# In Celery tasks
from ai_core.ids.http_scope import normalize_task_context

scope = normalize_task_context(
    tenant_id=tenant_id,
    service_id="crawler-worker",  # Required!
    trace_id=trace_id,
    invocation_id=invocation_id,
    run_id=run_id,
    ingestion_run_id=ingestion_run_id,
)
```

## API Changes

### Upload Ingestion
No API changes - existing endpoints work as before

### Crawler Ingestion
**New Requirement**: `case_id` header now mandatory
```python
# Old - would work without case_id
POST /v1/crawler/run
Headers: X-Tenant-ID

# New - requires case_id
POST /v1/crawler/run
Headers:
  X-Tenant-ID: tenant-123
  X-Case-ID: case-456  # Now required!
```

### Search Ingestion
New modes supported:
- `acquire_only` - Search and select, don't ingest
- `acquire_and_ingest` - Search, select, and ingest

## Code Migration Examples

### Example 1: Updating Crawler Calls
```python
# Before
from crawler.tasks import run_crawler
run_crawler.delay(origins=[...])

# After
from ai_core.services import run_crawler_runner
from ai_core.ids.http_scope import normalize_task_context

meta = {
    "scope_context": normalize_task_context(
        tenant_id=tenant_id,
        service_id="crawler-worker",  # Add this
        trace_id=trace_id,
        invocation_id=invocation_id,
        run_id=run_id,
        ingestion_run_id=ingestion_run_id,
    )
}
result = run_crawler_runner(
    meta=meta,
    request_model=request,
    lifecycle_store=None,
)
```

### Example 2: Updating Upload Ingestion
```python
# Before - direct service call
from documents.services import ingest_document
ingest_document(blob_data, metadata)

# After - via universal graph
from ai_core.graphs.technical.universal_ingestion_graph import (
    build_universal_ingestion_graph
)

graph = build_universal_ingestion_graph()
result = graph.invoke({
    "input": {
        "source": "upload",
        "mode": "ingest_only",
        "collection_id": "col-123",
        "upload_blob": blob_data,
        "metadata_obj": metadata,
    },
    "context": {
        "tenant_id": "tenant-1",
        "trace_id": "trace-1",
        "invocation_id": "inv-1",  # Required
        "case_id": "case-1",
        "ingestion_run_id": "run-1",
    },
})
```

## Testing Your Migration

1. **Run ID Contract Tests**:
```bash
npm run win:test:py:unit -- ai_core/contracts/ -v
```

2. **Run Integration Tests**:
```bash
npm run win:test:py:fast -- ai_core/tests/test_crawler_runner.py -v
```

3. **Verify Idempotency**:
```bash
# Test duplicate requests
npm run win:test:py:single -- ai_core/tests/test_crawler_runner.py::test_idempotency
```

## Rollback Plan

If issues arise, revert to previous branch:
```bash
git checkout main
git merge --abort  # If mid-merge
```

Previous ingestion endpoints remain available during transition period.

## Support

Questions? See:
- [ID Contract Review Checklist](id-contract-review-checklist.md)
- [ID Semantics](id-semantics.md)
- [AGENTS.md](../../AGENTS.md)
```

**Acceptance Criteria**:
- ✅ Documents all breaking changes
- ✅ Provides code migration examples
- ✅ Includes rollback instructions
- ✅ Lists all new requirements

---

## Phase 3: Medium Priority (Can Follow Up)

### 3.1 Add Memory Leak Mitigation

**File**: `ai_core/graphs/technical/universal_ingestion_graph.py`
**Priority**: ⚪ Medium

**Add cleanup function**:
```python
def _clear_cached_processing_graph():
    """Clear the cached processing graph (for testing/cleanup)."""
    global _CACHED_PROCESSING_GRAPH
    _CACHED_PROCESSING_GRAPH = None
    logger.info("Processing graph cache cleared")
```

**Add to test teardown**:
```python
# In conftest.py
@pytest.fixture(autouse=True)
def cleanup_graph_cache():
    """Clear graph cache after each test."""
    yield
    from ai_core.graphs.technical.universal_ingestion_graph import (
        _clear_cached_processing_graph
    )
    _clear_cached_processing_graph()
```

---

### 3.2 Add Rate Limiting for Search

**File**: `ai_core/graphs/technical/universal_ingestion_graph.py`
**Priority**: ⚪ Medium

**Add Django rate limiting**:
```python
from django_ratelimit.decorators import ratelimit
from django.core.cache import cache

def _check_search_rate_limit(tenant_id: str, query: str) -> bool:
    """Check if tenant has exceeded search rate limit."""
    cache_key = f"search_rate_limit:{tenant_id}"
    count = cache.get(cache_key, 0)

    max_searches_per_hour = getattr(settings, "MAX_SEARCHES_PER_TENANT_PER_HOUR", 100)

    if count >= max_searches_per_hour:
        return False

    cache.set(cache_key, count + 1, timeout=3600)
    return True

# In search_node:
tenant_id = context.get("tenant_id")
if not _check_search_rate_limit(tenant_id, query):
    return {
        "error": "Search rate limit exceeded",
        "search_results": []
    }
```

---

### 3.3 Remove Debug Code

**File**: `ai_core/services/crawler_runner.py`
**Location**: Lines 44-65
**Priority**: ⚪ Low

**Remove**:
```python
def debug_check_json_serializable(obj, path=""):
    # ... entire function
```

**And all calls to it**:
```python
# Remove this line:
debug_check_json_serializable(payload, "sync_payload_internal")
```

---

## Summary & Timeline

### Phase 1: Critical Fixes (3-5 days)
- Tasks 1.1-1.7 must be completed before merge
- Estimated: 24-40 hours of development + testing
- **Blocker**: Cannot merge until all Phase 1 tasks complete

### Phase 2: High Priority (2-3 days)
- Tasks 2.1-2.6 should be completed before merge
- Estimated: 16-24 hours
- **Recommended**: Complete before merge for quality

### Phase 3: Medium Priority (Follow-up PR)
- Tasks 3.1-3.3 can be done in separate PR
- Estimated: 4-8 hours
- **Optional**: Can merge main PR without these

### Total Estimated Effort
- **Before Merge**: 40-64 hours (5-8 days)
- **Follow-up**: 4-8 hours (0.5-1 day)
- **Total**: 44-72 hours (6-9 days)

---

## Daily Progress Tracking

Use the todo list to track daily progress:
```bash
# View current tasks
# Tasks are already loaded in the system

# Mark completed as you go
# Update status via TodoWrite tool
```

---

## Sign-Off Checklist

Before merging to `main`:

- [ ] All Phase 1 tasks completed and tested
- [ ] All Phase 2 tasks completed and tested
- [ ] Full test suite passes (`npm run win:test:py:unit`)
- [ ] Integration tests pass (`npm run win:test:py:fast`)
- [ ] ID Contract checklist verified
- [ ] Migration documentation reviewed
- [ ] Code review approved
- [ ] CI/CD pipeline green

---

**Last Updated**: 2025-12-23
**Document Owner**: Development Team
**Review Status**: Draft
