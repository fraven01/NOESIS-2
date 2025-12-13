# 🎯 CRITICAL FINDING: Persistence Happens in Parallel Registration ONLY

## Summary

**The graph does NOT persist the document.**  
**Only the parallel registration (`_register_document`) creates the document in the DB.**

This is the root cause of Drift #3.

---

## Evidence

### Upload Path (CORRECT - uses callback)

**File**: `ai_core/services/__init__.py:1737-1759`

```python
def _persist_via_repository(_: Mapping[str, object]) -> dict[str, object]:
    repository = _get_documents_repository()
    repository.upsert(normalized_document, scope=_build_scope())  # ← Persistence!
    return {"document_id": document_identifier, ...}

# Pass persistence handler to graph
graph = UploadIngestionGraph(persistence_handler=_persist_via_repository)
graph_result = graph.run(graph_payload, run_until="persist_complete")
```

**Result**: Upload graph receives a persistence callback and calls it at the right time.

---

### Crawler Path (BROKEN - no graph persistence)

**File**: `crawler/worker.py:258-274`

```python
# BEFORE graph:
resolved_document_id = self._register_document(...)  # ← Only persistence point!

# AFTER:
# ... publish to celery ...
# ... graph runs ...
# ❌ Graph never persists because no persistence_handler passed!
```

**File**: `ai_core/graphs/crawler_ingestion_graph.py:73-114`

```python
def __init__(self, ..., document_persistence: DocumentPersistenceService | None = None):
    self._document_persistence = persistence_candidate  # ← Set but never used!
```

**Problem**:

1. `_document_persistence` is stored but **NEVER CALLED**
2. Graph assumes document already exists (because parallel registration created it)
3. If parallel registration fails, graph has no way to create document

---

## Comparison

| Aspect | Upload | Crawler |
|--------|--------|---------|
| **Persistence Handler** | ✅ Passed to graph | ❌ Not passed |
| **Graph Persistence** | ✅ Calls handler | ❌ Assumes doc exists |
| **Parallel Creation** | ❌ None | ✅ Before graph |
| **Failure Safety** | ✅ Atomic | ❌ Orphans |

---

## The Smoking Gun

**Crawler assumes document exists** because:

1. Parallel registration creates it BEFORE graph
2. Graph was designed to UPDATE existing document, not CREATE
3. No fallback if parallel registration fails

**This is why removing parallel registration requires graph changes:**

- Graph needs to CREATE document if missing
- Graph needs persistence handler like Upload
- Or graph needs to use `_document_persistence` service

---

## Solution Path

### Option A: Make Graph Self-Sufficient (Recommended)

**Add persistence to crawler graph** (like Upload):

```python
# In crawler/worker.py (after removing parallel registration):
def _persist_document_callback(state):
    repository = self._get_documents_repository()
    normalized = state["normalized_document"]
    repository.upsert(normalized, workflow_id=...)
    return {"document_id": str(normalized.ref.document_id)}

# Pass to graph:
graph = CrawlerIngestionGraph(persistence_handler=_persist_document_callback)
result = graph.run(state, run_until="persist_complete")
```

**Pros**:

- ✅ Aligns with Upload pattern
- ✅ Graph controls when to persist
- ✅ Atomic (persist only if graph succeeds)

**Cons**:

- Requires modifying CrawlerIngestionGraph to accept persistence_handler

---

### Option B: Use Existing `_document_persistence`

**Call the existing service** that's already injected:

```python
# In CrawlerIngestionGraph.run():
def run(self, state, meta=None):
    normalized_payload = self._ensure_normalized_payload(state)
    
    # **NEW**: Persist using injected service
    if self._document_persistence:
        self._document_persistence.upsert_normalized(
            normalized=normalized_payload
        )
    
    # Continue with graph...
    result_state = self._document_graph.invoke(...)
```

**Pros**:

- ✅ Uses existing architecture
- ✅ No new parameters needed

**Cons**:

- ⚠️ Persists at graph START (not ideal)
- ⚠️ Still fails if parallel registration fails mid-way

---

### Option C: Hybrid Approach (Safest for Migration)

**Phase 1**: Add fallback to graph

```python
# In CrawlerIngestionGraph._ensure_normalized_payload():
def _ensure_normalized_payload(self, state):
    normalized = # ... existing normalization ...
    
    # **NEW**: Check if document exists, create if not
    existing = self._repository.get(
        tenant_id=normalized.document.ref.tenant_id,
        document_id=normalized.document.ref.document_id
    )
    
    if existing is None:
        # Parallel registration failed/skipped - create now
        self._repository.upsert(
            normalized.document,
            workflow_id=normalized.document.ref.workflow_id
        )
        logger.info("crawler.graph_created_missing_document")
    
    return normalized
```

**Phase 2**: Disable parallel registration with feature flag

**Phase 3**: Remove parallel registration code

**Pros**:

- ✅ Safe migration path
- ✅ Works with OR without parallel registration
- ✅ No breaking changes

**Cons**:

- Temporary complexity (both paths exist)

---

## Recommendation

**Use Option C (Hybrid)** for the following reasons:

1. **Zero Downtime**: Both paths work during migration
2. **Testable**: Can test each phase independently
3. **Rollback Safe**: Can re-enable parallel registration if issues
4. **Matches Phased Plan**: Aligns with 7-phase implementation

---

## Updated Flow Diagram

### CURRENT (Broken)

```
T1: Parallel Registration → Document in DB ✅
T2: Graph Runs → Assumes doc exists ⚠️
T3: Graph Completes → Updates metadata ✅

If T1 fails:
T2: Graph Runs → Crashes (doc not found) ❌
```

### AFTER FIX (Hybrid)

```
T1: Parallel Registration → Document in DB ✅ (if enabled)
T2: Graph Starts → Checks if doc exists
    ├─ Exists? → Continue ✅
    └─ Missing? → Create now ✅
T3: Graph Completes → Updates metadata ✅

If T1 fails or skipped:
T2: Graph Creates Document ✅
T3: Graph Completes ✅
```

---

## Next Steps

1. ✅ Document current flow - DONE
2. ✅ Find persistence logic - DONE
3. ⏳ Implement Option C Phase 1 (fallback)
4. ⏳ Add feature flag
5. ⏳ Test both paths
6. ⏳ Deprecate parallel registration
7. ⏳ Remove parallel registration

---

**Status**: Ready for implementation  
**Risk**: Low (hybrid approach is backwards-compatible)  
**Effort**: 2 days for Phase 1+2, 1 week total for complete migration
