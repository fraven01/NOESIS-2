# Embedding API Refactoring - Technical Debt

## Current State (Option A) ✅ Implemented

`trigger_embedding()` requires `normalized_document` as mandatory parameter even when chunks are provided.

**Problem**: `normalized_document` is used as **metadata transport object**, not for text content.

```python
def trigger_embedding(
    *,
    normalized_document: NormalizedDocumentPayload,  # Required for metadata extraction
    chunks: Optional[Sequence[Mapping[str, Any]]] = None,
    # ...
):
    # Extract metadata from normalized_document
    document = normalized_document.document
    tenant = tenant_id or document.ref.tenant_id
    # ... uses document.ref, document.checksum, document.source, etc.
    
    # If chunks provided, use those (not normalized_document.primary_text)
    if chunks:
        for chunk in chunks:
            chunk_content = chunk.get("text")
            # ...
    else:
        # Fallback: use normalized_document.content_normalized
        normalized_content = normalized_document.content_normalized
```

**Issue**: When chunks exist, we decode the entire blob payload just to access metadata fields.

**Workaround Impact**:

- ✅ **Pro**: Embedding works with all blob types (FileBlob, InlineBlob, ExternalBlob)
- ✅ **Pro**: No API breaking changes
- ❌ **Con**: Unnecessary blob decoding (performance overhead for large documents)
- ❌ **Con**: Couples metadata extraction to blob payload access

**Performance Impact**:

- Small documents (<100KB): Negligible
- Large documents (>10MB): Significant - blob is fetched from storage only for metadata

---

## Future Refactoring (Option B) 📋 Roadmap

### Goal

Decouple metadata extraction from blob payload access in `trigger_embedding()`.

### Proposed API Change

```python
def trigger_embedding(
    *,
    normalized_document: Optional[NormalizedDocumentPayload] = None,
    document_metadata: Optional[DocumentMetadata] = None,  # NEW
    chunks: Optional[Sequence[Mapping[str, Any]]] = None,
    # ...
) -> EmbeddingResult:
    """
    Args:
        normalized_document: Full normalized document (for backward compatibility)
        document_metadata: Lightweight metadata object (preferred when chunks provided)
        chunks: Pre-chunked text segments
    """
    # Extract metadata from either source
    if document_metadata:
        tenant = document_metadata.tenant_id
        document_id = document_metadata.document_id
        checksum = document_metadata.checksum
        # ...
    elif normalized_document:
        # Backward compatible path
        document = normalized_document.document
        tenant = document.ref.tenant_id
        # ...
    else:
        raise ValueError("Either normalized_document or document_metadata required")
    
    # Build chunks (no blob access needed if chunks provided)
    if chunks:
        # Use provided chunks
        # ...
    elif normalized_document:
        # Fallback: extract from normalized_document
        # ...
```

### New Data Structure

```python
@dataclass
class DocumentMetadata:
    """Lightweight metadata for embedding without full document payload."""
    tenant_id: str
    document_id: str
    workflow_id: str
    case_id: Optional[str] = None
    source: str = "crawler"
    checksum: str
    external_id: Optional[str] = None
    lifecycle_state: str = "active"
    # ... other metadata fields
```

### Migration Path

1. **Phase 1** (Current): Both `normalized_document` and `chunks` required
2. **Phase 2**: Add optional `document_metadata` parameter
3. **Phase 3**: Update callers to use `document_metadata` when chunks available
4. **Phase 4**: Deprecate `normalized_document` requirement when `document_metadata` provided
5. **Phase 5**: Remove `normalized_document` requirement (BREAKING CHANGE)

### Benefits

- ✅ **Performance**: No blob decoding when chunks available
- ✅ **Separation of Concerns**: Metadata != Payload
- ✅ **Simplicity**: Clearer API intent
- ✅ **Scalability**: Supports large documents without blob overhead

### Files to Modify

**Core API**:

- `ai_core/api.py` - Add `document_metadata` parameter to `trigger_embedding()`
- `ai_core/contracts/payloads.py` - Define `DocumentMetadata` dataclass

**Graph Updates**:

- `documents/processing_graph.py` - Update `_embed_action` to build lightweight metadata
- `ai_core/graphs/crawler_ingestion_graph.py` - Pass metadata instead of full normalized_document

**Tests**:

- `ai_core/tests/test_embedding.py` - Test both API paths
- Integration tests for backward compatibility

### Estimated Effort

- **Design**: 2 hours (define `DocumentMetadata` contract)
- **Implementation**: 4 hours (API changes, graph updates)
- **Testing**: 4 hours (unit + integration tests)
- **Migration**: 2 hours (update all callers)
- **Total**: ~12 hours (1.5 days)

### Priority

**P2 - Medium**: Performance optimization, not blocking functionality

**Triggers for Higher Priority**:

- Documents >50MB being processed frequently
- Blob storage latency becomes bottleneck
- Memory pressure from unnecessary blob caching

---

## Related Technical Debt

### Blob Strategy Optimization

Currently, repository converts ALL blobs to FileBlob. Consider size-based threshold:

- InlineBlob: ≤100KB
- FileBlob: >100KB

**See**: `docs/roadmap/REPOSITORY_BLOB_STRATEGY.md` (if created)

### Centralized Blob Creation

Create `BlobFactory` to avoid ad-hoc blob construction across codebase.

**See**: `implementation_plan.md` Phase 4

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2025-12-10 | Implement Option A (both params) | Quick fix, no breaking changes, embedding works |
| TBD | Plan Option B refactoring | Future optimization when performance becomes issue |

---

## References

- Implementation Plan: `C:\Users\vendo\.gemini\antigravity\brain\644c8be6-90a4-4d1f-a360-a392a96275c1\implementation_plan.md`
- Blob Architecture Review: `C:\Users\vendo\.gemini\antigravity\brain\644c8be6-90a4-4d1f-a360-a392a96275c1\blob_architecture_review_prompt.md`
- Original Issue: Embedding failed with `ValueError: unsupported_blob_type`
