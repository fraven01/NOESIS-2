# Blob Type Handling - Architecture Decision Record

## Status: ✅ NOT A DRIFT - Intentional Design

## Context

Different ingestion sources use different blob types:

- **Upload Path**: Uses `InlineBlob` (base64-encoded data)
- **Crawler Path**: Uses `FileBlob` (object store URI reference)
  
This appears as a "drift" but is actually intentional design.

## Decision

**Upload uses InlineBlob by design**:

- Small files (user uploads typically < 10MB)
- Already in memory from HTTP multipart upload
- Simplifies API contract (no separate storage step)
- Repository converts to FileBlob on `upsert()`

**Crawler uses FileBlob by design**:

- Large web pages (potentially > 100MB)
- Already persisted to object store before document creation
- Avoids double-encoding (bytes → base64 → bytes → storage)
- More efficient memory usage

## Implementation

The `DbDocumentsRepository` handles both transparently:

```python
def _materialize_document_safe(self, doc: NormalizedDocument):
    """Convert InlineBlob → FileBlob before persistence."""
    if isinstance(doc.blob, InlineBlob):
        data = base64.b64decode(doc.blob.base64)
        uri, _, _ = self._storage.put(data)
        new_blob = FileBlob(type="file", uri=uri, ...)
        return doc.model_copy(update={"blob": new_blob})
    return doc  # Already FileBlob, no change
```

**Result**: All documents in DB use `FileBlob`, regardless of ingestion source.

## Consequences

**Positive**:

- ✅ Optimized for each use case (small uploads vs large crawled pages)
- ✅ Single persistence layer handles both transparently
- ✅ DB always stores FileBlob (consistent schema)
- ✅ No unnecessary conversions or double-encoding

**Negative**:

- ⚠️ Different contracts for different entry points
- ⚠️ Requires understanding of conversion layer
- ⚠️ Testing needs to cover both blob types

## Alternatives Considered

**Alternative 1**: Force Upload to use FileBlob

- **Rejected**: Requires extra storage step before API call
- Would complicate upload API contract
- No real benefit (conversion happens anyway)

**Alternative 2**: Force Crawler to use InlineBlob

- **Rejected**: Memory inefficient for large pages
- Double encoding overhead (bytes → base64 → bytes)
- Would OOM on large documents

## Verification

```python
# Upload path (simplified)
upload_blob = InlineBlob(base64=encoded_payload)
doc = NormalizedDocument(blob=upload_blob)
stored_doc = repository.upsert(doc)  # ← Converts to FileBlob internally
assert isinstance(stored_doc.blob, FileBlob)  # ✅ Always FileBlob after persistence

# Crawler path (simplified)
crawler_blob = FileBlob(uri=object_store_path)
doc = NormalizedDocument(blob=crawler_blob)
stored_doc = repository.upsert(doc)  # ← No conversion needed
assert isinstance(stored_doc.blob, FileBlob)  # ✅ Already FileBlob
```

## Recommendation

**Status**: ✅ **NO ACTION NEEDED**

This is NOT a drift to fix - it's intentional architecture that:

1. Optimizes each path for its use case
2. Maintains consistent schema (FileBlob in DB)
3. Abstracts differences at persistence layer

**Documentation**: This ADR serves as documentation for future maintainers.

---

**Reviewed**: 2025-12-10  
**Decision**: Keep current design, document rationale  
**Next Review**: When adding new ingestion sources
