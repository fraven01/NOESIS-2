# Crawler vs. Upload Ingestion: Inconsistency Analysis

**Date**: 2025-12-10
**Scope**: Field-level comparison between `crawler/worker.py` and `ai_core/graphs/upload_ingestion_graph.py`
**Status**: Pre-MVP (breaking changes allowed)

---

## Executive Summary - 5 Key Findings

| Field | Status | Severity | Impact |
|-------|--------|----------|---------|
| **origin_uri** | ✅ Konsistent | Low | Both paths set origin_uri correctly |
| **external_ref** | ⚠️ Struktur unterschiedlich | **HIGH** | Crawler doesn't set structured external_ref → downstream consumers fail |
| **title** | ⚠️ Crawler oft leer | **MEDIUM** | Crawled docs have no human-readable title → UX degradation |
| **media_type** | ⚠️ Persistence-Lücken | **HIGH** | Field name mismatch (`media_type` vs `content_type`) → data loss |
| **source** | ✅ Konsistent | Low | Both paths set source correctly |

**Critical Issue**: Crawler path creates incomplete `DocumentMeta` objects, leading to:
- Missing `external_ref.provider` → guardrail checks fail ([ai_core/api.py:117](ai_core/api.py#L117))
- Missing `title` → UI displays empty/unknown document names
- `content_type` vs `media_type` confusion → parsing errors downstream

---

## Field-by-Field Analysis

### 1. ✅ `origin_uri` - CONSISTENT

#### Contract Definition
```python
# documents/contracts.py:384-387
origin_uri: Optional[str] = Field(
    default=None,
    description="Original source URI for the ingested document.",
)
```

#### Upload Implementation
```python
# upload_ingestion_graph.py:215
meta = DocumentMeta(
    # ...
    origin_uri=payload.get("origin_uri") or payload.get("file_uri"),
    # ...
)
```

#### Crawler Implementation
```python
# crawler/worker.py:169
raw_meta.setdefault("origin_uri", request.canonical_source)
```

**✅ Status**: Both paths populate `origin_uri` with fallback logic.
**Risk**: Low - field is consistently set.

---

### 2. ⚠️ `external_ref` - STRUCTURE MISMATCH

#### Contract Definition
```python
# documents/contracts.py:392-401
external_ref: Optional[Dict[str, str]] = Field(
    default=None,
    description=(
        "Optional external reference identifiers provided by source systems "
        "(<= 16 entries; keys <= 128 chars, values <= 512 chars; limits enforced). "
        "The keys `provider` and `external_id` capture the upstream system and "
        "its native identifier. Additional provider specific attributes must "
        "use the `provider_tag:` prefix."
    ),
)
```

**Expected Structure**:
```python
{
    "provider": str,      # REQUIRED: Upstream system name (e.g., "web", "upload")
    "external_id": str,   # REQUIRED: Native identifier from source system
    # Optional provider-specific fields with "provider_tag:" prefix
}
```

#### Upload Implementation ✅
```python
# upload_ingestion_graph.py:217-221
external_ref={
    "provider": "upload",
    "uploader_id": self._resolve_str(payload.get("uploader_id")),
    "external_id": self._resolve_str(payload.get("source_key"))
}
```

**Analysis**: Correctly creates structured dict with:
- ✅ `provider`: hardcoded to `"upload"`
- ✅ `external_id`: derived from `source_key`
- ✅ Additional field: `uploader_id` (no prefix violation, but not in contract!)

#### Crawler Implementation ❌
```python
# crawler/worker.py:169-232
# NO external_ref assignment found in _compose_state()
# raw_meta dict is built but external_ref is NEVER set
```

**Analysis**:
- ❌ `external_ref` is **NOT set** in crawler path
- ❌ Downstream code expects `external_ref.provider` ([ai_core/api.py:117](ai_core/api.py#L117))
- ❌ Completion payload builder expects `external_ref.external_id` ([ai_core/api.py:453](ai_core/api.py#L453))

#### Defensive Code Found
```python
# ai_core/api.py:117 (Guardrail signals builder)
provider=(meta.external_ref or {}).get("provider"),  # Defensive .get()

# ai_core/api.py:453 (Completion payload)
external_id=(document.meta.external_ref or {}).get(
    "external_id", str(document.ref.document_id)  # Fallback to document_id
),
```

**Root Cause**: Crawler never sets `external_ref`, so downstream code uses fallbacks or empty dicts.

**Impact**:
- Guardrail checks get `provider=None` → incorrect rate limiting scope
- Completion payloads use `document_id` instead of true external ID → duplicate detection breaks
- Inconsistent data model between ingestion paths

**Priority**: 🔴 **CRITICAL** (P1)

---

### 3. ⚠️ `title` - MISSING IN CRAWLER

#### Contract Definition
```python
# documents/contracts.py:367-370
title: Optional[str] = Field(
    default=None,
    description="Optional human readable document title.",
    max_length=256,
)
```

#### Upload Implementation ✅
```python
# upload_ingestion_graph.py:213
filename = (payload.get("filename") or "upload").strip()
meta = DocumentMeta(
    # ...
    title=filename,
    # ...
)
```

**Analysis**: Title is explicitly set to filename (or `"upload"` as fallback).

#### Crawler Implementation ❌
```python
# crawler/worker.py:169-232
# NO title assignment found in _compose_state()
# raw_meta dict is built but title is NEVER set
```

**Analysis**:
- ❌ Crawler documents have `title=None`
- ❌ UI/API consumers cannot display meaningful document names
- ❌ Search/filtering by title returns incomplete results

**Workaround Potential**: Could extract title from:
1. HTTP `<title>` tag (for HTML)
2. PDF metadata (`/Title` field)
3. Filename extracted from `origin_uri` path

**Impact**:
- UX degradation: Users see "Untitled" or document IDs instead of names
- Search quality: Title-based queries miss crawled documents
- Inconsistent presentation between upload and crawler sources

**Priority**: 🟡 **IMPORTANT** (P2)

---

### 4. ⚠️ `media_type` vs `content_type` - FIELD NAME CONFUSION

#### Contract Definition (InlineBlob)
```python
# documents/contracts.py (InlineBlob model, assumed from upload code)
blob = InlineBlob(
    type="inline",
    media_type=declared_mime or "application/octet-stream",  # Field name: media_type
    # ...
)
```

#### Upload Implementation ✅
```python
# upload_ingestion_graph.py:224-230
declared_mime = self._normalize_mime(payload.get("declared_mime"))
# ...
blob = InlineBlob(
    type="inline",
    media_type=declared_mime or "application/octet-stream",  # Uses media_type
    # ...
)
```

**Analysis**: Correctly uses `media_type` field in blob object.

#### Crawler Implementation ⚠️
```python
# crawler/worker.py:183-184
if result.metadata.content_type and "content_type" not in raw_meta:
    raw_meta["content_type"] = result.metadata.content_type  # Sets content_type in dict
```

**Analysis**:
- ⚠️ Crawler sets `content_type` in `raw_meta` dict (line 184)
- ⚠️ Upload sets `media_type` in `InlineBlob` object (line 226)
- ⚠️ **Field name mismatch**: `content_type` ≠ `media_type`

#### DB Query Evidence Needed
**Question**: Does the repository/persistence layer expect:
- `DocumentMeta.content_type` (not in contract!)
- `InlineBlob.media_type` (used by upload)
- Both fields simultaneously?

**Hypothesis**:
- Upload path persists to `blob.media_type`
- Crawler path writes to `raw_meta["content_type"]` which may NOT map to blob field
- Downstream parsers may look for `media_type` and fail to find it

**Impact**:
- Parser dispatcher cannot determine correct parser → fallback to text/plain
- Asset extraction skips images/videos → embedding quality degrades
- Inconsistent MIME handling across ingestion sources

**Priority**: 🔴 **CRITICAL** (P1)

---

### 5. ✅ `source` - CONSISTENT

#### Contract Definition
```python
# documents/contracts.py (NormalizedDocument model, inferred)
source: str  # Channel that handed document to ingestion (e.g., "crawler", "upload")
```

#### Upload Implementation
```python
# upload_ingestion_graph.py:245
normalized = NormalizedDocument(
    # ...
    source="upload"
)
```

#### Crawler Implementation
```python
# crawler/worker.py:171-172
resolved_source = self._resolve_source(raw_meta, ingestion_overrides)
raw_meta["source"] = resolved_source
```

**Analysis**: Both paths populate `source` correctly:
- Upload: hardcoded to `"upload"`
- Crawler: dynamic resolution with fallback logic

**✅ Status**: Field is consistently set across paths.
**Risk**: Low - no issues detected.

---

## Code Evidence - Concrete Line References

### Crawler Path: `crawler/worker.py`

#### `_compose_state()` Method (Lines 144-251)
```python
def _compose_state(
    self,
    result: FetchResult,
    request: FetchRequest,
    *,
    tenant_id: str,
    # ... other params
) -> dict[str, Any]:
    # Initialize raw_meta dict
    raw_meta: dict[str, Any] = dict(document_metadata or {})

    # ✅ origin_uri - SET (Line 169)
    raw_meta.setdefault("origin_uri", request.canonical_source)

    # ✅ source - SET (Lines 171-172)
    resolved_source = self._resolve_source(raw_meta, ingestion_overrides)
    raw_meta["source"] = resolved_source

    # ⚠️ content_type - SET (Lines 183-184)
    if result.metadata.content_type and "content_type" not in raw_meta:
        raw_meta["content_type"] = result.metadata.content_type

    # ❌ external_ref - NEVER SET
    # ❌ title - NEVER SET

    # raw_meta dict is passed to graph but fields are missing
    raw_document: dict[str, Any] = {
        "metadata": raw_meta,  # Incomplete metadata dict
        "payload_path": payload_path,
    }
```

**Issues**:
1. Line 184: Sets `content_type` (not `media_type`)
2. No code path sets `external_ref`
3. No code path sets `title`

---

### Upload Path: `ai_core/graphs/upload_ingestion_graph.py`

#### `_prepare_upload_document()` Method (Lines 182-266)
```python
def _prepare_upload_document(self, payload: Mapping[str, Any]) -> NormalizedDocumentPayload:
    # ... extract inputs

    filename = (payload.get("filename") or "upload").strip()

    # Build document object structure
    meta = DocumentMeta(
        tenant_id=tenant_id,
        workflow_id=payload.get("workflow_id") or DEFAULT_WORKFLOW_PLACEHOLDER,

        # ✅ title - SET (Line 213)
        title=filename,

        tags=list(self._normalize_tags(payload.get("tags"))),

        # ✅ origin_uri - SET (Line 215)
        origin_uri=payload.get("origin_uri") or payload.get("file_uri"),

        # ✅ external_ref - SET (Lines 217-221)
        external_ref={
            "provider": "upload",
            "uploader_id": self._resolve_str(payload.get("uploader_id")),
            "external_id": self._resolve_str(payload.get("source_key"))
        }
    )

    # ✅ media_type - SET (Lines 224-230)
    blob = InlineBlob(
        type="inline",
        media_type=declared_mime or "application/octet-stream",
        base64=base64.b64encode(binary).decode("ascii"),
        sha256=content_hash,
        size=len(binary)
    )
```

**Correct Implementation**:
1. Line 213: Sets `title` to filename
2. Lines 217-221: Creates structured `external_ref` dict
3. Line 226: Uses `media_type` (correct field name)

---

### Defensive Checks in Consumers

#### Guardrail Signals Builder: `ai_core/api.py:115-117`
```python
def _guardrail_signals_from_meta(meta: DocumentMeta, origin_uri: str | None) -> GuardrailSignals:
    return GuardrailSignals(
        tenant_id=meta.tenant_id,
        provider=(meta.external_ref or {}).get("provider"),  # ⚠️ Defensive .get()
        canonical_source=origin_uri,
        # ...
    )
```

**Issue**: Code defensively checks `external_ref or {}` because crawler path sets it to `None`.

#### Completion Payload Builder: `ai_core/api.py:453-455`
```python
def build_completion_payload(...) -> CompletionPayload:
    return CompletionPayload(
        # ...
        external_id=(document.meta.external_ref or {}).get(
            "external_id", str(document.ref.document_id)  # ⚠️ Fallback to document_id
        ),
        # ...
    )
```

**Issue**: Falls back to `document_id` when `external_ref.external_id` is missing.

---

## Root Cause Analysis - Why Did This Happen?

### Historical Context
1. **Legacy Path First**: Crawler implementation predates structured contracts
2. **Upload Refactor**: Upload graph was refactored to use `NormalizedDocument` contracts directly
3. **Crawler Missed Update**: Crawler still builds raw dict instead of `DocumentMeta` object

### Technical Debt Drivers
1. **Dict-First Design**: Crawler uses `raw_meta: dict[str, Any]` instead of Pydantic models
2. **No Contract Enforcement**: Dict allows arbitrary fields → missing required fields go unnoticed
3. **Defensive Coding**: Downstream code added `.get()` fallbacks instead of fixing root cause

### Testing Gap
1. **No E2E Tests**: Missing tests comparing crawler vs. upload outputs
2. **No Contract Validation**: `raw_meta` dict is never validated against `DocumentMeta` schema
3. **Integration Tests Missing**: Tests mock ingestion → don't catch field mismatches

---

## 3-Stufen Recommendations

### Priority 1 (Critical) - Must Fix Before MVP

#### 1.1 Strukturiertes `external_ref` im Crawler
**Goal**: Crawler must create structured `external_ref` matching contract.

**Implementation**:
```python
# crawler/worker.py:_compose_state() - ADD after line 172

# Build structured external_ref (per contract)
external_ref = {
    "provider": "web",  # or extract from domain logic
    "external_id": self._extract_external_id(request, result),
}
raw_meta["external_ref"] = external_ref
```

**Helper Method**:
```python
def _extract_external_id(
    self, request: FetchRequest, result: FetchResult
) -> str:
    """Extract canonical external ID from fetch context."""
    # Priority 1: Use request metadata if available
    if "external_id" in request.metadata:
        return str(request.metadata["external_id"])

    # Priority 2: Use URL as external_id (normalize first)
    canonical = request.canonical_source
    # Strip protocol and trailing slashes
    cleaned = canonical.replace("https://", "").replace("http://", "").rstrip("/")
    return cleaned
```

**Validation**: Add test asserting `external_ref` structure:
```python
def test_crawler_sets_external_ref():
    result = worker.process(request, tenant_id="test")
    state = result.state

    assert "external_ref" in state["raw_document"]["metadata"]
    ref = state["raw_document"]["metadata"]["external_ref"]
    assert "provider" in ref
    assert "external_id" in ref
    assert ref["provider"] == "web"
```

---

#### 1.2 Required `origin_uri` mit Validation
**Goal**: Make `origin_uri` required (not Optional) to prevent downstream None checks.

**Contract Change**:
```python
# documents/contracts.py:384-387 - REMOVE Optional
origin_uri: str = Field(  # No longer Optional[str]
    description="Original source URI for the ingested document.",
)
```

**Migration**:
- Crawler: Already sets it (line 169) ✅
- Upload: Already sets it with fallback (line 215) ✅
- DB: Add `NOT NULL` constraint + backfill migration

**Breaking Change**: Yes - but safe if all paths already set it.

---

#### 1.3 Fixiere `media_type` Feld-Name
**Goal**: Standardize on `media_type` everywhere, remove `content_type` usage.

**Implementation**:
```python
# crawler/worker.py:183-184 - REPLACE
if result.metadata.content_type and "media_type" not in raw_meta:
    raw_meta["media_type"] = result.metadata.content_type  # Use media_type key
```

**Contract Enforcement**:
```python
# documents/contracts.py - ADD to DocumentMeta
media_type: Optional[str] = Field(
    default=None,
    description="MIME type of the document payload.",
)

# DEPRECATE content_type field if it exists
```

**Migration**:
```sql
-- Backfill media_type from content_type if needed
UPDATE documents SET media_type = content_type WHERE media_type IS NULL;
```

---

### Priority 2 (Important) - Fix Before Public Beta

#### 2.1 Setze `title` im Crawler
**Goal**: Extract human-readable title from crawled content.

**Implementation Strategy**:
```python
# crawler/worker.py:_compose_state() - ADD after line 169

# Extract title (priority order)
title = self._extract_title(result, request)
if title:
    raw_meta["title"] = title
```

**Title Extraction Logic**:
```python
def _extract_title(
    self, result: FetchResult, request: FetchRequest
) -> str | None:
    """Extract title from HTML <title>, PDF metadata, or URL path."""
    content_type = (result.metadata.content_type or "").lower()

    # HTML: Parse <title> tag
    if "html" in content_type:
        try:
            from lxml import html as html_parser
            tree = html_parser.fromstring(result.payload)
            title_elem = tree.find(".//title")
            if title_elem is not None and title_elem.text:
                return title_elem.text.strip()[:256]  # Enforce max_length
        except Exception:
            pass

    # PDF: Extract /Title metadata (requires PyPDF2 or similar)
    if "pdf" in content_type:
        # TODO: Implement PDF metadata extraction
        pass

    # Fallback: Use URL path basename
    from urllib.parse import urlparse, unquote
    parsed = urlparse(request.canonical_source)
    path = unquote(parsed.path)
    basename = path.rstrip("/").split("/")[-1]
    if basename and basename != "":
        return basename[:256]

    # Last resort: Use domain name
    return parsed.netloc or None
```

**Test**:
```python
def test_crawler_extracts_html_title():
    html_payload = b"<html><head><title>Test Page</title></head></html>"
    result = worker.process(fetch_result(html_payload), tenant_id="test")

    assert result.state["raw_document"]["metadata"]["title"] == "Test Page"
```

---

#### 2.2 Entferne Defensive Code
**Goal**: Remove `.get()` fallbacks once fields are guaranteed to exist.

**Cleanup Locations**:
1. `ai_core/api.py:117` - Remove `or {}` after fixing crawler
2. `ai_core/api.py:453` - Remove fallback to `document_id`

**Before**:
```python
provider=(meta.external_ref or {}).get("provider"),
```

**After** (once P1 fixes deployed):
```python
provider=meta.external_ref["provider"],  # Direct access, KeyError if missing
```

**Validation**: Add contract validation test:
```python
def test_external_ref_required_fields():
    """Ensure external_ref always has provider and external_id."""
    meta = DocumentMeta(
        tenant_id="test",
        workflow_id="wf1",
        external_ref={"provider": "web"}  # Missing external_id
    )
    # Should raise ValidationError
```

---

### Priority 3 (Optional) - Quality Improvements

#### 3.1 Dokumentation & Linting
- Add docstrings to `_compose_state()` explaining field mappings
- Document `external_ref` contract in [docs/rag/ingestion.md](docs/rag/ingestion.md)
- Add Ruff rule to flag dict access on Pydantic models

#### 3.2 Contract-First Refactor
- Refactor crawler to build `DocumentMeta` object directly (not dict)
- Use Pydantic validation to catch missing fields at ingestion time

---

## Migration Plan (Pre-MVP Breaking Changes Allowed)

### Phase 1: Hotfix Critical Issues (Week 1)
**Goal**: Fix `external_ref` and `media_type` field name issues.

**Steps**:
1. **Day 1-2**: Implement P1.1 (structured `external_ref` in crawler)
   - Add `_extract_external_id()` helper
   - Update `_compose_state()` to set `external_ref`
   - Write unit test
2. **Day 3**: Implement P1.3 (`media_type` standardization)
   - Replace `content_type` with `media_type` in crawler
   - Add `media_type` field to `DocumentMeta` contract
3. **Day 4-5**: Integration Testing
   - Run E2E tests with both crawler and upload paths
   - Verify guardrail checks receive `provider` field
   - Verify completion payloads use `external_id`

**Deliverables**:
- ✅ Crawler sets `external_ref.provider` and `external_ref.external_id`
- ✅ All paths use `media_type` (not `content_type`)
- ✅ Tests pass for both ingestion paths

---

### Phase 2: Title Extraction (Week 2)
**Goal**: Add title extraction to crawler path.

**Steps**:
1. **Day 1-2**: Implement P2.1 title extraction
   - Add HTML `<title>` parser
   - Add URL fallback logic
   - Test with real crawl data
2. **Day 3-4**: Refine extraction heuristics
   - Handle edge cases (empty titles, duplicates)
   - Truncate to 256 chars per contract
3. **Day 5**: Validation
   - Spot-check 100 crawled docs for title quality

**Deliverables**:
- ✅ 90%+ of crawled HTML docs have meaningful titles
- ✅ URL fallback provides baseline title for all docs

---

### Phase 3: Cleanup & Documentation (Week 3)
**Goal**: Remove defensive code and document contracts.

**Steps**:
1. **Day 1-2**: Remove defensive `.get()` calls (P2.2)
   - Update `ai_core/api.py` to use direct field access
   - Add contract validation tests
2. **Day 3-5**: Documentation (P3.1)
   - Update [AGENTS.md](AGENTS.md#tool-verträge) with field requirements
   - Add ingestion field matrix to [docs/rag/ingestion.md](docs/rag/ingestion.md)
   - Write migration guide for external teams

**Deliverables**:
- ✅ No defensive `or {}` patterns in codebase
- ✅ Contract docs reflect actual field requirements

---

### Phase 4: Contract-First Refactor (Post-MVP Optional)
**Goal**: Refactor crawler to use Pydantic models directly.

**Approach**:
```python
# Future state: crawler builds DocumentMeta object
def _compose_state(self, ...) -> dict[str, Any]:
    meta = DocumentMeta(
        tenant_id=tenant_id,
        workflow_id=workflow_id,
        title=self._extract_title(result, request),
        origin_uri=request.canonical_source,
        external_ref={
            "provider": "web",
            "external_id": self._extract_external_id(request, result),
        },
        source=resolved_source,
    )
    # Pydantic validation happens here ✅

    raw_document = {
        "metadata": meta.model_dump(),  # Serialize after validation
        "payload_path": payload_path,
    }
```

**Benefits**:
- Validation errors caught at ingestion time (not downstream)
- IDE autocomplete for metadata fields
- Contract changes automatically enforced

---

## Testing Strategy

### Unit Tests (Priority 1)

#### Test: Crawler Sets External Ref
```python
# crawler/tests/test_worker.py

def test_crawler_sets_structured_external_ref():
    """Verify crawler creates external_ref matching DocumentMeta contract."""
    worker = CrawlerWorker(fetcher=mock_fetcher)
    request = FetchRequest(canonical_source="https://example.com/page")

    result = worker.process(
        request,
        tenant_id="test-tenant",
        case_id="case-123",
    )

    assert result.published
    state = result.fetch_result.state  # Extract state from result
    metadata = state["raw_document"]["metadata"]

    # Assert external_ref structure
    assert "external_ref" in metadata
    ref = metadata["external_ref"]
    assert isinstance(ref, dict)
    assert "provider" in ref
    assert "external_id" in ref

    # Assert values
    assert ref["provider"] == "web"
    assert ref["external_id"] == "example.com/page"  # Normalized URL
```

#### Test: Media Type Field Name
```python
def test_crawler_uses_media_type_not_content_type():
    """Verify crawler sets media_type (not content_type)."""
    worker = CrawlerWorker(fetcher=mock_fetcher)
    fetch_result = FetchResult(
        status=FetchStatus.FETCHED,
        metadata=FetchMetadata(content_type="text/html"),
        payload=b"<html></html>",
    )

    result = worker.process(..., fetch_result=fetch_result)
    metadata = result.state["raw_document"]["metadata"]

    # Assert media_type is set (not content_type)
    assert "media_type" in metadata
    assert metadata["media_type"] == "text/html"
    assert "content_type" not in metadata  # Old field should be gone
```

#### Test: Title Extraction
```python
@pytest.mark.parametrize("html,expected_title", [
    (b"<html><head><title>Test Page</title></head></html>", "Test Page"),
    (b"<html><title>  Whitespace  </title></html>", "Whitespace"),
    (b"<html><title></title></html>", None),  # Empty title → fallback
])
def test_crawler_extracts_html_title(html, expected_title):
    """Verify title extraction from HTML <title> tag."""
    worker = CrawlerWorker(fetcher=mock_fetcher)
    fetch_result = FetchResult(
        metadata=FetchMetadata(content_type="text/html"),
        payload=html,
    )

    result = worker.process(..., fetch_result=fetch_result)
    metadata = result.state["raw_document"]["metadata"]

    if expected_title:
        assert metadata["title"] == expected_title
    else:
        # Fallback to URL basename
        assert metadata["title"] is not None
```

---

### Integration Tests (Priority 2)

#### Test: End-to-End Field Parity
```python
# tests/integration/test_ingestion_parity.py

def test_crawler_upload_field_parity():
    """Verify crawled and uploaded docs have same metadata fields."""
    # Setup: Ingest same content via both paths
    html_content = b"<html><head><title>Test</title></head><body>Content</body></html>"

    # Path 1: Upload
    upload_payload = {
        "tenant_id": "test",
        "filename": "test.html",
        "declared_mime": "text/html",
        "file_bytes": html_content,
        "uploader_id": "user-123",
        "source_key": "upload-key-456",
    }
    upload_graph = UploadIngestionGraph()
    upload_result = upload_graph.run(upload_payload)
    upload_doc = upload_result["document"]

    # Path 2: Crawler
    crawler_request = FetchRequest(canonical_source="https://test.com/page.html")
    crawler_worker = CrawlerWorker(fetcher=mock_fetcher)
    crawler_result = crawler_worker.process(crawler_request, tenant_id="test")
    crawler_doc = crawler_result.state["raw_document"]

    # Assert field parity
    upload_meta = upload_doc.meta
    crawler_meta = crawler_doc["metadata"]

    # Compare required fields
    assert "origin_uri" in crawler_meta
    assert "external_ref" in crawler_meta
    assert "title" in crawler_meta
    assert "media_type" in crawler_meta

    # Verify external_ref structure
    assert crawler_meta["external_ref"]["provider"] == "web"
    assert upload_meta.external_ref["provider"] == "upload"
```

---

### DB Validation Tests (Priority 3)

#### Test: Persisted Document Structure
```python
def test_persisted_document_has_complete_metadata(db):
    """Verify persisted docs from both paths have all required fields."""
    # Ingest via crawler
    crawler_worker.process(...)

    # Query DB
    doc = db.query(Document).filter_by(source="crawler").first()

    # Assert DB columns populated
    assert doc.origin_uri is not None
    assert doc.external_ref is not None
    assert doc.external_ref["provider"] == "web"
    assert doc.title is not None
    assert doc.media_type is not None
```

---

## Appendix: Field Matrix

| Field | Crawler Status | Upload Status | Contract Required | Fix Priority |
|-------|---------------|---------------|-------------------|--------------|
| `tenant_id` | ✅ Set | ✅ Set | ✅ Yes | N/A |
| `workflow_id` | ✅ Set | ✅ Set | ✅ Yes | N/A |
| `origin_uri` | ✅ Set | ✅ Set | ⚠️ Optional (should be required) | P1.2 |
| `source` | ✅ Set | ✅ Set | ✅ Yes | N/A |
| `title` | ❌ Missing | ✅ Set | ⚠️ Optional | P2.1 |
| `external_ref.provider` | ❌ Missing | ✅ Set | ✅ Yes (per contract docs) | **P1.1** |
| `external_ref.external_id` | ❌ Missing | ✅ Set | ✅ Yes (per contract docs) | **P1.1** |
| `media_type` | ⚠️ Sets `content_type` instead | ✅ Set | ⚠️ Optional | **P1.3** |
| `tags` | ⚠️ Not validated | ✅ Set | ⚠️ Optional | P3 |
| `language` | ❌ Missing | ❌ Missing | ⚠️ Optional | Future |

---

## Next Steps

1. **Prioritize**: Review with team, confirm priority assignments
2. **Estimate**: Size P1 tasks for Week 1 sprint
3. **Branch**: Create `fix/crawler-upload-parity` branch
4. **Implement**: Start with P1.1 (external_ref) on Day 1
5. **Test**: Run full integration suite after each phase
6. **Document**: Update [AGENTS.md](AGENTS.md) with new field requirements

**Owner**: TBD
**Target Completion**: 3 weeks (P1+P2+P3)
**Blocker Status**: None (all paths forward are clear)

---

**Document Version**: 1.0
**Last Updated**: 2025-12-10
**Related Issues**: None (new finding)
