# workflow_id Normalization Drift Analysis - NOESIS-2

**Date**: 2025-12-07
**Reviewer**: Claude Code
**Task**: Identify and document workflow_id normalization inconsistencies across repositories

---

## Executive Summary

**Contract (from AGENTS.md & ID Semantics)**:
- `workflow_id` is a string identifier for workflows (e.g., `"Case:123/v2"`, `"ad-hoc"`)
- Must be normalized consistently for cross-repository queries
- Used as storage key and filter parameter across InMemory, DB, and ObjectStore repositories

**Status**: ⚠️ **DIVERGENCE ELIMINATED** (ObjectStore deleted), **MINOR IMPROVEMENTS RECOMMENDED**

### Key Findings

1. ✅ **ObjectStore repository deleted** - Primary divergence source removed
2. ✅ **Contract validation enforces strict charset** - `[A-Za-z0-9._-]+` only
3. ✅ **Storage normalization consistent** - InMemory & DB use same `_workflow_storage_key()`
4. ⚠️ **Redundant sanitization exists** - `sanitize_identifier()` still used for paths
5. ⚠️ **Case-sensitive workflows** - `"Workflow-A"` ≠ `"workflow-a"` (by design)

**No critical cross-repository incompatibility exists.** The current implementation is production-ready with layered normalization (validation → storage).

---

## 1. Normalization Functions - Complete Inventory

### 1.1 `_workflow_storage_key` (InMemory & DB Repositories)

**Location**: [documents/repository.py:368-369](documents/repository.py#L368-L369)

**Implementation**:
```python
def _workflow_storage_key(value: Optional[str]) -> str:
    return (str(value).strip() if value else "").strip()
```

**Behavior**:
- Converts to string via `str(value)`
- Strips leading/trailing whitespace (double strip is redundant but harmless)
- Returns empty string for `None`/empty values
- **NO character replacement**
- **Case-sensitive** (preserves original casing)

**Example Transformations**:
```python
_workflow_storage_key("  Case:123/v2  ")  # → "Case:123/v2"
_workflow_storage_key("Workflow-ID")      # → "Workflow-ID"
_workflow_storage_key(None)               # → ""
_workflow_storage_key("   ")              # → ""
_workflow_storage_key("test")             # → "test"
```

**Used By**:
- `InMemoryDocumentsRepository` ([documents/repository.py:891+](documents/repository.py#L891))
- `DbDocumentsRepository` ([ai_core/adapters/db_documents_repository.py:24](ai_core/adapters/db_documents_repository.py#L24))

**Import Pattern**:
```python
from documents.repository import _workflow_storage_key
```

---

### 1.2 `normalize_workflow_id` (Contract Validation)

**Location**: [documents/contract_utils.py:64-74](documents/contract_utils.py#L64-L74)

**Implementation**:
```python
def normalize_workflow_id(value: str) -> str:
    """Normalize and validate workflow identifiers."""

    normalized = normalize_string(value)
    if not normalized:
        raise ValueError("workflow_empty")
    if len(normalized) > _WORKFLOW_ID_MAX_LENGTH:
        raise ValueError("workflow_too_long")
    if not _WORKFLOW_ID_RE.fullmatch(normalized):
        raise ValueError("workflow_invalid_char")
    return normalized
```

**Supporting Functions**:
```python
# documents/contract_utils.py:25-30
def normalize_string(value: str) -> str:
    """Normalize string input by applying NFKC, trimming invisibles and whitespace."""
    normalized = unicodedata.normalize("NFKC", value)
    normalized = _strip_invisible(normalized)
    return normalized.strip()

# documents/contract_utils.py:19-22
def _strip_invisible(value: str) -> str:
    return "".join(
        ch for ch in value if unicodedata.category(ch) not in _INVISIBLE_CATEGORIES
    )

# Constants
_INVISIBLE_CATEGORIES = {"Cf", "Cc", "Cs"}  # Format, Control, Surrogate
_WORKFLOW_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$", re.ASCII)
_WORKFLOW_ID_MAX_LENGTH = 128
```

**Behavior**:
- **NFKC Unicode normalization** (compatibility decomposition)
- **Removes invisible characters** (categories Cf, Cc, Cs)
- **Strips whitespace**
- **Validates charset**: Must match `[A-Za-z0-9._-]+` (alphanumeric, dot, underscore, hyphen)
- **Max length**: 128 characters
- **Raises ValueError** on invalid input
- **Case-sensitive** (preserves casing but validates)

**Example Transformations**:
```python
normalize_workflow_id("\u200b Workflow\u200d-ID \u00a0")  # → "Workflow-ID"
normalize_workflow_id(" ingest_2024-01 ")                # → "ingest_2024-01"
normalize_workflow_id("invalid id")                      # → ValueError: workflow_invalid_char (space)
normalize_workflow_id("project:2024")                    # → ValueError: workflow_invalid_char (colon)
normalize_workflow_id("path/to/workflow")                # → ValueError: workflow_invalid_char (slash)
normalize_workflow_id("")                                # → ValueError: workflow_empty
normalize_workflow_id("a" * 129)                         # → ValueError: workflow_too_long
```

**Tests**: [tests/documents/test_contract_utils_workflow_ids.py](tests/documents/test_contract_utils_workflow_ids.py)

---

### 1.3 `resolve_workflow_id` (Optional Workflow Helper)

**Location**: [documents/contract_utils.py:77-94](documents/contract_utils.py#L77-L94)

**Implementation**:
```python
def resolve_workflow_id(
    value: Optional[str],
    *,
    required: bool = False,
    placeholder: str = DEFAULT_WORKFLOW_PLACEHOLDER,
) -> str:
    """
    Normalize a workflow_id or supply a consistent placeholder when optional.

    If ``required`` is True and the value is empty/None, a ValueError is raised.
    """

    candidate = normalize_optional_string(value)
    if candidate is not None:
        return normalize_workflow_id(candidate)
    if required:
        raise ValueError("workflow_required")
    return normalize_workflow_id(placeholder)
```

**Constants**:
```python
# common/constants.py:26
DEFAULT_WORKFLOW_PLACEHOLDER = "ad-hoc"
```

**Behavior**:
- Normalizes optional workflow_id values
- Falls back to `"ad-hoc"` when value is None/empty (unless `required=True`)
- Applies full `normalize_workflow_id()` validation to both input and placeholder
- Used for API entry points and graph initialization

**Example Transformations**:
```python
resolve_workflow_id("  test  ")              # → "test"
resolve_workflow_id(None)                    # → "ad-hoc"
resolve_workflow_id("")                      # → "ad-hoc"
resolve_workflow_id(None, required=True)     # → ValueError: workflow_required
resolve_workflow_id("", placeholder="dev")   # → "dev"
resolve_workflow_id("invalid:id")            # → ValueError: workflow_invalid_char
```

**Used By**:
- [ai_core/services/crawler_runner.py:87-90](ai_core/services/crawler_runner.py#L87-L90)
- [ai_core/graphs/upload_ingestion_graph.py:264, 527](ai_core/graphs/upload_ingestion_graph.py#L264)
- [documents/contracts.py:959](documents/contracts.py#L959) (DocumentMeta validator)

---

### 1.4 `sanitize_identifier` (ObjectStore - Filesystem Safety)

**Location**: [common/object_store_defaults.py:47-56](common/object_store_defaults.py#L47-L56)

**Implementation**:
```python
def sanitize_identifier(self, value: str) -> str:
    value = str(value)
    if ".." in value or "/" in value or os.sep in value:
        raise ValueError("unsafe_identifier")

    sanitized = re.sub(r"[^A-Za-z0-9._-]", "_", value)[:128]
    if not sanitized:
        print(f"DEBUG: unsafe_identifier value={value!r}")
        raise ValueError("unsafe_identifier")
    return sanitized
```

**Wrapper Function**:
```python
# common/object_store_defaults.py:108-109
def sanitize_identifier(value: str) -> str:
    return _DEFAULT_OBJECT_STORE.sanitize_identifier(value)
```

**Behavior**:
- **Rejects path traversal**: `..`, `/`, `\`
- **Replaces all non-alphanumeric chars** (except `.`, `_`, `-`) with `_`
- Truncates to 128 characters
- **Case-sensitive** (preserves original casing)
- Designed for **filesystem safety**, NOT semantic equivalence

**Example Transformations**:
```python
sanitize_identifier("tenant name!@#")     # → "tenant_name___"
sanitize_identifier("Case:123/v2")        # → ValueError: unsafe_identifier (contains /)
sanitize_identifier("project:2024")       # → "project_2024" (replaces colon)
sanitize_identifier("workflow.2024")      # → "workflow.2024" (preserves dot)
sanitize_identifier("..")                 # → ValueError: unsafe_identifier
sanitize_identifier("test-workflow")      # → "test-workflow"
```

**Tests**: [ai_core/tests/test_infra.py:114-122](ai_core/tests/test_infra.py#L114-L122)

**Current Usage** (Post-ObjectStore deletion):
```python
# documents/utils.py:20-21
tenant_segment = object_store.sanitize_identifier(tenant_id)
workflow_segment = object_store.sanitize_identifier(workflow_id or "default")

# Used for file path construction (NOT storage keys):
path = f"{tenant_segment}/{workflow_segment}/uploads/{filename}"
```

**Historical Usage** (DELETED ObjectStoreDocumentsRepository):
```python
# ai_core/adapters/object_store_repository.py:33-37 (DELETED)
def upsert(self, doc: NormalizedDocument, workflow_id: Optional[str] = None):
    tenant_segment = object_store.sanitize_identifier(doc.ref.tenant_id)
    workflow = workflow_id or doc.ref.workflow_id or "upload"
    workflow_segment = object_store.sanitize_identifier(workflow)  # ❌ DIVERGENT
    uploads_prefix = f"{tenant_segment}/{workflow_segment}/uploads"
```

---

## 2. Repository Implementations

### 2.1 InMemoryDocumentsRepository

**Location**: [documents/repository.py:891+](documents/repository.py#L891)

**Normalization Pattern**:
```python
# documents/repository.py:438, 500
workflow_key = _workflow_storage_key(normalized_workflow)
```

**Usage in Methods**:

#### `set_document_state` (Line 438-460)
```python
def set_document_state(
    self,
    *,
    tenant_id: str,
    document_id: UUID,
    state: str,
    workflow_id: Optional[str],
    # ...
) -> DocumentLifecycleRecord:
    normalized_workflow = (workflow_id or None) and str(workflow_id)
    workflow_key = _workflow_storage_key(normalized_workflow)  # ✅ Normalized

    # Store in DB with workflow_key
    instance = model.objects.get(
        tenant_id=tenant_id,
        document_id=document_id,
        workflow_id=workflow_key,  # Lookup uses normalized key
    )
```

#### `get_document_state` (Line 500-519)
```python
def get_document_state(
    self,
    *,
    tenant_id: str,
    document_id: UUID,
    workflow_id: Optional[str],
) -> Optional[DocumentLifecycleRecord]:
    workflow_key = _workflow_storage_key((workflow_id or None) and str(workflow_id))

    instance = model.objects.get(
        tenant_id=tenant_id,
        document_id=document_id,
        workflow_id=workflow_key,  # Filter uses normalized key
    )
```

**Storage Keys**:
- Documents: `(tenant_id, workflow, document_id, version)` tuple (in-memory dict)
- Lifecycle: Stored in Django model `DocumentLifecycleState.workflow_id` field
- Assets: Stored in Django model `DocumentAsset.workflow_id` field

---

### 2.2 DbDocumentsRepository

**Location**: [ai_core/adapters/db_documents_repository.py](ai_core/adapters/db_documents_repository.py)

**Import**: Line 24
```python
from documents.repository import (
    DocumentsRepository,
    _workflow_storage_key,  # ✅ Reuses InMemory normalization
)
```

**Normalization Pattern**: Identical to InMemory

**Usage in Methods**:

#### `upsert` (Line 70-80)
```python
def upsert(
    self, doc: NormalizedDocument, workflow_id: Optional[str] = None
) -> NormalizedDocument:
    workflow = workflow_id or doc_copy.ref.workflow_id
    if workflow != doc_copy.ref.workflow_id:
        raise ValueError("workflow_mismatch")

    workflow_key = _workflow_storage_key(workflow)  # ✅ Normalized

    # Used for lifecycle state lookup
    lifecycle_exists = models.Exists(
        lifecycle_model.objects.filter(
            tenant_id=tenant,
            workflow_id=workflow_key,  # Filter uses normalized key
            document_id=models.OuterRef("document__id"),
        )
    )
```

#### `_collection_queryset` (Line 516-532)
```python
def _collection_queryset(
    self,
    tenant_id: str,
    collection_id: UUID,
    workflow_id: Optional[str]
):
    lifecycle_model = apps.get_model("documents", "DocumentLifecycleState")
    workflow_key = _workflow_storage_key(workflow_id)  # ✅ Normalized

    if workflow_id is not None:
        lifecycle_exists = models.Exists(
            lifecycle_model.objects.filter(
                tenant_id=tenant,
                workflow_id=workflow_key,  # Subquery filter
                document_id=models.OuterRef("document__id"),
            )
        )
        queryset = queryset.annotate(has_lifecycle=lifecycle_exists).filter(
            has_lifecycle=True
        )
```

#### `_select_lifecycle_state` (Line 644-665)
```python
def _select_lifecycle_state(
    model, tenant, document_id: UUID, workflow_id: Optional[str]
):
    workflow_key = _workflow_storage_key(workflow_id)  # ✅ Normalized
    filters = {"tenant_id": tenant, "document_id": document_id}
    if workflow_id is not None:
        filters["workflow_id"] = workflow_key
    qs = model.objects.filter(**filters).order_by("-changed_at")
    return qs.first()
```

#### Asset Methods (Lines 443-461, 469-486, 501-517)
```python
# get_asset (Line 455)
qs = DocumentAsset.objects.filter(tenant=tenant, asset_id=asset_id)
if workflow_id:
    qs = qs.filter(workflow_id=workflow_id)  # Direct filter (assumes pre-normalized)

# list_assets_by_document (Line 480)
qs = DocumentAsset.objects.filter(tenant=tenant, document__id=document_id)
if workflow_id:
    qs = qs.filter(workflow_id=workflow_id)  # Direct filter

# delete_asset (Line 512)
qs = DocumentAsset.objects.filter(tenant=tenant, asset_id=asset_id)
if workflow_id:
    qs = qs.filter(workflow_id=workflow_id)  # Direct filter
```

**Note**: Asset methods filter directly without explicit normalization in the query, relying on workflow_id already being normalized when stored.

---

### 2.3 ObjectStoreDocumentsRepository (DELETED)

**Location**: `ai_core/adapters/object_store_repository.py` (DELETED from git)

**Status**: File deleted (visible in git status: `D ai_core/adapters/object_store_repository.py`)

**Git Status Evidence**:
```
D ai_core/adapters/object_store_repository.py
```

**Normalization Pattern** (Historical - from repository_divergence_analysis.md):
```python
# Line 33-37 (from historical analysis)
def upsert(self, doc: NormalizedDocument, workflow_id: Optional[str] = None):
    tenant_segment = object_store.sanitize_identifier(doc.ref.tenant_id)
    workflow = workflow_id or doc.ref.workflow_id or "upload"
    workflow_segment = object_store.sanitize_identifier(workflow)  # ❌ DIVERGENT!
    uploads_prefix = f"{tenant_segment}/{workflow_segment}/uploads"
```

**Path Construction** (Historical):
```
.ai_core_store/{tenant_sanitized}/{workflow_sanitized}/uploads/{document_id}_upload.bin
```

**Divergence Example** (Historical):
```python
workflow_id = "case_2024:Q1"

# After contract validation, colon would be rejected:
normalize_workflow_id("case_2024:Q1")  # → ValueError: workflow_invalid_char

# But if we had a valid workflow with special handling:
workflow_id = "case_2024-Q1"

# InMemory/DB:
_workflow_storage_key("case_2024-Q1")         # → "case_2024-Q1"

# ObjectStore (would have been):
object_store.sanitize_identifier("case_2024-Q1")  # → "case_2024-Q1" (same)
```

**Impact**: ObjectStore divergence is now **eliminated** by repository deletion.

---

## 3. Database Schema Constraints

### 3.1 DocumentLifecycleState Model

**Location**: [documents/models.py:143-177](documents/models.py#L143-L177)

```python
class DocumentLifecycleState(models.Model):
    """Latest lifecycle status for a document within a tenant workflow."""

    tenant_id = models.ForeignKey(
        "customers.Tenant",
        on_delete=models.CASCADE,
        to_field="schema_name",
        db_column="tenant_id",
    )
    document_id = models.UUIDField()
    workflow_id = models.CharField(max_length=255, blank=True, default="")  # ← Storage
    state = models.CharField(max_length=32)
    trace_id = models.CharField(max_length=255)
    run_id = models.UUIDField()
    changed_at = models.DateTimeField()
    # ...

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=("tenant_id", "document_id", "workflow_id"),
                name="document_lifecycle_unique_record",
            )
        ]
        indexes = [
            models.Index(
                fields=("tenant_id", "workflow_id"),
                name="doc_lifecycle_tenant_wf_idx",
            ),
        ]
```

**Key Points**:
- `workflow_id` is a **CharField(max_length=255)** with empty string default
- **Unique constraint** on `(tenant_id, document_id, workflow_id)` tuple
- Index on `(tenant_id, workflow_id)` for efficient filtering
- Stores the **normalized** workflow_id from `_workflow_storage_key()`

**Storage Format**:
- Direct string storage (no encoding)
- Empty string for `None`/empty workflow_id
- Case-sensitive storage (preserves original casing)

---

### 3.2 DocumentAsset Model

**Location**: [documents/models.py:222-268](documents/models.py#L222-L268)

```python
class DocumentAsset(models.Model):
    """
    Persistence for non-document assets (chunks, images, etc.) associated with a document.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    tenant = models.ForeignKey("customers.Tenant", on_delete=models.PROTECT)
    document = models.ForeignKey(Document, on_delete=models.CASCADE)
    workflow_id = models.CharField(max_length=255)  # ← Required field
    asset_id = models.UUIDField()
    collection_id = models.UUIDField(null=True, blank=True)
    asset_type = models.CharField(max_length=32)
    content_type = models.CharField(max_length=128, blank=True, default="")
    # ...

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=("tenant", "asset_id", "workflow_id"),
                name="document_asset_unique_identity",
            )
        ]
        indexes = [
            models.Index(fields=("tenant", "document"), name="doc_asset_tenant_doc_idx"),
            models.Index(fields=("tenant", "asset_id"), name="doc_asset_tenant_asset_idx"),
        ]
```

**Key Points**:
- `workflow_id` is **required** (no default, no null)
- **Unique constraint** on `(tenant, asset_id, workflow_id)` tuple
- Filters in DB repo use workflow_id directly (assumes pre-normalized)
- Assets are tied to specific workflow executions

---

## 4. Workflow_ID Comparison & Filtering

### 4.1 Equality Checks

**Pattern**: Direct string comparison after normalization

**InMemory Repository** ([documents/repository.py](documents/repository.py)):
```python
# Line 911-912
workflow = workflow_id or ref.workflow_id
if workflow != ref.workflow_id:
    raise ValueError("workflow_mismatch")

# Line 929-930 (asset validation)
if asset.ref.workflow_id != workflow:
    raise ValueError("asset_workflow_mismatch")

# Line 1160-1161 (add_asset)
workflow = workflow_id or asset_copy.ref.workflow_id
if workflow != asset_copy.ref.workflow_id:
    raise ValueError("workflow_mismatch")

# Line 1176-1177 (asset workflow check)
if asset_copy.ref.workflow_id != document.ref.workflow_id:
    raise ValueError("asset_workflow_mismatch")
```

**DB Repository** ([ai_core/adapters/db_documents_repository.py](ai_core/adapters/db_documents_repository.py)):
```python
# Line 66-68
workflow = workflow_id or doc_copy.ref.workflow_id
if workflow != doc_copy.ref.workflow_id:
    raise ValueError("workflow_mismatch")
```

**Contract Validation** ([documents/contracts.py](documents/contracts.py)):
```python
# Line 1523 (NormalizedDocument validation)
if self.ref.workflow_id != self.meta.workflow_id:
    raise ValueError("meta_workflow_mismatch")

# Line 1532 (asset validation)
if asset.ref.workflow_id != self.ref.workflow_id:
    raise ValueError("asset_workflow_mismatch")
```

**Analysis**:
- All comparisons use Python `==` operator (case-sensitive)
- No case-insensitive matching
- Whitespace must match (but normalization strips it)

---

### 4.2 Database Filtering

**Lifecycle State Queries**:
```python
# ai_core/adapters/db_documents_repository.py:526-532
lifecycle_exists = models.Exists(
    lifecycle_model.objects.filter(
        tenant_id=tenant,
        workflow_id=workflow_key,  # Uses _workflow_storage_key() normalized value
        document_id=models.OuterRef("document__id"),
    )
)
queryset = queryset.annotate(has_lifecycle=lifecycle_exists).filter(
    has_lifecycle=True
)
```

**Asset Queries**:
```python
# ai_core/adapters/db_documents_repository.py:443-444, 469, 501
qs = DocumentAsset.objects.filter(tenant=tenant, asset_id=asset_id)
if workflow_id:
    qs = qs.filter(workflow_id=workflow_id)  # Direct filter (assumes normalized)
```

**Analysis**:
- All database filters assume workflow_id is **already normalized**
- No normalization applied at query time (relies on storage normalization)
- Django ORM uses exact string matching (case-sensitive)

---

### 4.3 List Operations with workflow_id Filter

**DbDocumentsRepository.list_latest_by_collection**:
```python
# ai_core/adapters/db_documents_repository.py:283-314
def list_latest_by_collection(
    self,
    tenant_id: str,
    collection_id: UUID,
    limit: int = 100,
    cursor: Optional[str] = None,
    *,
    workflow_id: Optional[str] = None,
) -> Tuple[List[DocumentRef], Optional[str]]:
    memberships = self._collection_queryset(tenant_id, collection_id, workflow_id)

    # ... iteration over memberships
    for membership in memberships[:candidate_limit]:
        document = membership.document
        normalized = _build_document_from_metadata(document)

        doc_ref = normalized.ref
        if workflow_id and doc_ref.workflow_id != workflow_id:  # ✅ String equality
            continue
```

**Dual Filtering Pattern**:
1. `_collection_queryset()` - filters at DB level using `_workflow_storage_key(workflow_id)`
2. Python loop - additional check via string equality on `ref.workflow_id`

**Why dual filtering?**
- DB filter narrows down candidates efficiently
- Python loop ensures exact match after deserialization
- Defense-in-depth pattern (belt-and-suspenders)

---

## 5. Test Coverage

### 5.1 Contract Validation Tests

**File**: [tests/documents/test_contract_utils_workflow_ids.py](tests/documents/test_contract_utils_workflow_ids.py)

```python
def test_normalize_workflow_id_strips_invisibles_and_whitespace():
    assert normalize_workflow_id("\u200b Workflow\u200d-ID \u00a0") == "Workflow-ID"

@pytest.mark.parametrize(
    "value,code",
    [
        ("", "workflow_empty"),
        ("   ", "workflow_empty"),
        ("\u200b\u200b", "workflow_empty"),
        ("a" * 129, "workflow_too_long"),
    ],
)
def test_normalize_workflow_id_rejects_empty_and_too_long(value, code):
    with pytest.raises(ValueError) as exc:
        normalize_workflow_id(value)
    assert str(exc.value) == code

def test_normalize_workflow_id_rejects_invalid_characters():
    with pytest.raises(ValueError) as exc:
        normalize_workflow_id("invalid id")  # Space is invalid
    assert str(exc.value) == "workflow_invalid_char"

def test_normalize_workflow_id_allows_valid_identifier():
    assert normalize_workflow_id(" ingest_2024-01 ") == "ingest_2024-01"
```

**Coverage**:
- ✅ Whitespace stripping
- ✅ Invisible character removal
- ✅ Empty string rejection
- ✅ Length validation (128 chars max)
- ✅ Charset validation (`[A-Za-z0-9._-]+`)
- ❌ **Missing**: Case sensitivity tests
- ❌ **Missing**: NFKC normalization tests (e.g., ligatures)

---

### 5.2 Repository Tests

**File**: [ai_core/tests/test_db_documents_repository.py](ai_core/tests/test_db_documents_repository.py)

**Test Methods**:
- `test_upsert_fails_on_missing_collection` - Uses `workflow_id="test"`
- `test_list_latest_by_collection_deduplication` - Uses `workflow_id="test"`
- `test_asset_persistence` - Uses `workflow_id="test"` for assets

**Example**:
```python
@pytest.mark.django_db
class TestDbDocumentsRepository:
    def test_upsert_fails_on_missing_collection(self, repository, tenant):
        doc = NormalizedDocument(
            ref=DocumentRef(
                tenant_id=tenant.schema_name,
                document_id=doc_id,
                workflow_id="test",  # ← Simple alphanumeric ID
                collection_id=missing_collection_id,
                version="v1"
            ),
            # ...
        )
        with pytest.raises(ValueError, match=f"Collection not found"):
            repository.upsert(doc)
```

**Coverage Analysis**:
- ✅ Simple alphanumeric workflow_id values
- ❌ **Missing**: workflow_id with special chars (`.`, `_`, `-`)
- ❌ **Missing**: workflow_id normalization edge cases
- ❌ **Missing**: Cross-repository workflow_id consistency tests

---

### 5.3 ObjectStore Tests

**File**: [ai_core/tests/test_infra.py:114-122](ai_core/tests/test_infra.py#L114-L122)

```python
def test_sanitize_identifier_replaces_invalid_characters():
    assert object_store.sanitize_identifier("tenant name!@#") == "tenant_name___"

def test_sanitize_identifier_rejects_unsafe_sequences():
    with pytest.raises(ValueError):
        object_store.sanitize_identifier("..")
    with pytest.raises(ValueError):
        object_store.sanitize_identifier("tenant/abc")
```

**Coverage**:
- ✅ Character replacement behavior
- ✅ Path traversal rejection
- ❌ **Missing**: Tests comparing `sanitize_identifier()` vs `normalize_workflow_id()`
- ❌ **Missing**: Tests for workflow_id usage in paths vs storage

---

### 5.4 Contract Mismatch Tests

**File**: [tests/documents/test_document_contracts.py](tests/documents/test_document_contracts.py)

```python
def test_normalized_document_meta_workflow_mismatch():
    # Tests that ref.workflow_id != meta.workflow_id raises error
    # (Lines 578-585)

def test_normalized_document_asset_workflow_mismatch():
    # Tests that asset.ref.workflow_id != doc.ref.workflow_id raises error
    # (Lines 589-596)
```

**Coverage**:
- ✅ workflow_id consistency validation across DocumentRef/DocumentMeta
- ✅ Asset workflow_id matching document workflow_id
- ❌ **Missing**: Normalized vs non-normalized workflow_id mismatch tests

---

## 6. Cross-Repository Compatibility Analysis

### 6.1 Current State (Post-ObjectStore Deletion)

**Active Normalizations**: 2

1. **Contract Layer** (`normalize_workflow_id`):
   - Purpose: Input validation at API boundaries
   - Rules: NFKC, strip invisibles/whitespace, validate charset `[A-Za-z0-9._-]`
   - Max length: 128 characters
   - Used: Entry points (APIs, graphs, contracts)

2. **Storage Layer** (`_workflow_storage_key`):
   - Purpose: Database key normalization
   - Rules: Strip whitespace only
   - Used: InMemory & DB repositories for queries

**Relationship**: Contract normalization happens **before** storage normalization:

```
User Input → normalize_workflow_id() → ref.workflow_id → _workflow_storage_key() → DB
           (validation)                 (domain model)     (storage key)
```

**Consistency Check**:
```python
# Given user input:
user_input = "  project_2024-Q1  "

# Step 1: Contract validation
workflow_validated = normalize_workflow_id(user_input)  # → "project_2024-Q1"

# Step 2: Storage normalization
workflow_key = _workflow_storage_key(workflow_validated)  # → "project_2024-Q1"

# Result: Both layers agree ✅
```

**Why two layers?**
- **Contract layer**: Enforces business rules (charset, length)
- **Storage layer**: Defensive cleanup (whitespace, None handling)
- Enables contract evolution without breaking storage

---

### 6.2 Historical Divergence (ObjectStore Repository)

**Problem**: ObjectStore used `sanitize_identifier()` which:
- Replaced characters like `:` with `_`
- Was designed for filesystem safety, not semantic equivalence
- Created **different storage keys** for the same workflow_id

**Example of Historical Divergence**:
```python
# Given a workflow_id that passes contract validation:
workflow_id = "project_2024.Q1"

# InMemory/DB would store:
_workflow_storage_key("project_2024.Q1")  # → "project_2024.Q1"

# ObjectStore would create path:
sanitize_identifier("project_2024.Q1")    # → "project_2024.Q1" (same, dot allowed)

# But if someone used invalid chars before validation:
workflow_id = "project:2024/Q1"  # Would be rejected by normalize_workflow_id()

# If it somehow got through:
_workflow_storage_key("project:2024/Q1")    # → "project:2024/Q1"
sanitize_identifier("project:2024/Q1")      # → ValueError (contains /)
```

**Impact Analysis**:
- ObjectStore deletion **eliminates** this divergence
- Remaining `sanitize_identifier()` usage is for **filesystem paths**, not storage keys
- No cross-repository incompatibility exists in current codebase

---

### 6.3 Case Sensitivity Analysis

**Current Behavior**: workflow_id is **case-sensitive** across all systems

```python
workflow_a = "Ingestion-2024"
workflow_b = "ingestion-2024"

# All systems preserve case:
normalize_workflow_id(workflow_a)              # → "Ingestion-2024"
normalize_workflow_id(workflow_b)              # → "ingestion-2024"

_workflow_storage_key(workflow_a)              # → "Ingestion-2024"
_workflow_storage_key(workflow_b)              # → "ingestion-2024"

# Database comparison (Django ORM):
"Ingestion-2024" != "ingestion-2024"  # ✅ Different workflows

# PostgreSQL comparison (default collation):
SELECT 'Ingestion-2024' = 'ingestion-2024';  -- FALSE (case-sensitive)
```

**Implications**:
- Users can create workflows with different casing: `"Test"`, `"test"`, `"TEST"`
- Queries must use exact case match
- No case-folding or case-insensitive lookups

**Is this a problem?**
- ⚠️ Potential user confusion (expected behavior for identifiers)
- ✅ Consistent with UUIDs and other identifiers
- ✅ Allows semantic distinction (e.g., `"Dev"` vs `"dev"`)

**Recommendation**: Document case-sensitivity in API docs

---

## 7. Normalization Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Input Layer                         │
│  - HTTP API (X-Workflow-ID header)                             │
│  - Graph initialization (upload_ingestion_graph)                │
│  - Service layer (crawler_runner)                               │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
           ┌──────────────────────┐
           │ resolve_workflow_id()│
           │  - None → "ad-hoc"   │
           │  - Empty → "ad-hoc"  │
           │  - Else → validate   │
           └──────────┬───────────┘
                      │
                      ▼
        ┌──────────────────────────────┐
        │   normalize_workflow_id()    │
        │  - NFKC normalization        │
        │  - Strip invisibles          │
        │  - Validate [A-Za-z0-9._-]+  │
        │  - Max 128 chars             │
        │  - Raise on invalid          │
        └──────────┬───────────────────┘
                   │
                   ▼
        ┌──────────────────────────┐
        │   DocumentRef.workflow_id│  ← Contract/Domain Layer
        │   (validated string)     │
        └──────────┬───────────────┘
                   │
                   ▼
        ┌──────────────────────────┐
        │ _workflow_storage_key()  │  ← Storage Layer
        │  - str(value).strip()    │
        │  - None → ""             │
        └──────────┬───────────────┘
                   │
                   ▼
    ┌──────────────────────────────────────┐
    │       Database Storage               │
    │  - DocumentLifecycleState.workflow_id│  ← CharField(255)
    │  - DocumentAsset.workflow_id         │  ← CharField(255)
    └──────────────────────────────────────┘

                   │
                   ▼
    ┌──────────────────────────────────────┐
    │       Query/Filter Operations        │
    │  - .filter(workflow_id=workflow_key) │
    │  - if doc.ref.workflow_id != wf_id   │
    └──────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════

       Separate Path (File Storage Only)

┌──────────────────────────────────────────┐
│     documents/utils.py                   │
│  get_upload_file_path()                  │
└──────────────┬───────────────────────────┘
               │
               ▼
    ┌──────────────────────────┐
    │ sanitize_identifier()    │  ← Filesystem Safety
    │  - Replace non-safe chars│
    │  - Reject path traversal │
    └──────────┬───────────────┘
               │
               ▼
    ┌──────────────────────────────────┐
    │   File Path Construction         │
    │   {tenant}/{workflow}/uploads/   │
    └──────────────────────────────────┘
```

---

## 8. Key Files Reference

### Normalization Functions
| File | Lines | Function | Purpose |
|------|-------|----------|---------|
| [documents/repository.py](documents/repository.py) | 368-369 | `_workflow_storage_key()` | InMemory/DB storage key normalization |
| [documents/contract_utils.py](documents/contract_utils.py) | 64-74 | `normalize_workflow_id()` | Contract validation & normalization |
| [documents/contract_utils.py](documents/contract_utils.py) | 77-94 | `resolve_workflow_id()` | Optional workflow resolution with placeholder |
| [common/object_store_defaults.py](common/object_store_defaults.py) | 47-56 | `sanitize_identifier()` | Filesystem-safe sanitization |

### Repository Implementations
| File | Lines | Class | Normalization Strategy |
|------|-------|-------|----------------------|
| [documents/repository.py](documents/repository.py) | 891+ | `InMemoryDocumentsRepository` | `_workflow_storage_key()` |
| [ai_core/adapters/db_documents_repository.py](ai_core/adapters/db_documents_repository.py) | 32+ | `DbDocumentsRepository` | `_workflow_storage_key()` (imported) |
| ~~ai_core/adapters/object_store_repository.py~~ | **DELETED** | ~~`ObjectStoreDocumentsRepository`~~ | ~~`sanitize_identifier()`~~ |

### Database Models
| File | Lines | Model | workflow_id Field |
|------|-------|-------|------------------|
| [documents/models.py](documents/models.py) | 143-177 | `DocumentLifecycleState` | CharField(255), default="", unique constraint |
| [documents/models.py](documents/models.py) | 222-268 | `DocumentAsset` | CharField(255), required, unique constraint |

### Tests
| File | Lines | Test Coverage |
|------|-------|--------------|
| [tests/documents/test_contract_utils_workflow_ids.py](tests/documents/test_contract_utils_workflow_ids.py) | 1-33 | `normalize_workflow_id()` validation |
| [ai_core/tests/test_db_documents_repository.py](ai_core/tests/test_db_documents_repository.py) | 1-180 | DB repository operations |
| [ai_core/tests/test_infra.py](ai_core/tests/test_infra.py) | 114-122 | `sanitize_identifier()` behavior |

### Constants
| File | Line | Constant | Value |
|------|------|----------|-------|
| [common/constants.py](common/constants.py) | 26 | `DEFAULT_WORKFLOW_PLACEHOLDER` | `"ad-hoc"` |
| [common/constants.py](common/constants.py) | 12 | `X_WORKFLOW_ID_HEADER` | `"X-Workflow-ID"` |
| [documents/contract_utils.py](documents/contract_utils.py) | 13 | `_WORKFLOW_ID_RE` | `r"^[A-Za-z0-9._-]+$"` |
| [documents/contract_utils.py](documents/contract_utils.py) | 14 | `_WORKFLOW_ID_MAX_LENGTH` | `128` |

---

## 9. Recommendations

### Priority 1: Consolidate Normalization Logic (MEDIUM)

**Issue**: Two normalization functions with overlapping concerns

**Current State**:
```python
# Contract Layer
normalize_workflow_id(value)  # → NFKC + strip + validate

# Storage Layer
_workflow_storage_key(value)  # → str + strip
```

**Option A - Make storage layer use contract validation**:
```python
def _workflow_storage_key(value: Optional[str]) -> str:
    if not value:
        return ""
    # Reuse contract validation (will raise on invalid input)
    return normalize_workflow_id(value.strip())
```

**Pros**:
- Single source of truth
- Catches invalid workflow_id at storage layer
- Enforces contract compliance

**Cons**:
- **Breaking change** if any existing workflow_id in DB has invalid chars
- Need migration to fix existing data
- More strict than current defensive approach

**Recommendation**: **Do NOT implement** - risk of data migration issues

---

**Option B - Document current layering as intentional** (RECOMMENDED):
```python
# documents/repository.py

def _workflow_storage_key(value: Optional[str]) -> str:
    """
    Normalize workflow_id for database storage.

    This function provides DEFENSIVE normalization (whitespace trimming only).
    It assumes the value has already passed contract validation via
    `normalize_workflow_id()` which enforces strict charset rules.

    Design rationale:
    - Contract layer (normalize_workflow_id): Business rule enforcement
    - Storage layer (this function): Defensive cleanup + None handling

    This separation allows contract evolution without breaking storage queries.
    """
    return (str(value).strip() if value else "").strip()
```

**Pros**:
- No breaking changes
- Documents existing behavior
- Preserves backward compatibility

**Cons**:
- Still two functions to maintain

**Recommendation**: **Implement Option B** - Add docstring clarifying design intent

---

### Priority 2: Remove Redundant `sanitize_identifier()` Usage (LOW)

**Issue**: `sanitize_identifier()` used for path construction when workflow_id is already safe

**Current Code** ([documents/utils.py:20-21](documents/utils.py#L20-L21)):
```python
tenant_segment = object_store.sanitize_identifier(tenant_id)
workflow_segment = object_store.sanitize_identifier(workflow_id or "default")
```

**Analysis**:
- `normalize_workflow_id()` already restricts to `[A-Za-z0-9._-]` (all filesystem-safe)
- No `/`, `\`, `:`, or other unsafe chars can exist after contract validation
- `sanitize_identifier()` is redundant

**Recommendation**:
```python
# documents/utils.py

def get_upload_file_path(...):
    """
    Construct file path for document uploads.

    Note: workflow_id has already passed normalize_workflow_id() validation,
    which only allows [A-Za-z0-9._-] characters (all filesystem-safe).
    No additional sanitization needed.
    """
    tenant_segment = str(tenant_id)  # Already validated UUID
    workflow_segment = workflow_id or "default"  # Already normalized

    return f"{tenant_segment}/{workflow_segment}/uploads/{filename}"
```

**Pros**:
- Removes unnecessary function call
- Simplifies code
- Documents safety assumptions

**Cons**:
- Removes defense-in-depth layer
- If contract validation ever changes, paths could break

**Recommendation**: **Keep current code** but add comment explaining it's defensive:
```python
# Defensive sanitization (workflow_id already validated by normalize_workflow_id)
workflow_segment = object_store.sanitize_identifier(workflow_id or "default")
```

---

### Priority 3: Add Test Coverage for Edge Cases (HIGH)

**Missing Tests**:

#### Test 1: Case Sensitivity
```python
# tests/documents/test_workflow_id_normalization.py

def test_workflow_id_is_case_sensitive():
    """workflow_id maintains case sensitivity across all layers."""
    workflow_upper = "Project-2024"
    workflow_lower = "project-2024"

    # Contract layer preserves case
    assert normalize_workflow_id(workflow_upper) == "Project-2024"
    assert normalize_workflow_id(workflow_lower) == "project-2024"

    # Storage layer preserves case
    assert _workflow_storage_key(workflow_upper) == "Project-2024"
    assert _workflow_storage_key(workflow_lower) == "project-2024"

    # Different workflows
    assert workflow_upper != workflow_lower
```

#### Test 2: NFKC Normalization
```python
def test_workflow_id_nfkc_normalization():
    """NFKC normalization handles compatibility equivalents."""
    # Ligature "ﬁ" (U+FB01) → "fi"
    workflow_ligature = "pro\ufb01le"  # "proﬁle"
    normalized = normalize_workflow_id(workflow_ligature)
    assert normalized == "profile"

    # Fullwidth digits (U+FF10-FF19) → ASCII digits
    workflow_fullwidth = "\uff12\uff10\uff12\uff14"  # "２０２４"
    normalized = normalize_workflow_id(workflow_fullwidth)
    assert normalized == "2024"
```

#### Test 3: Cross-Repository Consistency
```python
@pytest.mark.django_db
def test_workflow_id_consistency_across_repositories(tenant_factory):
    """workflow_id remains consistent when stored/retrieved across repositories."""
    tenant = tenant_factory(schema_name="test-tenant")
    workflow_input = "  Test-Workflow_2024.v1  "

    # Step 1: Contract normalization
    workflow_normalized = normalize_workflow_id(workflow_input)
    assert workflow_normalized == "Test-Workflow_2024.v1"

    # Step 2: Create document with normalized workflow_id
    doc = NormalizedDocument(
        ref=DocumentRef(
            tenant_id=tenant.schema_name,
            workflow_id=workflow_normalized,
            document_id=uuid4(),
        ),
        # ...
    )

    # Step 3: Store in DB repository
    repo = DbDocumentsRepository()
    stored = repo.upsert(doc)

    # Step 4: Verify storage key
    lifecycle_state = DocumentLifecycleState.objects.get(
        tenant_id=tenant,
        document_id=stored.ref.document_id,
    )
    assert lifecycle_state.workflow_id == "Test-Workflow_2024.v1"

    # Step 5: Retrieve and verify consistency
    retrieved = repo.get(
        tenant_id=tenant.schema_name,
        document_id=stored.ref.document_id,
        workflow_id=workflow_normalized,
    )
    assert retrieved.ref.workflow_id == workflow_normalized
```

#### Test 4: Special Characters Validation
```python
@pytest.mark.parametrize(
    "invalid_workflow_id,reason",
    [
        ("project:2024", "colon not allowed"),
        ("project/v1", "slash not allowed"),
        ("project@2024", "at-sign not allowed"),
        ("project 2024", "space not allowed"),
        ("project#2024", "hash not allowed"),
    ],
)
def test_workflow_id_rejects_special_characters(invalid_workflow_id, reason):
    """normalize_workflow_id rejects invalid special characters."""
    with pytest.raises(ValueError, match="workflow_invalid_char"):
        normalize_workflow_id(invalid_workflow_id)
```

---

### Priority 4: Document Case Sensitivity in API Docs (HIGH)

**Issue**: Case sensitivity not documented for users

**Recommendation**: Add to API documentation:

```markdown
# Workflow ID Specification

## Format
- **Charset**: Alphanumeric, dot, underscore, hyphen (`[A-Za-z0-9._-]+`)
- **Max Length**: 128 characters
- **Case Sensitive**: Yes (`"Workflow-A"` ≠ `"workflow-a"`)
- **Placeholder**: `"ad-hoc"` (when not specified)

## Examples

### Valid workflow_id values:
- `"ingestion-2024"`
- `"Case_123.v2"`
- `"project.2024-Q1"`
- `"ad-hoc"` (default)

### Invalid workflow_id values:
- `"project:2024"` ❌ (colon not allowed)
- `"path/to/workflow"` ❌ (slash not allowed)
- `"project 2024"` ❌ (space not allowed)
- `""` ❌ (empty string not allowed)
- `"a" * 129` ❌ (exceeds 128 chars)

## Case Sensitivity

workflow_id is **case-sensitive**. The following are treated as **different workflows**:

```json
{
  "workflow_id": "Ingestion-2024"  // Different from...
}
{
  "workflow_id": "ingestion-2024"  // ...this workflow
}
```

## Normalization

User input undergoes two normalization steps:

1. **Contract Validation** (`normalize_workflow_id`):
   - NFKC Unicode normalization
   - Remove invisible characters
   - Strip leading/trailing whitespace
   - Validate charset and length

2. **Storage Normalization** (`_workflow_storage_key`):
   - Strip whitespace (defensive)
   - Handle None/empty values

Both steps preserve case sensitivity.
```

---

### Priority 5: Add Observability for workflow_id Usage (MEDIUM)

**Issue**: No metrics tracking workflow_id patterns

**Recommendation**: Add structured logging and metrics

```python
# documents/contract_utils.py

from ai_core.infra.observability import record_metric

def normalize_workflow_id(value: str) -> str:
    """Normalize and validate workflow identifiers."""

    normalized = normalize_string(value)

    # Track normalization metrics
    if normalized != value:
        record_metric(
            "workflow_id.normalization.changed",
            value=1,
            tags={
                "original_length": len(value),
                "normalized_length": len(normalized),
            }
        )

    if not normalized:
        record_metric("workflow_id.validation.empty", value=1)
        raise ValueError("workflow_empty")

    if len(normalized) > _WORKFLOW_ID_MAX_LENGTH:
        record_metric("workflow_id.validation.too_long", value=1)
        raise ValueError("workflow_too_long")

    if not _WORKFLOW_ID_RE.fullmatch(normalized):
        record_metric(
            "workflow_id.validation.invalid_char",
            value=1,
            tags={"workflow_id_sample": normalized[:20]}  # First 20 chars
        )
        raise ValueError("workflow_invalid_char")

    return normalized
```

**Dashboards**:
- Track most common workflow_id patterns
- Alert on validation failures (potential API abuse)
- Monitor workflow_id length distribution

---

## 10. Migration Considerations

### Existing Data Integrity

**Question**: Are there any workflow_id values in the database that violate current contract rules?

**Audit Query**:
```sql
-- Find lifecycle states with invalid workflow_id
SELECT DISTINCT
    workflow_id,
    COUNT(*) as occurrences
FROM documents_documentlifecyclestate
WHERE workflow_id !~ '^[A-Za-z0-9._-]+$'  -- Regex match for allowed chars
   OR LENGTH(workflow_id) > 128
   OR workflow_id = ''
GROUP BY workflow_id
ORDER BY occurrences DESC;
```

**Cleanup Strategy** (if violations found):
```python
# management/commands/audit_workflow_ids.py

def audit_invalid_workflow_ids():
    """Report workflow_id values that violate contract rules."""
    from documents.contract_utils import normalize_workflow_id

    invalid_states = []

    for state in DocumentLifecycleState.objects.all():
        try:
            normalize_workflow_id(state.workflow_id or "")
        except ValueError as exc:
            invalid_states.append({
                "tenant_id": str(state.tenant_id),
                "document_id": str(state.document_id),
                "workflow_id": state.workflow_id,
                "error": str(exc),
            })

    # Also check DocumentAsset
    for asset in DocumentAsset.objects.all():
        try:
            normalize_workflow_id(asset.workflow_id or "")
        except ValueError as exc:
            invalid_states.append({
                "tenant": str(asset.tenant_id),
                "asset_id": str(asset.asset_id),
                "workflow_id": asset.workflow_id,
                "error": str(exc),
            })

    return invalid_states
```

---

## 11. Performance Considerations

### Index Analysis

**Current Indexes** ([documents/models.py](documents/models.py)):

#### DocumentLifecycleState
```python
indexes = [
    models.Index(
        fields=("tenant_id", "workflow_id"),
        name="doc_lifecycle_tenant_wf_idx",
    ),
]
```

**Analysis**:
- ✅ Composite index on `(tenant_id, workflow_id)` supports filtering
- ✅ Efficient for queries like: `filter(tenant_id=..., workflow_id=...)`
- ✅ Supports partial match: `filter(tenant_id=...)` uses index prefix

**Query Pattern** ([ai_core/adapters/db_documents_repository.py:526-532](ai_core/adapters/db_documents_repository.py#L526-L532)):
```python
lifecycle_exists = models.Exists(
    lifecycle_model.objects.filter(
        tenant_id=tenant,
        workflow_id=workflow_key,  # Uses composite index
        document_id=models.OuterRef("document__id"),
    )
)
```

**Performance**: ✅ **Optimal** - composite index fully utilized

---

#### DocumentAsset
```python
indexes = [
    models.Index(fields=("tenant", "document"), name="doc_asset_tenant_doc_idx"),
    models.Index(fields=("tenant", "asset_id"), name="doc_asset_tenant_asset_idx"),
]

# Missing: index on (tenant, workflow_id)
```

**Query Pattern** ([ai_core/adapters/db_documents_repository.py:443-444](ai_core/adapters/db_documents_repository.py#L443-L444)):
```python
qs = DocumentAsset.objects.filter(tenant=tenant, asset_id=asset_id)
if workflow_id:
    qs = qs.filter(workflow_id=workflow_id)  # ⚠️ No index on workflow_id
```

**Performance Impact**:
- First filter uses `(tenant, asset_id)` index ✅
- Second filter on `workflow_id` requires sequential scan of filtered results ⚠️
- Impact depends on number of assets per tenant/asset_id

**Recommendation**: Monitor query performance. Add index if:
- Asset queries consistently > 50ms
- Many assets per tenant with different workflow_id values

```python
# documents/models.py

class DocumentAsset(models.Model):
    # ...

    class Meta:
        indexes = [
            models.Index(fields=("tenant", "document"), name="doc_asset_tenant_doc_idx"),
            models.Index(fields=("tenant", "asset_id"), name="doc_asset_tenant_asset_idx"),
            # NEW: Add workflow_id index for filtering
            models.Index(
                fields=("tenant", "workflow_id"),
                name="doc_asset_tenant_wf_idx",
            ),
        ]
```

---

## 12. Conclusion

### ✅ Strengths

1. **Divergence Eliminated**: ObjectStore repository deletion removed primary inconsistency
2. **Layered Normalization**: Contract validation + storage normalization provides defense-in-depth
3. **Strict Validation**: `normalize_workflow_id()` enforces consistent charset rules
4. **Test Coverage**: Contract validation tests comprehensive
5. **Consistent Storage**: InMemory & DB repositories use same `_workflow_storage_key()`

### ⚠️ Improvements Recommended

1. **Documentation**: Add docstrings explaining layered normalization design
2. **Test Coverage**: Add case sensitivity, NFKC, and cross-repository consistency tests
3. **API Docs**: Document workflow_id case sensitivity and charset rules
4. **Observability**: Add metrics for workflow_id validation patterns
5. **Index Optimization**: Consider adding `(tenant, workflow_id)` index to DocumentAsset

### 🎯 Risk Assessment

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| Cross-repository query mismatch | **HIGH** | **VERY LOW** | Layered normalization, ObjectStore deleted |
| Case sensitivity confusion | **MEDIUM** | **MEDIUM** | Document in API specs, add tests |
| Invalid workflow_id in existing data | **MEDIUM** | **LOW** | Audit query, migration script if needed |
| Performance degradation on asset queries | **LOW** | **LOW** | Monitor, add index if needed |

**Overall Status**: ✅ **PRODUCTION-READY**

The current implementation is **safe and consistent**. Recommended improvements are documentation and observability enhancements, not bug fixes.

---

## 13. References

1. **NOESIS-2 AGENTS.md**: [AGENTS.md](AGENTS.md) - ID semantics and contracts
2. **ID Propagation Guide**: [docs/architecture/id-propagation.md](docs/architecture/id-propagation.md)
3. **Unicode Normalization**: [UAX #15 - Unicode Normalization Forms](https://unicode.org/reports/tr15/)
4. **Django ORM Indexes**: [Django Index Reference](https://docs.djangoproject.com/en/stable/ref/models/indexes/)
5. **Filesystem Safety**: [OWASP Path Traversal](https://owasp.org/www-community/attacks/Path_Traversal)

---

## Appendix: Normalization Function Comparison Matrix

| Aspect | `normalize_workflow_id()` | `_workflow_storage_key()` | `sanitize_identifier()` |
|--------|--------------------------|---------------------------|------------------------|
| **Purpose** | Contract validation | Storage normalization | Filesystem safety |
| **Layer** | API/Contract | Repository/Storage | Path construction |
| **Unicode** | NFKC normalization | No normalization | No normalization |
| **Whitespace** | Strip leading/trailing | Strip leading/trailing | No stripping |
| **Invisible chars** | Remove (Cf, Cc, Cs) | No removal | No removal |
| **Charset** | Validate `[A-Za-z0-9._-]+` | No validation | Replace non-safe with `_` |
| **Path safety** | Validates (rejects `/`, `:`) | No validation | Rejects `/`, `\`, `..` |
| **Max length** | 128 chars (enforced) | No limit | 128 chars (truncate) |
| **Empty input** | Raises `ValueError` | Returns `""` | Raises `ValueError` |
| **None input** | Raises `TypeError` | Returns `""` | N/A (str required) |
| **Case** | Preserves | Preserves | Preserves |
| **Errors** | Raises on invalid | Never raises | Raises on unsafe |
| **Usage** | Entry points | DB queries | File paths |

**Key Insight**: The three functions serve **different purposes** and are **not interchangeable**. Current usage is appropriate for each context.

---

**Report Generated**: 2025-12-07
**Codebase**: NOESIS-2, Branch `main`
**Analysis Scope**: Complete workflow_id normalization across all repositories
**Status**: ObjectStore divergence eliminated, minor documentation improvements recommended
