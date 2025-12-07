# run_id / ingestion_run_id XOR Constraint Review - Findings

**Date**: 2025-12-07
**Reviewer**: Claude Code
**Task**: Validate XOR constraint compliance for `run_id` vs `ingestion_run_id` across the codebase

---

## Executive Summary

**Contract (from AGENTS.md)**:
- Every Tool invocation requires exactly **ONE** runtime ID: `run_id` **XOR** `ingestion_run_id`
- Both fields must never be set simultaneously
- Both fields must never be empty simultaneously

**Status**: ⚠️ **PARTIAL COMPLIANCE**

### Key Findings

1. ✅ **ScopeContext enforces XOR correctly** - Validation at tool/graph level works
2. ❌ **DocumentLifecycleState model violates XOR** - Database persistence layer does NOT enforce or propagate these IDs
3. ❌ **Missing ID propagation** - `trace_id`, `run_id`, and `ingestion_run_id` are not saved to `DocumentLifecycleState` records

---

## Detailed Analysis

### 1. Contract Definition (AGENTS.md)

From [AGENTS.md:115](AGENTS.md#L115):
```
Pflicht-Tags: `tenant_id`, `trace_id`, `invocation_id` sowie genau eine Laufzeit-ID
(`run_id` **oder** `ingestion_run_id`); optional `idempotency_key`.
```

From [AGENTS.md:242](AGENTS.md#L242):
```
Jeder Tool-Aufruf erfordert `tenant_id`, `trace_id`, `invocation_id` und
genau eine Laufzeit-ID (`run_id` oder `ingestion_run_id`).
```

**Glossar Matrix** (AGENTS.md:193-194):
| Begriff/Feld | Bedeutung | Status | Vorkommen |
|---|---|---|---|
| `run_id` | Laufzeit-ID für einen Graph-Lauf | Pflicht (eine von) | Graph, Tool |
| `ingestion_run_id` | Laufzeit-ID für einen Ingestion-Lauf | Pflicht (eine von) | Graph, Tool |

---

### 2. ScopeContext Validation ✅

**Location**: [ai_core/contracts/scope.py:55-67](ai_core/contracts/scope.py#L55-L67)

```python
@model_validator(mode="after")
def validate_run_scope(self) -> "ScopeContext":
    """Ensure exactly one runtime identifier is provided."""

    has_run_id = bool(self.run_id)
    has_ingestion_run_id = bool(self.ingestion_run_id)

    if has_run_id == has_ingestion_run_id:
        raise ValueError(
            "Exactly one of run_id or ingestion_run_id must be provided"
        )

    return self
```

**Analysis**:
- ✅ Correctly enforces XOR constraint
- ✅ Raises `ValueError` if both are truthy OR both are falsy
- ✅ Used by all tools and graphs that create `ScopeContext`

**Verification in Graphs**:

**Crawler Ingestion Graph** ([ai_core/graphs/crawler_ingestion_graph.py:688-699](ai_core/graphs/crawler_ingestion_graph.py#L688-L699)):
```python
scope = ScopeContext(
    tenant_id=str(chunk_info.meta["tenant_id"]),
    trace_id=str(state.get("trace_id") or uuid4()),
    invocation_id=str(uuid4()),
    ingestion_run_id=str(state.get("ingestion_run_id") or uuid4()),  # ✅ Sets ingestion_run_id
    case_id=(...),
    tenant_schema=str(tenant_schema_value) if tenant_schema_value else None,
)
```

**Upload Ingestion Graph** ([ai_core/graphs/upload_ingestion_graph.py:329-343](ai_core/graphs/upload_ingestion_graph.py#L329-L343)):
```python
ingestion_run_id = self._require_str(payload, "ingestion_run_id")  # ✅ Required from payload
state["meta"].update({
    "tenant_id": tenant_id,
    # ...
    "ingestion_run_id": ingestion_run_id,  # ✅ Propagated to state
})
```

---

### 3. DocumentLifecycleState Model ❌

**Location**: [documents/models.py:143-177](documents/models.py#L143-L177)

```python
class DocumentLifecycleState(models.Model):
    """Latest lifecycle status for a document within a tenant workflow."""

    tenant_id = models.ForeignKey(...)
    document_id = models.UUIDField()
    workflow_id = models.CharField(max_length=255, blank=True, default="")
    state = models.CharField(max_length=32)
    trace_id = models.CharField(max_length=255, blank=True, default="")      # ❌ Can be empty
    run_id = models.CharField(max_length=255, blank=True, default="")        # ❌ Can be empty
    ingestion_run_id = models.CharField(max_length=255, blank=True, default="")  # ❌ Can be empty
    changed_at = models.DateTimeField()
    reason = models.TextField(blank=True, default="")
    policy_events = models.JSONField(blank=True, default=list)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
```

**Issues**:
1. ❌ Both `run_id` and `ingestion_run_id` are optional (`blank=True`)
2. ❌ Both default to empty string (`default=""`)
3. ❌ No database-level constraint enforcing XOR
4. ❌ No model-level validation enforcing XOR
5. ❌ `trace_id` is also optional, violating the "Pflicht" requirement

**Consequences**:
- Can create records with both fields empty (**violates XOR**)
- Can theoretically create records with both fields set (**violates XOR**)
- Cannot trace which graph execution created a lifecycle state
- Breaks observability contract from AGENTS.md

---

### 4. DocumentLifecycleState Persistence ❌

**Location**: [ai_core/adapters/db_documents_repository.py:159-169](ai_core/adapters/db_documents_repository.py#L159-L169)

```python
DocumentLifecycleState.objects.update_or_create(
    tenant_id=tenant,
    document_id=document.id,
    workflow_id=workflow_key,
    defaults={
        "state": doc_copy.lifecycle_state,
        "changed_at": doc_copy.created_at,
        "reason": "",
        "policy_events": [],
    },
)
```

**Critical Issues**:
1. ❌ Does NOT set `trace_id`
2. ❌ Does NOT set `run_id`
3. ❌ Does NOT set `ingestion_run_id`
4. ❌ All three IDs remain as empty strings (default)

**Result**:
- Every `DocumentLifecycleState` record created has:
  - `trace_id = ""`
  - `run_id = ""`
  - `ingestion_run_id = ""`
- **Direct violation** of XOR constraint (both runtime IDs are empty)
- **Direct violation** of AGENTS.md contract (missing mandatory `trace_id`)

---

### 5. HTTP Scope Handling ✅

**Location**: [ai_core/ids/http_scope.py:126-145](ai_core/ids/http_scope.py#L126-L145)

```python
# Logic for run_id/ingestion_run_id:
# If ingestion_run_id is present, we use it.
# If NOT present, we MUST have a run_id. If run_id is also missing, generate one.
# ScopeContext validation will ensure XOR.
if not ingestion_run_id and not run_id:
    run_id = uuid.uuid4().hex

scope_kwargs = {
    "tenant_id": tenant_id,
    "trace_id": trace_id,
    "invocation_id": invocation_id,
    "run_id": run_id,
    "ingestion_run_id": ingestion_run_id,
    # ...
}

return ScopeContext.model_validate(scope_kwargs)  # ✅ Validates XOR
```

**Analysis**:
- ✅ Ensures at least one ID is present before validation
- ✅ Delegates XOR enforcement to `ScopeContext.model_validate()`
- ✅ Generates `run_id` if neither is provided (fallback)

---

## Gap Analysis

### Layer-by-Layer Compliance

| Layer | Component | XOR Enforced? | IDs Propagated? | Status |
|---|---|---|---|---|
| **Tool/Graph** | `ScopeContext` | ✅ Yes (Pydantic validator) | N/A | ✅ Compliant |
| **HTTP API** | `http_scope.py` | ✅ Yes (delegates to ScopeContext) | N/A | ✅ Compliant |
| **Domain** | `DocumentDomainService` | ⚠️ N/A (doesn't create lifecycle states) | ❌ No | ⚠️ Partial |
| **Persistence** | `DbDocumentsRepository` | ❌ No | ❌ No | ❌ Non-compliant |
| **Database** | `DocumentLifecycleState` model | ❌ No | N/A | ❌ Non-compliant |

### Contract Violations

1. **Missing ID Propagation**:
   - `trace_id`, `run_id`, `ingestion_run_id` from `ScopeContext` are NOT saved to `DocumentLifecycleState`
   - Breaks observability: Cannot trace which execution created a lifecycle state

2. **XOR Constraint Not Enforced in DB**:
   - Model allows both fields to be empty (current behavior)
   - Model would allow both fields to be set (if code changed)

3. **Missing Mandatory Fields**:
   - `trace_id` is marked as `Pflicht` (mandatory) in AGENTS.md but is optional in the model

---

## Semantic Analysis

### run_id vs ingestion_run_id Usage

**From AGENTS.md Glossar**:
- `run_id`: "Laufzeit-ID für einen Graph-Lauf" (Runtime ID for a graph run)
- `ingestion_run_id`: "Laufzeit-ID für einen Ingestion-Lauf" (Runtime ID for an ingestion run)

**Usage Pattern**:
- **Ingestion Graphs** (crawler, upload): Use `ingestion_run_id`
- **RAG Graphs** (external_knowledge, collection_search): Use `run_id`
- **Business Logic**: Both are "runtime identifiers" but for different execution contexts

**Relationship** (from AGENTS.md:210):
```
Tools benötigen zusätzlich zu den oben genannten Kontexten immer
``trace_id``, ``invocation_id`` und genau eine Laufzeit-ID
(``run_id`` oder ``ingestion_run_id``)
```

### Why XOR?

From the codebase analysis:
1. **Single Execution Context**: A tool invocation happens in exactly ONE graph execution
2. **Context Disambiguation**: Either it's a regular graph run OR an ingestion run, never both
3. **Observability**: Each runtime ID traces back to a specific graph execution type

---

## Recommendations

### Priority 1: Fix DocumentLifecycleState Persistence (CRITICAL)

**Issue**: IDs not propagated from `ScopeContext` to database

**Solution A - Add IDs to upsert() signature** (Recommended):
```python
# ai_core/adapters/db_documents_repository.py
def upsert(
    self,
    doc: NormalizedDocument,
    workflow_id: Optional[str] = None,
    scope: Optional[ScopeContext] = None,  # ← NEW
) -> NormalizedDocument:
    # ...

    defaults = {
        "state": doc_copy.lifecycle_state,
        "changed_at": doc_copy.created_at,
        "reason": "",
        "policy_events": [],
    }

    # ← NEW: Propagate IDs from scope
    if scope:
        defaults["trace_id"] = scope.trace_id
        if scope.run_id:
            defaults["run_id"] = scope.run_id
        if scope.ingestion_run_id:
            defaults["ingestion_run_id"] = scope.ingestion_run_id

    DocumentLifecycleState.objects.update_or_create(
        tenant_id=tenant,
        document_id=document.id,
        workflow_id=workflow_key,
        defaults=defaults,
    )
```

**Solution B - Extract IDs from NormalizedDocument**:
- Add `trace_id`, `run_id`, `ingestion_run_id` to `DocumentRef` or `DocumentMeta`
- Requires contract changes in `documents.contracts`

**Recommendation**: Use **Solution A** - less invasive, maintains separation of concerns

---

### Priority 2: Add Model-Level Validation (HIGH)

**Issue**: Model allows XOR violations

**Solution**: Add Pydantic-style validation to Django model

```python
# documents/models.py
class DocumentLifecycleState(models.Model):
    # ... fields ...

    def clean(self):
        """Validate XOR constraint for runtime IDs."""
        super().clean()

        has_run_id = bool(self.run_id)
        has_ingestion_run_id = bool(self.ingestion_run_id)

        if has_run_id == has_ingestion_run_id:
            raise ValidationError(
                "Exactly one of run_id or ingestion_run_id must be provided"
            )

    def save(self, *args, **kwargs):
        self.full_clean()  # Enforce validation on save
        super().save(*args, **kwargs)
```

**Note**: This only works for model-level operations, NOT for bulk operations or raw SQL

---

### Priority 3: Add Database Constraint (MEDIUM)

**Issue**: No database-level enforcement

**Solution**: Add a CHECK constraint

```python
# documents/migrations/00XX_add_runtime_id_xor_constraint.py
from django.db import migrations, models

class Migration(migrations.Migration):
    dependencies = [
        ('documents', '0011_documentasset'),
    ]

    operations = [
        migrations.AddConstraint(
            model_name='documentlifecyclestate',
            constraint=models.CheckConstraint(
                check=(
                    models.Q(run_id='', ingestion_run_id__gt='') |
                    models.Q(run_id__gt='', ingestion_run_id='')
                ),
                name='lifecycle_runtime_id_xor',
            ),
        ),
    ]
```

**Note**:
- PostgreSQL supports CHECK constraints
- Constraint ensures: `(run_id empty AND ingestion_run_id not empty) OR (run_id not empty AND ingestion_run_id empty)`

---

### Priority 4: Update All Callers (HIGH)

**Issue**: `upsert()` is called without `scope` parameter

**Affected Files**:
1. `ai_core/graphs/crawler_ingestion_graph.py`
2. `ai_core/graphs/upload_ingestion_graph.py`
3. `documents/domain_service.py` (if it calls repository directly)

**Example Fix**:
```python
# Before
result = repository.upsert(normalized_doc, workflow_id=workflow)

# After
result = repository.upsert(normalized_doc, workflow_id=workflow, scope=scope)
```

---

### Priority 5: Consider Deprecating DocumentLifecycleState.run_id/ingestion_run_id (LOW)

**Question**: Should lifecycle states even track runtime IDs?

**Analysis**:
- `DocumentLifecycleState` tracks the **latest** state per tenant/document/workflow
- Runtime IDs are **ephemeral** - they identify a specific execution
- If we only track the latest state, the runtime ID becomes **stale** after the next state change

**Options**:
1. **Keep IDs** - Useful for debugging "which execution last changed this state?"
2. **Remove IDs** - Simplify model, rely on logs/Langfuse for execution tracking
3. **Add execution history** - Create `DocumentLifecycleHistory` model to track all transitions

**Recommendation**:
- **Keep IDs** for now (observability benefit)
- Consider adding audit trail in future if needed
- Document that these IDs refer to the "last execution that changed this state"

---

## Implementation Plan

### Phase 1: Fix Critical Issues (Sprint 1)

1. ✅ Add `scope` parameter to `DbDocumentsRepository.upsert()`
2. ✅ Propagate `trace_id`, `run_id`, `ingestion_run_id` to `DocumentLifecycleState.defaults`
3. ✅ Update all callers to pass `scope` parameter
4. ✅ Add unit tests for ID propagation
5. ✅ Add integration tests verifying XOR constraint

### Phase 2: Add Model Validation (Sprint 1-2)

1. ✅ Add `clean()` method to `DocumentLifecycleState`
2. ✅ Override `save()` to call `full_clean()`
3. ✅ Add model-level tests for XOR validation
4. ✅ Document validation behavior in model docstring

### Phase 3: Add Database Constraint (Sprint 2)

1. ✅ Create migration with CHECK constraint
2. ✅ Test migration on staging environment
3. ✅ Verify existing data doesn't violate constraint (may need data migration)
4. ✅ Deploy to production

### Phase 4: Documentation (Sprint 2)

1. ✅ Update `docs/architecture/id-guide-for-agents.md` with lifecycle state examples
2. ✅ Document XOR constraint in `documents/models.py` docstrings
3. ✅ Add troubleshooting guide for XOR violations
4. ✅ Update AGENTS.md if needed

---

## Testing Strategy

### Unit Tests

```python
# tests/documents/test_lifecycle_state_xor.py

def test_lifecycle_state_requires_exactly_one_runtime_id():
    """Both runtime IDs empty should raise ValidationError."""
    state = DocumentLifecycleState(
        tenant_id=tenant,
        document_id=uuid4(),
        workflow_id="test",
        state="pending",
        changed_at=datetime.now(timezone.utc),
        trace_id="trace-123",
        run_id="",  # Empty
        ingestion_run_id="",  # Empty
    )

    with pytest.raises(ValidationError, match="Exactly one of run_id"):
        state.full_clean()

def test_lifecycle_state_rejects_both_runtime_ids():
    """Both runtime IDs set should raise ValidationError."""
    state = DocumentLifecycleState(
        tenant_id=tenant,
        document_id=uuid4(),
        workflow_id="test",
        state="pending",
        changed_at=datetime.now(timezone.utc),
        trace_id="trace-123",
        run_id="run-123",  # Set
        ingestion_run_id="ing-456",  # Also set
    )

    with pytest.raises(ValidationError, match="Exactly one of run_id"):
        state.full_clean()

def test_lifecycle_state_accepts_run_id_only():
    """Only run_id should be valid."""
    state = DocumentLifecycleState(
        tenant_id=tenant,
        document_id=uuid4(),
        workflow_id="test",
        state="pending",
        changed_at=datetime.now(timezone.utc),
        trace_id="trace-123",
        run_id="run-123",
        ingestion_run_id="",
    )

    state.full_clean()  # Should not raise
    state.save()

def test_lifecycle_state_accepts_ingestion_run_id_only():
    """Only ingestion_run_id should be valid."""
    state = DocumentLifecycleState(
        tenant_id=tenant,
        document_id=uuid4(),
        workflow_id="test",
        state="pending",
        changed_at=datetime.now(timezone.utc),
        trace_id="trace-123",
        run_id="",
        ingestion_run_id="ing-456",
    )

    state.full_clean()  # Should not raise
    state.save()
```

### Integration Tests

```python
# ai_core/tests/test_db_documents_repository.py

def test_upsert_propagates_scope_ids_for_ingestion():
    """Repository should propagate ingestion_run_id to lifecycle state."""
    scope = ScopeContext(
        tenant_id="test-tenant",
        trace_id="trace-123",
        invocation_id="inv-456",
        ingestion_run_id="ing-789",
    )

    doc = NormalizedDocument(...)
    result = repository.upsert(doc, scope=scope)

    lifecycle = DocumentLifecycleState.objects.get(
        tenant_id=tenant,
        document_id=result.ref.document_id,
    )

    assert lifecycle.trace_id == "trace-123"
    assert lifecycle.ingestion_run_id == "ing-789"
    assert lifecycle.run_id == ""

def test_upsert_propagates_scope_ids_for_regular_run():
    """Repository should propagate run_id to lifecycle state."""
    scope = ScopeContext(
        tenant_id="test-tenant",
        trace_id="trace-123",
        invocation_id="inv-456",
        run_id="run-789",
    )

    doc = NormalizedDocument(...)
    result = repository.upsert(doc, scope=scope)

    lifecycle = DocumentLifecycleState.objects.get(
        tenant_id=tenant,
        document_id=result.ref.document_id,
    )

    assert lifecycle.trace_id == "trace-123"
    assert lifecycle.run_id == "run-789"
    assert lifecycle.ingestion_run_id == ""
```

---

## Migration Considerations

### Existing Data

**Question**: What about existing `DocumentLifecycleState` records with empty IDs?

**Options**:

1. **Backfill Strategy** (if possible):
   - Query Langfuse for historical traces
   - Match by `document_id` + `changed_at` timestamp
   - Populate IDs from trace data

2. **Default Placeholder**:
   - Set `run_id = "unknown"` for all existing records
   - Document that pre-migration records have placeholder IDs

3. **Soft Constraint**:
   - Add constraint but make it NOT VALID initially
   - Only enforce for new records
   - Gradually backfill and validate

**Recommendation**: Use **Option 2** (default placeholder) - simplest and safest

### Migration Script

```python
# documents/migrations/00XX_backfill_runtime_ids.py
from django.db import migrations

def backfill_runtime_ids(apps, schema_editor):
    """Set placeholder runtime IDs for existing lifecycle states."""
    DocumentLifecycleState = apps.get_model('documents', 'DocumentLifecycleState')

    # Update all records with empty runtime IDs
    DocumentLifecycleState.objects.filter(
        run_id='',
        ingestion_run_id=''
    ).update(
        run_id='legacy-unknown'
    )

class Migration(migrations.Migration):
    dependencies = [
        ('documents', '00XX_add_runtime_id_xor_constraint'),
    ]

    operations = [
        migrations.RunPython(backfill_runtime_ids, migrations.RunPython.noop),
    ]
```

---

## Risk Assessment

### High Risk

1. **Breaking Change**: Adding `scope` parameter to `upsert()` affects all callers
   - **Mitigation**: Make parameter optional with default `None`, add deprecation warning

2. **Data Migration**: Existing records violate new constraint
   - **Mitigation**: Backfill before adding constraint, test on staging

### Medium Risk

1. **Performance**: `full_clean()` on every save adds overhead
   - **Mitigation**: Only validate fields that changed, skip for bulk operations if needed

2. **Observability Gap**: Historical states lose execution context
   - **Mitigation**: Document limitation, rely on Langfuse for historical traces

### Low Risk

1. **Type Changes**: May need to update type hints
   - **Mitigation**: Run mypy, fix type errors incrementally

---

## Questions for Team Discussion

1. **Scope Propagation**: Should `scope` be mandatory or optional in `upsert()`?
   - Pro optional: Backward compatible, gradual rollout
   - Pro mandatory: Enforces contract immediately, prevents accidental omissions

2. **Historical Data**: How should we handle existing lifecycle states?
   - Keep as-is with placeholder IDs?
   - Attempt backfill from Langfuse?
   - Delete and recreate?

3. **Model Validation**: Should we enforce `full_clean()` on all saves?
   - May break bulk operations
   - Alternative: Add validation only for API-triggered saves?

4. **Lifecycle State Purpose**: Do we need runtime IDs on lifecycle states?
   - Current use case: Observability
   - Alternative: Rely purely on Langfuse traces?
   - Future use case: Audit trail?

---

## References

1. **AGENTS.md**: [f:\NOESIS-2\NOESIS-2\AGENTS.md](f:\NOESIS-2\NOESIS-2\AGENTS.md)
   - Lines 115, 193-194, 210, 242

2. **ScopeContext**: [ai_core/contracts/scope.py](ai_core/contracts/scope.py)
   - Lines 39-67 (model definition)
   - Lines 55-67 (XOR validation)

3. **DocumentLifecycleState**: [documents/models.py](documents/models.py)
   - Lines 143-177 (model definition)

4. **DbDocumentsRepository**: [ai_core/adapters/db_documents_repository.py](ai_core/adapters/db_documents_repository.py)
   - Lines 159-169 (lifecycle state persistence)

5. **HTTP Scope**: [ai_core/ids/http_scope.py](ai_core/ids/http_scope.py)
   - Lines 126-145 (runtime ID handling)

6. **Crawler Ingestion Graph**: [ai_core/graphs/crawler_ingestion_graph.py](ai_core/graphs/crawler_ingestion_graph.py)
   - Lines 688-699 (scope creation with ingestion_run_id)

7. **Upload Ingestion Graph**: [ai_core/graphs/upload_ingestion_graph.py](ai_core/graphs/upload_ingestion_graph.py)
   - Lines 329-343 (ingestion_run_id propagation)

---

## Conclusion

The XOR constraint for `run_id` / `ingestion_run_id` is **correctly enforced at the tool/graph layer** via `ScopeContext` validation, but **completely missing at the database persistence layer**.

This creates a **compliance gap** where:
- ✅ Tools and graphs cannot create invalid contexts
- ❌ Database can contain invalid lifecycle states
- ❌ Observability is broken (no trace IDs saved)

**Recommended Action**: Implement Priority 1-3 fixes in Sprint 1-2 to achieve full compliance with AGENTS.md contracts.
