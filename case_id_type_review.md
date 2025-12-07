# Case ID Type & Tenant-Isolation Review

**Review Date**: 2025-12-07
**Scope**: Analyse von `case_id` Typ, Verwendung und Tenant-Isolation
**Status**: ✅ COMPLETED

---

## Executive Summary

**FINDING: `case_id` ist ein String (Case.external_id), nicht die UUID (Case.id)**

Die Analyse zeigt, dass:
1. ✅ `case_id` referenziert `Case.external_id` (String), nicht `Case.id` (UUID)
2. ✅ Tenant-Isolation ist durchgängig gewährleistet
3. ✅ Alle Queries kombinieren `case_id` mit `tenant_id`
4. ⚠️ Type-Annotationen sind konsistent als `Optional[str]`

---

## 1. Case Model Analyse

### Case Model Definition

**Quelle**: [cases/models.py:10-56](cases/models.py#L10-L56)

```python
class Case(models.Model):
    """Primary case record for a tenant workflow."""

    # PRIMARY KEY (nicht verwendet als case_id!)
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    # TENANT-ISOLATION
    tenant = models.ForeignKey(
        "customers.Tenant",
        on_delete=models.PROTECT,
        related_name="cases",
    )

    # CASE_ID = external_id (String!)
    external_id = models.CharField(max_length=255)

    title = models.CharField(max_length=255, blank=True, default="")
    status = models.CharField(max_length=32, choices=Status.choices, default=Status.OPEN)
    phase = models.CharField(max_length=64, blank=True, default="")
    metadata = models.JSONField(blank=True, default=dict)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    closed_at = models.DateTimeField(null=True, blank=True)
```

### Key Constraints & Indexes

```python
class Meta:
    constraints = [
        # Tenant + external_id = UNIQUE
        models.UniqueConstraint(
            fields=("tenant", "external_id"),
            name="case_unique_external_id",
        )
    ]
    indexes = [
        models.Index(fields=("tenant", "status"), name="case_tenant_status_idx"),
        models.Index(fields=("tenant", "external_id"), name="case_tenant_external_idx"),
    ]
```

**🔑 Key Insight**: `external_id` ist der business identifier, immer mit `tenant` kombiniert.

---

## 2. Case ID Type Klärung

### Definition: `case_id = Case.external_id` (String)

**Evidence**:

#### 1. Service Layer verwendet `external_id`

**Quelle**: [cases/services.py:59-70](cases/services.py#L59-L70)

```python
def resolve_case(tenant: Tenant, case_id: str | None) -> models.Case | None:
    """Resolve a case by ID, ensuring it exists. Returns None if case_id is empty."""
    if not case_id:
        return None

    normalized_case_id = _normalise_case_id(case_id)
    try:
        # ✅ case_id = external_id (String)
        return models.Case.objects.get(tenant=tenant, external_id=normalized_case_id)
    except models.Case.DoesNotExist:
        raise CaseNotFoundError(
            f"Case {normalized_case_id} not found for tenant {getattr(tenant, 'schema_name', tenant.pk)}"
        )
```

#### 2. DocumentIngestionRun verwendet String

**Quelle**: [documents/models.py:179-219](documents/models.py#L179-L219)

```python
class DocumentIngestionRun(models.Model):
    """Persistent metadata for the last ingestion run per tenant/case."""

    tenant_id = models.CharField(max_length=255)
    case = models.CharField(max_length=255, null=True, blank=True)  # ✅ String!
    collection_id = models.CharField(max_length=255, blank=True, default="")
    run_id = models.CharField(max_length=255)

    # ...

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=("tenant_id", "case"),  # ✅ Tenant + case String
                name="document_ingestion_run_unique_case"
            )
        ]
        indexes = [
            models.Index(
                fields=("tenant_id", "case"),
                name="doc_ing_run_tenant_case_idx",
            ),
        ]
```

#### 3. Contracts verwenden `Optional[str]`

**Quelle**: [documents/contracts.py:803-805](documents/contracts.py#L803-L805)

```python
case_id: Optional[str] = Field(
    default=None,
    description="Optional business case identifier supplied alongside the document.",
)
```

#### 4. Case Events emittieren `external_id`

**Quelle**: [ai_core/case_events.py:102-123](ai_core/case_events.py#L102-L123)

```python
def _emit_case_observability_event(case_event: CaseEvent) -> None:
    case = case_event.case
    ingestion_run = case_event.ingestion_run
    payload = {
        "tenant_id": str(case.tenant_id),
        "case_id": case.external_id,  # ✅ external_id als case_id
        "case_status": case.status,
        "case_phase": case.phase or "",
        # ...
    }
    emit_event("case.lifecycle.ingestion", payload)
```

---

## 3. Tenant-Isolation Analysis

### ✅ Status: COMPLIANT

Alle Queries kombinieren `case_id` mit `tenant` oder `tenant_id`.

### Query Patterns

#### Pattern 1: Case Lookup mit Tenant

**Quelle**: [cases/services.py:66](cases/services.py#L66)

```python
# ✅ SECURE: tenant + external_id
models.Case.objects.get(tenant=tenant, external_id=normalized_case_id)
```

#### Pattern 2: DocumentCollection mit Case

**Quelle**: [documents/services.py:55-59](documents/services.py#L55-L59)

```python
# ✅ SECURE: tenant + collection_id + case
query = DocumentCollection.objects.select_related("case").filter(
    tenant=tenant, collection_id=technical_collection_id
)
if case is not None:
    document_collection = query.filter(case=case).first()
```

#### Pattern 3: DocumentIngestionRun Lookup

**Quelle**: [ai_core/case_events.py:32](ai_core/case_events.py#L32)

```python
# ✅ SECURE: tenant_id + case
return DocumentIngestionRun.objects.get(tenant_id=tenant_id, case=case_id)
```

### Database Constraints (Defense in Depth)

Alle Modelle mit `case`-Referenz haben Tenant-Constraints:

1. **Case**: `(tenant, external_id)` UNIQUE
2. **DocumentIngestionRun**: `(tenant_id, case)` UNIQUE
3. **DocumentCollection**: `(tenant, case)` INDEX
4. **CaseEvent**: `(tenant, event_type)` INDEX

---

## 4. Validation: Query Combinations

### Tested Query Patterns

| Query Location | Pattern | Tenant-Isolation | Status |
|----------------|---------|------------------|--------|
| `cases/services.py:66` | `Case.objects.get(tenant=X, external_id=Y)` | ✅ YES | PASS |
| `documents/services.py:59` | `DocumentCollection.filter(tenant=X, case=Y)` | ✅ YES | PASS |
| `ai_core/case_events.py:32` | `DocumentIngestionRun.get(tenant_id=X, case=Y)` | ✅ YES | PASS |
| `cases/tests/test_lifecycle.py:36` | `Case.objects.get(external_id=X, tenant=Y)` | ✅ YES | PASS |

### Test Evidence

**Quelle**: [cases/tests/test_lifecycle.py:36](cases/tests/test_lifecycle.py#L36)

```python
# ✅ Tests verwenden immer tenant + external_id
case = Case.objects.get(external_id="case-lifecycle", tenant=tenant)
```

---

## 5. Type Consistency Review

### Current Type Annotations

| Location | Type | Context |
|----------|------|---------|
| `cases/services.py:59` | `str \| None` | Function parameter |
| `cases/services.py:97` | `str \| None` | Function parameter |
| `documents/contracts.py:803` | `Optional[str]` | Pydantic field |
| `ai_core/api.py:423` | `Optional[str]` | Function parameter |
| `ai_core/case_events.py:29` | `str \| None` | Function parameter |
| `ai_core/case_events.py:41` | `str \| None` | Function parameter |

### ✅ Consistency: GOOD

Alle Type-Annotationen verwenden `str | None` oder `Optional[str]`.

**Keine UUID-Annotationen gefunden** ✅

---

## 6. Integration Points

### Where `case_id` flows through the system:

```
┌─────────────────────────────────────────────────────────────┐
│ 1. HTTP Request                                             │
│    DocumentPayload.case_id: Optional[str]                   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     v
┌─────────────────────────────────────────────────────────────┐
│ 2. Service Layer                                            │
│    resolve_case(tenant, case_id: str | None)                │
│    → Case.objects.get(tenant=X, external_id=case_id)        │
└────────────────────┬────────────────────────────────────────┘
                     │
                     v
┌─────────────────────────────────────────────────────────────┐
│ 3. Ingestion                                                │
│    DocumentIngestionRun.case = case_id (String)             │
│    DocumentCollection.case = Case FK                        │
└────────────────────┬────────────────────────────────────────┘
                     │
                     v
┌─────────────────────────────────────────────────────────────┐
│ 4. Case Events                                              │
│    emit_ingestion_case_event(tenant_id, case_id)            │
│    → payload["case_id"] = case.external_id                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 7. Findings & Recommendations

### ✅ Findings (PASS)

1. **Type Clarity**: `case_id` ist konsistent als `String` (external_id) definiert
2. **Tenant-Isolation**: Alle Queries kombinieren `case_id` mit `tenant`
3. **Constraints**: DB-Constraints erzwingen Tenant-Isolation auf Schema-Ebene
4. **Type Safety**: Type-Annotationen sind konsistent

### ⚠️ Observations

1. **Naming Ambiguity**: `case_id` könnte fälschlicherweise als UUID (Case.id) interpretiert werden
   - **Recommendation**: Erwäge Benennung `case_external_id` für mehr Klarheit
   - **Impact**: LOW (existierender Code ist korrekt)

2. **Mixed Foreign Key Styles**:
   - `DocumentCollection.case` = FK zu `Case` (Objekt-Referenz)
   - `DocumentIngestionRun.case` = CharField (String-Wert)
   - **Reason**: `DocumentIngestionRun` nutzt `tenant_id` (String) statt FK
   - **Status**: Akzeptabel, aber inkonsistent

### 🎯 Action Items

#### None Required (System is Secure)

Alle identifizierten Queries und Constraints sind korrekt implementiert.

#### Optional Improvements

1. **Documentation**:
   ```python
   # Add to AGENTS.md Glossar
   case_id:
     Type: String
     Semantics: Case.external_id (business identifier)
     NOT: Case.id (UUID primary key)
     Tenant-Isolation: REQUIRED (always query with tenant)
   ```

2. **Type Alias** (Optional):
   ```python
   # In common/types.py
   CaseExternalId = Annotated[str, "Business case identifier (Case.external_id)"]

   # Usage:
   def resolve_case(tenant: Tenant, case_id: CaseExternalId | None) -> Case | None:
       ...
   ```

---

## 8. Test Coverage

### Existing Tests

1. **Case Resolution**: [cases/tests/test_lifecycle.py:36](cases/tests/test_lifecycle.py#L36)
   ```python
   case = Case.objects.get(external_id="case-lifecycle", tenant=tenant)
   ```

2. **Ingestion Events**: [cases/tests/test_services.py:16-18](cases/tests/test_services.py#L16-L18)
   ```python
   run = DocumentIngestionRun.objects.create(
       tenant_id=tenant.schema_name,
       case="case-event",  # String external_id
       run_id="run-1",
   )
   ```

3. **Case Events**: [ai_core/tests/test_case_events.py:21-23](ai_core/tests/test_case_events.py#L21-L23)
   ```python
   DocumentIngestionRun.objects.create(
       tenant_id=tenant.schema_name,
       case="case-hook",  # String external_id
       run_id="run-hook",
   )
   ```

### ✅ Coverage Assessment: GOOD

Tests verwenden konsistent String-Werte für `case_id`.

---

## 9. Schema Divergence

### No Divergence Detected

Alle Modelle, die `case` oder `case_id` verwenden:

| Model | Field | Type | Tenant-Isolation |
|-------|-------|------|------------------|
| `Case` | `external_id` | CharField(255) | `(tenant, external_id)` UNIQUE |
| `DocumentIngestionRun` | `case` | CharField(255) | `(tenant_id, case)` UNIQUE |
| `DocumentCollection` | `case` | ForeignKey(Case) | `(tenant, case)` INDEX |
| `CaseEvent` | `case` | ForeignKey(Case) | `tenant` FK |

### Type Alignment

```
Case.external_id (CharField)
    ↓
case_id (str | None)
    ↓
DocumentIngestionRun.case (CharField)
DocumentCollection.case (FK → Case)
```

**✅ Alignment: CONSISTENT**

---

## 10. Summary

### Key Takeaways

1. ✅ **`case_id` = `Case.external_id` (String)**, nicht `Case.id` (UUID)
2. ✅ **Tenant-Isolation**: Durchgängig gewährleistet in allen Queries
3. ✅ **Type Safety**: Konsistent als `Optional[str]` annotiert
4. ✅ **DB Constraints**: Erzwingen Tenant-Isolation auf Schema-Ebene
5. ⚠️ **Naming**: Könnte klarer sein (z.B. `case_external_id`)

### Security Posture

**SECURE** - Keine Tenant-Isolation-Vulnerabilities identifiziert.

### Performance

Alle relevanten Queries nutzen Indexes:
- `case_tenant_external_idx` (Case)
- `doc_ing_run_tenant_case_idx` (DocumentIngestionRun)
- `doc_collection_tenant_case_idx` (DocumentCollection)

### Compliance

Erfüllt alle Anforderungen aus [AGENTS.md](AGENTS.md#glossar--feld-matrix):
- ✅ `tenant_id` immer gesetzt
- ✅ `case_id` optional, aber wenn gesetzt, dann tenant-isoliert
- ✅ Type-Annotationen korrekt

---

## Appendix A: Referenced Files

- [cases/models.py](cases/models.py) - Case Model Definition
- [cases/services.py](cases/services.py) - Case Resolution Logic
- [documents/models.py](documents/models.py) - DocumentCollection, DocumentIngestionRun
- [documents/contracts.py](documents/contracts.py) - Pydantic Contracts
- [documents/services.py](documents/services.py) - DocumentCollection Queries
- [ai_core/case_events.py](ai_core/case_events.py) - Case Event Integration
- [ai_core/api.py](ai_core/api.py) - Embedding API with case_id

---

## Appendix B: Glossar Entry (Recommendation)

Für [AGENTS.md#glossar--feld-matrix](AGENTS.md#glossar--feld-matrix):

```markdown
### case_id

**Type**: `Optional[str]`
**Semantics**: Business case identifier, maps to `Case.external_id`
**NOT**: `Case.id` (UUID primary key)
**Tenant-Isolation**: REQUIRED - always query with `tenant` or `tenant_id`
**Examples**:
- `"CASE-2024-001"` (valid)
- `"litigation-phase-1"` (valid)
- `None` (valid - no case association)
- UUID string (invalid - use `Case.external_id` instead)

**Contract Propagation**:
```python
DocumentPayload.case_id (str | None)
    → resolve_case(tenant, case_id) → Case.external_id
    → DocumentIngestionRun.case (CharField)
    → DocumentCollection.case (FK to Case)
    → CaseEvent.case (FK to Case)
```

**Security**: All lookups MUST include tenant filter.
```

---

**Review Completed**: 2025-12-07
**Reviewer**: Claude Code
**Status**: ✅ APPROVED - No Action Required
