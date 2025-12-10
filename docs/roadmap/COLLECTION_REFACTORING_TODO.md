# Collection Architecture Refactoring - TODO

**Status:** Planning
**Priority:** HIGH
**Estimated Effort:** 2-3 Sprints
**Owner:** TBD
**Created:** 2025-12-10
**Last Updated:** 2025-12-10

---

## Executive Summary

Collections sollten **workflow-agnostische "logische Aktenschränke"** sein, die Dokumente aus allen Ingestion-Quellen (Upload, Crawler, Confluence, etc.) organisieren. Aktuell sind sie de facto auf Upload-Workflows beschränkt.

**Ziel:** Collections als universelles Organisationsprinzip etablieren und Drift zwischen Dokumentation und Implementation beseitigen.

---

## Problem Statement

### 1. Drift: Doku vs. Implementation

**Dokumentation sagt:**
> "collection_id ergänzt tenant_id und case_id als **optionale Persistenz- und Filterdimension**"
> → Collections sollten für **alle Workflows** verfügbar sein

**Implementation ist:**
- Nur Upload-Graph setzt Collections aktiv
- Crawler/Integration haben keine Collection-Assignment-Mechanismen
- Kein Routing von Crawler-Dokumenten in spezifische Collections

### 2. Dual-Field-Verwirrung

```python
DocumentRef.collection_id              # Was ist das?
DocumentRef.document_collection_id     # Wozu zwei Felder?
DocumentMeta.document_collection_id    # Redundanz!
```

**Validator erzwingt:** `collection_id == document_collection_id`
→ Warum existieren beide, wenn sie gleich sein müssen?

### 3. Fehlende Workflow-Integration

- **Crawler:** Liest `document_collection_id` aus State, nutzt es aber nicht
- **Integration:** Kein Collection-Support vorhanden
- **Upload:** Einziger Workflow mit aktiver Collection-Nutzung

### 4. DB-Schema-Lücken

- `unique_collection_id_per_tenant` ist modelliert, aber **nicht migriert**
- Duplikate sind möglich, keine DB-seitige Garantie
- `DocumentCollection.id` (PK) vs. `collection_id` (logical) Divergenz

---

## Goals & Non-Goals

### Goals ✅

1. **Workflow-Agnostische Collections:**
   - Upload → Collection "manual-search"
   - Crawler → Collection "web-crawler"
   - Confluence → Collection "confluence-import"
   - Alle durchsuchbar mit einheitlichem Collection-Filter!

2. **Field Consolidation:**
   - Ein einziges `collection_id` Feld (nicht zwei!)
   - Klare Semantik: References auf `DocumentCollection` Modell

3. **DB Integrity:**
   - `unique_collection_id_per_tenant` Constraint migriert
   - Keine Duplikate möglich

4. **Crawler Collection Support:**
   - Crawler kann Dokumente in Collections routen
   - System-Collections auto-bootstrapped (web-crawler, etc.)

5. **Documentation Alignment:**
   - AGENTS.md erweitert um collection_id in ID-Matrix
   - RAG-Overview aktualisiert mit Workflow-Beispielen

### Non-Goals ❌

- **NICHT:** Complete Collection Registry UI (separates Epic)
- **NICHT:** Hierarchische Collections (v2 Feature)
- **NICHT:** Cross-Tenant Collections (Security-Risk)
- **NICHT:** Collection-Level Permissions (separates Epic)

---

## Implementation Plan

### Phase 1: Foundation (Sprint 1) - CRITICAL

#### 1.1 Add DB Constraint Migration ✅ Priority: P0

**File:** `documents/migrations/00XX_add_collection_uniqueness.py`

```python
operations = [
    migrations.AddConstraint(
        model_name='documentcollection',
        constraint=models.UniqueConstraint(
            fields=['tenant', 'collection_id'],
            name='unique_collection_id_per_tenant'
        ),
    ),
]
```

**Risk:** Existing data may have duplicates!

**Mitigation:**
```sql
-- Pre-migration check
SELECT tenant_id, collection_id, COUNT(*)
FROM documents_documentcollection
GROUP BY tenant_id, collection_id
HAVING COUNT(*) > 1;

-- If duplicates exist, manual resolution needed
```

#### 1.2 Regression Tests ✅ DONE

**File:** `documents/tests/test_domain_service_collection_ids.py`

- [x] Test: Ingestion uses logical collection_id (not PK)
- [x] Test: Bulk ingest preserves logical IDs
- [x] Test: Collection resolution returns correct ID
- [x] Test: Multiple collections all use logical IDs

**Status:** Tests written, ready to run

#### 1.3 Field Deprecation Plan 📝 Priority: P0

**Strategy:**
1. Add `DeprecationWarning` for `document_collection_id`
2. Backfill: `collection_id = document_collection_id` where null
3. Update all internal code to use `collection_id` only
4. Mark `document_collection_id` as deprecated in schema
5. **Breaking Change:** Remove in v3.0 (6 months deprecation period)

**Migration:**
```python
# Step 1: Backfill collection_id from document_collection_id
DocumentRef.objects.filter(
    collection_id__isnull=True,
    document_collection_id__isnull=False
).update(collection_id=F('document_collection_id'))

# Step 2: Add deprecation warning
@property
def document_collection_id(self):
    warnings.warn(
        "document_collection_id is deprecated, use collection_id",
        DeprecationWarning,
        stacklevel=2
    )
    return self.collection_id
```

---

### Phase 2: Crawler Integration (Sprint 1-2)

#### 2.1 Crawler Collection Assignment 🔧 Priority: P1

**File:** `ai_core/graphs/crawler_ingestion_graph.py`

**Changes:**
```python
def run(self, state, meta):
    # NEW: Extract collection key from state
    collection_key = state.get("collection_key", "web-crawler")

    # NEW: Resolve collection
    if collection_key:
        collection = self._ensure_collection(
            tenant=tenant,
            key=collection_key,
            label=f"Web Crawler: {collection_key}"
        )
        document_collection_id = collection.collection_id

    # Pass to document graph context
    context = DocumentProcessingContext.from_document(
        document,
        case_id=...,
        collection_id=document_collection_id,  # NEW!
    )
```

**Impact:** Crawler-Dokumente können jetzt in Collections zugeordnet werden!

#### 2.2 System Collections Bootstrap 🔧 Priority: P1

**File:** `documents/management/commands/bootstrap_system_collections.py`

```python
SYSTEM_COLLECTIONS = {
    "manual-search": {
        "label": "Manual Uploads",
        "description": "User-uploaded documents",
    },
    "web-crawler": {
        "label": "Web Crawler",
        "description": "Crawled web content",
    },
    "confluence": {
        "label": "Confluence Import",
        "description": "Imported Confluence pages",
    },
    "email": {
        "label": "Email Import",
        "description": "Imported email content",
    },
}

def handle(self, *args, **options):
    for tenant in Tenant.objects.all():
        for key, config in SYSTEM_COLLECTIONS.items():
            ensure_collection(tenant, key=key, **config)
```

**Trigger:** Add to `bootstrap_tenant_schema()` in `conftest.py`

#### 2.3 Integration Tests 📝 Priority: P1

**File:** `ai_core/tests/test_crawler_collections.py`

- [ ] Test: Crawler assigns documents to web-crawler collection
- [ ] Test: Collection key from state overrides default
- [ ] Test: Documents are queryable by collection filter
- [ ] Test: System collections auto-created on tenant setup

---

### Phase 3: Upload Graph Alignment (Sprint 2)

#### 3.1 Upload Collection Assignment 🔍 Priority: P2

**Investigation:** Wo wird `collection_id` für Uploads aktuell gesetzt?
- Web-Layer untersuchen (außerhalb von Graph)
- Sicherstellen: Konsistente Nutzung von `collection_id`

**Expected Path:**
```
Web Request → UploadView → UploadIngestionGraph → DocumentProcessingContext
                                                      ↓
                                                collection_id setzen
```

#### 3.2 Align Upload with Unified Pattern 🔧 Priority: P2

**Ensure:**
- Upload nutzt `collection_id` (nicht `document_collection_id`)
- Collection-Assignment erfolgt im Graph (nicht Web-Layer)
- Default: "manual-search" Collection

---

### Phase 4: Integration Workflow Support (Sprint 2-3)

#### 4.1 Generic Integration Pattern 📝 Priority: P2

**Design:**
```python
# integration_ingestion_graph.py
def run(self, state, meta):
    source = state.get("source")  # "confluence", "sharepoint", etc.
    collection_key = f"{source}-import"

    collection = ensure_collection(
        tenant=tenant,
        key=collection_key,
        label=f"{source.title()} Import"
    )
```

**Benefit:** Jede Integration bekommt automatisch eigene Collection!

#### 4.2 Confluence Connector Example 🔧 Priority: P3

**File:** `integrations/confluence/connector.py`

```python
def sync_space(self, space_key: str):
    for page in confluence_client.get_pages(space_key):
        self.ingest_document(
            content=page.content,
            metadata={
                "title": page.title,
                "source": "confluence",
                "collection_key": f"confluence-{space_key}",
            }
        )
```

**Result:** Confluence Space → Eigene Collection!

---

### Phase 5: Documentation & Cleanup (Sprint 3)

#### 5.1 Update AGENTS.md 📝 Priority: P2

**Changes:**
- [ ] Add `collection_id` to ID-Matrix (Glossar)
- [ ] Clarify: Collections sind workflow-agnostisch
- [ ] Example: Upload, Crawler, Integration alle nutzen Collections

#### 5.2 Update RAG Documentation 📝 Priority: P2

**Files:**
- `docs/rag/overview.md` - Erweitern mit Workflow-Beispielen
- `docs/rag/ingestion.md` - Collection-Assignment für alle Workflows
- `docs/architecture/collection-registry-sota.md` - Mark as IN PROGRESS

#### 5.3 Deprecation Notices 📝 Priority: P2

**Add to:**
- `CHANGELOG.md` - Deprecation von `document_collection_id`
- `MIGRATION_GUIDE.md` - How to migrate from old to new field
- API Docs - Deprecation warnings in OpenAPI schema

---

## Migration Strategy

### For Existing Data

#### Step 1: Pre-Migration Validation

```python
# Check for duplicates
duplicates = DocumentCollection.objects.values(
    'tenant', 'collection_id'
).annotate(count=Count('id')).filter(count__gt=1)

if duplicates.exists():
    raise ValueError("Duplicates found, manual resolution required")
```

#### Step 2: Backfill collection_id

```python
# Ensure all DocumentRef have collection_id
DocumentRef.objects.filter(
    collection_id__isnull=True,
    document_collection_id__isnull=False
).update(collection_id=F('document_collection_id'))
```

#### Step 3: Apply Constraint Migration

```bash
python manage.py migrate documents 00XX_add_collection_uniqueness
```

#### Step 4: Bootstrap System Collections

```bash
python manage.py bootstrap_system_collections
```

### For Client Code

**Old Code (Deprecated):**
```python
doc_ref = DocumentRef(
    document_id=doc_id,
    document_collection_id=collection_id,  # ❌ Deprecated
)
```

**New Code:**
```python
doc_ref = DocumentRef(
    document_id=doc_id,
    collection_id=collection_id,  # ✅ Correct
)
```

**Migration Window:** 6 months deprecation before removal

---

## Testing Strategy

### Unit Tests

- [x] Regression: collection.collection_id vs. collection.id
- [ ] Field consolidation: collection_id only
- [ ] Crawler collection assignment
- [ ] System collections bootstrap

### Integration Tests

- [ ] E2E: Upload → manual-search collection
- [ ] E2E: Crawler → web-crawler collection
- [ ] E2E: Integration → source-specific collection
- [ ] E2E: Query by collection filter (all sources)

### Migration Tests

- [ ] Duplicate detection
- [ ] Backfill validation
- [ ] Constraint enforcement
- [ ] Rollback safety

---

## Risks & Mitigation

### Risk 1: Breaking Change for Clients

**Impact:** HIGH
**Probability:** MEDIUM
**Mitigation:**
- 6-month deprecation period
- Clear migration guide
- Automated migration script for common patterns
- CI/CD warnings for deprecated field usage

### Risk 2: Duplicate Collections in Production

**Impact:** HIGH
**Probability:** LOW
**Mitigation:**
- Pre-migration validation script
- Manual resolution process for duplicates
- Post-migration verification

### Risk 3: Performance Impact

**Impact:** MEDIUM
**Probability:** LOW
**Mitigation:**
- Collection lookups already indexed
- System collections cached at startup
- Benchmark before/after migration

### Risk 4: Crawler Collection Assignment Breaks Existing Workflows

**Impact:** MEDIUM
**Probability:** MEDIUM
**Mitigation:**
- Feature flag: `ENABLE_CRAWLER_COLLECTIONS`
- Gradual rollout per tenant
- Fallback to tenant-wide pool if disabled

---

## Success Criteria

### Must Have ✅

- [ ] All workflows can assign documents to collections
- [ ] DB uniqueness constraint enforced
- [ ] No usage of `document_collection_id` in new code
- [ ] System collections auto-bootstrapped
- [ ] Documentation updated (AGENTS.md, RAG docs)
- [ ] Migration guide published

### Should Have 📋

- [ ] Regression tests passing
- [ ] Integration tests for all workflows
- [ ] CI/CD checks for deprecated field
- [ ] Performance benchmarks green

### Nice to Have 🎁

- [ ] Admin UI for collection management
- [ ] Collection-level analytics dashboard
- [ ] Collection export/import tools

---

## Timeline

### Sprint 1 (Weeks 1-2)
- [x] Regression tests
- [ ] DB constraint migration
- [ ] Field deprecation plan
- [ ] Crawler collection assignment

### Sprint 2 (Weeks 3-4)
- [ ] System collections bootstrap
- [ ] Upload alignment
- [ ] Integration pattern design
- [ ] Documentation updates

### Sprint 3 (Weeks 5-6)
- [ ] Confluence connector example
- [ ] Migration guide
- [ ] Deprecation notices
- [ ] Final testing & validation

---

## Open Questions

1. **Should collections be optional or required?**
   - Current: Optional
   - Proposed: Required with default "uncategorized"

2. **How to handle historical documents without collections?**
   - Option A: Backfill into "legacy" collection
   - Option B: Keep as null (query as "no collection")
   - **Decision:** TBD

3. **Collection deletion: Hard or soft delete?**
   - Current: Soft delete (lifecycle_state)
   - Impact: Documents retain deleted collection references
   - **Decision:** Keep soft delete, filter in queries

4. **Cross-tenant collections for shared knowledge bases?**
   - Security risk: Tenant isolation violation
   - Use case: Company-wide knowledge base
   - **Decision:** NOT in v1, evaluate for v2

---

## References

- [Collection ID Review](../architecture/collection_id_review.md)
- [AGENTS.md](../../AGENTS.md)
- [RAG Overview](../rag/overview.md)
- [Collection Registry SOTA](../architecture/collection-registry-sota.md)
- Commit de4e089: "address collection ID consistency issues"

---

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2025-12-10 | Initial TODO document created | Claude |

---

**Next Action:** Review & prioritize with team, assign ownership for Phase 1 tasks.
