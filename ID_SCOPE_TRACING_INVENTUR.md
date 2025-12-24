# ID/Scope/Tracing-Konsistenz Inventur

**Datum**: 2025-12-21
**Zweck**: Vollständige Analyse aller Identifier (tenant_id, tenant_schema, case_id, workflow_id, collection_id, idempotency_key, trace_id, run_id, ingestion_run_id, invocation_id), Header-/Meta-Normalisierung, Scope-/ToolContext-Propagation und DB-Persistenz.

**Methodik**: Code ist Quelle (nicht Doku). End-to-End-Flows geprüft: HTTP Request → ScopeContext/ToolContext → Graph/Services → Persistence/Logs → Responses.

---

## 1. Canonical Contracts (Sollzustand)

### 1.1 HTTP Headers (common/constants.py)

**Definiert (Zeile 4-12)**:
- `X-Tenant-ID` (Zeile 4)
- `X-Tenant-Schema` (Zeile 5)
- `X-Case-ID` (Zeile 6)
- `X-Trace-ID` (Zeile 7)
- `X-Collection-ID` (Zeile 8)
- `X-Key-Alias` (Zeile 9)
- `Idempotency-Key` (Zeile 10)
- `X-Retry-Attempt` (Zeile 11)
- `X-Workflow-ID` (Zeile 12)

**Fehlende Header-Konstanten**:
❌ **DIVERGENZ #1**: `X-Invocation-ID`, `X-Run-ID`, `X-Ingestion-Run-ID` **fehlen** in `common/constants.py`, werden aber verwendet in:
  - `ai_core/ids/http_scope.py:132-133` (X-Invocation-ID)
  - `ai_core/ids/http_scope.py:139` (X-Run-ID)
  - `ai_core/ids/http_scope.py:142` (X-Ingestion-Run-ID)
  - `ai_core/graph/schemas.py:124, 129` (beide)

**Nicht in AGENTS.md erwähnt**:
⚠️ **DIVERGENZ #2**: `X-Retry-Attempt` ist in `common/constants.py` definiert, aber **nicht in AGENTS.md#47-51** gelistet.

### 1.2 ScopeContext (ai_core/contracts/scope.py)

**Pflichtfelder**:
- `tenant_id: TenantId` (str)
- `trace_id: TraceId` (str)
- `invocation_id: InvocationId` (str)

**XOR-Regel (validator, Zeile 66-77)**:
- Genau eines von: `run_id` XOR `ingestion_run_id` (Pflicht!)

**Optionale Felder**:
- `case_id: CaseId | None` (str)
- `tenant_schema: TenantSchema | None` (str)
- `workflow_id: WorkflowId | None` (str)
- `idempotency_key: IdempotencyKey | None` (str)
- `collection_id: CollectionId` (UUID | None) ← **Typ beachten: UUID!**
- `timestamp: Timestamp` (auto-generiert)

**Frozen**: `ConfigDict(frozen=True)` (Zeile 63)

### 1.3 ToolContext (ai_core/tool_contracts/base.py)

**Pflichtfelder**:
- `tenant_id: Union[UUID, str]`
- `trace_id: str` (default_factory: "trace-test")
- `invocation_id: UUID` (default_factory: uuid4)

**XOR-Regel (validator, Zeile 81-86)**:
- Genau eines von: `run_id` XOR `ingestion_run_id` (Pflicht!)

**Optionale Felder**:
- `case_id: Optional[str]` (Zeile 67)
- `workflow_id: Optional[str]` (Zeile 63)
- `collection_id: Optional[str]` (Zeile 64) ← **Typ beachten: str!**
- `document_id: Optional[str]` (Zeile 65)
- `document_version_id: Optional[str]` (Zeile 66)
- `idempotency_key: Optional[str]` (Zeile 50)
- `tenant_schema: Optional[str]` (Zeile 51)
- + weitere (timeouts_ms, budget_tokens, locale, safety_mode, auth, metadata)

**Frozen**: `ConfigDict(frozen=True)` (Zeile 31)

❌ **DIVERGENZ #3**: `collection_id` Typ-Inkonsistenz:
  - **ScopeContext**: `CollectionId = UUID | None` (scope.py:37)
  - **ToolContext**: `Optional[str]` (base.py:64)

❌ **DIVERGENZ #4**: `document_id`, `document_version_id` existieren in ToolContext, aber **nicht in ScopeContext**. Werden in `tool_context_from_scope` nicht kopiert (base.py:89-117).

---

## 2. HTTP → ScopeContext Normalisierung

### 2.1 Duplikation der Normalisierungslogik

❌ **DIVERGENZ #5**: **Zwei parallel existierende Normalizer** mit ähnlicher, aber nicht identischer Logik:

**Normalizer 1**: `ai_core/ids/http_scope.py:normalize_request(HttpRequest) -> ScopeContext`
  - Verwendet `TenantContext.from_request()` (Zeile 93)
  - Liest X-Invocation-ID, X-Run-ID, X-Ingestion-Run-ID aus Headers (Zeilen 130-143)
  - Auto-generiert `run_id = uuid4().hex` falls beide fehlen (Zeile 150)
  - Liest X-Collection-ID als UUID (Zeile 153-155)

**Normalizer 2**: `ai_core/graph/schemas.py:_build_scope_context(request) -> ScopeContext`
  - Fallback für non-HttpRequest (Zeile 95-100)
  - Verwendet `_coalesce()` für Header-Extraktion (Zeile 104-108)
  - Liest X-Invocation-ID, X-Run-ID, X-Ingestion-Run-ID (Zeilen 113-131)
  - Auto-generiert `run_id = uuid4().hex` falls beide fehlen (Zeile 141)
  - **Collection-ID wird NICHT gelesen!** (fehlt komplett)

**Problem**: Code-Duplikation, unterschiedliches Verhalten (collection_id), unterschiedliche Tenant-Resolution.

### 2.2 Tenant-Resolution

**TenantContext.from_request()** (http_scope.py:93):
  - Priorisiert Domain/URL/Middleware, **nicht** direkt aus Header (Zeile 89-107)
  - Falls kein tenant_id → `TenantRequiredError` (Zeile 107)

**_build_scope_context()** (graph/schemas.py:104):
  - Liest direkt aus Header `X-Tenant-ID` via `_coalesce()` (Zeile 104)
  - Wirft `ValueError("missing required meta keys: tenant_id")` (Zeile 134)

⚠️ **DIVERGENZ #6**: Unterschiedliche Error-Typen (`TenantRequiredError` vs. `ValueError`).

---

## 3. Graph-Meta-Normalisierung

### 3.1 normalize_meta (ai_core/graph/schemas.py:159-215)

**case_id ist PFLICHT** (Zeile 164-165):
```python
if not scope.case_id:
    raise ValueError("Case header is required and must use the documented format.")
```

❌ **DIVERGENZ #7**: **case_id ist Optional in ScopeContext, aber Pflicht in normalize_meta!**
  - ScopeContext: `case_id: CaseId | None` (scope.py:53)
  - normalize_meta: `if not scope.case_id: raise ValueError(...)` (schemas.py:164)
  - **Widerspruch**: Contract sagt Optional, Implementierung erzwingt Pflicht.

**Propagation**:
- Serialisiert `scope_context` in `meta["scope_context"]` (Zeile 182)
- Baut `tool_context` via `scope.to_tool_context(metadata=context_metadata)` (Zeile 211)
- Serialisiert `tool_context` in `meta["tool_context"]` (Zeile 213)

---

## 4. ID-Generierung vs. Header-Lesung

### 4.1 run_id

**Header-Lesung** (http_scope.py:138-150):
```python
run_id = _normalize_header_value(
    headers.get("X-Run-ID") or meta.get("HTTP_X_RUN_ID")
)
# ...
if not ingestion_run_id and not run_id:
    run_id = uuid4().hex  # Auto-generate
```

**Graph-Ausführung** (ai_core/services/__init__.py:793):
```python
run_id = uuid4().hex  # Immer neu generiert!
```

❌ **DIVERGENZ #8**: `run_id` aus Header wird **gelesen**, aber in `execute_graph` **ignoriert und neu generiert**!
  - Header-Wert wird verworfen
  - Graph bekommt immer einen neuen, unabhängigen `run_id`

### 4.2 ingestion_run_id

**Header-Lesung** (http_scope.py:141-143):
```python
ingestion_run_id = _normalize_header_value(
    headers.get("X-Ingestion-Run-ID") or meta.get("HTTP_X_INGESTION_RUN_ID")
)
```

**Crawler-Runner** (ai_core/services/crawler_runner.py:243):
```python
canonical_ingestion_run_id = str(uuid4())  # Lokal generiert!
```

**Upload-Handler** (ai_core/services/__init__.py:1725):
```python
ingestion_run_id = uuid4().hex  # Lokal generiert!
```

❌ **DIVERGENZ #9**: `ingestion_run_id` aus Header wird **gelesen**, aber **nicht verwendet**. Koordinatoren generieren eigene IDs.

### 4.3 invocation_id

**Header-Lesung** (http_scope.py:130-135):
```python
invocation_id = (
    _normalize_header_value(
        headers.get("X-Invocation-ID") or meta.get("HTTP_X_INVOCATION_ID")
    )
    or uuid4().hex  # Fallback
)
```

✅ **KORREKT**: Wenn Header fehlt, wird generiert. Wenn vorhanden, wird verwendet.

---

## 5. DB-Persistenz (welche IDs werden gespeichert?)

### 5.1 Django-ORM-Modelle

#### documents/models.py - Document (Zeile 71-143)

**Persistiert**:
- `id: UUIDField` (PK, Zeile 78)
- `tenant: ForeignKey(Tenant)` (Zeile 79)
- `workflow_id: CharField(max_length=255, db_index=True)` (Zeile 90-97)
- `trace_id: CharField(max_length=255)` (Zeile 98-104)
- `case_id: CharField(max_length=255, db_index=True)` (Zeile 105-111)
- `hash, source, external_id, metadata, lifecycle_state, created_at, updated_at, soft_deleted_at`

**NICHT persistiert**:
- `tenant_id` (nur via FK zu Tenant)
- `run_id`, `ingestion_run_id`, `invocation_id`
- `collection_id` (nur in M2M via DocumentCollectionMembership)

**Indexes**:
- `(tenant, workflow_id)` (Zeile 141)
- `(tenant, case_id)` (Zeile 140)

#### documents/models.py - DocumentIngestionRun (Zeile 176-216)

**Persistiert**:
- `tenant_id: CharField` (Zeile 179) ← **String!**
- `case: CharField` (Zeile 180)
- `collection_id: CharField` (Zeile 181)
- `run_id: CharField` (Zeile 182) ← **ACHTUNG: heißt `run_id`, nicht `ingestion_run_id`!**
- `trace_id: CharField` (Zeile 194)
- `status, queued_at, started_at, finished_at, duration_ms, inserted_documents, replaced_documents, skipped_documents, inserted_chunks, invalid_document_ids, document_ids, embedding_profile, source, error`

**Indexes**:
- `(tenant_id, case)` unique constraint (Zeile 204)
- `(tenant_id, run_id)` (Zeile 213)

❌ **DIVERGENZ #10**: **Feldname-Inkonsistenz**:
  - DB-Feld heißt `run_id` (Zeile 182)
  - Laut Contract sollte es `ingestion_run_id` heißen (für Ingestion-spezifische Runs)
  - Code verwendet `ingestion_run_id` (services/__init__.py:1260, crawler_runner.py:243)
  - **Mapping-Problem zwischen Contract und Schema!**

#### documents/models.py - DocumentCollection (Zeile 13-69)

**Persistiert**:
- `id: UUIDField` (PK, Zeile 16)
- `tenant: ForeignKey(Tenant)` (Zeile 17)
- `case: ForeignKey(Case, null=True)` (Zeile 22)
- `collection_id: UUIDField` (Zeile 31) ← **UUID!**
- `name, key, type, visibility, metadata, embedding_profile, created_at, updated_at, soft_deleted_at`

**Unique Constraints**:
- `(tenant, key)` (Zeile 48)
- `(tenant, collection_id)` (Zeile 52)

#### cases/models.py - CaseEvent (Zeile 58-97)

**Persistiert**:
- `case: ForeignKey(Case)` (Zeile 61)
- `tenant: ForeignKey(Tenant)` (Zeile 66)
- `ingestion_run: ForeignKey(DocumentIngestionRun, null=True)` (Zeile 74) ← **FK, nicht String!**
- `workflow_id: CharField` (Zeile 81)
- `collection_id: CharField` (Zeile 82)
- `trace_id: CharField` (Zeile 83)
- `event_type, source, graph_name, payload, created_at`

**NICHT persistiert**:
- `run_id`, `ingestion_run_id` (nur via FK zu DocumentIngestionRun)
- `invocation_id`

### 5.2 Vector-DB-Schema (docs/rag/schema.sql)

#### {{SCHEMA_NAME}}.documents (Zeile 90-163)

**Persistiert**:
- `id: UUID` (PK, Zeile 91)
- `tenant_id: UUID NOT NULL` (Zeile 92)
- `collection_id: UUID` (Zeile 93, nullable)
- `workflow_id: TEXT` (Zeile 94, nullable)
- `source: TEXT NOT NULL` (Zeile 95)
- `hash: TEXT NOT NULL` (Zeile 97)
- `metadata: JSONB DEFAULT '{}'` (Zeile 98)
- `lifecycle: TEXT DEFAULT 'active'` (Zeile 99)
- `external_id: TEXT` (Zeile 120)
- `created_at, deleted_at`

**NICHT persistiert** (als eigene Felder):
- `trace_id`, `case_id`, `run_id`, `ingestion_run_id`, `invocation_id`

**Indexes**:
- `(tenant_id, workflow_id)` (Zeile 142)
- `(tenant_id, source, hash)` unique WHERE workflow_id IS NULL (Zeile 145)
- `(tenant_id, workflow_id, source, hash)` unique WHERE workflow_id IS NOT NULL (Zeile 149)

#### {{SCHEMA_NAME}}.chunks (Zeile 165-214)

**Persistiert**:
- `id: UUID` (PK, Zeile 166)
- `document_id: UUID NOT NULL` (FK, Zeile 167)
- `tenant_id: UUID` (Zeile 172)
- `collection_id: UUID` (Zeile 173)
- `ord: INTEGER NOT NULL` (Zeile 168)
- `text: TEXT NOT NULL` (Zeile 169)
- `tokens: INTEGER NOT NULL` (Zeile 170)
- `metadata: JSONB DEFAULT '{}'` (Zeile 171)
- `text_norm: TEXT GENERATED ALWAYS AS (...)` (Zeile 178)

**Index auf metadata-JSON** (Zeile 192):
```sql
CREATE INDEX IF NOT EXISTS chunks_metadata_case_idx
    ON {{SCHEMA_NAME}}.chunks ((metadata->>'case'));
```

⚠️ **DIVERGENZ #11**: `case_id` wird **im JSON gespeichert** (`metadata->>'case'`), **nicht als eigenes Feld**!
  - Index existiert, aber kein Schema-Feld
  - Inkonsistenz zu Django-ORM, wo `Document.case_id` ein eigenes Feld ist

#### {{SCHEMA_NAME}}.embeddings (Zeile 215-276)

**Persistiert**:
- `id: UUID` (PK, Zeile 216)
- `chunk_id: UUID NOT NULL` (FK, Zeile 217)
- `embedding: vector({{VECTOR_DIM}}) NOT NULL` (Zeile 218)
- `tenant_id: UUID` (Zeile 219)
- `collection_id: UUID` (Zeile 220)

**NICHT persistiert**:
- `trace_id`, `case_id`, `workflow_id`, `run_id`, `ingestion_run_id`, `invocation_id`

---

## 6. Logging & Tracing

### 6.1 Logging-Context-Binding (common/middleware.py)

**RequestLogContextMiddleware** (Zeile 59-135):

**Bindet an Logging-Context** (Zeile 62-67):
- `trace_id` (via `META_TRACE_ID_KEY`)
- `case_id` (via `META_CASE_ID_KEY`)
- `tenant_id` (via `META_TENANT_ID_KEY`)
- `key_alias` (via `META_KEY_ALIAS_KEY`)

❌ **DIVERGENZ #12**: **Folgende IDs werden NICHT an Logging-Context gebunden**:
- `workflow_id`
- `run_id`
- `ingestion_run_id`
- `invocation_id`
- `collection_id`

**Auswirkung**: Diese IDs erscheinen nicht automatisch in strukturierten Logs (außer manuell hinzugefügt).

### 6.2 Langfuse-Tags (ai_core/services/__init__.py:821-828)

**Tags**:
```python
tags=[
    "graph",
    f"graph:{context.graph_name}",
    f"version:{context.graph_version}",
]
```

**Metadata** (Zeile 828):
```python
metadata={
    "trace_id": context.trace_id,
    "tenant.id": context.tenant_id,
    "case.id": context.case_id,
    "graph.version": context.graph_version,
    "workflow.id": context.workflow_id,
    "run.id": context.run_id,
}
```

✅ **KORREKT**: Alle wichtigen IDs werden in Langfuse-Metadata persistiert.

---

## 7. Idempotency-Handling

### 7.1 Crawler-Runner (ai_core/services/crawler_runner.py:131-204)

**Cache-Prefix**: `"crawler_idempotency:"` (Zeile 168)
**TTL**: `3600` (1 Stunde, Zeile 169)

**Cache-Key-Bildung** (Zeilen 171-183):
1. **Wenn `idempotency_key` vorhanden**: `f"{CACHE_PREFIX}key:{idempotency_key}"`
2. **Sonst, wenn `fingerprint` berechnet**: `f"{CACHE_PREFIX}fp:{fingerprint}"`

**Fingerprint-Payload** (Zeilen 145-158):
```python
fingerprint_payload = {
    "tenant_id": str(tenant_id_for_fp),  # Pflicht!
    "case_id": meta["scope_context"].get("case_id"),  # Optional
    "workflow_id": str(workflow_resolved),
    "collection_id": request_model.collection_id,
    "mode": request_model.mode,
    "origins": sorted([...], key=lambda o: o.get("uri", "")),
}
fingerprint = hashlib.sha256(
    json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")
).hexdigest()
```

**Response bei Idempotenz** (Zeilen 195-204):
```python
return CrawlerRunnerCoordinatorResult(
    payload={
        "idempotent": True,
        "skipped": True,
        "origins": [],
        "message": "Request already processed (idempotent)",
    },
    status_code=status.HTTP_200_OK,
    idempotency_key=idempotency_key,
)
```

⚠️ **DIVERGENZ #13**: **Kein zentraler Idempotency-Handler**!
  - Crawler hat eigene Implementierung
  - Andere Flows haben keine Idempotency-Checks (Upload, RAG Ingestion Run)
  - Keine einheitliche Cache-Strategie

---

## 8. End-to-End-Flow-Analyse

### 8.1 Upload-Flow

**Einstieg**: `ai_core/services/__init__.py:handle_document_upload()`

**ID-Propagation**:
1. `meta` enthält: `scope_context` (tenant/case/trace/etc.), plus `tenant_schema` (via normalize_meta)
2. `ingestion_run_id = uuid4().hex` generiert (Zeile 1725)
3. **NormalizedDocument** erstellt mit `DocumentRef`:
   - `tenant_id: str` (Zeile 1696)
   - `workflow_id: str` (Zeile 1697)
   - `document_id: UUID` (Zeile 1698)
   - `collection_id: str | UUID | None` (Zeile 1699-1703)
4. **UniversalIngestionGraph** invoked mit `context`:
   - `tenant_id, case_id, trace_id, workflow_id, ingestion_run_id` (Zeilen 1748-1751)
5. **Persist-Knoten** speichert in:
   - Django ORM: `Document` (workflow_id, trace_id, case_id)
   - Vector DB: `documents` (tenant_id, workflow_id, collection_id)

**Lücken**:
- `run_id` wird **nicht** propagiert (nur `ingestion_run_id`)
- `invocation_id` aus Request wird **nicht** an Graph weitergegeben

### 8.2 Crawler-Flow

**Einstieg**: `ai_core/services/crawler_runner.py:run_crawler_runner()`

**ID-Propagation**:
1. `meta` enthält: `tenant_id, trace_id, case_id, collection_id` (Zeile 79)
2. `canonical_ingestion_run_id = str(uuid4())` generiert (Zeile 243)
3. **Context** für Graph:
   - `tenant_id, case_id, trace_id, workflow_id, ingestion_run_id` (Zeilen 247-252)
4. **UniversalIngestionGraph** invoked
5. **Output-Validierung** (Zeilen 321-335):
   - Falls `output_run_id != canonical_ingestion_run_id`: **Warning geloggt**
   - Coordinator ist "source of truth" (Zeile 334)

**Lücken**:
- `invocation_id` wird **nicht** aus Request gelesen oder generiert
- `run_id` wird **nicht** verwendet (nur `ingestion_run_id`)

### 8.3 Graph-Execution-Flow

**Einstieg**: `ai_core/services/__init__.py:execute_graph()`

**ID-Propagation**:
1. `normalized_meta = _normalize_meta(request)` (Zeile 770)
2. `run_id = uuid4().hex` **neu generiert** (Zeile 793) ← **Nicht aus Request!**
3. `GraphContext` erstellt (Zeilen 796-804):
   - `tenant_id, case_id, trace_id, workflow_id, run_id, graph_name, graph_version`
4. `ToolContext` aus `normalized_meta["tool_context"]` gelesen (Zeilen 779-791)
5. Graph invoked, Result returned

**Lücken**:
- `run_id` aus Header wird **ignoriert** (immer neu generiert)
- `invocation_id` wird **nicht** in GraphContext propagiert

---

## 9. Typ-Inkonsistenzen (Zusammenfassung)

| Identifier       | ScopeContext        | ToolContext         | Django ORM          | Vector DB           | Inkonsistenz? |
|------------------|---------------------|---------------------|---------------------|---------------------|---------------|
| `tenant_id`      | `str`               | `UUID \| str`       | `FK(Tenant)`        | `UUID`              | ✅ OK (via Tenant) |
| `trace_id`       | `str`               | `str`               | `CharField`         | -                   | ✅ OK |
| `invocation_id`  | `str`               | `UUID`              | -                   | -                   | ⚠️ str vs UUID |
| `run_id`         | `str \| None`       | `str \| None`       | -                   | -                   | ✅ OK |
| `ingestion_run_id` | `str \| None`     | `str \| None`       | `CharField` (als `run_id`!) | -              | ❌ Feldname! |
| `case_id`        | `str \| None`       | `str \| None`       | `CharField`         | - (im JSON)         | ⚠️ Optional vs Pflicht |
| `workflow_id`    | `str \| None`       | `str \| None`       | `CharField`         | `TEXT`              | ✅ OK |
| `collection_id`  | `UUID \| None`      | `str \| None`       | `UUIDField`         | `UUID`              | ❌ **UUID vs str!** |
| `tenant_schema`  | `str \| None`       | `str \| None`       | - (via schema_name) | -                   | ✅ OK |
| `idempotency_key`| `str \| None`       | `str \| None`       | -                   | -                   | ✅ OK |

**Kritische Typen-Inkonsistenzen**:
1. ❌ **collection_id**: UUID in ScopeContext, str in ToolContext (DIVERGENZ #3)
2. ⚠️ **invocation_id**: str in ScopeContext, UUID in ToolContext (minor)
3. ❌ **ingestion_run_id**: Feld heißt `run_id` in DB (DIVERGENZ #10)

---

## 10. Zusammenfassung der Divergenzen

### Kritische Divergenzen (Funktionsbruch)

| # | Typ | Beschreibung | Datei:Zeile | Empfehlung |
|---|-----|--------------|-------------|------------|
| **#3** | Typ-Inkonsistenz | `collection_id` ist `UUID \| None` in ScopeContext, aber `str \| None` in ToolContext | scope.py:57, base.py:64 | **Harmonisieren**: Entweder beide `str` (empfohlen) oder beide `UUID`. UUID-Serialisierung in tool_context_from_scope hinzufügen. |
| **#7** | Contract-Divergenz | `case_id` ist Optional in ScopeContext, aber Pflicht in `normalize_meta` | scope.py:53, schemas.py:164 | **Entscheiden**: Entweder ScopeContext macht case_id Pflicht (breaking change), oder normalize_meta akzeptiert None (aber dann Graph-Flows brechen). Empfehlung: case_id Pflicht machen in ScopeContext. |
| **#8** | ID-Generierung | `run_id` aus Header wird gelesen, aber in `execute_graph` ignoriert und neu generiert | http_scope.py:138, services/__init__.py:793 | **Konsistenz**: Entweder Header respektieren (X-Run-ID verwenden) oder Header-Lesung entfernen. Empfehlung: Header verwenden, wenn vorhanden. |
| **#9** | ID-Generierung | `ingestion_run_id` aus Header wird gelesen, aber nicht verwendet (Koordinatoren generieren eigene IDs) | http_scope.py:141, crawler_runner.py:243, services/__init__.py:1725 | **Konsistenz**: Koordinatoren sollten Header-Wert respektieren, wenn vorhanden. Oder Header-Lesung entfernen. |
| **#10** | DB-Feldname | `DocumentIngestionRun.run_id` sollte `ingestion_run_id` heißen | documents/models.py:182 | **Migration**: Feld umbenennen zu `ingestion_run_id`. Oder Code anpassen, um `run_id` zu verwenden (aber Contract bricht). |

### Mittlere Divergenzen (Wartbarkeit/Konsistenz)

| # | Typ | Beschreibung | Datei:Zeile | Empfehlung |
|---|-----|--------------|-------------|------------|
| **#1** | Fehlende Konstanten | `X-Invocation-ID`, `X-Run-ID`, `X-Ingestion-Run-ID` fehlen in `common/constants.py` | constants.py, http_scope.py:132-142 | **Ergänzen**: Konstanten hinzufügen. |
| **#4** | Fehlende Felder | `document_id`, `document_version_id` in ToolContext, aber nicht in ScopeContext | base.py:65-66 | **Dokumentieren**: Klarstellen, dass diese Felder nur Tool-spezifisch sind, nicht Scope-Level. Oder in ScopeContext aufnehmen (breaking change). |
| **#5** | Code-Duplikation | Zwei Normalizer (http_scope.py, graph/schemas.py) mit unterschiedlichem Verhalten | http_scope.py:63, schemas.py:88 | **Konsolidieren**: Einen Normalizer behalten, anderen entfernen oder delegieren. |
| **#11** | Schema-Inkonsistenz | `case_id` im Vector-DB als JSON (`metadata->>'case'`), in Django-ORM als Feld | schema.sql:192, models.py:105 | **Entscheiden**: Entweder Vector-DB Feld hinzufügen (Migration) oder Django-ORM auch JSON nutzen. Empfehlung: Feld in Vector-DB hinzufügen für Konsistenz. |
| **#12** | Logging-Lücken | `workflow_id`, `run_id`, `ingestion_run_id`, `invocation_id`, `collection_id` nicht in Logging-Context | middleware.py:62-67 | **Erweitern**: Middleware um weitere IDs ergänzen. |
| **#13** | Idempotency-Fragmentierung | Kein zentraler Idempotency-Handler, nur Crawler hat eigene Implementierung | crawler_runner.py:131-204 | **Zentralisieren**: Einheitlichen Idempotency-Service erstellen, von allen Flows nutzen. |

### Minore Divergenzen (Dokumentation/Hinweise)

| # | Typ | Beschreibung | Datei:Zeile | Empfehlung |
|---|-----|--------------|-------------|------------|
| **#2** | Doku-Lücke | `X-Retry-Attempt` nicht in AGENTS.md erwähnt | constants.py:11, AGENTS.md:47-51 | **Dokumentieren**: In AGENTS.md ergänzen oder aus constants.py entfernen, falls nicht genutzt. |
| **#6** | Error-Typen | `TenantRequiredError` vs. `ValueError` bei fehlendem tenant_id | http_scope.py:107, schemas.py:134 | **Harmonisieren**: Einheitlichen Error-Typ nutzen. |

---

## 11. Empfehlungen (Priorisierung)

### Sofort (Breaking Changes, aber Pre-MVP erlaubt)

1. **collection_id Typ harmonisieren** (DIVERGENZ #3):
   - **Ziel**: `collection_id` überall als `str` (UUID-String)
   - **Änderungen**:
     - `ScopeContext.collection_id: str | None` (statt `UUID | None`)
     - Parsing in Normalizern: `str(uuid)` statt `UUID(str)`
     - Tests anpassen

2. **case_id Pflicht machen in ScopeContext** (DIVERGENZ #7):
   - **Ziel**: Contract und Implementierung angleichen
   - **Änderungen**:
     - `ScopeContext.case_id: CaseId` (Pflicht, kein `| None`)
     - Alle Normalizer müssen case_id validieren oder generieren
     - Tests anpassen (alle Requests brauchen case_id)

3. **DocumentIngestionRun.run_id umbenennen** (DIVERGENZ #10):
   - **Ziel**: Feldname = Semantik
   - **Änderungen**:
     - Migration: `run_id` → `ingestion_run_id`
     - Index umbenennen
     - Code anpassen (ingestion_status.py, services/__init__.py)

### Kurzfristig (Konsistenz-Verbesserungen)

4. **run_id & ingestion_run_id Header respektieren** (DIVERGENZ #8, #9):
   - **Ziel**: Header-Werte nutzen, falls vorhanden
   - **Änderungen**:
     - `execute_graph`: `run_id` aus `normalized_meta.scope_context` lesen
     - `crawler_runner`: `ingestion_run_id` aus `context` lesen
     - `handle_document_upload`: `ingestion_run_id` aus `meta` lesen

5. **Fehlende Header-Konstanten ergänzen** (DIVERGENZ #1):
   - **Änderungen**:
     - `common/constants.py`: `X_INVOCATION_ID_HEADER`, `X_RUN_ID_HEADER`, `X_INGESTION_RUN_ID_HEADER`
     - `META_INVOCATION_ID_KEY`, `META_RUN_ID_KEY`, `META_INGESTION_RUN_ID_KEY`

6. **Normalizer konsolidieren** (DIVERGENZ #5):
   - **Ziel**: Ein Normalizer für alle Fälle
   - **Änderungen**:
     - `graph/schemas.py:_build_scope_context` delegiert an `http_scope.py:normalize_request`
     - Oder: Beide nutzen gemeinsame Helper-Funktionen

### Mittelfristig (Architektur-Verbesserungen)

7. **Logging-Context erweitern** (DIVERGENZ #12):
   - **Änderungen**:
     - `RequestLogContextMiddleware`: Bind `workflow_id`, `run_id`, `ingestion_run_id`, `invocation_id`, `collection_id`

8. **Zentralen Idempotency-Service** (DIVERGENZ #13):
   - **Änderungen**:
     - Neues Modul: `ai_core/infra/idempotency.py`
     - Service mit Cache-Key-Bildung, TTL-Konfiguration, Fingerprinting
     - Alle Flows nutzen denselben Service

9. **Vector-DB: case_id als Feld** (DIVERGENZ #11):
   - **Änderungen**:
     - `docs/rag/schema.sql`: `chunks` Tabelle: `case_id TEXT` Spalte hinzufügen
     - Index: `(tenant_id, case_id)`
     - Migration für bestehende Daten: `metadata->>'case'` nach `case_id` kopieren

### Langfristig (Doku & Cleanup)

10. **AGENTS.md aktualisieren** (DIVERGENZ #2):
    - X-Retry-Attempt dokumentieren oder entfernen

11. **Tests für ID-Propagation**:
    - End-to-End-Tests für alle Flows (Upload, Crawler, Graph-Execution)
    - Assertions: Header → ScopeContext → ToolContext → DB → Response
    - Idempotency-Tests für alle Flows

---

## 12. Anhang: Code-Pointer (Vollständig)

### Contracts & Normalizer

- **ScopeContext**: `ai_core/contracts/scope.py:41-100`
- **ToolContext**: `ai_core/tool_contracts/base.py:28-87`
- **tool_context_from_scope**: `ai_core/tool_contracts/base.py:89-117`
- **normalize_request (HttpRequest)**: `ai_core/ids/http_scope.py:63-170`
- **_build_scope_context (generic)**: `ai_core/graph/schemas.py:88-156`
- **normalize_meta**: `ai_core/graph/schemas.py:159-215`
- **Header-Konstanten**: `common/constants.py:1-68`
- **Header-Aliase**: `ai_core/ids/headers.py:12-43`

### Services & Flows

- **execute_graph**: `ai_core/services/__init__.py:762-1213`
- **handle_document_upload**: `ai_core/services/__init__.py:1495-1902`
- **run_crawler_runner**: `ai_core/services/crawler_runner.py:77-359`
- **start_ingestion_run**: `ai_core/services/__init__.py:1216-1316`

### DB-Modelle

- **Tenant**: `customers/models.py:9-50`
- **Document**: `documents/models.py:71-143`
- **DocumentIngestionRun**: `documents/models.py:176-216`
- **DocumentCollection**: `documents/models.py:13-69`
- **Case**: `cases/models.py:10-55`
- **CaseEvent**: `cases/models.py:58-97`

### Vector-DB-Schema

- **schema.sql**: `docs/rag/schema.sql:1-296`
- **documents Tabelle**: `docs/rag/schema.sql:90-163`
- **chunks Tabelle**: `docs/rag/schema.sql:165-214`
- **embeddings Tabelle**: `docs/rag/schema.sql:215-276`

### Middleware & Logging

- **RequestLogContextMiddleware**: `common/middleware.py:59-135`
- **TenantSchemaMiddleware**: `common/middleware.py:16-26`
- **HeaderTenantRoutingMiddleware**: `common/middleware.py:29-56`

### Graphs

- **UniversalIngestionGraph**: `ai_core/graphs/technical/universal_ingestion_graph.py`
  - Input: Zeile 51-61
  - Context: Zeile 85
  - Normalize-Knoten: Zeile 98-156
  - Persist-Knoten: Zeile 440-530

---

**Ende der Inventur**
