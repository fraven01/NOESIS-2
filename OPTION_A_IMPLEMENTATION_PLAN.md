# Option A: Implementierungsplan

**Datum**: 2025-12-27
**Ziel**: 100% Vibe Coding Ready - Strikte Trennung von Scope/Business/ToolContext
**Methode**: Schrittweise Migration mit Tests nach jedem Schritt
**Pre-MVP**: Breaking Changes erlaubt

---

## 🎯 Goldene Regel (Operationalisiert)

> **Tool-Inputs enthalten nur funktionale Parameter.**
> **Context enthält Scope, Business und Runtime Permissions.**
> **Tool-Run-Funktionen lesen Identifiers und Permissions ausschließlich aus context, nicht aus params.**

---

## 📋 Phasen-Übersicht

| Phase | Beschreibung | Dateien | Breaking | Tests |
|-------|--------------|---------|----------|-------|
| **1** | **Contracts erstellen** | 4 | ✅ | Unit |
| **2** | **Tool-Inputs säubern** | ~6 | ✅ | Unit + Integration |
| **3** | **Normalizer anpassen** | ~3 | ✅ | Integration |
| **4** | **Dokumentation** | ~3 | ❌ | - |

**Geschätzter Aufwand**: 1-2 Tage fokussierte Arbeit

---

## Phase 1: Contracts erstellen (Foundation)

**Ziel**: Neue Contract-Struktur etablieren mit Backward Compatibility

### 1.1 BusinessContext erstellen ✅
**Status**: ✅ DONE

**Datei**: `ai_core/contracts/business.py`

**Inhalt**:
- `BusinessContext` mit `case_id`, `collection_id`, `workflow_id`, `document_id`, `document_version_id`
- Alle Felder optional
- Immutable (`frozen=True`)
- Type aliases exportieren

**Tests**:
```python
# ai_core/tests/contracts/test_business_context.py
def test_business_context_creation():
    ctx = BusinessContext(case_id="test", collection_id="col")
    assert ctx.case_id == "test"

def test_business_context_immutable():
    ctx = BusinessContext(case_id="test")
    with pytest.raises(ValidationError):
        ctx.case_id = "new"  # Should fail (frozen)

def test_business_context_all_optional():
    ctx = BusinessContext()  # Should work
    assert ctx.case_id is None
```

---

### 1.2 ScopeContext refactoren (Business-IDs raus)

**Datei**: `ai_core/contracts/scope.py`

**Änderungen**:
```python
# RAUS:
- case_id: CaseId | None = None
- collection_id: CollectionId | None = None
- workflow_id: WorkflowId | None = None

# BLEIBT:
+ tenant_id: TenantId
+ trace_id: TraceId
+ invocation_id: InvocationId
+ user_id: UserId | None
+ service_id: ServiceId | None
+ run_id: RunId | None
+ ingestion_run_id: IngestionRunId | None
+ tenant_schema: TenantSchema | None
+ idempotency_key: IdempotencyKey | None
+ timestamp: Timestamp
```

**Tests**:
```python
# ai_core/tests/contracts/test_scope_context.py
def test_scope_context_no_business_ids():
    # Sollte NICHT mehr möglich sein:
    with pytest.raises(ValidationError):
        ScopeContext(
            tenant_id="t", trace_id="tr", invocation_id="inv",
            run_id="r", case_id="case"  # ← Field doesn't exist
        )

def test_scope_context_minimal():
    ctx = ScopeContext(
        tenant_id="t", trace_id="tr", invocation_id="inv", run_id="r"
    )
    assert ctx.tenant_id == "t"
```

**Breaking Change**: ✅ Bestätigt
- Code, der `ScopeContext(case_id=...)` erstellt, bricht
- Migration: BusinessContext separat erstellen

---

### 1.3 ToolContext mit Komposition umbauen

**Datei**: `ai_core/tool_contracts/base.py`

**Neue Struktur**:
```python
class ToolContext(BaseModel):
    """Complete tool invocation context with separated concerns."""

    # Komposition (NEU)
    scope: ScopeContext
    business: BusinessContext

    # Runtime Metadata
    locale: str | None = None
    timeouts_ms: PositiveInt | None = None
    budget_tokens: int | None = None
    safety_mode: str | None = None
    auth: dict[str, Any] | None = None
    visibility_override_allowed: bool = False  # ← Aus RetrieveInput hierher!
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)

    # === Backward Compatibility Properties (DEPRECATED) ===
    @property
    def tenant_id(self) -> str:
        """Deprecated: Use context.scope.tenant_id instead."""
        return self.scope.tenant_id

    @property
    def case_id(self) -> str | None:
        """Deprecated: Use context.business.case_id instead."""
        return self.business.case_id

    # ... alle anderen Properties
```

**Tests**:
```python
# ai_core/tests/test_tool_context.py
def test_tool_context_composition():
    scope = ScopeContext(tenant_id="t", trace_id="tr", invocation_id="i", run_id="r")
    business = BusinessContext(case_id="c", collection_id="col")

    ctx = ToolContext(scope=scope, business=business)

    assert ctx.scope.tenant_id == "t"
    assert ctx.business.case_id == "c"

def test_tool_context_backward_compat_properties():
    scope = ScopeContext(tenant_id="t", trace_id="tr", invocation_id="i", run_id="r")
    business = BusinessContext(case_id="c")
    ctx = ToolContext(scope=scope, business=business)

    # Properties sollten funktionieren:
    assert ctx.tenant_id == "t"  # via property
    assert ctx.case_id == "c"    # via property

def test_tool_context_visibility_override():
    scope = ScopeContext(tenant_id="t", trace_id="tr", invocation_id="i", run_id="r")
    business = BusinessContext()
    ctx = ToolContext(scope=scope, business=business, visibility_override_allowed=True)

    assert ctx.visibility_override_allowed is True
```

**Breaking Change**: ✅ Bestätigt
- Direkter Constructor-Call `ToolContext(tenant_id=..., case_id=...)` bricht
- Migration: `ToolContext(scope=..., business=...)`
- Properties halten alten Code am Laufen

---

### 1.4 tool_context_from_scope erweitern

**Datei**: `ai_core/tool_contracts/base.py`

**Neue Signatur**:
```python
def tool_context_from_scope(
    scope: ScopeContext,
    business: BusinessContext | None = None,
    *,
    now: datetime | None = None,
    **overrides: Any,
) -> ToolContext:
    """Build ToolContext from ScopeContext and BusinessContext."""

    if business is None:
        business = BusinessContext()  # Empty

    payload: dict[str, Any] = {
        "scope": scope,
        "business": business,
    }

    payload.update(overrides)

    return ToolContext(**payload)
```

**ScopeContext.to_tool_context erweitern**:
```python
# In ScopeContext:
def to_tool_context(
    self,
    business: BusinessContext | None = None,
    **overrides: object
) -> ToolContext:
    """Project this scope into a ToolContext with optional business context."""
    from ai_core.tool_contracts.base import tool_context_from_scope
    return tool_context_from_scope(self, business, **overrides)
```

**Tests**:
```python
def test_tool_context_from_scope():
    scope = ScopeContext(tenant_id="t", trace_id="tr", invocation_id="i", run_id="r")
    business = BusinessContext(case_id="c")

    ctx = tool_context_from_scope(scope, business, locale="de-DE")

    assert ctx.scope.tenant_id == "t"
    assert ctx.business.case_id == "c"
    assert ctx.locale == "de-DE"

def test_scope_to_tool_context():
    scope = ScopeContext(tenant_id="t", trace_id="tr", invocation_id="i", run_id="r")
    business = BusinessContext(case_id="c")

    ctx = scope.to_tool_context(business=business, budget_tokens=512)

    assert ctx.scope.tenant_id == "t"
    assert ctx.business.case_id == "c"
    assert ctx.budget_tokens == 512
```

**Breaking Change**: ⚠️ Minor
- Alte Calls ohne `business` funktionieren (leerer BusinessContext)
- Aber: Code sollte Business-Context explizit übergeben

---

### Phase 1 Checkpoint

**Deliverables**:
- ✅ `ai_core/contracts/business.py` (NEU)
- ✅ `ai_core/contracts/scope.py` (GEÄNDERT)
- ✅ `ai_core/tool_contracts/base.py` (GEÄNDERT)
- ✅ Unit-Tests für alle drei

**Commit Message**:
```
feat(contracts): strict separation - introduce BusinessContext

BREAKING CHANGE: ScopeContext no longer contains business domain IDs
(case_id, collection_id, workflow_id). These are now in BusinessContext.

ToolContext now uses composition (scope + business) instead of flat structure.
Backward compatibility via @property accessors (deprecated, will be removed).

Related: OPTION_A_IMPLEMENTATION_PLAN.md, OPTION_A_SOURCE_CODE_ANALYSIS.md

🤖 Generated with Claude Code
Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
```

---

## Phase 2: Tool-Inputs säubern

**Ziel**: Alle Context-IDs aus Tool-Input-Modellen entfernen

### 2.1 RetrieveInput säubern + retrieve.py:run() migrieren

**Dateien**:
- `ai_core/nodes/retrieve.py` (RetrieveInput)
- `ai_core/nodes/retrieve.py` (run function)

**Änderungen**:
```python
# VORHER:
class RetrieveInput(BaseModel):
    query: str = ""
    collection_id: str | None = None          # ❌ RAUS
    workflow_id: str | None = None            # ❌ RAUS
    visibility_override_allowed: bool | None = None  # ❌ RAUS (jetzt in ToolContext)
    # ...

# NACHHER:
class RetrieveInput(BaseModel):
    """Pure functional parameters for retrieval.

    BREAKING CHANGE (Option A):
    Removed context IDs (collection_id, workflow_id, visibility_override_allowed).
    These are now in ToolContext.business and ToolContext respectively.
    """
    query: str = ""
    filters: Mapping[str, Any] | None = None
    process: str | None = None
    doc_class: str | None = None
    visibility: str | None = None
    hybrid: Mapping[str, Any] | None = None
    top_k: int | None = None
```

**run() Funktion migrieren**:
```python
# VORHER:
def run(context: ToolContext, params: RetrieveInput) -> RetrieveOutput:
    tenant_id = str(context.tenant_id)  # via property (deprecated)
    collection_id = params.collection_id  # ❌ RED FLAG!

    # Fallback-Logik:
    override_flag = params.visibility_override_allowed
    if override_flag is None:
        override_flag = context.visibility_override_allowed

# NACHHER:
def run(context: ToolContext, params: RetrieveInput) -> RetrieveOutput:
    # Alle IDs aus context (neue Struktur):
    tenant_id = context.scope.tenant_id
    tenant_schema = context.scope.tenant_schema
    case_id = context.business.case_id
    collection_id = context.business.collection_id  # ✅
    workflow_id = context.business.workflow_id      # ✅

    # Permissions aus context (kein Fallback):
    visibility_override_allowed = context.visibility_override_allowed
```

**Tests**:
```python
# ai_core/tests/nodes/test_retrieve.py
def test_retrieve_input_no_context_ids():
    # Sollte NICHT mehr möglich sein:
    with pytest.raises(ValidationError):
        RetrieveInput(query="test", collection_id="col")  # ← Field removed

def test_retrieve_run_with_new_context():
    scope = ScopeContext(tenant_id="t", trace_id="tr", invocation_id="i", run_id="r")
    business = BusinessContext(case_id="c", collection_id="col", workflow_id="wf")
    context = ToolContext(scope=scope, business=business, visibility_override_allowed=True)

    params = RetrieveInput(query="test", filters={}, hybrid={...})

    output = run(context, params)

    # Verify it uses context, not params
```

**Breaking Change**: ✅ Bestätigt
- RetrieveInput-Konstruktion mit `collection_id` bricht
- retrieve.py:run() nutzt neue Context-Struktur

---

### 2.2 FrameworkAnalysisInput säubern + Graph migrieren

**Dateien**:
- `ai_core/tools/framework_contracts.py`
- `ai_core/graphs/business/framework_analysis_graph.py`

**Änderungen**:
```python
# VORHER:
class FrameworkAnalysisInput(BaseModel):
    document_collection_id: UUID  # ❌ RAUS
    document_id: UUID | None = None  # ❌ RAUS
    force_reanalysis: bool = False
    confidence_threshold: float = 0.70

# NACHHER:
class FrameworkAnalysisInput(BaseModel):
    """Framework analysis functional parameters.

    BREAKING CHANGE (Option A):
    Removed document_collection_id, document_id.
    These are now in ToolContext.business.
    """
    force_reanalysis: bool = False
    confidence_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
```

**Graph migrieren**:
```python
# VORHER (in Graph):
document_id = input_params.document_id  # ❌

# NACHHER:
document_id = context.business.document_id  # ✅
collection_id = context.business.collection_id  # ✅
```

**Tests**:
```python
def test_framework_analysis_input_no_context_ids():
    with pytest.raises(ValidationError):
        FrameworkAnalysisInput(document_collection_id=uuid.uuid4())

def test_framework_analysis_graph_with_new_context():
    scope = ScopeContext(...)
    business = BusinessContext(document_id="doc-1", collection_id="col-1")
    context = ToolContext(scope=scope, business=business)

    params = FrameworkAnalysisInput(force_reanalysis=True)

    # Graph sollte context.business nutzen
```

---

### 2.3 WebSearchContext löschen + WebSearch migrieren

**Dateien**:
- `ai_core/tools/web_search.py` (WebSearchContext LÖSCHEN)
- Alle WebSearch-Caller (umstellen auf ToolContext)

**Änderungen**:
```python
# LÖSCHEN:
class WebSearchContext(BaseModel):
    tenant_id: str
    trace_id: str
    workflow_id: str
    case_id: str | None
    run_id: str
    worker_call_id: str | None = None

# ERSETZEN durch ToolContext:
def web_search_function(context: ToolContext, params: WebSearchInput):
    # Alle IDs aus context:
    tenant_id = context.scope.tenant_id
    workflow_id = context.business.workflow_id
    case_id = context.business.case_id

    # worker_call_id → metadata:
    worker_call_id = context.metadata.get("worker_call_id")
```

**Tests**:
```python
def test_web_search_uses_tool_context():
    scope = ScopeContext(...)
    business = BusinessContext(workflow_id="wf", case_id="c")
    context = ToolContext(
        scope=scope,
        business=business,
        metadata={"worker_call_id": "worker-123"}
    )

    params = WebSearchInput(query="test")

    # Web-Search sollte ToolContext nutzen
```

**Breaking Change**: ✅ Bestätigt
- WebSearchContext existiert nicht mehr
- Alle Caller müssen ToolContext verwenden

---

### Phase 2 Checkpoint

**Deliverables**:
- ✅ `RetrieveInput` gesäubert
- ✅ `FrameworkAnalysisInput` gesäubert
- ✅ `WebSearchContext` gelöscht
- ✅ Alle Tool-Run-Funktionen migriert
- ✅ Integration-Tests passen

**Commit Message**:
```
feat(tools): remove context IDs from tool inputs

BREAKING CHANGE: Tool-Input models no longer contain context identifiers.
- RetrieveInput: removed collection_id, workflow_id, visibility_override_allowed
- FrameworkAnalysisInput: removed document_collection_id, document_id
- WebSearchContext: DELETED, replaced with ToolContext

Golden Rule enforced: Tool-Inputs contain only functional parameters.
Context IDs and permissions are exclusively in ToolContext.

Related: OPTION_A_SOURCE_CODE_ANALYSIS.md #5 (Architecture Decisions)

🤖 Generated with Claude Code
Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
```

---

## Phase 3: Normalizer anpassen

**Ziel**: HTTP/Graph Normalizer für neue Context-Struktur anpassen

### 3.1 normalize_meta() erweitern

**Datei**: `ai_core/graph/schemas.py`

**Änderungen** (Option 1: Graph-spezifische Validierung):
```python
def normalize_meta(request: Any) -> dict:
    scope = _build_scope_context(request)

    # BusinessContext aus Request extrahieren (ALLES OPTIONAL!)
    case_id = request.headers.get(X_CASE_ID_HEADER)
    collection_id = request.headers.get(X_COLLECTION_ID_HEADER)
    workflow_id = request.headers.get(X_WORKFLOW_ID_HEADER)
    document_id = request.headers.get(X_DOCUMENT_ID_HEADER)

    # BusinessContext erstellen (keine Validierung hier!)
    business = BusinessContext(
        case_id=case_id,
        collection_id=collection_id,
        workflow_id=workflow_id,
        document_id=document_id,
    )

    # KEIN ValueError mehr! ✅
    # Graphs validieren selbst, was sie brauchen

    # ToolContext mit optionalem Business Context
    tool_context = scope.to_tool_context(business=business, metadata=context_metadata)

    meta = {
        "graph_name": graph_name,
        "scope_context": scope.model_dump(exclude_none=True),
        "business_context": business.model_dump(exclude_none=True),  # NEU, optional
        "tool_context": tool_context.model_dump(exclude_none=True),
    }

    return meta
```

**Graph-spezifische Validierung** (Beispiel):
```python
# ai_core/graphs/business/framework_analysis_graph.py
def run(self, state, meta):
    # Graph validiert selbst, was er braucht:
    business = BusinessContext.model_validate(meta["business_context"])

    if not business.case_id:
        raise ValueError(
            "Framework Analysis Graph requires case_id. "
            "Ensure X-Case-ID header is set."
        )

    if not business.document_id:
        raise ValueError(
            "Framework Analysis Graph requires document_id. "
            "Ensure X-Document-ID header is set."
        )

    # Proceed with analysis...
```

**Rationale** (User Decision):
- ✅ Technical Graphs können **ohne** case_id laufen (z.B. Collection Search)
- ✅ Business Graphs enforced selbst, was sie brauchen (z.B. Framework Analysis)
- ✅ Flexibilität für verschiedene Graph-Typen
- ✅ Clean Separation: normalize_meta ist Infrastruktur, nicht fachlich

**Tests**:
```python
def test_normalize_meta_with_business_context():
    request = MockRequest(headers={
        X_TENANT_ID_HEADER: "tenant-1",
        X_CASE_ID_HEADER: "case-1",
        X_COLLECTION_ID_HEADER: "col-1",
    })

    meta = normalize_meta(request)

    assert "scope_context" in meta
    assert "business_context" in meta
    assert meta["business_context"]["case_id"] == "case-1"
    assert meta["tool_context"]["business"]["case_id"] == "case-1"
```

---

### 3.2 HTTP Request Normalizer

**Datei**: `ai_core/ids/http_scope.py` (normalize_request)

**Änderungen**:
Ähnlich wie normalize_meta - BusinessContext aus Request extrahieren.

---

### Phase 3 Checkpoint

**Deliverables**:
- ✅ `normalize_meta()` angepasst
- ✅ `normalize_request()` angepasst
- ✅ Integration-Tests passen

**Commit Message**:
```
feat(normalizers): adapt for BusinessContext separation

Graph and HTTP request normalizers now create BusinessContext separately
from ScopeContext. Tool context composition preserved.

Related: OPTION_A_IMPLEMENTATION_PLAN.md Phase 3

🤖 Generated with Claude Code
Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
```

---

## Phase 4: Dokumentation

**Ziel**: AGENTS.md, CLAUDE.md, README updates

### 4.1 AGENTS.md updaten

**Sections to update**:
1. Scope Context Contract (Business-IDs entfernt)
2. Tool Context Contract (Komposition erklärt)
3. **NEU**: Business Context Contract
4. **NEU**: Golden Rule section

### 4.2 CLAUDE.md updaten

**Updates**:
- Contracts-Section anpassen
- Beispiele mit neuer Struktur

### 4.3 Migration Guide schreiben

**Datei**: `docs/migrations/OPTION_A_MIGRATION_GUIDE.md`

**Inhalt**:
- Was hat sich geändert
- Wie migriere ich meinen Code
- Beispiele vorher/nachher
- FAQs

---

## Rollback-Strategie

Falls etwas schiefgeht:

**Phase 1**: Einfach - neue Dateien löschen, alte ScopeContext/ToolContext wiederherstellen
**Phase 2**: Mittel - Tool-Inputs zurückrollen, Run-Funktionen zurück
**Phase 3**: Komplex - Normalizer zurück, Meta-Dictionary-Struktur zurück

**Aber**: Pre-MVP, keine Prod-Daten → Rollback nicht kritisch!

---

## Test-Strategie

**Unit-Tests**: Nach jedem Schritt
- Neue Contracts: Erstellung, Immutability, Validation
- Tool-Inputs: Field Removal, ValidationErrors
- Tool-Run-Funktionen: Context-Zugriff

**Integration-Tests**: Nach Phase 2
- End-to-End Tool-Aufrufe
- Graph-Executions mit neuer Context-Struktur

**E2E-Tests**: Nach Phase 3
- HTTP-Request → Graph → Tool → Response
- Vollständige Flows mit neuer Struktur

---

## Success Criteria

**Phase 1**: ✅
- [ ] BusinessContext existiert und ist getestet
- [ ] ScopeContext enthält KEINE Business-IDs mehr
- [ ] ToolContext nutzt Komposition
- [ ] Properties für Backward Compatibility funktionieren
- [ ] Alle Unit-Tests grün

**Phase 2**: ✅
- [ ] Tool-Inputs enthalten KEINE Context-IDs mehr
- [ ] Tool-Run-Funktionen nutzen `context.scope.X` / `context.business.X`
- [ ] Keine Fallback-Logik mehr (`params.X or context.X`)
- [ ] WebSearchContext gelöscht
- [ ] Alle Integration-Tests grün

**Phase 3**: ✅
- [ ] Normalizer erstellen BusinessContext korrekt
- [ ] Meta-Dictionary enthält separate business_context
- [ ] E2E-Tests grün

**Phase 4**: ✅
- [ ] AGENTS.md aktualisiert
- [ ] Migration Guide existiert
- [ ] Code-Kommentare deprecaten alte Patterns

---

## Next Steps

1. ✅ **Phase 1.1 done**: BusinessContext erstellt
2. ⬜ **Phase 1.2**: ScopeContext refactoren + Tests
3. ⬜ **Phase 1.3**: ToolContext Komposition + Tests
4. ⬜ **Phase 1.4**: tool_context_from_scope + Tests
5. 🚀 **Phase 1 Checkpoint**: Commit

**Soll ich mit Phase 1.2 weitermachen?** 🚀

---

**Erstellt von**: Claude Sonnet 4.5
**Status**: Phase 1.1 done, ready for 1.2
**Nächster Schritt**: ScopeContext refactoren
