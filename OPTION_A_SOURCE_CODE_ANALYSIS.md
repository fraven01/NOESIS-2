# Option A: Source Code Analysis

**Datum**: 2025-12-27
**Scope**: Konkrete Quelltext-Analyse für Option A Migration
**Methode**: Grep/RipGrep über aktuelle Codebase

---

## 1. Tool-Input-Modelle mit Context-IDs (gefunden im Code)

### ✅ Bestätigt: RetrieveInput
**Datei**: `ai_core/nodes/retrieve.py:29-42`
```python
class RetrieveInput(BaseModel):
    query: str = ""
    filters: Mapping[str, Any] | None = None
    process: str | None = None
    doc_class: str | None = None
    collection_id: str | None = None          # ❌ RAUS
    workflow_id: str | None = None            # ❌ RAUS
    visibility: str | None = None
    visibility_override_allowed: bool | None = None  # ⚠️ Grenzfall (bleibt?)
    hybrid: Mapping[str, Any] | None = None
    top_k: int | None = None
```

**Migration**:
- `collection_id` → `ToolContext.business.collection_id`
- `workflow_id` → `ToolContext.business.workflow_id`
- `visibility_override_allowed`: Diskussion ob ToolContext oder Input

---

### ✅ Bestätigt: FrameworkAnalysisInput
**Datei**: `ai_core/tools/framework_contracts.py:153-161`
```python
class FrameworkAnalysisInput(BaseModel):
    document_collection_id: UUID  # ❌ RAUS → BusinessContext.collection_id
    document_id: UUID | None = None  # ❌ RAUS → BusinessContext.document_id
    force_reanalysis: bool = False
    confidence_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
```

**Migration**:
- `document_collection_id` → `ToolContext.business.collection_id`
- `document_id` → `ToolContext.business.document_id`

---

### ✅ Neu entdeckt: WebSearchContext (SEPARATE Context!)
**Datei**: `ai_core/tools/web_search.py:103-121`
```python
class WebSearchContext(BaseModel):
    """Validated runtime context for web search executions."""

    tenant_id: str          # ❌ Duplikat von ScopeContext
    trace_id: str           # ❌ Duplikat von ScopeContext
    workflow_id: str        # ❌ Duplikat von ScopeContext/BusinessContext
    case_id: str | None     # ❌ Duplikat von BusinessContext
    run_id: str             # ❌ Duplikat von ScopeContext
    worker_call_id: str | None = None  # ⚠️ Tool-spezifisch?
```

**Problem**: Komplett redundant! Web-Search sollte `ToolContext` verwenden.

**Migration-Optionen**:
1. **Option A**: `WebSearchContext` komplett entfernen, nur `ToolContext` verwenden
2. **Option B**: `WebSearchContext` als Wrapper um `ToolContext` (deprecated)

**Empfehlung**: Option A - `WebSearchContext` löschen!

---

### ✅ Sauber: GraphInput (collection_search)
**Datei**: `ai_core/graphs/technical/collection_search.py:178-191`
```python
class GraphInput(BaseModel):
    question: str
    collection_scope: str
    quality_mode: str = "standard"
    max_candidates: int = 20
    purpose: str
    execute_plan: bool = False
    auto_ingest: bool = False
    auto_ingest_top_k: int = 10
    auto_ingest_min_score: float = 60.0
```

**Status**: ✅ **PERFEKT!** Keine Context-IDs, nur fachliche Parameter.

---

## 2. Tool-Run-Funktionen (Nutzung von Context-IDs)

### ✅ retrieve.py:run()
**Datei**: `ai_core/nodes/retrieve.py:569-648`

**Aktuelle Nutzung**:
```python
def run(context: ToolContext, params: RetrieveInput) -> RetrieveOutput:
    # Aus ToolContext:
    tenant_id = str(context.tenant_id)
    tenant_schema = context.tenant_schema
    case_id = context.case_id

    # Aus RetrieveInput (DUPLIKAT!):
    collection_id = params.collection_id  # ⚠️
    workflow_id = params.workflow_id      # ⚠️

    # Fallback-Logik:
    override_flag = params.visibility_override_allowed
    if override_flag is None:
        override_flag = context.visibility_override_allowed
```

**Migration**:
```python
def run(context: ToolContext, params: RetrieveInput) -> RetrieveOutput:
    # Alle IDs aus context:
    tenant_id = context.scope.tenant_id
    tenant_schema = context.scope.tenant_schema
    case_id = context.business.case_id
    collection_id = context.business.collection_id  # ✅
    workflow_id = context.business.workflow_id      # ✅

    # Visibility override aus context (definitiv):
    override_flag = context.visibility_override_allowed
```

---

### ⚠️ WebSearch (spezielle Architektur)

**Nutzung**: Web-Search verwendet KEIN `ToolContext`, sondern eigenen `WebSearchContext`!

**Fundstelle**: Weitere Analyse erforderlich in `llm_worker/` oder wo WebSearch aufgerufen wird.

---

## 3. Gefundene ID-Zugriffe im Code (Stichprobe)

Running: `rg "\.collection_id|\.workflow_id|\.document_id|\.case_id" ai_core --type py`

**Wird fortgesetzt mit vollständiger Liste...**

---

## 4. Zusammenfassung: Betroffene Dateien

### Phase 1: Contracts (neu/ändern)
- [ ] **NEU**: `ai_core/contracts/business.py` (BusinessContext)
- [ ] **ÄNDERN**: `ai_core/contracts/scope.py` (ScopeContext reduzieren)
- [ ] **ÄNDERN**: `ai_core/tool_contracts/base.py` (ToolContext Komposition)

### Phase 2: Tool-Input-Modelle (Context-IDs entfernen)
- [ ] `ai_core/nodes/retrieve.py` (RetrieveInput)
- [ ] `ai_core/tools/framework_contracts.py` (FrameworkAnalysisInput)
- [ ] **DISKUSSION**: `ai_core/tools/web_search.py` (WebSearchContext komplett ersetzen?)

### Phase 3: Tool-Run-Funktionen (auf neue Struktur migrieren)
- [ ] `ai_core/nodes/retrieve.py:run()`
- [ ] Framework Analysis Graph (finden!)
- [ ] Weitere Run-Funktionen (Liste fortsetzen)

### Phase 4: Graph Normalizer
- [ ] `ai_core/graph/schemas.py:normalize_meta()` (BusinessContext extrahieren)
- [ ] `ai_core/ids/http_scope.py` (HTTP Request → ScopeContext)

### Phase 5: Tests
- [ ] Alle Tests für RetrieveInput
- [ ] Alle Tests für FrameworkAnalysisInput
- [ ] Tests für ScopeContext/ToolContext
- [ ] Integration-Tests

---

## 5. Architektur-Entscheidungen (User Confirmed)

### ✅ ENTSCHIEDEN: WebSearchContext löschen
**Entscheidung**: `WebSearchContext` komplett LÖSCHEN und durch `ToolContext` ersetzen.

**Begründung** (User):
> WebSearchContext ist ein zweites Context-Modell mit identischen Feldern wie Scope plus Business und damit strukturell redundant. Das erzeugt genau die Mehrdeutigkeit, die du bereits in retrieve.py siehst, weil Code dann zwischen params und context wählen muss. Ein einziges Context-Objekt ist die richtige Invariante.

**Migration**:
- Alle WebSearch-Aufrufe auf `ToolContext` umstellen
- `WebSearchContext` komplett entfernen

---

### ✅ ENTSCHIEDEN: visibility_override_allowed gehört in ToolContext
**Entscheidung**: `visibility_override_allowed` aus `RetrieveInput` RAUS, nur in `ToolContext`.

**Begründung** (User):
> Das ist eine Laufzeitberechtigung, also eine Policy-Entscheidung des Aufrufkontexts, nicht ein fachlicher Parameter der Retrieval-Funktion. Wenn es in RetrieveInput bleibt, kann ein Caller es pro Request beliebig setzen und ihr müsst an jeder Stelle erneut prüfen, ob der Caller überhaupt override darf. Das ist eine Permission Boundary und gehört in den Context. In deinem Code existiert bereits die Fallback-Logik von params zu context, das ist ein klares Red Flag und sollte entfernt werden.

**Migration**:
- `RetrieveInput.visibility_override_allowed` LÖSCHEN
- Nur `ToolContext.visibility_override_allowed` behalten
- Fallback-Logik in `retrieve.py:run()` ENTFERNEN

---

### ✅ ENTSCHIEDEN: worker_call_id → ToolContext.metadata (initial)
**Entscheidung**: `worker_call_id` initial als tool-spezifisches Feld in `ToolContext.metadata`.

**Begründung** (User):
> Wenn worker_call_id wirklich nur WebSearch-intern ist, bleibt es als tool-spezifisches Feld in ToolContext.metadata oder als eigenes optionales Feld im ToolContext, aber nicht als separates Context-Modell. Wenn ihr worker_call_id als observability und tracing identifier über Tools hinweg braucht, dann gehört es in ScopeContext als weiteres runtime correlation Feld. Ich würde initial metadata wählen, später promoten falls ihr es mehrfach verwendet.

**Migration**:
- `worker_call_id` → `ToolContext.metadata["worker_call_id"]`
- Falls später mehrfach verwendet: Promote zu `ScopeContext.worker_call_id`

---

### 🎯 Goldene Regel (operationalisiert)

**User-Formulierung**:
> Tool-Inputs enthalten nur funktionale Parameter.
> Context enthält Scope, Business und Runtime Permissions.
> Tool-Run-Funktionen lesen Identifiers und Permissions ausschließlich aus context, nicht aus params.

**Konsequenzen**:
1. ✅ Tool-Inputs: NUR fachliche Parameter (query, filters, confidence_threshold, etc.)
2. ✅ Context: Scope (WHO/WHEN) + Business (WHAT) + Runtime Permissions (MAY)
3. ✅ Tool-Run-Funktionen: Kein Zugriff auf `params.collection_id` etc. - nur `context`
4. ❌ Fallback-Logik (`params.X or context.X`) ist ein **Red Flag** und wird eliminiert

---

### ✅ ENTSCHIEDEN: Graph-spezifische Validierung (Option 1)
**Entscheidung**: `normalize_meta` enforced `case_id` **NICHT** mehr global. Graphs validieren selbst.

**Begründung**:
- `BusinessContext` ist **komplett optional** (alle Felder `str | None`)
- `normalize_meta` ist Infrastruktur-Code, nicht fachlich
- Technical Graphs (z.B. Collection Search) brauchen kein `case_id`
- Business Graphs (z.B. Framework Analysis) validieren selbst, was sie brauchen

**Migration**:
```python
# VORHER (global hard requirement):
if not business.case_id:
    raise ValueError("Case header is required...")  # ❌

# NACHHER (graph-spezifisch):
def normalize_meta(request):
    business = BusinessContext(case_id=request.headers.get("X-Case-ID"))
    # Kein ValueError! ✅

# Graph validiert selbst:
class FrameworkAnalysisGraph:
    def run(self, state, meta):
        business = BusinessContext.model_validate(meta["business_context"])
        if not business.case_id:
            raise ValueError("This graph requires case_id")
```

**Vorteile**:
- ✅ Clean Separation of Concerns
- ✅ Flexibilität für verschiedene Graph-Typen
- ✅ Konsistent mit "BusinessContext ist optional"-Philosophie

---

## 6. Nächste Schritte (konkret)

1. ✅ **Vollständige ID-Zugriff-Analyse** (rg-Output vervollständigen)
2. ⬜ **WebSearch-Architektur verstehen** (wo wird WebSearchContext verwendet?)
3. ⬜ **Alle Run-Funktionen finden** (vollständige Liste)
4. ⬜ **User-Entscheidungen** (WebSearchContext, visibility_override_allowed)
5. 🚀 **Start Phase 1** (BusinessContext erstellen)

---

**Status**: In Arbeit - wird fortgesetzt...
