# Backlog: kwargs & ID-Propagation Harmonisierung

**Status**: Pre-MVP Breaking Changes erlaubt
**Ziel**: Harmonisierung aller Worker/Task-Calls auf explizite ToolContext-Propagation
**Kontext**: Nach Graph-Harmonisierung (ScopeContext/BusinessContext) müssen alle Services/Workers/Tasks auf konsistente ID-Propagation umgestellt werden.

**Hauptproblem**: Viele Services propagieren IDs als einzelne kwargs-Parameter statt strukturiert über `ToolContext`. Dies versteckt die ID-Propagation, macht Code-Reviews schwieriger und verletzt die in Graphen etablierten Contracts.

---

## Executive Summary

**Status**: Phase 1+2+3+4+5+6 ✅ VOLLSTÄNDIG ABGESCHLOSSEN (2026-01-09)

**Identifizierte Probleme**: 11 (1 davon bereits korrekt, hard_delete)
- **Kritisch**: 6 Probleme (Worker-Signaturen, Celery-Dispatch, Task-Signaturen, Context-Factories)
  - ✅ **ALLE 6 GELÖST** (alle kritischen Probleme behoben)
- **Hoch**: 5 Probleme (Test-Helpers, Service-Layer)
  - ✅ **ALLE 5 GELÖST** (2 waren bereits korrekt, 3 behoben)

**Breaking Changes**: 9 total (9 abgeschlossen)
- ✅ **Phase 1**: `with_scope_apply_async()` Signatur
- ✅ **Phase 2+3**: 4 Signaturen (UploadWorker, CrawlerWorker, 2 Tasks)
- ✅ **Phase 4**: 2 Signaturen (tool_context_from_scope, to_tool_context)
- ✅ **Phase 5**: 1 Signatur (GraphTestMixin.make_scope_context) - 2 weitere waren bereits korrekt
- ✅ **Phase 6**: 1 Schema (IngestionOverrides mit extra="allow")

**Verbleibender Aufwand**: ✅ **KEINER** - Projekt vollständig abgeschlossen

**Tests**: ✅ Alle 1520 Tests bestanden (Phase 1+2+3+4+5+6 verifiziert)

**Größter Impact**:
- `UploadWorker.process()`: 7 redundante ID-Parameter → `ToolContext`
- `CrawlerWorker.process()`: 8 redundante ID-Parameter → `ToolContext`
- `with_scope_apply_async()`: `**kwargs` → explizite Parameter
- 3 Task-Dispatches ohne deterministische `task_id` (Idempotenz-Risiko!)

**Wichtigste Design-Entscheidungen**:
1. `crawl_id` → `ScopeContext.crawl_run_id` (empfohlen) vs. `ToolContext.metadata`
2. `ingestion_overrides` Schema-Validierung (empfohlen)
3. `meta_overrides` entfernen (empfohlen, ist Workaround)

---

## Übersicht der Problembereiche

### 🔴 Kritisch (Blocking für neue Features)

1. **Worker-Process-Signaturen** - ID-Explosion in `process()` Methoden (7-8 redundante Parameter)
2. **Celery Task-Dispatch** - `**kwargs` versteckt Task-IDs (3 Stellen ohne deterministische Task-ID)
3. **Task-Signaturen** - Redundante ID-Parameter parallel zu `meta` (2 Tasks)
4. **Context-Factories** - `**overrides` erlaubt Contract-Bruch (1 Factory)

### 🟡 Hoch (Consistency & Maintainability)

5. **Test-Helpers** - Dynamische Field-Extraktion fehler-anfällig (1 Helper)
6. **Service-Layer Task-Calls** - Missing task_id in Dispatches (2 Manager)

---

## 🔴 Kritisch: Worker-Process-Signaturen

### Problem 1: UploadWorker.process() - 7 redundante ID-Parameter ✅ GELÖST

**Datei**: `documents/upload_worker.py:58-71`

**IST-Zustand**:
```python
def process(
    self,
    upload_file: Any,
    *,
    tenant_id: str,              # ❌ → context.scope.tenant_id
    case_id: Optional[str],      # ❌ → context.business.case_id
    workflow_id: Optional[str],  # ❌ → context.business.workflow_id
    trace_id: Optional[str],     # ❌ → context.scope.trace_id
    invocation_id: Optional[str],# ❌ → context.scope.invocation_id
    user_id: Optional[str],      # ❌ → context.scope.user_id
    ingestion_run_id: Optional[str], # ❌ → context.scope.ingestion_run_id
    document_metadata: Optional[Mapping[str, Any]],
    ingestion_overrides: Optional[Mapping[str, Any]],
) -> WorkerPublishResult:
```

**SOLL-Zustand**:
```python
def process(
    self,
    upload_file: Any,
    context: ToolContext,  # ✅ Alle IDs hier!
    *,
    document_metadata: Optional[Mapping[str, Any]] = None,
    ingestion_overrides: Optional[Mapping[str, Any]] = None,
) -> WorkerPublishResult:
    """Process upload and dispatch to ingestion graph."""
    # IDs aus Context extrahieren
    tenant_id = context.scope.tenant_id
    case_id = context.business.case_id
    workflow_id = context.business.workflow_id
    trace_id = context.scope.trace_id
    invocation_id = context.scope.invocation_id
    user_id = context.scope.user_id
    ingestion_run_id = context.scope.ingestion_run_id
```

**Auswirkung**:
- Breaking Change in `documents/tasks.py:66-77` (Task-Call)
- Analog zu Graph-Pattern (siehe `ai_core/graphs/`)

**Geschätzter Aufwand**: 1-2h
- Update Worker-Signatur
- Fix Task-Call
- Update Tests (~5 Files)

---

### Problem 2: CrawlerWorker.process() - 6 redundante ID-Parameter ✅ GELÖST

**Datei**: `crawler/worker.py:84-98`

**IST-Zustand**:
```python
def process(
    self,
    request: FetchRequest,
    *,
    tenant_id: str,              # ❌ → context.scope.tenant_id
    case_id: Optional[str],      # ❌ → context.business.case_id
    crawl_id: Optional[str],     # ❓ Business-ID oder infrastruktur? TBD
    trace_id: Optional[str],     # ❌ → context.scope.trace_id
    frontier_state: Optional[Mapping[str, Any]], # ✅ OK (business logic)
    document_id: Optional[str],  # ✅ OK (override)
    document_metadata: Optional[Mapping[str, Any]], # ✅ OK
    ingestion_overrides: Optional[Mapping[str, Any]], # ✅ OK
    meta_overrides: Optional[Mapping[str, Any]], # ❌ TBD: sollte aus context kommen
    idempotency_key: Optional[str] = None, # ❌ → context.scope.idempotency_key?
) -> WorkerPublishResult:
```

**SOLL-Zustand**:
```python
def process(
    self,
    request: FetchRequest,
    context: ToolContext,  # ✅ Alle IDs hier!
    *,
    crawl_id: Optional[str] = None,  # TBD: In BusinessContext verschieben?
    frontier_state: Optional[Mapping[str, Any]] = None,
    document_id: Optional[str] = None,
    document_metadata: Optional[Mapping[str, Any]] = None,
    ingestion_overrides: Optional[Mapping[str, Any]] = None,
) -> WorkerPublishResult:
    """Fetch request and publish to ingestion graph."""
    tenant_id = context.scope.tenant_id
    case_id = context.business.case_id
    trace_id = context.scope.trace_id
    idempotency_key = context.scope.idempotency_key
```

**Offene Fragen**:
- `crawl_id`: Business-ID (→ `BusinessContext`) oder Request-spezifisch?
- `meta_overrides`: Sollte komplett entfernt werden (Anti-Pattern)

**Auswirkung**:
- Breaking Change in `crawler/tasks.py:74-83` (Task-Call)
- Eventuell `BusinessContext` Extension für `crawl_id`

**Geschätzter Aufwand**: 2-3h
- Update Worker-Signatur
- Fix Task-Call
- Update Tests
- Design-Entscheidung für `crawl_id`

---

## 🔴 Kritisch: Celery Task-Dispatch

### Problem 3: with_scope_apply_async() - **kwargs versteckt Task-IDs

**Datei**: `common/celery.py:804-842`

**IST-Zustand**:
```python
def with_scope_apply_async(
    signature: Signature,
    scope: TypingMapping[str, Any],
    *args: Any,
    **kwargs: Any,  # ❌ Versteckt task_id und andere Parameter
):
    """Clone signature and schedule with trace headers."""
    # ...
    return scoped_signature.apply_async(*args, **kwargs)  # kwargs durchgereicht
```

**Problem**: Task-IDs werden implizit übergeben:
```python
# documents/upload_worker.py:196
apply_kwargs: Dict[str, Any] = {}
if ingestion_run_id:
    apply_kwargs["task_id"] = ingestion_run_id
async_result = with_scope_apply_async(signature, scope, **apply_kwargs)
```

**SOLL-Zustand**:
```python
def with_scope_apply_async(
    signature: Signature,
    scope: Mapping[str, Any],
    *,
    task_id: str | None = None,  # ✅ Explizit
    countdown: int | None = None,
    eta: datetime | None = None,
    expires: datetime | None = None,
    retry: bool | None = None,
    retry_policy: dict | None = None,
) -> AsyncResult:
    """Clone and schedule signature with explicit apply_async params."""
    apply_kwargs = {}
    if task_id:
        apply_kwargs["task_id"] = task_id
    if countdown is not None:
        apply_kwargs["countdown"] = countdown
    if eta:
        apply_kwargs["eta"] = eta
    if expires:
        apply_kwargs["expires"] = expires
    if retry is not None:
        apply_kwargs["retry"] = retry
    if retry_policy:
        apply_kwargs["retry_policy"] = retry_policy

    if not otel_headers:
        return signature.apply_async(**apply_kwargs)
    scoped = _clone_with_scope(signature, {}, otel_headers)
    return scoped.apply_async(**apply_kwargs)
```

**Auswirkung**:
- Breaking Change in allen `with_scope_apply_async()` Calls:
  - `documents/upload_worker.py:196`
  - `ai_core/services/ingestion.py:73`
  - `ai_core/services/document_upload.py:275-278`

**Geschätzter Aufwand**: 2-3h
- Update `with_scope_apply_async()` Signatur
- Fix alle Caller (~10 Stellen)
- Update Tests

---

### Problem 4: Task-Dispatch ohne task_id - Missing Idempotency

**Datei**: `ai_core/services/document_upload.py:267-278`

**IST-Zustand**:
```python
signature = upload_document_task.s(
    file_bytes=file_bytes,
    filename=original_name,
    content_type=detected_mime,
    metadata=document_metadata_payload,
    meta=meta,
    ingestion_run_id=ingestion_run_id,  # ❌ Redundant (schon in meta)
)
async_result = with_scope_apply_async(
    signature,
    scope_context.model_dump(mode="json", exclude_none=True),
    # ❌ MISSING: task_id=ingestion_run_id
)
```

**Problem**:
1. `ingestion_run_id` wird als Task-Parameter übergeben (redundant)
2. Celery Task-ID wird nicht gesetzt → random Task-ID statt deterministisch

**SOLL-Zustand**:
```python
signature = upload_document_task.s(
    file_bytes=file_bytes,
    filename=original_name,
    content_type=detected_mime,
    metadata=document_metadata_payload,
    meta=meta,
    # ✅ REMOVE: ingestion_run_id parameter
)
async_result = with_scope_apply_async(
    signature,
    scope_context.model_dump(mode="json", exclude_none=True),
    task_id=ingestion_run_id,  # ✅ SET: Deterministische Task-ID
)
```

**Auswirkung**:
- Celery Task-IDs werden vorhersagbar (wichtig für Idempotenz)
- Task-Signatur vereinfacht

**Geschätzter Aufwand**: 30min
- Fix Dispatch-Call
- Remove redundanter Parameter

---

## 🔴 Kritisch: Task-Signaturen

### Problem 5: upload_document_task - Redundanter ingestion_run_id Parameter

**Datei**: `documents/tasks.py:26-33`

**IST-Zustand**:
```python
@shared_task(base=ScopedTask, queue="ingestion")
def upload_document_task(
    file_bytes: bytes,
    filename: str,
    content_type: str,
    metadata: Dict[str, Any],
    meta: Dict[str, Any],
    ingestion_run_id: str | None = None,  # ❌ Redundant! Bereits in meta
) -> Dict[str, Any]:
    context = tool_context_from_meta(meta)
    # ...
    ingestion_run_id = ingestion_run_id or context.scope.ingestion_run_id  # Fallback
```

**Problem**: `ingestion_run_id` wird **doppelt** übergeben:
1. Als expliziter Task-Parameter
2. Im `meta` dict via `scope_context.ingestion_run_id`

**SOLL-Zustand**:
```python
@shared_task(base=ScopedTask, queue="ingestion")
def upload_document_task(
    file_bytes: bytes,
    filename: str,
    content_type: str,
    metadata: Dict[str, Any],
    meta: Dict[str, Any],
    # ✅ REMOVE: ingestion_run_id parameter
) -> Dict[str, Any]:
    context = tool_context_from_meta(meta)
    ingestion_run_id = context.scope.ingestion_run_id  # ✅ Immer aus Context

    result = worker.process(
        upload,
        context,  # ✅ Ganzer Context statt einzelne IDs
        document_metadata=metadata,
        ingestion_overrides=metadata.get("ingestion_overrides"),
    )
```

**Auswirkung**:
- Breaking Change in `ai_core/services/document_upload.py:267-278`
- Breaking Change in `documents/upload_manager.py:44-50`

**Geschätzter Aufwand**: 1h
- Remove Parameter
- Fix Caller
- Update Tests

---

### Problem 6: crawl_url_task - Manuelle ID-Extraktion

**Datei**: `crawler/tasks.py:14-83`

**IST-Zustand**:
```python
@shared_task(bind=True, queue="crawler", name="crawler.tasks.crawl_url_task")
def crawl_url_task(
    self,
    url: str,
    meta: Mapping[str, Any],
    ingestion_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    context = tool_context_from_meta(meta)
    tenant_id = context.scope.tenant_id  # ❌ Manuell extrahiert

    # ... später:
    result = worker.process(
        request,
        tenant_id=tenant_id,        # ❌ Einzeln übergeben
        case_id=context.business.case_id,
        crawl_id=meta.get("crawl_id"),
        trace_id=context.scope.trace_id,
        document_metadata=doc_meta,
        ingestion_overrides=ingestion_overrides,
        meta_overrides=meta,
    )
```

**SOLL-Zustand**:
```python
@shared_task(bind=True, queue="crawler", name="crawler.tasks.crawl_url_task")
def crawl_url_task(
    self,
    url: str,
    meta: Mapping[str, Any],
    ingestion_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    context = tool_context_from_meta(meta)

    # Validate tenant early
    if not context.scope.tenant_id:
        logger.error("crawl_url_task.missing_tenant_id", extra={"url": url})
        return {"status": "error", "reason": "missing_tenant_id"}

    # ... später:
    result = worker.process(
        request,
        context,  # ✅ Ganzer Context
        crawl_id=meta.get("crawl_id"),  # TBD: In BusinessContext?
        document_metadata=doc_meta,
        ingestion_overrides=ingestion_overrides,
    )
```

**Auswirkung**:
- Abhängig von `CrawlerWorker.process()` Harmonisierung (Problem 2)

**Geschätzter Aufwand**: 1h (nach Problem 2)
- Update Task
- Update Tests

---

### ~~Problem 7: hard_delete - Undokumentierter actor kwarg~~ ✅ KEIN PROBLEM

**Datei**: `ai_core/rag/hard_delete.py:165-170`

**Status**: ✅ **BEREITS KORREKT** - Signatur ist explizit, kein kwargs-Problem

**IST-Zustand**:
```python
@shared_task(
    base=ScopedTask,
    name="rag.hard_delete",
    queue="rag_delete",
)
def hard_delete(  # type: ignore[override]
    state: Mapping[str, object],
    meta: Mapping[str, object] | None = None,
    *,
    actor: Mapping[str, object] | None = None,  # ✅ Explizit dokumentiert
) -> Mapping[str, object]:
```

**Caller**: `ai_core/views.py:2140-2144`
```python
async_result = hard_delete.delay(
    state,
    meta,
    actor=actor,  # ✅ Korrekt - Parameter ist explizit in Signatur
)
```

**Warum kein Problem**:
- Parameter `actor` ist **explizit** in Signatur (Zeile 169)
- Caller nutzt benannten Parameter korrekt
- Keine `**kwargs` die etwas verstecken

**Empfehlung**: ✅ **KEIN FIX NÖTIG**

**Geschätzter Aufwand**: 0min

---

## 🔴 Kritisch: Context-Factories

### Problem 8: tool_context_from_scope() - **overrides erlaubt Contract-Bruch ✅ GELÖST

**Datei**: `ai_core/tool_contracts/base.py:95-140`

**IST-Zustand**:
```python
def tool_context_from_scope(
    scope: ScopeContext,
    business: BusinessContext | None = None,
    *,
    now: datetime | None = None,
    **overrides: Any,  # ❌ Erlaubt beliebige Felder
) -> ToolContext:
    """Build ToolContext from ScopeContext and BusinessContext."""
    if business is None:
        business = BusinessContext()
    payload: dict[str, Any] = {
        "scope": scope,
        "business": business,
    }
    payload.update(overrides)  # ❌ Kann scope/business überschreiben!
    return ToolContext(**payload)
```

**Problem**: Caller kann `scope` oder `business` überschreiben:
```python
# Möglich aber FALSCH:
tool_context_from_scope(scope, business, scope=evil_scope)  # Überschreibt scope!
```

**SOLL-Zustand**:
```python
def tool_context_from_scope(
    scope: ScopeContext,
    business: BusinessContext | None = None,
    *,
    now: datetime | None = None,
    locale: str | None = None,
    timeouts_ms: dict[str, int] | None = None,
    # ✅ KEIN **overrides mehr - nur dokumentierte Felder
) -> ToolContext:
    """Build ToolContext with explicit field overrides only."""
    if business is None:
        business = BusinessContext()
    return ToolContext(
        scope=scope,
        business=business,
        now=now,
        locale=locale,
        timeouts_ms=timeouts_ms,
    )
```

**Auswirkung**:
- Breaking Change in `ai_core/contracts/scope.py:209` (Override-Durchpass)
- Breaking Change in allen Test-Calls mit `**overrides`

**Geschätzter Aufwand**: 2-3h
- Update Factory-Signatur
- Fix alle Caller (~10 Stellen)
- Update Tests (~20 Stellen)

---

## 🟡 Hoch: Test-Helpers

### Problem 9: make_test_meta() - **kwargs versteckt Signatur

**Datei**: `ai_core/tests/utils.py:114-144`

**IST-Zustand**:
```python
def make_test_meta(
    extra: Mapping[str, Any] | None = None,
    **kwargs: Any,  # ❌ Versteckt erlaubte Parameter
) -> dict[str, Any]:
    """Create test metadata for graph execution.
    Accepts all arguments from `make_test_ids` via kwargs.
    """
    ids = make_test_ids(**kwargs)  # kwargs durchgereicht
```

**Problem**:
- **Signatur nicht selbst-dokumentierend**: Man muss `make_test_ids()` lesen um zu wissen welche Parameter erlaubt sind
- **Typos crashen zwar** (weil `make_test_ids()` explizite Parameter hat), aber **erst zur Laufzeit** statt bei IDE-Autocomplete
- **Keine Type-Hints** für Caller

**Beispiel**:
```python
# Typo crasht zur Laufzeit (TypeError), aber IDE warnt nicht
make_test_meta(tenant_idd="test")  # "tenant_idd" statt "tenant_id"
```

**SOLL-Zustand**:
```python
def make_test_meta(
    *,
    tenant_id: str | None = None,
    trace_id: str | None = None,
    case_id: str | None = None,
    workflow_id: str | None = None,
    run_id: str | None = None,
    ingestion_run_id: str | None = None,
    invocation_id: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create test meta with explicit IDs."""
    ids = make_test_ids(
        tenant_id=tenant_id,
        trace_id=trace_id,
        case_id=case_id,
        workflow_id=workflow_id,
        run_id=run_id,
        ingestion_run_id=ingestion_run_id,
        invocation_id=invocation_id,
    )
    # ...
```

**Auswirkung**:
- Breaking Change in ~50 Test-Files (grep `make_test_meta`)

**Geschätzter Aufwand**: 2-3h
- Update Helper-Signatur
- Fix alle Test-Calls (~50 Stellen)

---

### Problem 10: make_tool_context() - Dynamische Field-Extraktion

**Datei**: `ai_core/tests/utils.py:63-77`

**IST-Zustand**:
```python
def make_tool_context(self, **overrides: Any) -> ToolContext:
    """Create valid ToolContext with defaults."""
    business_keys = {
        "case_id",
        "workflow_id",
        "collection_id",
        "document_id",
        "document_version_id",
    }
    business_payload = {
        key: overrides.pop(key) for key in tuple(business_keys) if key in overrides
    }  # ❌ pop() kann crashen, Typos ignoriert
    business = BusinessContext(**business_payload)
```

**Problem**:
- `pop()` Fehler-anfällig
- Typos in Feldnamen werden ignoriert

**SOLL-Zustand**:
```python
def make_tool_context(
    self,
    *,
    # Scope-Felder
    tenant_id: str | None = None,
    trace_id: str | None = None,
    invocation_id: str | None = None,
    run_id: str | None = None,
    ingestion_run_id: str | None = None,
    # Business-Felder
    case_id: str | None = None,
    workflow_id: str | None = None,
    collection_id: str | None = None,
    document_id: str | None = None,
    document_version_id: str | None = None,
    # Weitere Felder
    now: datetime | None = None,
    locale: str | None = None,
) -> ToolContext:
    """Create valid ToolContext with explicit fields."""
    scope = self.make_scope(
        tenant_id=tenant_id,
        trace_id=trace_id,
        invocation_id=invocation_id,
        run_id=run_id,
        ingestion_run_id=ingestion_run_id,
    )
    business = BusinessContext(
        case_id=case_id,
        workflow_id=workflow_id,
        collection_id=collection_id,
        document_id=document_id,
        document_version_id=document_version_id,
    )
    return ToolContext(
        scope=scope,
        business=business,
        now=now,
        locale=locale,
    )
```

**Auswirkung**:
- Breaking Change in ~30 Test-Files

**Geschätzter Aufwand**: 1-2h
- Update Helper-Signatur
- Fix alle Test-Calls (~30 Stellen)

---

## 🟡 Hoch: Service-Layer Task-Calls

### Problem 11: crawler/manager.py - ingestion_overrides unvalidiert

**Datei**: `crawler/manager.py:64-68`

**IST-Zustand**:
```python
task_result = crawl_url_task.delay(
    url=origin.url,
    meta=dict(meta),  # ❌ Warum dict() Kopie?
    ingestion_overrides=ingestion_overrides,  # ❌ Dynamisches dict ohne Validierung
)
```

**Problem**:
- `meta` wird als `dict()` kopiert (warum?)
- `ingestion_overrides` ist dynamisches dict ohne Type-Check

**SOLL-Zustand**:
```python
# Option A: Validierung via Pydantic
from ai_core.schemas import IngestionOverrides

validated_overrides = IngestionOverrides.model_validate(ingestion_overrides)
task_result = crawl_url_task.delay(
    url=origin.url,
    meta=meta,  # ✅ Keine unnötige Kopie
    ingestion_overrides=validated_overrides.model_dump(),
)
```

**Offene Fragen**:
- Existiert `IngestionOverrides` Schema? Wenn nicht → erstellen

**Geschätzter Aufwand**: 2-3h
- Erstelle `IngestionOverrides` Schema (falls nicht existiert)
- Update alle Caller
- Update Tests

---

### Problem 12: documents/upload_manager.py - Missing task_id

**Datei**: `documents/upload_manager.py:44-50`

**IST-Zustand**:
```python
task_result = upload_document_task.delay(
    file_bytes=file_bytes,
    filename=upload.name,
    content_type=getattr(upload, "content_type", "application/octet-stream"),
    metadata=metadata,
    meta=meta,
)
# ❌ Keine task_id gesetzt → random Task-ID
```

**SOLL-Zustand**:
```python
# Analog zu document_upload.py (Problem 4)
context = tool_context_from_meta(meta)
ingestion_run_id = context.scope.ingestion_run_id or uuid4().hex

signature = upload_document_task.s(
    file_bytes=file_bytes,
    filename=upload.name,
    content_type=getattr(upload, "content_type", "application/octet-stream"),
    metadata=metadata,
    meta=meta,
)
async_result = with_scope_apply_async(
    signature,
    context.scope.model_dump(mode="json", exclude_none=True),
    task_id=ingestion_run_id,  # ✅ Deterministische Task-ID
)
```

**Auswirkung**:
- Konsistent mit `document_upload.py` Pattern

**Geschätzter Aufwand**: 30min
- Update Task-Dispatch
- Update Tests

---

## Migration-Plan (Empfohlene Reihenfolge)

**⚠️ WICHTIG**: Phase 2/3 sollten **zusammen** durchgeführt werden, um zu vermeiden dass Task- und Worker-Signaturen kurzzeitig auseinanderlaufen.

### ✅ Phase 1: Celery-Layer (ABGESCHLOSSEN)

**Ziel**: Explizite Task-Dispatch-Parameter

1. ✅ **Problem 3**: `with_scope_apply_async()` explizite Parameter
   - `common/celery.py:804-867` - explizite Parameter statt `**kwargs`
   - Unterstützt: `task_id`, `countdown`, `eta`, `expires`, `retry`, `retry_policy`, `queue`, `priority`
2. ✅ **Problem 4**: `document_upload.py` set `task_id`
   - `ai_core/services/document_upload.py:277` - `task_id=ingestion_run_id`
3. ✅ **Problem 12**: `upload_manager.py` set `task_id`
   - `documents/upload_manager.py:58-62` - `task_id=ingestion_run_id`
4. ✅ **BONUS**: `crawler_manager.py` nutzt `with_scope_apply_async()`
   - `crawler/manager.py:70-74` - OTEL-Tracing enthalten

**Tests**:
```bash
npm run test:py -- tests/chaos/test_tool_context_contracts.py
npm run test:py -- documents/tests/test_upload_manager.py
npm run test:py -- ai_core/tests/test_ingestion_view.py
```

**Deliverable**: ✅ Deterministische Celery Task-IDs (wichtig für Idempotenz!)

---

### ✅ Phase 2+3: Worker & Task-Signaturen (ABGESCHLOSSEN - 2026-01-08)

**Ziel**: ToolContext als primärer ID-Carrier in Workers UND Tasks

**Warum zusammen?**
- Task-Signatur-Änderungen (Problem 5,6) hängen von Worker-Signatur (Problem 1,2) ab
- Verhindert kurzzeitiges Auseinanderlaufen der Signaturen
- Atomic Breaking Change

**Reihenfolge innerhalb Phase 2+3**:
1. ✅ **Problem 1**: `UploadWorker.process()` ToolContext
   - `documents/upload_worker.py:58-65` - 7 Parameter → `context: ToolContext`
   - Neue Methode: `_compose_meta_from_context()`
2. ✅ **Problem 5**: `upload_document_task` remove `ingestion_run_id`
   - `documents/tasks.py:62-67` - Context-Propagation statt einzelne IDs
   - Parameter war bereits entfernt
3. ✅ **Problem 2**: `CrawlerWorker.process()` ToolContext
   - `crawler/worker.py:86-94` - 8 Parameter → `context: ToolContext`
   - **Design-Entscheidung**: `crawl_id` → `ToolContext.metadata["crawl_id"]`
   - **Removal**: `meta_overrides` Parameter komplett entfernt
   - Neue Methode: `_compose_meta_from_context()`
4. ✅ **Problem 6**: `crawl_url_task` ToolContext-Propagation
   - `crawler/tasks.py:80-85` - Context-Propagation, `crawl_id` in metadata

**Tests**:
```bash
npm run test:py:parallel
npm run test:py -- documents/tests/test_upload_worker.py
npm run test:py -- crawler/tests/test_worker.py
```

**Deliverable**: ✅ Konsistente Context-Propagation durch alle Worker/Tasks

---

### Phase 4: Context-Factories (Hoch, 2-3h)

**Ziel**: Contract-Sicherheit

8. ✅ **Problem 8**: `tool_context_from_scope()` explizite Parameter (2-3h)

**Tests**: `npm run test:py -- ai_core/tests/`

---

### ✅ Phase 5: Test-Helpers (ABGESCHLOSSEN)

**Ziel**: Typo-Prävention

9. ✅ **Problem 9**: `make_test_meta()` - **BEREITS KORREKT** (hatte bereits explizite Parameter)
10. ✅ **Problem 10**: `make_tool_context()` - **BEREITS KORREKT** (hatte bereits explizite Parameter)
11. ✅ **Bonus**: `GraphTestMixin.make_scope_context()` - `**extra` entfernt, explizite Parameter hinzugefügt

**Tests**: ✅ `npm run test:py:parallel` - Alle 1520 Tests bestanden

**Ergebnis**: Phase 5 war schneller als erwartet, da Problems 9 und 10 bereits gelöst waren. Nur eine zusätzliche Methode (`make_scope_context`) musste harmonisiert werden.

---

### ✅ Phase 6: Service-Layer Validierung (ABGESCHLOSSEN)

**Ziel**: Typsicherheit für Overrides

11. ✅ **Problem 11**: `ingestion_overrides` Schema-Validierung
    - **Schema erstellt**: `ai_core/schemas.py:IngestionOverrides`
    - **Design**: Permissive (`extra="allow"`) für Backward-Kompatibilität
    - **Dokumentierte Felder**: `collection_id`, `embedding_profile`, `scope`, `guardrails`, `source`, `raw_document`
    - **Agentic Coder Warning**: Docstring enthält explizite Anweisung an Claude/Gemini, neue Felder zu melden
    - **Validierung**: Nur in Entry-Points (`crawler/manager.py`), nicht in Workers/Tasks
    - **Breaking Change**: Minimal - nur Typo-Erkennung für dokumentierte Felder

**Tests**: ✅ `npm run test:py:parallel` - Alle 1520 Tests bestanden

**Ergebnis**: Schema bietet Type-Safety für dokumentierte Felder bei voller Flexibilität für Graph-Extensions.

---

## Geschätzter Gesamt-Aufwand

- **Kritisch (Phase 1)**: 3-4 Stunden (Celery-Layer)
- **Kritisch (Phase 2+3)**: 5-8 Stunden (Worker & Task-Signaturen, ZUSAMMEN)
- **Hoch (Phase 4-5)**: 5-8 Stunden (Context-Factories + Test-Helpers)
- **Optional (Phase 6)**: 2-3 Stunden (ingestion_overrides Schema)

**Gesamt Kritisch**: 8-12 Stunden (~1-2 Tage)
**Gesamt mit Hoch**: 13-20 Stunden (~2-3 Tage)
**Gesamt mit Optional**: 15-23 Stunden (~2-3 Tage)

---

## Inventarisierung: crawl_id & meta_overrides Nutzung

### crawl_id Nutzung (25 Stellen in `crawler/worker.py`)

**Alle Verwendungen**:
1. **Parameter**: `process()` Zeile 91
2. **State-Building**: `_compose_state()` Zeile 184, 192-193 → `state["crawl_id"]`
3. **Meta-Building**: `_compose_meta()` Zeile 537, 602 → `meta["crawl_id"]`
4. **Object-Store-Pfade**:
   - Blob-Writer-Factory Zeile 464, 482, 488, 493, 495, 501, 508
   - Asset-Extraction Zeile 709, 747, 832, 838
5. **Caller**: `crawler/tasks.py:78` → `crawl_id=meta.get("crawl_id")`

**Semantik**:
- `crawl_id` ist ein **Session/Run-Identifier** für Crawler-Runs
- Gruppiert mehrere Fetch-Requests zusammen (analog zu `ingestion_run_id`)
- Verwendet für Object-Store-Pfade: `tenant/{case_or_crawl}/uploads/...`

**Inkonsistenz**:
- Kommt aus `meta.get("crawl_id")` im Task
- Wird aber als **expliziter Parameter** durch alle Methoden gereicht
- **Nicht** in ScopeContext oder BusinessContext

---

### meta_overrides Nutzung (9 Stellen nur in `crawler/`)

**Alle Verwendungen**:
1. **Parameter**: `CrawlerWorker.process()` Zeile 98
2. **Extraction in `_compose_meta()`** Zeile 541-609:
   - `scope_context` aus overrides (Zeile 548-550)
   - `business_context` aus overrides (Zeile 551-553)
   - `initiated_by_user_id` aus overrides (Zeile 555-557)
   - Rest in `filtered_overrides` gemerged (Zeile 608-609)
3. **Caller**: `crawler/tasks.py:82` → `meta_overrides=meta`
4. **Tests**: `crawler/tests/test_worker.py:192,206` → `meta_overrides={"trace_id": "trace-1"}`

**Root Cause**:
- `CrawlerWorker.process()` hat **KEINE** `meta` oder `context` Parameter
- Daher muss bereits existierender `meta` Context via `meta_overrides` durchgeschleust werden
- **Workaround** für fehlende Context-Propagation!

**Lösung**: Wenn `process(context: ToolContext)` bekommt, ist `meta_overrides` **komplett überflüssig**

---

## Offene Design-Entscheidungen

### 1. `crawl_id` Placement

**Frage**: Wo sollte `crawl_id` leben?

**⚠️ Aktualisierte Analyse** (basierend auf Inventarisierung):

**Option A**: In `ScopeContext.crawl_run_id` (empfohlen)
```python
# ScopeContext erweitern (analog zu ingestion_run_id)
class ScopeContext(BaseModel):
    # ... existierende Felder
    run_id: Optional[str] = None
    ingestion_run_id: Optional[str] = None
    crawl_run_id: Optional[str] = None  # ✅ Run/Session-Identifier
```
**Pro**:
- `crawl_id` ist **kein Business-Konzept** sondern Run-Context (wie `ingestion_run_id`)
- Konsistent mit bestehenden Run-Identifiern
- Keine BusinessContext-Contract-Änderung nötig

**Contra**:
- ScopeContext hat schon `run_id` und `ingestion_run_id` - warum noch einen?
- Eventuell genügt `run_id` für Crawler?

**Option B**: In `ToolContext.metadata["crawl_id"]` (flexibel)
```python
context = ToolContext(
    scope=scope,
    business=business,
    metadata={"crawl_id": "crawl-123"},  # ✅ Flexibel, kein Contract-Change
)
```
**Pro**:
- Kein Contract-Breaking Change
- Flexibel für weitere Run-Context-Felder

**Contra**:
- Weniger typsicher (dict statt Field)
- Keine IDE-Autocomplete

**Option C**: In `BusinessContext` (NEIN - falsche Semantik)
```python
# BusinessContext erweitern
class BusinessContext(BaseModel):
    crawl_id: Optional[str] = None  # ❌ NEIN - ist kein Business-Konzept
```
**Contra**:
- `crawl_id` ist **kein Business-Domain-Konzept** (nicht wie `case_id`, `workflow_id`)
- Ist Infrastructure/Run-Context

**Empfehlung**: **Option A** (ScopeContext) - `crawl_id` ist Run-Context, nicht Business-Domain

---

### 2. `ingestion_overrides` Schema

**Frage**: Sollte `ingestion_overrides` ein typisiertes Schema haben?

**Status Quo**: Dynamisches `dict[str, Any]` ohne Validierung

**Vorschlag**: Erstelle `IngestionOverrides` Schema
```python
# ai_core/schemas.py
class IngestionOverrides(BaseModel):
    """Validated overrides for ingestion process."""
    collection_id: Optional[str] = None
    embedding_profile: Optional[str] = None
    scope: Optional[str] = None
    chunking_strategy: Optional[str] = None
    # ... weitere Felder

    model_config = ConfigDict(extra="forbid")  # ✅ Keine unbekannten Felder
```

**Vorteil**:
- Type-Safety
- Dokumentiert erlaubte Overrides
- Verhindert Typos

**Nachteil**: Breaking Change für unstrukturierte Overrides

**Empfehlung**: ✅ **Implementieren** - Pre-MVP erlaubt Breaking Changes

---

### 3. `meta_overrides` in CrawlerWorker

**Frage**: Sollte `meta_overrides` komplett entfernt werden?

**⚠️ Aktualisierte Analyse** (basierend auf Inventarisierung):

**Status Quo**: `meta_overrides: Optional[Mapping[str, Any]]` wird als **Workaround** genutzt um Context durchzuschleusen

**Root Cause**:
- `CrawlerWorker.process()` hat KEINE `context` Parameter
- `crawler/tasks.py` übergibt `meta_overrides=meta` um Context weiterzugeben
- Worker extrahiert `scope_context`, `business_context`, `initiated_by_user_id` aus Overrides

**Lösung**: ✅ **ENTFERNEN** nach Worker-Signatur-Harmonisierung

**Migration**:
1. `CrawlerWorker.process(context: ToolContext)` statt einzelne IDs
2. `crawler/tasks.py` übergibt `context` direkt
3. `meta_overrides` Parameter komplett entfernen
4. `initiated_by_user_id` → in `ToolContext.metadata` verschieben

**Abhängigkeit**: Muss **nach** Problem 2 (CrawlerWorker-Signatur) erfolgen

**Empfehlung**: ✅ **REMOVE** - ist Workaround für fehlende Context-Propagation

---

## Test-Strategie

### Unit-Tests (Phase 1-4)

```bash
# Problem 3-4: Celery-Dispatch
npm run test:py -- tests/chaos/test_tool_context_contracts.py

# Problem 5-6: Task-Signaturen
npm run test:py -- documents/tests/test_upload_worker.py
npm run test:py -- crawler/tests/

# Problem 1-2: Worker-Signaturen
npm run test:py -- documents/tests/test_upload_worker.py
npm run test:py -- crawler/tests/test_worker.py

# Problem 8: Context-Factories
npm run test:py -- ai_core/tests/test_tool_contracts.py
```

### Integration-Tests (Phase 3-5)

```bash
# Vollständig (inkl. slow tests)
npm run test:py:parallel
```

### Chaos-Tests (Regression)

```bash
# Contract-Validierung
npm run test:py -- tests/chaos/test_tool_context_contracts.py -v
npm run test:py -- tests/chaos/test_graph_io_contracts.py -v
```

---

## Breaking Changes Checklist

### ✅ Phase 1: Celery-Layer (ABGESCHLOSSEN)
- [x] `with_scope_apply_async()` Signatur - explizite Parameter statt `**kwargs`
  - ✅ `common/celery.py:804-867`
  - ✅ `task_id` gesetzt in 3 Dispatches

### Phase 2+3: Worker & Task-Signaturen (ZUSAMMEN) ✅ ABGESCHLOSSEN
- [x] `UploadWorker.process()` Signatur - `ToolContext` statt 7 Parameter
- [x] `upload_document_task` Signatur - remove `ingestion_run_id` Parameter
- [x] `CrawlerWorker.process()` Signatur - `ToolContext` statt 8 Parameter, remove `meta_overrides`
- [x] `crawl_url_task` Signatur - `ToolContext`-Propagation
- [x] Design-Entscheidung: `crawl_id` → `ToolContext.metadata["crawl_id"]` (Option B)
- [x] Tests aktualisiert: 11 Worker-Tests refaktoriert mit `_make_test_context()` Helper
- [x] Alle Tests bestanden: 1520 passed (1396 fast + 124 slow)

### Phase 4: Context-Factories ✅ ABGESCHLOSSEN
- [x] `tool_context_from_scope()` Signatur - explizite Parameter statt `**overrides`
- [x] `ScopeContext.to_tool_context()` Signatur - explizite Parameter
- [x] Breaking Change: 8 explizite Parameter statt `**overrides`
- [x] Type Safety: Verhindert versehentliches Überschreiben von scope/business
- [x] Alle Tests bestanden: 1520 passed (1396 fast + 124 slow)

### ✅ Phase 5: Test-Helpers (ABGESCHLOSSEN)
- [x] `make_test_meta()` Signatur - **BEREITS KORREKT** (explizite Parameter)
- [x] `make_tool_context()` Signatur - **BEREITS KORREKT** (explizite Parameter)
- [x] `GraphTestMixin.make_scope_context()` Signatur - `**extra` entfernt, explizite Parameter
- [x] Alle Tests bestanden: 1520 passed (1396 fast + 124 slow)

### ✅ Phase 6: Service-Layer Validierung (ABGESCHLOSSEN)
- [x] `IngestionOverrides` Schema erstellt in `ai_core/schemas.py`
- [x] Schema Design: Permissive (`extra="allow"`) für Backward-Kompatibilität
- [x] 6 dokumentierte Felder mit Type-Hints
- [x] Agentic Coder Warning im Docstring
- [x] Validierung in `crawler/manager.py` Entry-Point
- [x] Workers/Tasks behalten flexible `Mapping[str, Any]` Signatur
- [x] Alle Tests bestanden: 1520 passed (1396 fast + 124 slow)

**Gesamt**: 9 Breaking Changes abgeschlossen (crawl_id via metadata, ingestion_overrides mit Schema)

---

## Erfolgs-Kriterien

Nach Abschluss sollten:

1. ✅ **Alle IDs explizit** propagiert werden (keine kwargs für IDs) - **ERLEDIGT**
2. ✅ **ToolContext primärer Carrier** für Worker/Service-Calls - **ERLEDIGT**
3. ✅ **Celery Task-IDs deterministisch** (via `task_id` Parameter) - **ERLEDIGT**
4. ✅ **Test-Helpers typsicher** (Typos crashen sofort) - **ERLEDIGT**
5. ✅ **Overrides validiert** (Schema statt arbitrary dict) - **ERLEDIGT (Phase 6)**
6. ✅ **Alle Chaos-Tests grün** (Contract-Validierung) - **ERLEDIGT**

**Status**: ✅ **6 von 6 Kriterien erfüllt** - Projekt vollständig abgeschlossen!

---

## Referenzen

- **Master Reference**: [AGENTS.md](../AGENTS.md)
- **ID-Semantik**: [docs/architecture/id-semantics.md](../docs/architecture/id-semantics.md)
- **ID-Propagation**: [docs/architecture/id-propagation.md](../docs/architecture/id-propagation.md)
- **Tool-Contracts**: [docs/agents/tool-contracts.md](../docs/agents/tool-contracts.md)
- **Graph I/O Specs**: [ai_core/graphs/io.py](../ai_core/graphs/io.py)

---

**Version**: 2.0
**Erstellt**: 2026-01-08
**Abgeschlossen**: 2026-01-09
**Status**: ✅ **VOLLSTÄNDIG ABGESCHLOSSEN** - Alle 6 Phasen implementiert und getestet
**Breaking**: Ja (Pre-MVP erlaubt) - 9 Breaking Changes implementiert
**Tests**: Alle 1520 Tests bestehen (1396 fast + 124 slow)
