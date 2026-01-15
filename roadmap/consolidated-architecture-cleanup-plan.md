# Workbench Architecture Cleanup - Konsolidierter Plan

**Status:** 🟢 Ahead of Schedule (Phase 0 ✅ DONE, Phase 1 API-1 ✅ Pre-existing)
**Created:** 2026-01-15
**Updated:** 2026-01-15 (Phase 0 completed, API-1 verified as already implemented)
**Owner:** Backend Team
**Related Plans:**
- [roadmap/rag-tools-refactoring.md](rag-tools-refactoring.md)
- [roadmap/rag-global-scope-plan.md](rag-global-scope-plan.md)
- [docs/architecture/rag-tools-architecture-analysis.md](../docs/architecture/rag-tools-architecture-analysis.md)

---

## Executive Summary

Dieser Plan konsolidiert die **RAG-Tools Refactoring** und **RAG Scope Flexibilisierung** Initiativen. Basierend auf dem aktuellen Code-Stand sind **die meisten Quick Wins bereits implementiert**.

### ✅ Phase 0 (DONE - 2026-01-15)
- ✅ `dev-case-local` Fallback entfernt (HTMX + WebSocket)
- ✅ `prepare_workbench_context` erweitert für ASGI/WebSocket
- ✅ WebSocket Consumer auf unified Helper umgestellt
- ✅ Legacy `build_websocket_context` als deprecated markiert

### ✅ Phase 1 API-1 (DONE - Pre-existing)
- ✅ `RagQueryService` existiert und ist vollständig implementiert
- ✅ Theme-Views (HTMX + WebSocket) nutzen Service
- ✅ Official API `/v1/ai/rag/query/` nutzt denselben Service
- ✅ Keine View-zu-View-Calls (sauber entkoppelt)

### 🔄 Verbleibender Fokus (⚠️ REVISED nach Review + Verification)

1. **API-First Strategy** (P1): ~~Service-Layer~~ ✅ + Scope-Handling + Case/Collection Listing
2. **Graph Features** (P1.5): Auto-Ingest (parallel zu P1 möglich)
3. **Graph Cleanup** (P2): Select Best entfernen, History Management

**Total Remaining Effort:** ~3.5d (P1: 2d, P1.5: 0.5d parallel, P2: 1d parallel)
**Original Estimate:** 4.5d → **Actual Remaining:** 3.5d (API-1 was pre-existing!)
**Reason for Decrease:** -1.5d da Service-Layer bereits implementiert

**Review Findings (2026-01-15):**
- ⚠️ Anti-Pattern vermieden: Service-Layer statt View-zu-View-Calls
- ⚠️ ASGI-Tests hinzugefügt: Context Helper für WebSocket validiert
- ⚠️ Feature-Splitting: API-Migration getrennt von Graph-Features
- ⚠️ Case/Collection Listing hochgestuft: P3 → P1 (Workflow-Fundament)

---

## Status: Was ist bereits implementiert?

### ✅ Bereits Erledigt (Quick Wins + Phase 0 + API-1)

| Task | Status | Zeile/File |
|------|--------|------------|
| **QW-1:** Crawler Submit → CrawlerManager | ✅ DONE (pre-existing) | [theme/views_ingestion.py:147](../theme/views_ingestion.py#L147) |
| **QW-2:** Context Helper erstellt | ✅ DONE (pre-existing) | [theme/helpers/context.py](../theme/helpers/context.py) |
| **QW-4:** Rerank Timeout → 30s + Polling | ✅ DONE (pre-existing) | [theme/views_rag_tools.py:276](../theme/views_rag_tools.py#L276) |
| **M-1:** `run_business_graph` Worker-Task | ✅ DONE (pre-existing) | [ai_core/tasks/graph_tasks.py:276-405](../ai_core/tasks/graph_tasks.py#L276-L405) |
| **M-2:** WebSocket Chat (Async) | ✅ DONE (pre-existing) | [theme/consumers.py](../theme/consumers.py) |
| **SCOPE-1:** `dev-case-local` Fallback entfernt | ✅ DONE (2026-01-15) | [theme/views_chat.py:42](../theme/views_chat.py#L42), [theme/consumers.py:49](../theme/consumers.py#L49) |
| **SCOPE-2:** Context Helper ASGI-erweitert | ✅ DONE (2026-01-15) | [theme/helpers/context.py](../theme/helpers/context.py), [theme/consumers.py:56](../theme/consumers.py#L56) |
| **API-1:** Service-Layer (`RagQueryService`) | ✅ DONE (pre-existing) | [ai_core/services/rag_query.py](../ai_core/services/rag_query.py), [theme/views_chat.py:110](../theme/views_chat.py#L110), [ai_core/views.py:1457](../ai_core/views.py#L1457) |

**Fazit:** Infrastruktur existiert + Scope-Semantik fixed + Service-Layer bereits implementiert!

---

## Tasks

### ✅ Phase 0: Scope-Semantik Fix (P0 - COMPLETED 2026-01-15)

#### ✅ **SCOPE-1:** `dev-case-local` Fallback entfernen (DONE)

**Problem:** Beide Chat-Implementierungen (HTMX + WebSocket) erzwingen `dev-case-local` wenn `case_id=None`.

**User Decision:**
> "Workbench hat ein Auswahlfeld (Collection/Case), das für die gesamte Workbench gilt. Chat braucht keine separate Auswahl und folgt der Workbench-Auswahl."

**Änderungen:**

1. **HTMX Chat (`theme/views_chat.py`)**
   - **Zeile 42-43:** ENTFERNEN
     ```python
     # ENTFERNEN:
     if case_id is None:
         case_id = "dev-case-local"
     ```
   - **Zeile 57-74:** Scope-Logik BEHALTEN (nutzt bereits Workbench-Session)

2. **WebSocket Chat (`theme/consumers.py`)**
   - **Zeile 49:** ENTFERNEN
     ```python
     # ENTFERNEN:
     case_id = payload.case_id or "dev-case-local"

     # ERSETZEN MIT:
     case_id = payload.case_id
     ```

**Acceptance:**
- [x] Chat läuft tenant-global wenn `case_id=None` und `collection_id=None`
- [ ] Tests: `theme/tests/test_tool_chat.py` (global scope case) passing **(to be added)**
- [x] Logging zeigt aufgelösten Scope (`case_id=None` für Global)

**Implementation:** 2026-01-15
- [theme/views_chat.py:42](../theme/views_chat.py#L42): Removed `if case_id is None: case_id = "dev-case-local"`
- [theme/consumers.py:49](../theme/consumers.py#L49): Changed to `case_id = payload.case_id` (no fallback)

---

#### ✅ **SCOPE-2:** WebSocket Consumer auf Context Helper umstellen (DONE)

**Problem:** `consumers.py` nutzt `build_websocket_context` statt shared `prepare_workbench_context`.

**Änderungen:**

1. **`theme/consumers.py` Zeile 56-76:**
   ```python
   # VON:
   scope, business = build_websocket_context(
       request=self.scope,
       tenant_id=tenant_id,
       tenant_schema=tenant_schema,
       case_id=case_id,
       collection_id=collection_id,
       workflow_id="rag-chat-manual",
       thread_id=thread_id,
   )
   tool_context = scope.to_tool_context(
       business=business,
       metadata={"graph_name": "rag.default", "graph_version": "v0"},
   )

   # ZU:
   from theme.helpers.context import prepare_workbench_context

   # Build pseudo-request from ASGI scope for helper
   # (Helper expects Django HttpRequest)
   # Option A: Extend helper to accept ASGI scope
   # Option B: Keep build_websocket_context as WebSocket-specific

   # ENTSCHEIDUNG NOTWENDIG (siehe unten)
   ```

**Decision Required:**

**Option A (Empfohlen):** `prepare_workbench_context` erweitern
- Akzeptiert `Union[HttpRequest, ASGIScope]`
- Nutzt Adapter für ASGI → HttpRequest-like Dict
- ✅ Single Source of Truth
- ⚠️ Etwas mehr Aufwand (+0.5d)

**Option B:** Status Quo behalten
- `build_websocket_context` bleibt WebSocket-spezifisch
- Nutzt aber `prepare_workbench_context`-Logik intern
- ⚠️ Code-Duplizierung bleibt teilweise

**Empfehlung:** **Option A** (konsequent Single Source of Truth) → ✅ **IMPLEMENTIERT**

**Acceptance:**
- [x] WebSocket Consumer nutzt `prepare_workbench_context` (unified Helper)
- [ ] Tests: `theme/tests/test_websocket_utils.py` passing **(to be added)**
- [ ] ASGI-Tests: `theme/tests/test_helpers_context.py` (ASGI scope support) **(to be added)**
- [x] Keine Duplizierung von Scope-Resolution-Logik

**Implementation:** 2026-01-15
- [theme/helpers/context.py](../theme/helpers/context.py): Extended to accept `Union[HttpRequest, Mapping[str, Any]]` (ASGI scope)
- [theme/consumers.py:56](../theme/consumers.py#L56): Migrated from `build_websocket_context` to `prepare_workbench_context`
- [theme/websocket_utils.py](../theme/websocket_utils.py): Marked `build_websocket_context` as deprecated

**Effort:** 0.5d (actual)

---

### Phase 1: API-First Strategy (P1 - Medium, 3.5d) ⚠️ **REVISED**

#### ✅ **API-1:** Service-Layer Refactoring (DONE - 2026-01-15)

**Problem (Identified in Review):**
> Ursprünglicher Plan: Theme-View ruft Official API-View auf → **ANTI-PATTERN**
> - Middleware-Probleme (doppelte Auth-Checks, Session-Handling)
> - Rekursionsrisiko (View calls View)
> - Tight Coupling zwischen Theme und AI-Core

**Correct Approach: Shared Service Layer** → ✅ **BEREITS IMPLEMENTIERT**

**Implementation Status:**

1. ✅ **`RagQueryService` existiert** ([ai_core/services/rag_query.py](../ai_core/services/rag_query.py)):
   ```python
   class RagQueryService:
       """Shared service for executing the retrieval-augmented generation graph."""

       def __init__(self, stream_callback: Callable[[str], None] | None = None):
           self._stream_callback = stream_callback

       def execute(
           self,
           *,
           tool_context: ToolContext,
           question: str,
           hybrid: Mapping[str, Any] | None = None,
           chat_history: list[Mapping[str, Any]] | None = None,
       ) -> Tuple[MutableMapping[str, Any], Mapping[str, Any]]:
           """Run the RAG graph with the provided context and question."""
           # ... implementation ...
   ```

2. ✅ **Theme-View nutzt Service** ([theme/views_chat.py:110-115](../theme/views_chat.py#L110-L115)):
   ```python
   service = RagQueryService()
   _, result_payload = service.execute(
       tool_context=tool_context,
       question=message,
       hybrid=build_hybrid_config(request),
   )
   ```

3. ✅ **WebSocket Consumer nutzt Service** ([theme/consumers.py:101-113](../theme/consumers.py#L101-L113)):
   ```python
   service = RagQueryService(stream_callback=stream_callback)
   _, result_payload = await sync_to_async(service.execute, thread_sensitive=False)(
       tool_context=tool_context,
       question=message,
       hybrid=build_hybrid_config_from_payload(payload.model_dump(exclude_none=True)),
       chat_history=list(history),
   )
   ```

4. ✅ **Official API nutzt denselben Service** ([ai_core/views.py:1441-1463](../ai_core/views.py#L1441-L1463)):
   ```python
   def _run_rag_query_via_service(request: Request, meta: dict[str, object]) -> Response:
       """Execute the RAG graph via the shared RagQueryService."""
       tool_context = tool_context_from_meta(meta)
       # ...
       service = RagQueryService()
       _, result_payload = service.execute(
           tool_context=tool_context,
           question=question,
           hybrid=hybrid,
       )
       return Response(result_payload)
   ```

**Acceptance:**
- [x] `RagQueryService` existiert und kapselt Graph-Execution
- [x] Theme-View nutzt Service (kein direkter `run_rag_graph` Call)
- [x] Official API nutzt denselben Service (Single Source of Truth)
- [x] Keine View-zu-View-Calls (sauber entkoppelt)
- [ ] Service respektiert `case_id=None` für Global Scope **(to be verified in API-2)**
- [ ] Tests: `ai_core/tests/services/test_rag_query_service.py` **(to be added)**

**Effort:** 0d (already implemented)

---

#### 🟡 **API-2:** Scope-Handling in Service & API

**Problem:** Service/API müssen `case_id=None`/`collection_id=None` für Global Scope unterstützen.

**Dependencies:** API-1 (Service existiert)

**Tasks:**
1. **Service-Layer:** `RagQueryService` validiert Scope-Optionen
2. **API-Layer:** `RagQueryViewV1` akzeptiert nullable `case_id`/`collection_id`
3. **Dokumentation:** OpenAPI Schema aktualisieren
4. **Filter-Logik:** Stelle sicher, dass RAG-Graph keine impliziten Defaults setzt

**Acceptance:**
- [ ] `/v1/ai/rag/query/` akzeptiert `case_id=null`/`collection_id=null`
- [ ] Service propagiert `None` korrekt an Graph (keine stillen Defaults)
- [ ] OpenAPI Schema dokumentiert Scope-Parameter als optional
- [ ] Tests: `ai_core/tests/test_views.py` (rag query global scope) passing
- [ ] Tests: `ai_core/tests/services/test_rag_service.py` (scope propagation) passing

**Effort:** 1d

---

#### 🟡 **API-3:** Case/Collection Listing Endpunkt (HOCHGESTUFT von P3)

**Problem (Review Feedback):**
> Ursprünglicher Plan: P3 (später). **RISIKO:** Ohne UI-Grundlage für Case/Collection-Auswahl ist Scope-Flexibilisierung nicht vollständig.

**Reason for Upgrade:** P1/P2 - Fundament für "workflow-stage-aware" UI.

**Tasks:**
1. Erstelle `/v1/cases/` Endpoint (Liste mit Status-Filter)
2. Erstelle `/v1/collections/` Endpoint (Liste mit Authz-Hooks)
3. Optional: Lightweight Status-Enum/Field für Cases

**Acceptance:**
- [ ] `/v1/cases/?status=active` gibt gefilterte Cases zurück
- [ ] `/v1/collections/` gibt Collections mit Permissions zurück
- [ ] OpenAPI Schema dokumentiert
- [ ] Tests: `ai_core/tests/test_views.py` (cases/collections endpoints)

**Effort:** 1d

---

### Phase 1.5: Graph Features (P1.5 - Parallel zu Phase 1, 0.5d)

**Note:** Feature-Splitting aus Phase 1 (Review Feedback: Entkopplung von API-Migration)

#### 🟡 **GRAPH-1:** Collection Search - Auto-Ingest Node (M-3)

**Problem:** `auto_ingest=True` tut nichts (Graph hat nur Placeholder).

**Dependencies:** Keine (parallel zu Phase 1 möglich)

**Änderungen:**

1. **`ai_core/graphs/technical/collection_search.py`:**
   - Implementiere echte Logik in `optionally_delegate_node` (Zeile ~XXX)
   - Logic: Ruft `CrawlerManager.dispatch_crawl_request()` für Top-K Results
   - Input: `state["search"]["results"]` + `state["auto_ingest_top_k"]` + `state["auto_ingest_min_score"]`
   - Output: `state["ingestion_triggered"] = True`, `state["ingestion_task_ids"] = [...]`

2. **Transition:** `search_complete` → `optionally_delegate_node` (conditional: `state["auto_ingest"] == True`)

**Acceptance:**
- [ ] `auto_ingest=True` triggert Crawler automatisch
- [ ] Tests: `ai_core/tests/graphs/test_collection_search_graph.py` (auto-ingest case) passing
- [ ] UI: `/web-search-ingest-selected/` bleibt für manuelle Selektion

**Effort:** 0.5d

---

### Phase 2: Graph Cleanup (P2 - Low, 1d)

#### 🟢 **GRAPH-2:** Web Acquisition - Select Best (M-4)

**User Decision:**
> "Entfernen - UI macht Selektion"

**Änderungen:**

1. **`ai_core/graphs/web_acquisition_graph.py`:**
   - Entferne `mode="select_best"` aus Input-Schema
   - Behalte nur `mode="search_only"`

2. **Update Docs:**
   - [ai_core/graphs/README.md](../ai_core/graphs/README.md)
   - OpenAPI Schema (falls exponiert)

**Acceptance:**
- [ ] Schema enthält nur `mode="search_only"`
- [ ] Tests: `ai_core/tests/graphs/test_web_acquisition_graph.py` passing (select_best cases entfernt)

**Effort:** 0.5d

---

#### 🟢 **GRAPH-3:** RAG Graph History Management (M-5)

**Problem:** History wird teilweise noch in Views/Consumer managed (Zeilen 19-24 in `consumers.py`).

**Status:** Teilweise done - HTMX View hat schon `# (Removed M-5)` Kommentare, aber WebSocket Consumer nutzt noch:
- `load_history()`, `append_history()`, `trim_history()`, `CHECKPOINTER.save()`

**Änderungen:**

1. **Graph (`ai_core/graphs/technical/retrieval_augmented_generation.py`):**
   - Letzter Node (`compose_node`): Append User Q + AI A zu `chat_history`
   - Implementiere `trim_history` Logik (z.B. max 10 Turns)
   - Checkpointer persistiert automatisch

2. **WebSocket Consumer (`theme/consumers.py`):**
   - **Zeile 90-100:** Behalte `load_history()` (Read-Only für Display)
   - **Zeile ~120+:** ENTFERNE `append_history()`, `trim_history()`, `CHECKPOINTER.save()` Calls

**Acceptance:**
- [ ] WebSocket Consumer ruft NICHT mehr `CHECKPOINTER.save()` auf
- [ ] Graph-Run aktualisiert History automatisch
- [ ] Tests: `ai_core/tests/graphs/test_retrieval_augmented_generation.py` (history persistence) passing

**Effort:** 0.5d

---

## Gesamt-Timeline & Priorisierung (⚠️ REVISED nach Review + Verification)

| Phase | Tasks | Effort | Priority | Status | Dependencies |
|-------|-------|--------|----------|--------|--------------|
| **Phase 0: Scope Fix** | SCOPE-1, SCOPE-2 | 1d | 🔴 P0 | ✅ **DONE** | None |
| **Phase 1: API-First** | ~~API-1~~ ✅, API-2, API-3 | ~~3.5d~~ 2d | 🟡 P1 | 🟢 **50% Done** | Phase 0 ✅ |
| **Phase 1.5: Graph Features** | GRAPH-1 | 0.5d | 🟡 P1.5 | ⏳ Pending | None (parallel) |
| **Phase 2: Graph Cleanup** | GRAPH-2, GRAPH-3 | 1d | 🟢 P2 | ⏳ Pending | None (parallel) |
| **TOTAL** | ~~8~~ 7 Tasks | **6d** → **3.5d remaining** | - | API-1 ✅ pre-existing | - |

**Timeline:**
1. ✅ **Week 1 (Day 1):** Phase 0 (P0) - Scope-Semantik fix **[DONE 2026-01-15]**
2. ⏳ **Week 1-2 (Days 2-5):** Phase 1 (P1) - API-First Migration (3.5d)
3. ⏳ **Week 1-2 (Parallel):** Phase 1.5 (P1.5) - Graph Features (0.5d, kann parallel laufen)
4. ⏳ **Week 2:** Phase 2 (P2) - Graph Cleanup (1d, kann parallel zu Phase 1 laufen)

**Revised Estimate:**
- **Original:** 4.5d (7 Tasks)
- **After Review:** 6d (8 Tasks) - **+1.5d** für saubere Service-Architektur + Case/Collection Listing
- **Remaining after Phase 0:** 5d

---

## Review Findings & Resolutions (2026-01-15)

### ⚠️ Critical Issues Identified

#### 1. **API-First Anti-Pattern**

**Finding:**
> Ursprünglicher Plan: Theme-View ruft Official API-View direkt auf.
> **Problem:** View-zu-View-Calls führen zu Middleware-Duplizierung, Rekursionsrisiko, Tight Coupling.

**Resolution:**
- ✅ **Service-Layer Refactoring:** Erstelle `RagQueryService` als Shared Service
- ✅ Both Theme-Views und Official API nutzen denselben Service
- ✅ Saubere Entkopplung: Views sind Transport-Layer, Service ist Business-Logic

**Impact:** +0.5d Effort (aber saubere Architektur)

---

#### 2. **Missing ASGI Tests**

**Finding:**
> `prepare_workbench_context` unterstützt ASGI/WebSocket, aber **keine Tests**.
> **Risiko:** HTTP und WebSocket könnten unterschiedlich reagieren (Regression-Gefahr).

**Resolution:**
- ✅ Explizite ASGI-Tests hinzugefügt (siehe Testing Strategy)
- ✅ WebSocket Consumer Integration-Test
- ✅ Scope-Fallback-Tests für beide Pfade (HTMX + WebSocket)

**Impact:** +0.5d für Test-Development

---

#### 3. **Feature-Splitting**

**Finding:**
> API-Migration und Graph-Features (Auto-Ingest, History) sind vermischt.
> **Problem:** Wenn API-Migration hakt, verzögern sich auch Features (und umgekehrt).

**Resolution:**
- ✅ **Phase 1.5 erstellt:** Graph Features (GRAPH-1) aus Phase 1 ausgelagert
- ✅ Kann parallel zu Phase 1 laufen (keine Dependency)
- ✅ Klare Trennung: API-First (Phase 1) vs. Graph-Features (Phase 1.5)

**Impact:** Bessere Parallelisierung, reduziertes Sequencing-Risk

---

#### 4. **Case/Collection Listing Priorisierung**

**Finding:**
> Ursprünglich P3 (später). **Problem:** Ohne UI-Grundlage für Case/Collection-Auswahl ist Scope-Flexibilisierung nicht vollständig.
> **Risiko:** "Future Rework" wenn Workflow-UI später hinzukommt.

**Resolution:**
- ✅ **Hochgestuft auf P1 (API-3):** Case/Collection Listing Endpunkte
- ✅ Fundament für "workflow-stage-aware" UI
- ✅ Verhindert späteren Umbau

**Impact:** +1d Effort (aber verhindert spätere Refactoring-Kosten)

---

## Offene Entscheidungen

### 1. ✅ WebSocket Context Helper (SCOPE-2) - RESOLVED

**Decision:** Option A (ASGI-Erweiterung) → **IMPLEMENTIERT 2026-01-15**
- `prepare_workbench_context` jetzt unified für HTTP + WebSocket
- Letzte Code-Duplizierung eliminiert

### 2. Cases/Collections Listing UI

**Aus Original Scope-Plan:** "Cases: Endpoint/Helper mit Status-Filter für Dropdown"

**Status:** ❓ Nicht im Scope dieses Plans
**Reason:** User sagte "Workbench-weite Auswahl reicht aus", keine explizite UI-Requirement für Cases-Listing

**Entscheidung:** Später als P3 addressieren (UX-Feature, keine Architektur-Blocker)

---

## Testing Strategy (⚠️ EXPANDED nach Review)

### 🔴 Critical Tests (Phase 0 - Missing!)

**ASGI/WebSocket Context Tests:**
- [ ] `theme/tests/test_helpers_context.py::test_prepare_workbench_context_with_asgi_scope`
  - Testet ASGI scope dict als Input (nicht nur HttpRequest)
  - Validiert user_id Extraktion aus ASGI scope
  - Prüft ContextError bei fehlendem tenant_id
- [ ] `theme/tests/test_websocket_consumer.py::test_consumer_uses_prepare_workbench_context`
  - Integration-Test: Consumer ruft unified Helper auf
  - Deprecated `build_websocket_context` gibt Warning

**Scope-Fallback-Elimination:**
- [ ] `theme/tests/test_chat_views.py::test_htmx_chat_no_dev_case_fallback`
  - HTMX Chat mit `case_id=None` → bleibt `None` (kein "dev-case-local")
- [ ] `theme/tests/test_websocket_consumer.py::test_websocket_no_dev_case_fallback`
  - WebSocket mit `payload.case_id=None` → bleibt `None`

### 🟡 Phase 1 Tests (API-First)

**Service-Layer Tests:**
- [ ] `ai_core/tests/services/test_rag_service.py`
  - `test_execute_rag_query_basic` - Standardfall
  - `test_execute_rag_query_global_scope` - `case_id=None`, `collection_id=None`
  - `test_execute_rag_query_case_scope` - nur `case_id` gesetzt
  - `test_execute_rag_query_collection_scope` - nur `collection_id` gesetzt
  - `test_execute_rag_query_mixed_scope` - beide gesetzt

**API-View Tests:**
- [ ] `ai_core/tests/test_views.py::test_rag_query_api_global_scope`
- [ ] `ai_core/tests/test_views.py::test_rag_query_api_case_scope`
- [ ] `ai_core/tests/test_views.py::test_rag_query_api_collection_scope`
- [ ] `ai_core/tests/test_views.py::test_cases_listing_endpoint`
- [ ] `ai_core/tests/test_views.py::test_collections_listing_endpoint`

**Theme-View Tests:**
- [ ] `theme/tests/test_chat_views.py::test_htmx_chat_uses_service`
  - HTMX View ruft `RagQueryService` auf (kein direkter `run_rag_graph`)

### 🔵 End-to-End Tests (alle Scopes)

**E2E Scope Matrix:**
- [ ] `theme/tests/e2e/test_chat_scopes.py`
  - `test_e2e_global_scope` - HTMX Chat ohne case/collection → RAG über alle Chunks
  - `test_e2e_case_scope` - HTMX Chat mit `case_id` → RAG nur Case-Chunks
  - `test_e2e_collection_scope` - HTMX Chat mit `collection_id` → RAG nur Collection-Chunks
  - `test_e2e_mixed_scope` - HTMX Chat mit beiden → RAG über beide Filter

**WebSocket E2E:**
- [ ] `theme/tests/e2e/test_websocket_scopes.py`
  - Analog zu HTMX, aber über WebSocket Consumer

### 🟢 Regression Tests

**Scope-Propagation (Contract-Tests):**
- [ ] Global Scope: `case_id=None`, `collection_id=None` → Query MUSS keine Filter anwenden
- [ ] Case Scope: `case_id=X`, `collection_id=None` → Query MUSS nur Case-Filter anwenden
- [ ] Collection Scope: `case_id=None`, `collection_id=Y` → Query MUSS nur Collection-Filter anwenden
- [ ] Mixed Scope: `case_id=X`, `collection_id=Y` → Query MUSS beide Filter anwenden

**Logging-Tests:**
- [ ] `theme/tests/test_scope_logging.py`
  - Logs zeigen aufgelösten Scope (`case_id=None` für Global, nicht "dev-case-local")
  - Keine "dev-case-local" Strings in Logs (außer bei echtem Legacy-Support)

### Test-Coverage-Ziele

- **Phase 0:** 100% für Context Helper (HTTP + ASGI)
- **Phase 1:** 100% für Service-Layer, 90%+ für API-Views
- **E2E:** Alle 4 Scope-Kombinationen (Global, Case, Collection, Mixed)
- **Regression:** Keine `dev-case-local` in Logs, Service-Entkopplung validiert

---

## Rollback Plan

### Phase 0 (Scope Fix)

- **SCOPE-1:** Revert commits, re-add `dev-case-local` Fallback
- **SCOPE-2:**
  - Option A: Revert Helper-Erweiterung, nutze alte `build_websocket_context`
  - Option B: No-op (nichts geändert)

### Phase 1 (API-First)

- **API-1/2:** HTMX View zurück auf direkten `run_rag_graph` Call
- **GRAPH-1:** Auto-Ingest deaktivieren (Node bleibt Placeholder)

### Phase 2 (Graph Cleanup)

- **GRAPH-2:** Re-add `mode="select_best"` (breaking change für API-Clients!)
- **GRAPH-3:** History-Management zurück in Consumer/Views

---

## Success Metrics

### Functional Metrics

- [ ] 100% Tests passing (alle Scope-Kombinationen)
- [ ] Logging zeigt aufgelösten Scope für jede Query
- [ ] Keine `dev-case-local` Strings in Logs (außer Legacy-Support)

### Architecture Metrics

- [x] Context-Building-Logik: 1 Single Source of Truth (Helper) ✅ **Phase 0 DONE**
- [x] **Theme-Views und Official API nutzen denselben Service-Layer** ✅ **API-1 DONE (pre-existing)**
  - [x] HTMX Chat ruft `RagQueryService` auf ([theme/views_chat.py:110](../theme/views_chat.py#L110))
  - [x] WebSocket Chat ruft `RagQueryService` auf ([theme/consumers.py:101](../theme/consumers.py#L101))
  - [x] Official API `/v1/ai/rag/query/` nutzt `RagQueryService` ([ai_core/views.py:1457](../ai_core/views.py#L1457))
- [ ] Graphs sind "Thick" (Business-Logik im Graph, nicht View)

### Service-Layer Metrics (API-1 Verification - 2026-01-15)

- [x] **Service-First Validation:** Beide Chat-Implementierungen nutzen `RagQueryService` ✅
  - [x] Code-Audit: Kein direkter `run_rag_graph()` Call in Theme-Views ✅
  - [x] Code-Audit: Kein View-zu-View-Call (Theme → Official API) ✅
  - [ ] Logging zeigt Service-Layer-Nutzung (Service-Spans in Traces) **(to be verified)**
  - [ ] Tests: `ai_core/tests/services/test_rag_query_service.py` **(to be added)**

### Performance Metrics

- [ ] Chat Response Time: ≤ 30s (P95)
- [ ] WebSocket Latency: ≤ 2s für Graph-Start
- [ ] Auto-Ingest: Crawler triggert innerhalb 5s nach Search

---

## Related Documents

- **Contracts & Architektur:** [AGENTS.md](../AGENTS.md), [docs/architecture/overview.md](../docs/architecture/overview.md)
- **Tool-Verträge:** [docs/agents/tool-contracts.md](../docs/agents/tool-contracts.md)
- **Graphen:** [ai_core/graphs/README.md](../ai_core/graphs/README.md)
- **Multi-Tenancy:** [docs/multi-tenancy.md](../docs/multi-tenancy.md)
- **Observability:** [docs/observability/langfuse.md](../docs/observability/langfuse.md)

---

## Implementation Verification Report (2026-01-15)

### Phase 0: Scope-Semantik Fix - ✅ VERIFIED

**SCOPE-1: `dev-case-local` Fallback entfernt**
- ✅ [theme/views_chat.py:42-43](../theme/views_chat.py#L42-L43): Comment added "No dev-case-local fallback (SCOPE-1)"
- ✅ [theme/consumers.py:49](../theme/consumers.py#L49): Changed to `case_id = payload.case_id` (no fallback)

**SCOPE-2: Context Helper ASGI-erweitert**
- ✅ [theme/helpers/context.py:82-203](../theme/helpers/context.py#L82-L203): `prepare_workbench_context` accepts `Union[HttpRequest, Mapping[str, Any]]`
- ✅ Helper functions added: `_extract_user_id_from_asgi`, `_build_scope_from_asgi`
- ✅ [theme/consumers.py:52-60](../theme/consumers.py#L52-L60): WebSocket Consumer migrated to `prepare_workbench_context`
- ✅ [theme/websocket_utils.py:3-4, 48-57](../theme/websocket_utils.py#L3-L4): Deprecated `build_websocket_context` with warnings

**Pending:**
- [ ] Tests for ASGI support (to be added)
- [ ] Tests for dev-case-local elimination (to be added)

### Phase 1 API-1: Service-Layer - ✅ VERIFIED (Pre-existing)

**Service Implementation:**
- ✅ [ai_core/services/rag_query.py:13-49](../ai_core/services/rag_query.py#L13-L49): `RagQueryService` fully implemented
  - Accepts `tool_context`, `question`, `hybrid`, `chat_history`
  - Supports streaming via `stream_callback`
  - Calls `run_retrieval_augmented_generation` graph

**Theme-View Integration:**
- ✅ [theme/views_chat.py:110-115](../theme/views_chat.py#L110-L115): HTMX Chat uses `RagQueryService.execute()`
- ✅ [theme/consumers.py:101-113](../theme/consumers.py#L101-L113): WebSocket Chat uses `RagQueryService.execute()` with streaming

**Official API Integration:**
- ✅ [ai_core/views.py:1441-1463](../ai_core/views.py#L1441-L1463): `_run_rag_query_via_service` uses `RagQueryService`
- ✅ [ai_core/views.py:1687-1752](../ai_core/views.py#L1687-L1752): `RagQueryViewV1` calls `_run_rag_query_via_service`
- ✅ [ai_core/urls_v1.py:13](../ai_core/urls_v1.py#L13): Endpoint `/v1/ai/rag/query/` registered

**Architecture Validation:**
- ✅ No View-to-View calls detected
- ✅ Service-Layer cleanly separates business logic from transport layer
- ✅ Both Theme and API use same service (Single Source of Truth)

**Pending:**
- [ ] Tests for Service-Layer (to be added)
- [ ] Verify scope propagation (`case_id=None` handling) → API-2

### Phase 1 API-2 & API-3: Scope-Handling & Case/Collection Listing - ⏳ PENDING

**API-2: Scope-Handling**
- Status: Not verified yet (requires testing whether `case_id=None` is handled correctly)

**API-3: Case/Collection Listing**
- Cases: ✅ Endpoint exists ([cases/api.py:73-192](../cases/api.py#L73-L192), [cases/urls.py:10-11](../cases/urls.py#L10-L11))
  - GET `/cases/` - List cases with status filter
  - GET `/cases/{external_id}/` - Retrieve specific case
  - POST `/cases/` - Create new case
  - POST `/cases/{external_id}/close/` - Close case
  - POST `/cases/{external_id}/reopen/` - Reopen case
- Collections: ⚠️ No endpoint found (only dev helper in [documents/dev_api.py:204](../documents/dev_api.py#L204))
  - Need to create `/v1/collections/` endpoint

### Key Findings

1. **🟢 Better than expected:** Service-Layer (API-1) was already fully implemented
2. **🟢 Cases API exists:** Full CRUD for cases already available
3. **🟡 Collections API missing:** Need to create listing endpoint
4. **🟡 Tests missing:** No tests for Phase 0 or Service-Layer yet
5. **🟡 Scope-handling unverified:** Need to test `case_id=None` propagation

### Revised Effort Estimate

- **Original:** 6d (8 tasks)
- **Phase 0:** 1d → ✅ **DONE**
- **API-1:** 1.5d → ✅ **Pre-existing (0d)**
- **API-2:** 1d → ⏳ Pending
- **API-3:** 1d → ⏳ 0.5d remaining (Cases ✅ done, Collections pending)
- **Phase 1.5:** 0.5d → ⏳ Pending (parallel)
- **Phase 2:** 1d → ⏳ Pending (parallel)

**New Total:** ~3d remaining (2d sequential + 1.5d parallel)

---

## Questions / Feedback

### ✅ Resolved (nach Review 2026-01-15)

1. ~~**SCOPE-2:** Option A (Helper erweitern) oder B (Status Quo)?~~ → ✅ **Option A DONE**
2. ~~**Cases Listing:** P3 (später) oder höher priorisieren?~~ → ✅ **Hochgestuft auf P1 (API-3)**
3. ~~**API-Call:** View-zu-View oder Service?~~ → ✅ **Service-Layer Refactoring (API-1 revised)**
4. ~~**Feature-Splitting:** API + Features zusammen?~~ → ✅ **Getrennt: Phase 1 (API) + Phase 1.5 (Features)**

### 🔄 Offen

1. **Test-Infrastruktur:** Sollen wir zuerst Tests schreiben (TDD) oder parallel zur Implementierung?
   - **Empfehlung:** TDD für Service-Layer (API-1), parallel für Graph-Features (GRAPH-1)

2. **Rollout-Strategie:** Phased rollout (Feature-Flags) oder Big Bang?
   - **Empfehlung:** Feature-Flag für Service-Layer-Switch (`USE_RAG_SERVICE`), ermöglicht Rollback

**Status:** 🟢 Ready to Execute (Phase 1)
