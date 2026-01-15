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

1. **API-First Strategy** (P1): DONE (Scope-Handling + Case/Collection Listing complete)
2. **Graph Features** (P1.5): Auto-Ingest done
3. **Graph Cleanup** (P2): Select Best removed; history tests done

**Total Remaining Effort:** 0d
**Original Estimate:** 4.5d → **Actual Remaining:** 3.5d (API-1 was pre-existing!)
**Reason for Decrease:** API-2/3 abgeschlossen, GRAPH-2 erledigt

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
- [x] Tests: `theme/tests/test_tool_chat.py::test_tool_chat_allows_global_scope_without_case` passing
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
- [x] Tests: `theme/tests/test_websocket_utils.py` passing
- [x] ASGI-Tests: `theme/tests/test_workbench_context.py` (ASGI scope support)
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
- [x] Service respektiert `case_id=None` fǬr Global Scope
- [x] Tests: `ai_core/tests/services/test_rag_query_service.py`

**Effort:** 0d (already implemented)

---

#### DONE **API-2:** Scope-Handling in Service & API

**Problem:** Service/API müssen `case_id=None`/`collection_id=None` für Global Scope unterstützen.

**Dependencies:** API-1 (Service existiert)

**Tasks:**
1. **Service-Layer:** `RagQueryService` validiert Scope-Optionen
2. **API-Layer:** `RagQueryViewV1` akzeptiert nullable `case_id`/`collection_id`
3. **Dokumentation:** OpenAPI Schema aktualisieren
4. **Filter-Logik:** Stelle sicher, dass RAG-Graph keine impliziten Defaults setzt

**Acceptance:**
- [x] `/v1/ai/rag/query/` akzeptiert `case_id=null`/`collection_id=null`
- [x] Service propagiert `None` korrekt an Graph (keine stillen Defaults)
- [x] OpenAPI Schema dokumentiert Scope-Parameter als optional
- [x] Tests: `ai_core/tests/test_views.py` (rag query global scope) passing
- [x] Tests: `ai_core/tests/services/test_rag_query_service.py` (scope propagation) passing

**Effort:** 1d

---

#### DONE **API-3:** Case/Collection Listing Endpunkt (HOCHGESTUFT von P3)

**Problem (Review Feedback):**
> Ursprünglicher Plan: P3 (später). **RISIKO:** Ohne UI-Grundlage für Case/Collection-Auswahl ist Scope-Flexibilisierung nicht vollständig.

**Reason for Upgrade:** P1/P2 - Fundament für "workflow-stage-aware" UI.

**Tasks:**
1. Erstelle `/v1/cases/` Endpoint (Liste mit Status-Filter)
2. Erstelle `/v1/collections/` Endpoint (Liste mit Authz-Hooks)
3. Optional: Lightweight Status-Enum/Field für Cases

**Acceptance:**
- [x] `/v1/cases/?status=active` gibt gefilterte Cases zurǬck
- [x] `/v1/collections/` gibt Collections mit Permissions zurǬck
- [x] OpenAPI Schema dokumentiert
- [x] Tests: `ai_core/tests/test_views.py` (cases/collections endpoints)

**Effort:** 1d

---

### Phase 1.5: Graph Features (P1.5 - Parallel zu Phase 1, 0.5d)

**Note:** Feature-Splitting aus Phase 1 (Review Feedback: Entkopplung von API-Migration)

#### ✅ **GRAPH-1:** Collection Search - Auto-Ingest Node (M-3)

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
- [x] `auto_ingest=True` triggert Crawler automatisch
- [x] Tests: `ai_core/tests/graphs/test_collection_search_graph.py::TestCollectionSearchGraph::test_auto_ingest_triggers_crawler` passing
- [x] UI: `/web-search-ingest-selected/` bleibt für manuelle Selektion

**Effort:** 0.5d

---

### Phase 2: Graph Cleanup (P2 - Low, 1d)

#### DONE **GRAPH-2:** Web Acquisition - Select Best (M-4)

**User Decision:**
> "Entfernen - UI macht Selektion"

**Änderungen:**

1. **`ai_core/graphs/web_acquisition_graph.py`:**
   - ~~Entferne `mode="select_best"` aus Input-Schema~~
   - ~~Behalte nur `mode="search_only"`~~
   - `mode` Feld komplett entfernt (keine Funktionalität mehr)
   - Schema: `WebAcquisitionInputModel` mit `extra="forbid"`

2. **Caller Updates (2026-01-15):**
   - `theme/views_web_search.py`: `mode` aus `input_payload` entfernt
   - `theme/tests/test_rag_tools_view.py`: `mode` Assertion entfernt

**Acceptance:**
- [x] Schema enthält kein `mode` Feld mehr
- [x] Tests: `ai_core/tests/graphs/test_web_acquisition_graph.py` passing
- [x] Caller: `views_web_search.py` sendet kein `mode` mehr

**Effort:** 0.5d

---

#### DONE **GRAPH-3:** RAG Graph History Management (M-5)

**Problem:** History wurde teilweise noch in Views/Consumer gemanaged.

**Status:** Consumer-Cleanup erledigt; Tests fehlen noch.


**Änderungen:**

1. **Graph (`ai_core/graphs/technical/retrieval_augmented_generation.py`):**
   - Letzter Node (`compose_node`): Append User Q + AI A zu `chat_history`
   - Implementiere `trim_history` Logik (z.B. max 10 Turns)
   - Checkpointer persistiert automatisch

2. **WebSocket Consumer (`theme/consumers.py`):**
   - **Zeile 90-100:** Behalte `load_history()` (Read-Only für Display)
   - **Zeile ~120+:** ENTFERNE `append_history()`, `trim_history()`, `CHECKPOINTER.save()` Calls

**Acceptance:**
- [x] WebSocket Consumer ruft NICHT mehr `CHECKPOINTER.save()` auf
- [x] Graph-Run aktualisiert History automatisch
- [x] Tests: `ai_core/tests/test_graph_retrieval_augmented_generation.py::test_graph_persists_history_with_thread_id` passing

**Effort:** 0.5d

---

## Gesamt-Timeline & Priorisierung (⚠️ REVISED nach Review + Verification)

| Phase | Tasks | Effort | Priority | Status | Dependencies |
|-------|-------|--------|----------|--------|--------------|
| **Phase 0: Scope Fix** | SCOPE-1, SCOPE-2 | 1d | 🔴 P0 | ✅ **DONE** | None |
| **Phase 1: API-First** | ~~API-1~~ DONE, API-2, API-3 | ~~3.5d~~ 2d | P1 | DONE | Phase 0 DONE |
| **Phase 1.5: Graph Features** | GRAPH-1 | 0.5d | 🟡 P1.5 | ✅ Done | None (parallel) |
| **Phase 2: Graph Cleanup** | GRAPH-2, GRAPH-3 | 1d | P2 | ✅ Done | None (parallel) |
| **TOTAL** | ~~8~~ 7 Tasks | **6d** -> **0d remaining** | - | API-1 pre-existing | - |

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
- [x] `theme/tests/test_workbench_context.py::test_prepare_workbench_context_with_asgi_scope`
  - Testet ASGI scope dict als Input (nicht nur HttpRequest)
  - Validiert user_id Extraktion aus ASGI scope
  - Prüft ContextError bei fehlendem tenant_id
- [x] `theme/tests/test_chat_consumer.py::test_rag_chat_consumer_uses_prepare_workbench_context`
  - Integration-Test: Consumer ruft unified Helper auf
  - Deprecated `build_websocket_context` gibt Warning

**Scope-Fallback-Elimination:**
- [x] `theme/tests/test_chat_submit_global_search.py::test_chat_submit_no_dev_case_fallback`
  - HTMX Chat mit `case_id=None` → bleibt `None` (kein "dev-case-local")
- [x] `theme/tests/test_chat_consumer.py::test_rag_chat_consumer_allows_null_case_id`
  - WebSocket mit `payload.case_id=None` → bleibt `None`

### 🟡 Phase 1 Tests (API-First)

**Service-Layer Tests:**
- [x] `ai_core/tests/services/test_rag_query_service.py::test_execute_allows_global_scope_without_case_id`
- [x] Additional scope matrix tests (case/collection/mixed)

**API-View Tests:**
- [x] `ai_core/tests/test_views.py::test_rag_query_endpoint_allows_global_scope_without_case`
- [x] `ai_core/tests/test_views.py::test_rag_query_api_case_scope`
- [x] `ai_core/tests/test_views.py::test_rag_query_api_collection_scope`
- [x] `ai_core/tests/test_views.py::test_cases_listing_endpoint`
- [x] `ai_core/tests/test_views.py::test_collections_listing_endpoint_filters_by_case_membership`

**Theme-View Tests:**
- [x] `theme/tests/test_chat_views.py::test_htmx_chat_uses_service`
  - HTMX View ruft `RagQueryService` auf (kein direkter `run_rag_graph`)



