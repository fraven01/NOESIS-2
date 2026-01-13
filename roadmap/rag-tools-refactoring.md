# RAG-Tools Refactoring: Dev-Workbench Architecture Cleanup

**Status:** � In Progress (P0 Done, P1 Partial)
**Created:** 2026-01-13
**Updated:** 2026-01-13
**Analysis:** [docs/architecture/rag-tools-architecture-analysis.md](../docs/architecture/rag-tools-architecture-analysis.md)
**Owner:** Backend Team

---

## Problem Statement

Die RAG-Tools-Workbench (`/rag-tools/`) hat **kritische Architektur-Divergenzen**:

1. ❌ Views führen Business-Graphen **synchron** aus (umgehen Worker-Queue)
2. ❌ Massive **Logik-Duplizierung** (Context-Building in jeder View)
3. ❌ **Fehlende Worker-Tasks** für Business-Graphen
4. ❌ **Inkonsistente API-Nutzung** (Theme-Views vs. Official API)

**Impact:** Keine Skalierung, keine Retry-Logik, blockiert HTTP-Threads, hohe Maintenance-Kosten.

---

## Quick Wins (P0 - Critical, 1-2 Tage)

### ✅ QW-1: Crawler Submit auf CrawlerManager umstellen

**Problem:** `crawler_submit()` nutzt `run_crawler_runner()` direkt (synchron), statt `CrawlerManager` (async).

**Pointer:** `theme/views_ingestion.py:crawler_submit()` Zeile 148-153

**Change:**

```python
# VON:
result = run_crawler_runner(
    meta=meta,
    request_model=request_model,
    lifecycle_store=views._resolve_lifecycle_store(),
    graph_factory=None,
)

# ZU:
from crawler.manager import CrawlerManager
manager = CrawlerManager()
result = manager.dispatch_crawl_request(request_model, meta)
```

**Acceptance:**

- `crawler_submit` nutzt `CrawlerManager.dispatch_crawl_request()`
- Tests: `theme/tests/test_rag_tools_view.py` (crawler submit case) passing
- Konsistent mit `web_search_ingest_selected` Pattern

**Effort:** 0.5d
**Priority:** 🔴 Critical

---

### ✅ QW-2: Context Building deduplizieren

**Problem:** Jede View baut manuell `ScopeContext`, `BusinessContext`, `ToolContext` (30-50 Zeilen Duplizierung pro View).

**Pointers:**

- Duplizierung: `theme/views_chat.py:80-116`, `theme/views_web_search.py:190-208,263-280,437-457` (3x!), `theme/views_ingestion.py:107-135,216-234` (2x), `theme/views_documents.py:239-260`
- Existierende Lösung (nicht genutzt): `ai_core/views.py:_prepare_request()` Zeilen 482-780

**Implementation:**

1. Erstelle `theme/helpers/context.py::prepare_workbench_context(request)`
2. Extrahiere Shared Helper aus `_prepare_request()` ODER reuse direkt
3. Nutze in allen Theme-Views

**Acceptance:**

- `theme/helpers/context.py` existiert mit `prepare_workbench_context(request)`
- Alle Theme-Views nutzen Helper (keine manuelle Context-Building-Duplizierung)
- Tests: `theme/tests/test_rag_tools_view.py`, `theme/tests/test_chat_submit_global_search.py` passing

**Effort:** 1d
**Priority:** 🔴 Critical

---

### ✅ QW-3: Chat Submit - API-Reuse oder Service-Reuse

**Problem:** `chat_submit()` führt RAG Graph direkt aus (synchron), nutzt NICHT Official API `/v1/ai/rag/query/`.

**Pointers:**

- Theme-View: `theme/views_chat.py:chat_submit()` Zeile 73-131 (ruft `run_rag_graph()` direkt)
- Official API: `ai_core/views.py:RagQueryViewV1` Zeile 1709-1774 (nutzt `_GraphView` → `services.execute_graph`)

**Options:**

**Option A (API-Reuse, empfohlen):**

- HTMX ruft `/v1/ai/rag/query/` direkt auf
- `chat_submit` wird zu einfachem Proxy oder entfernt
- Vorteil: Single Source of Truth, keine Duplizierung

**Option B (Service-Reuse):**

- `chat_submit` nutzt `services.execute_graph` (wie `RagQueryViewV1`)
- Vorteil: Konsistent mit Official API, behält Theme-View-Logik

**Acceptance:**

- `chat_submit` nutzt entweder Official API ODER `services.execute_graph`
- Kein direkter `run_rag_graph()` Aufruf
- Tests: `theme/tests/test_chat_submit_global_search.py` passing

**Effort:** 0.5d
**Priority:** 🔴 Critical

---

### ✅ QW-4: Rerank Workflow - Timeout reduzieren + Polling

**Problem:** `start_rerank_workflow()` nutzt 120s Timeout (blockiert HTTP-Thread).

**Pointer:** `theme/views_rag_tools.py:start_rerank_workflow()` Zeile 256-261

**Change:**

- Timeout von 120s → 30s
- Neuer Polling-Endpoint: `/rag-tools/workflow-status/<task_id>/` (analog zu Ingestion Status)

**Acceptance:**

- Timeout reduziert auf 30s max
- Polling-Endpoint implementiert (returns task status + result when completed)
- Tests: `theme/tests/test_rag_tools_simulation.py` (rerank workflow case) passing

**Effort:** 0.5d
**Priority:** 🟡 Medium

---

## Mittelfristige Maßnahmen (P1 - Architektur, 3-5 Tage)

### ✅ M-1: Generic Graph Worker-Task erstellen

**Problem:** Fehlende Celery Worker-Tasks für Business-Graphen (Collection Search, Web Acquisition, Framework Analysis, RAG Query).

**Pointers:**

- Existierend: `ai_core/tasks/graph_tasks.py:run_ingestion_graph` (Zeilen 72-168)
- Fehlend: Worker für Collection Search, Web Acquisition, Framework Analysis, RAG Query

**Implementation:**

1. Erstelle `ai_core/tasks/graph_tasks.py::run_business_graph(graph_name, state, meta)`
2. Pattern analog zu `run_ingestion_graph`:
   - Graph Registry Lookup (`graph_name`)
   - Context Building (ScopeContext, BusinessContext, ToolContext)
   - Observability (Langfuse Trace)
   - Error Handling (ToolError envelope)
3. Registriere in Celery (`queue="agents"`)

**Acceptance:**

- `run_business_graph` Task existiert
- Unterstützt Graphen: `collection_search`, `web_acquisition`, `rag.default`, `framework_analysis`
- Tests: `ai_core/tests/tasks/test_graph_tasks.py` (run_business_graph cases) passing

**Effort:** 2d
**Priority:** 🟡 Medium

---

### ✅ M-2: Theme-Views auf Async Worker umstellen

**Problem:** Views führen Graphen synchron aus (blockieren HTTP-Threads).

**Dependencies:** M-1 (run_business_graph Task)

**Pointers:**

- `theme/views_web_search.py:web_search()` Zeile 218-222, 288-289
- `theme/views_rag_tools.py:start_rerank_workflow()` Zeile 256-261
- `theme/views_framework.py:framework_analysis_submit()` Zeile 86-91

**Implementation:**

1. **Web Search:**
   - Nutze `run_business_graph.delay("collection_search", state, meta)` für Collection Search
   - Nutze `run_business_graph.delay("web_acquisition", state, meta)` für Web Acquisition
   - Optional: Synchron-Modus für schnelle Queries (< 10s timeout)

2. **Chat Submit:**
   - Optional Async-Mode (Checkbox "Run in Background")
   - Nutze `run_business_graph.delay("rag.default", state, meta)` für Async
   - Polling-Endpoint: `/rag-tools/chat-status/<task_id>/`

3. **Framework Analysis:**
   - Nutze `run_business_graph.delay("framework_analysis", state, meta)`
   - Polling-Endpoint: `/framework-analysis/status/<task_id>/`

**Acceptance:**

- Alle Views haben Optional Async-Modus (oder Default Async für langsame Graphen)
- Polling-Endpoints implementiert
- Tests: `theme/tests/test_rag_tools_view.py`, `theme/tests/test_rag_tools_simulation.py` passing

**Effort:** 2d
**Priority:** 🟡 Medium

---

### ✅ M-3: Collection Search Graph - Auto-Ingest Node

**Problem:** `auto_ingest=True` existiert in Schema, aber Graph hat keinen Downstream-Node für Crawler-Trigger.

**Pointer:** `ai_core/graphs/technical/collection_search.py`

**Implementation:**

1. Neuer Node: `trigger_ingestion(state) -> state`
   - Input: `state["search"]["results"]` + `state["auto_ingest_top_k"]` + `state["auto_ingest_min_score"]`
   - Logic: Ruft `CrawlerManager.dispatch_crawl_request()` für Top-K Results über Min-Score
   - Output: `state["ingestion_triggered"] = True`, `state["ingestion_task_ids"] = [...]`
2. Transition: `search_complete` → `trigger_ingestion` (conditional: `state["auto_ingest"] == True`)
3. Transition: `trigger_ingestion` → `END`

**Acceptance:**

- `auto_ingest=True` triggert Crawler automatisch (kein manueller `/web-search-ingest-selected/` Call nötig)
- Tests: `ai_core/tests/graphs/test_collection_search_graph.py` (auto-ingest case) passing

**Effort:** 1d
**Priority:** 🟢 Low

---

### ✅ M-4: Web Acquisition Graph - Select Best Mode implementieren

**Problem:** `mode="select_best"` existiert in Schema, aber keine echte Best-Selection-Logik.

**Pointer:** `ai_core/graphs/web_acquisition_graph.py`

**Implementation:**

**Option A (Implementieren):**

1. Node `select_best(state) -> state`:
   - Input: `state["output"]["search_results"]`
   - Logic: Top-1 Result mit höchstem Score über Confidence-Threshold (z.B. 0.8)
   - Output: `state["output"]["selected_result"] = {...}`, `state["output"]["search_results"] = [top_1]`
2. Transition: `search_complete` → `select_best` (conditional: `state["input"]["mode"] == "select_best"`)

**Option B (Entfernen):**

- Remove `mode="select_best"` aus Schema (UI macht Selektion)

**Acceptance:**

- **Option A:** `mode="select_best"` gibt nur Top-1 Result zurück
- **Option B:** Schema enthält nur `mode="search_only"`
- Tests: `ai_core/tests/graphs/test_web_acquisition_graph.py` passing

**Effort:** 0.5d
**Priority:** 🟢 Low

---

### ✅ M-5: RAG Graph History Governance ("Thick Graph")

**Problem:** Logic Leakage. View `chat_submit` manually loads, appends, trims, and saves history. This prevents true Async/Worker execution (worker cannot update history if View owns it).

**Pointer:** `ai_core/graphs/technical/retrieval_augmented_generation.py`, `theme/views_chat.py`

**Implementation:**

1. **Graph:** Add `persist_history` logic (via `compose_node` returning history update, or Checkpointer automatic state persistence).
2. **State:** Ensure `chat_history` is updated with User Q + AI A *inside* the graph.
3. **View:** Remove `append_history`, `trim_history`, `save`. View inputs `question`, Graph handles the rest.

**Acceptance:**

- `chat_submit` does not call `CHECKPOINTER.save()`
- Graph run results in updated history in DB
- Tests: `ai_core/tests/test_graph_retrieval_augmented_generation.py` passing

**Effort:** 1.5d
**Priority:** 🟡 Medium (Required for Async Chat)
**Note:** validiert gegen `roadmap/rag_thread_registry_backlog.md`. M-5 regelt die *Persistenz* der Nachrichten im Graph (Checkpointer). Die *Registry* (Metadaten/Listing) ist ein separates Thema, setzt aber verlässliche Persistenz voraus.

---

## Langfristige Maßnahmen (P2 - Cleanup, 5-10 Tage)

### 📋 L-1: Framework Analysis in Production API integrieren

**Problem:** Framework Analysis Graph existiert, ist aber NICHT in Production API verfügbar (nur Dev-Workbench).

**Pointers:**

- Graph: `ai_core/graphs/business/framework_analysis_graph.py`
- Dev-View: `theme/views_framework.py:framework_analysis_submit()`
- Missing: API-Endpunkt in `ai_core/urls.py`, API-View in `ai_core/views.py`

**Implementation:**

1. Erstelle `/v1/ai/framework/analyze/` Endpoint (analog zu `/v1/ai/rag/query/`)
2. Erstelle `FrameworkAnalysisViewV1` in `ai_core/views.py` (analog zu `RagQueryViewV1`)
3. Nutze `run_business_graph.delay("framework_analysis", ...)` für Async

**Acceptance:**

- `/v1/ai/framework/analyze/` Endpoint existiert
- OpenAPI Schema dokumentiert
- Tests: `ai_core/tests/test_views.py` (framework analysis endpoint cases) passing

**Effort:** 2d
**Priority:** 🟢 Low

---

### 📋 L-2: Document Delete/Restore - Async Bulk-Mode

**Problem:** `document_delete()` und `document_restore()` laufen synchron (könnten bei vielen Chunks langsam sein).

**Pointers:**

- Views: `theme/views_documents.py:document_delete()` Zeile 297, `document_restore()` Zeile 421
- Repository: `documents/repository.py:mark_deleted()`, `restore_document()`

**Implementation:**

1. Behalte synchron für einzelne Dokumente (< 1s typisch)
2. Neuer Endpoint: `/rag-tools/document-bulk-delete/` (für Bulk-Operationen)
3. Worker: `bulk_delete_documents.delay(document_ids, meta)`

**Acceptance:**

- Single Delete/Restore bleibt synchron
- Bulk Delete/Restore nutzt Celery Worker
- Tests: `theme/tests/test_document_space_view.py` (bulk delete case) passing

**Effort:** 1d
**Priority:** 🟢 Low

---

### 📋 L-3: Theme-Views - API-First Strategy

**Problem:** Theme-Views duplizieren Logik aus Official API-Views.

**Pointers:**

- Theme-Views: `theme/views_web_search.py`, `theme/views_chat.py`, `theme/views_ingestion.py`, `theme/views_framework.py`
- Official API: `ai_core/views.py` (RagQueryViewV1, RagUploadView, etc.)

**Implementation (schrittweise):**

1. **Phase 1:** Chat Submit → ruft `/v1/ai/rag/query/` via HTMX auf
2. **Phase 2:** Web Search → ruft hypothetischen `/v1/ai/search/` Endpoint auf (erst erstellen)
3. **Phase 3:** Framework Analysis → ruft `/v1/ai/framework/analyze/` auf

**Acceptance:**

- Theme-Views rufen NUR Official API auf (keine eigene Graph-Ausführung)
- Konsistenz: Single Source of Truth
- Tests: `theme/tests/test_rag_tools_view.py`, `theme/tests/test_chat_submit_global_search.py` passing

**Effort:** 3d
**Priority:** 🟢 Low

---

### 📋 L-4: RAG Query - Async Option für Production API

**Problem:** `/v1/ai/rag/query/` läuft synchron (blockiert HTTP-Thread bei langen Queries).

**Pointer:** `ai_core/views.py:RagQueryViewV1` Zeile 1709-1774

**Implementation:**

1. Neuer Endpoint: `/v1/ai/rag/query/async/` (gibt `task_id` zurück)
2. Polling-Endpoint: `/v1/ai/rag/query/status/<task_id>/` (analog zu Ingestion Status)
3. Worker: `run_business_graph.delay("rag.default", state, meta)`

**Acceptance:**

- `/v1/ai/rag/query/async/` Endpoint existiert
- Polling-Endpoint implementiert
- OpenAPI Schema dokumentiert
- Tests: `ai_core/tests/test_views.py` (async rag query cases) passing

**Effort:** 2d
**Priority:** 🟢 Low

---

## Zusammenfassung: Effort & Priorität

| **Kategorie** | **Tasks** | **Total Effort** | **Priority** |
|---------------|-----------|------------------|--------------|
| **Quick Wins (P0)** | QW-1 bis QW-4 | 2.5d | 🔴 Critical |
| **Mittelfristig (P1)** | M-1 bis M-4 | 5.5d | 🟡 Medium |
| **Langfristig (P2)** | L-1 bis L-4 | 8d | 🟢 Low |
| **TOTAL** | 12 Tasks | **16d** | - |

**Empfehlung:** Start mit Quick Wins (2.5d), dann schrittweise P1 Migration.

---

## Testing Strategy

### Unit Tests

- `theme/tests/test_rag_tools_view.py` - Alle Theme-Views
- `ai_core/tests/tasks/test_graph_tasks.py` - run_business_graph Worker
- `ai_core/tests/graphs/test_*_graph.py` - Graph-spezifische Tests

### Integration Tests

- `theme/tests/test_rag_tools_simulation.py` - End-to-End Workbench Flows
- `ai_core/tests/test_views.py` - API-Endpunkt-Tests

### Performance Tests

- Async Worker vs. Sync: Response Time, Throughput, Error Rate
- Bulk Operations: Document Delete (100+ docs), Ingestion (100+ URLs)

---

## Rollback Plan

**Quick Wins (QW-1 bis QW-4):**

- Revert commits (keine Breaking Changes bei korrektem Pattern-Reuse)
- Tests sollten alle passing bleiben

**Mittelfristig (M-1 bis M-4):**

- M-1/M-2: Rollback auf synchrone Execution (Views rufen Graphen direkt)
- M-3/M-4: Feature-Flag `ENABLE_AUTO_INGEST`, `ENABLE_SELECT_BEST` (deaktiviert = alter Zustand)

**Langfristig (L-1 bis L-4):**

- L-1/L-4: Neue Endpunkte entfernen (keine Breaking Changes für existierende API)
- L-2: Bulk-Endpunkt entfernen, sync bleibt
- L-3: Theme-Views können weiterhin eigene Logik behalten (Migration optional)

---

## Migration Timeline (Vorschlag)

### Sprint 1 (Week 1): Quick Wins

- Tag 1-2: QW-1 (Crawler Submit) + QW-2 (Context Helper)
- Tag 3: QW-3 (Chat Submit API-Reuse)
- Tag 4: QW-4 (Rerank Timeout + Polling)
- Tag 5: Testing + Review

**Deliverable:** Kritische Sync-Probleme gelöst, Context-Duplizierung eliminiert.

### Sprint 2 (Week 2): Async Worker Infrastructure

- Tag 1-2: M-1 (Generic Worker-Task)
- Tag 3-4: M-2 (Theme-Views auf Async umstellen)
- Tag 5: Testing + Review

**Deliverable:** Alle Graphen können async laufen, skalierbar.

### Sprint 3 (Week 3-4): Graph Features + API Integration

- Tag 1: M-3 (Collection Search Auto-Ingest)
- Tag 2: M-4 (Web Acquisition Select Best)
- Tag 3-4: L-1 (Framework Analysis API)
- Tag 5: L-2 (Document Bulk Operations)

**Deliverable:** Graph-Features komplett, Framework Analysis in Production API.

### Sprint 4 (Week 4-5): API-First Strategy (Optional)

- Tag 1-2: L-3 (Theme-Views auf API umstellen)
- Tag 3-4: L-4 (RAG Query Async)
- Tag 5: Final Testing + Documentation

**Deliverable:** API-First Architecture, Theme-Views nutzen Official API, vollständig async.

---

## Related Documents

- **Analyse:** [docs/architecture/rag-tools-architecture-analysis.md](../docs/architecture/rag-tools-architecture-analysis.md)
- **Contracts:** [docs/agents/tool-contracts.md](../docs/agents/tool-contracts.md)
- **Graphen:** [ai_core/graphs/README.md](../ai_core/graphs/README.md)
- **Multi-Tenancy:** [docs/multi-tenancy.md](../docs/multi-tenancy.md)
- **Observability:** [docs/observability/langfuse.md](../docs/observability/langfuse.md)

---

## Questions / Decisions Required

1. **QW-3 (Chat Submit):** Option A (API-Reuse) oder Option B (Service-Reuse)? Empfehlung: **Option A** (API-First).
2. **M-4 (Web Acquisition):** Implementieren ODER Entfernen? Empfehlung: **Entfernen** (UI macht Selektion).
3. **L-3 (API-First):** Full Migration ODER Optional? Empfehlung: **Optional** (schrittweise Migration, kein Force).
4. **Timeline:** Sprint 1+2 zwingend (P0+P1), Sprint 3+4 optional (P2)?

**Status:** Awaiting Team Review & Approval
