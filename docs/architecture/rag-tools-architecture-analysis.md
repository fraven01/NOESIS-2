# RAG-Tools Architecture Analysis

**Datum:** 2026-01-13
**Scope:** UI-Layer (`rag-tools/`) → Graphen/Workers/API-Calls Mapping
**Ziel:** Architektur-Divergenz identifizieren, Logik-Duplizierung aufdecken, fehlende Features finden

---

## Executive Summary

Die RAG-Tools-Workbench (`rag-tools/`) ist eine Dev-UI mit **erheblichen Architektur-Problemen**:

1. **Views führen Graphen synchron aus** (umgehen Worker-Queue)
2. **Massive Logik-Duplizierung** (Context-Building in jeder View)
3. **Fehlende Worker-Tasks** für Business-Graphen (RAG Query, Collection Search, Web Acquisition, Framework Analysis)
4. **Inkonsistente API-Nutzung** (Theme-Views vs. Official API-Views)
5. **Fehlende Graph-Features** (Auto-Ingest, Best-Selection)

**Empfehlung:** Refactoring zu Async Worker-Pattern + API-Reuse erforderlich.

---

## 1. View → Graph/Worker/API Mapping

### 1.1 Web Search & Acquisition

| **View** | **Funktion** | **Ruft auf** | **Pattern** | **Problem** |
|----------|-------------|--------------|-------------|-------------|
| `theme/views_web_search.py::web_search()` | Web-Suche + Collection Search | **DIREKT:** `build_collection_search_graph().run()` (Zeile 218-222) ODER `build_web_acquisition_graph().invoke()` (Zeile 288-289) | Synchron (blockiert Request) | ❌ Umgeht Worker-Queue, keine Retry/Monitoring, blockiert Thread |
| `theme/views_web_search.py::web_search_ingest_selected()` | Ingest ausgewählter URLs | `CrawlerManager.dispatch_crawl_request()` (Zeile 461) | Async (Worker Queue) | ✅ Korrekt (nutzt Celery Worker) |

**Fehlende Worker-Tasks:**
- ❌ Kein `run_collection_search_graph` Celery Task
- ❌ Kein `run_web_acquisition_graph` Celery Task

**Fehlende Graph-Features:**
- ❌ Collection Search Graph: Kein `auto_ingest` Downstream (muss manuell via `web_search_ingest_selected` getriggert werden)
- ❌ Web Acquisition Graph: `mode="select_best"` existiert in Schema, aber keine echte Implementierung

---

### 1.2 Ingestion & Crawler

| **View** | **Funktion** | **Ruft auf** | **Pattern** | **Problem** |
|----------|-------------|--------------|-------------|-------------|
| `theme/views_ingestion.py::ingestion_submit()` | Datei-Upload + Ingestion | `ai_core.services.handle_document_upload()` (Zeile 249) | Async (triggert Worker) | ✅ Korrekt (nutzt Service → Worker) |
| `theme/views_ingestion.py::crawler_submit()` | Crawler-Form Submit | **DIREKT:** `run_crawler_runner()` (Zeile 148-153) | Synchron (blockiert Request) | ❌ Sollte `CrawlerManager` nutzen (wie `web_search_ingest_selected`) |
| `ai_core/views.py::crawl_selected()` | Crawl Selected (HTMX) | `CrawlerManager.dispatch_crawl_request()` (Zeile 1685) | Async (Worker Queue) | ✅ Korrekt |

**Offizielle API-Endpunkte:**
- `/v1/rag/documents/upload/` → `RagUploadView` (nutzt `handle_document_upload` Service)
- `/v1/rag/ingestion/run/` → `RagIngestionRunView` (nutzt `start_ingestion_run` Service)
- `/ai/rag/crawler/run/` → `CrawlerIngestionRunnerView` (DEBUG only, nutzt `run_crawler_runner` direkt)

**Problem:** `crawler_submit` nutzt NICHT `CrawlerManager`, sondern `run_crawler_runner` direkt.
**Lösung:** Umstellen auf `CrawlerManager.dispatch_crawl_request()` (wie `web_search_ingest_selected`).

---

### 1.3 RAG Chat

| **View** | **Funktion** | **Ruft auf** | **Pattern** | **Problem** |
|----------|-------------|--------------|-------------|-------------|
| `theme/views_chat.py::chat_submit()` | RAG Chat (Frage + Antwort) | **DIREKT:** `run_rag_graph(state, meta)` (Zeile 131) | Synchron (blockiert Request) | ❌ Umgeht Worker-Queue, keine Async-Option |

**Offizielle API-Endpunkte:**
- `/v1/ai/rag/query/` → `RagQueryViewV1` (nutzt `_GraphView` → `services.execute_graph` → LangGraph)

**Fehlende Worker-Tasks:**
- ❌ Kein `run_rag_query_graph` Celery Task für asynchrone Queries

**Problem:** RAG-Queries laufen synchron, blockieren HTTP-Thread.
**Lösung:** Optional async Worker-Mode (wie in `start_rerank_workflow` mit Timeout).

---

### 1.4 Document Management

| **View** | **Funktion** | **Ruft auf** | **Pattern** | **Problem** |
|----------|-------------|--------------|-------------|-------------|
| `theme/views_documents.py::document_explorer()` | Dokumente anzeigen | `DOCUMENT_SPACE_SERVICE.build_context()` | Synchron (Read-Only Service) | ✅ Korrekt (Read-Only) |
| `theme/views_documents.py::document_reingest()` | Re-Ingestion triggern | `ai_core.services.start_ingestion_run()` (Zeile 240) | Async (Worker Queue) | ✅ Korrekt |
| `theme/views_documents.py::document_delete()` | Soft Delete | `repository.mark_deleted()` | Synchron (DB Write) | ⚠️ Sollte ggf. asynchron sein (DB Lock bei vielen Chunks) |
| `theme/views_documents.py::document_restore()` | Restore | `repository.restore_document()` | Synchron (DB Write) | ⚠️ Sollte ggf. asynchron sein |

**Offizielle API-Endpunkte:**
- `/ai/rag/admin/hard-delete/` → `RagHardDeleteAdminView` (nutzt `hard_delete.delay()` Celery Task)

**Problem:** Soft Delete/Restore sind synchron, könnten bei vielen Chunks langsam sein.
**Lösung:** Optional async Worker-Mode für Bulk-Operationen.

---

### 1.5 Framework Analysis

| **View** | **Funktion** | **Ruft auf** | **Pattern** | **Problem** |
|----------|-------------|--------------|-------------|-------------|
| `theme/views_framework.py::framework_analysis_submit()` | Framework-Analyse | **DIREKT:** `build_framework_graph().run()` (Zeile 86-91) | Synchron (blockiert Request) | ❌ Umgeht Worker-Queue |

**Fehlende Worker-Tasks:**
- ❌ Kein `run_framework_analysis_graph` Celery Task

**Fehlende API-Endpunkte:**
- ❌ Framework Analysis ist NUR in Dev-Workbench verfügbar, NICHT in Production API

**Problem:** Framework Analysis Graph existiert, ist aber nicht in Production API integriert.
**Lösung:** API-Endpunkt erstellen + Celery Worker-Task.

---

### 1.6 Rerank Workflow

| **View** | **Funktion** | **Ruft auf** | **Pattern** | **Problem** |
|----------|-------------|--------------|-------------|-------------|
| `theme/views_rag_tools.py::start_rerank_workflow()` | Rerank Workflow | `views.submit_worker_task()` → `run_collection_search_graph` (Zeile 256-261) | **Hybrid:** Worker mit 120s Timeout (Zeile 260) | ⚠️ Timeout ist hoch, blockiert Thread 120s |

**Problem:** 120s Timeout ist zu hoch für HTTP-Request (sollte async mit Polling sein).
**Lösung:** Async Worker + Polling-Endpoint (wie Ingestion Status).

---

## 2. Logik-Duplizierung: Context Building

**Problem:** Jede View baut manuell `ScopeContext`, `BusinessContext`, `ToolContext`, obwohl `_prepare_request()` in `ai_core/views.py` schon existiert.

**Beispiele:**

| **Datei** | **Zeilen** | **Code** |
|-----------|------------|----------|
| `views_chat.py` | 80-116 | Manuelles Building von `scope`, `business`, `tool_context`, `meta` |
| `views_web_search.py` | 190-208, 263-280, 437-457 | Dreifache Duplizierung! |
| `views_ingestion.py` | 107-135, 216-234 | Doppelte Duplizierung |
| `views_documents.py` | 239-260 | Manuelles Building |
| `views_framework.py` | 55-90 | Manuelles Building (aber simplified) |

**Vorhandene Lösung:** `ai_core/views.py::_prepare_request()` (Zeilen 482-780)
- Validiert Headers (Tenant, Case, Key Alias)
- Baut `ScopeContext`, `BusinessContext`, `ToolContext`
- Enriched `request.META`
- Bind Log Context

**Empfehlung:** Theme-Views sollten `_prepare_request()` wiederverwenden oder einen gemeinsamen Helper erstellen.

---

## 3. Fehlende Worker-Tasks

**Vorhanden:**
- ✅ `ai_core.tasks.graph_tasks.run_ingestion_graph` (für Universal Ingestion)
- ✅ `ai_core.rag.hard_delete.hard_delete` (für Admin Hard Delete)

**Fehlend:**
- ❌ `run_collection_search_graph` (für Collection Search)
- ❌ `run_web_acquisition_graph` (für Web Acquisition)
- ❌ `run_rag_query_graph` (für RAG Chat)
- ❌ `run_framework_analysis_graph` (für Framework Analysis)

**Empfehlung:** Generic `run_business_graph` Worker-Task erstellen (ähnlich wie `run_ingestion_graph`), der beliebige Graphen async ausführt.

---

## 4. Inkonsistente API-Nutzung

**Official API (ai_core/views.py):**
- `/v1/ai/rag/query/` → `RagQueryViewV1` (nutzt `_GraphView` → `services.execute_graph`)
- `/v1/rag/documents/upload/` → `RagUploadView` (nutzt `services.handle_document_upload`)
- `/v1/rag/ingestion/run/` → `RagIngestionRunView` (nutzt `services.start_ingestion_run`)
- `/ai/rag/crawler/run/` → `CrawlerIngestionRunnerView` (DEBUG only)

**Theme-Views (theme/urls.py):**
- `/rag-tools/web-search/` → `web_search` (eigene Implementierung, nutzt NICHT Official API)
- `/rag-tools/crawler-submit/` → `crawler_submit` (eigene Implementierung)
- `/rag-tools/ingestion-submit/` → `ingestion_submit` (nutzt `handle_document_upload` Service, OK)
- `/rag-tools/chat-submit/` → `chat_submit` (eigene Implementierung, nutzt NICHT `/v1/ai/rag/query/`)
- `/rag-tools/start-rerank-workflow/` → `start_rerank_workflow` (eigene Implementierung)

**Problem:** Theme-Views duplizieren Logik aus API-Views, nutzen aber unterschiedliche Patterns.

**Empfehlung:**
1. **Option A (API-Reuse):** Theme-Views sollten Official API via HTMX aufrufen (z.B. `hx-post="/v1/ai/rag/query/"`)
2. **Option B (Service-Reuse):** Theme-Views sollten mindestens die gleichen Services nutzen (`services.execute_graph`, `handle_document_upload`)

**Aktuell:**
- ✅ `ingestion_submit` nutzt `handle_document_upload` (korrekt)
- ❌ `chat_submit` nutzt `run_rag_graph` direkt (sollte `services.execute_graph` nutzen)
- ❌ `web_search` nutzt Graph direkt (sollte Service nutzen)
- ❌ `crawler_submit` nutzt `run_crawler_runner` direkt (sollte `CrawlerManager` nutzen)

---

## 5. Fehlende Graph-Features

### 5.1 Collection Search Graph

**Schema:** `ai_core/graphs/technical/collection_search.py`

**Input:**
```python
class GraphInput(BaseModel):
    question: str
    collection_scope: str
    quality_mode: str = "standard"
    max_candidates: int = 20
    purpose: str
    execute_plan: bool = False
    auto_ingest: bool = False  # ⚠️ Existiert, aber nicht implementiert
    auto_ingest_top_k: int = 10
    auto_ingest_min_score: float = 60.0
```

**Problem:** `auto_ingest=True` existiert in Schema, aber Graph hat KEINEN Downstream-Node, der Crawler triggert.

**Aktueller Workaround:** UI muss `web_search_ingest_selected` manuell aufrufen (separater Request).

**Empfehlung:**
- Option A: Collection Search Graph bekommt `trigger_ingestion` Node (triggert `CrawlerManager`)
- Option B: `auto_ingest` aus Schema entfernen (UI-only Feature)

---

### 5.2 Web Acquisition Graph

**Schema:** `ai_core/graphs/web_acquisition_graph.py`

**Input:**
```python
class WebAcquisitionInputModel(BaseModel):
    query: str | None = None
    search_config: dict[str, Any] | None = None
    preselected_results: list[dict[str, Any]] | None = None
    mode: Literal["search_only", "select_best"] | None = None  # ⚠️ "select_best" nicht implementiert
```

**Problem:** `mode="select_best"` existiert in Schema, aber Graph hat keine echte Logik für "Best Result" Selektion.

**Aktueller Code (Zeilen 142-180):**
- `search_only`: Gibt alle Results zurück
- `select_best`: Ruft `select_search_candidates()` auf, aber gibt trotzdem alle Results zurück (Zeile 178: `state["output"]["search_results"] = search_results`)

**Empfehlung:**
- Implementiere echte Best-Selection (Top-1 mit Confidence Threshold)
- Oder entferne `select_best` Mode aus Schema

---

### 5.3 Framework Analysis Graph

**Schema:** `ai_core/graphs/business/framework_analysis_graph.py`

**Problem:** Graph existiert, ist aber NICHT in Production API integriert (nur in Dev-Workbench).

**Fehlend:**
- ❌ API-Endpunkt in `ai_core/urls.py`
- ❌ API-View in `ai_core/views.py`
- ❌ Celery Worker-Task

**Empfehlung:**
- Erstelle `/v1/ai/framework/analyze/` Endpoint (analog zu `/v1/ai/rag/query/`)
- Erstelle `run_framework_analysis_graph` Celery Task

---

## 6. API-Calls Übersicht

### 6.1 Official API (ai_core/urls.py)

| **Endpoint** | **Method** | **View** | **Graph/Service** | **Async** |
|--------------|-----------|----------|-------------------|-----------|
| `/v1/ai/ping/` | GET | `PingViewV1` | - | ✅ Instant |
| `/v1/ai/rag/query/` | POST | `RagQueryViewV1` | `services.execute_graph` → RAG Graph | ❌ Synchron |
| `/v1/rag/documents/upload/` | POST | `RagUploadView` | `services.handle_document_upload` → Worker | ✅ Async |
| `/v1/rag/ingestion/run/` | POST | `RagIngestionRunView` | `services.start_ingestion_run` → Worker | ✅ Async |
| `/v1/rag/ingestion/status/` | GET | `RagIngestionStatusView` | Read-Only | ✅ Instant |
| `/ai/rag/admin/hard-delete/` | POST | `RagHardDeleteAdminView` | `hard_delete.delay()` → Worker | ✅ Async |
| `/ai/rag/crawler/run/` | POST | `CrawlerIngestionRunnerView` | `run_crawler_runner` → Crawler Graph | ❌ Synchron (DEBUG only) |
| `/ai/crawl-selected/` | POST | `crawl_selected` | `CrawlerManager.dispatch_crawl_request` → Worker | ✅ Async |

---

### 6.2 Theme-Views (theme/urls.py)

| **Endpoint** | **Method** | **View** | **Graph/Service** | **Async** | **Problem** |
|--------------|-----------|----------|-------------------|-----------|-------------|
| `/rag-tools/web-search/` | POST | `web_search` | `build_collection_search_graph().run()` ODER `build_web_acquisition_graph().invoke()` | ❌ Synchron | Umgeht Worker |
| `/rag-tools/web-search/ingest-selected/` | POST | `web_search_ingest_selected` | `CrawlerManager.dispatch_crawl_request` | ✅ Async | ✅ Korrekt |
| `/rag-tools/crawler-submit/` | POST | `crawler_submit` | `run_crawler_runner` | ❌ Synchron | Sollte `CrawlerManager` nutzen |
| `/rag-tools/ingestion-submit/` | POST | `ingestion_submit` | `handle_document_upload` Service | ✅ Async | ✅ Korrekt |
| `/rag-tools/chat-submit/` | POST | `chat_submit` | `run_rag_graph()` | ❌ Synchron | Sollte `/v1/ai/rag/query/` nutzen ODER async Worker |
| `/rag-tools/start-rerank-workflow/` | POST | `start_rerank_workflow` | `submit_worker_task()` (120s Timeout) | ⚠️ Hybrid | Timeout zu hoch |
| `/framework-analysis/submit/` | POST | `framework_analysis_submit` | `build_framework_graph().run()` | ❌ Synchron | Fehlt in Official API |
| `/rag-tools/document-delete/` | POST | `document_delete` | `repository.mark_deleted()` | ❌ Synchron | ⚠️ Könnte langsam sein |
| `/rag-tools/document-restore/` | POST | `document_restore` | `repository.restore_document()` | ❌ Synchron | ⚠️ Könnte langsam sein |
| `/rag-tools/document-reingest/` | POST | `document_reingest` | `start_ingestion_run` Service | ✅ Async | ✅ Korrekt |

---

## 7. Empfehlungen

### 7.1 Sofortige Maßnahmen (Quick Wins)

1. **Crawler Submit umstellen**
   - **Datei:** `theme/views_ingestion.py::crawler_submit()`
   - **Ändern:** Von `run_crawler_runner()` zu `CrawlerManager.dispatch_crawl_request()`
   - **Vorteil:** Konsistenz mit `web_search_ingest_selected`, async Worker

2. **Context Building deduplizieren**
   - **Erstelle:** `theme/helpers/context.py::prepare_workbench_context(request)`
   - **Reuse:** `_prepare_request()` aus `ai_core/views.py` ODER extrahiere Shared Helper
   - **Nutze in:** Alle Theme-Views (`views_chat.py`, `views_web_search.py`, etc.)

3. **Chat Submit: API-Reuse**
   - **Option A:** `chat_submit` ruft `/v1/ai/rag/query/` via HTMX auf
   - **Option B:** `chat_submit` nutzt `services.execute_graph` (wie `RagQueryViewV1`)

4. **Rerank Workflow: Timeout reduzieren**
   - **Datei:** `theme/views_rag_tools.py::start_rerank_workflow()`
   - **Ändern:** `timeout_s=120` → `timeout_s=30` + Polling-Endpoint
   - **Polling:** Neuer Endpoint `/rag-tools/workflow-status/<task_id>/` (analog zu Ingestion Status)

---

### 7.2 Mittelfristige Maßnahmen (Architektur)

1. **Generic Graph Worker-Task erstellen**
   - **Erstelle:** `ai_core/tasks/graph_tasks.py::run_business_graph(graph_name, state, meta)`
   - **Nutze für:** Collection Search, Web Acquisition, Framework Analysis, RAG Query
   - **Pattern:** Analog zu `run_ingestion_graph` (Zeilen 72-168)

2. **Theme-Views auf Async Worker umstellen**
   - **web_search:** Nutze `run_business_graph.delay("collection_search", state, meta)`
   - **chat_submit:** Optional async Mode (Checkbox "Run in Background")
   - **framework_analysis_submit:** Nutze `run_business_graph.delay("framework_analysis", state, meta)`

3. **Collection Search Graph: Auto-Ingest Node**
   - **Datei:** `ai_core/graphs/technical/collection_search.py`
   - **Ergänze:** `trigger_ingestion` Node (ruft `CrawlerManager` auf, wenn `auto_ingest=True`)
   - **Transition:** `search_complete` → `trigger_ingestion` (wenn `auto_ingest=True`)

4. **Web Acquisition Graph: Select Best Mode**
   - **Datei:** `ai_core/graphs/web_acquisition_graph.py`
   - **Ergänze:** Echte Best-Selection (Top-1 mit Confidence Threshold)
   - **Oder:** Entferne `select_best` Mode aus Schema (UI macht Selektion)

---

### 7.3 Langfristige Maßnahmen (Cleanup)

1. **Framework Analysis in Production API**
   - **Erstelle:** `/v1/ai/framework/analyze/` Endpoint (analog zu `/v1/ai/rag/query/`)
   - **Erstelle:** `FrameworkAnalysisViewV1` (analog zu `RagQueryViewV1`)
   - **Worker:** `run_business_graph.delay("framework_analysis", ...)`

2. **Document Delete/Restore: Async Bulk-Mode**
   - **Erstelle:** `/rag-tools/document-bulk-delete/` (für Bulk-Operationen)
   - **Worker:** `bulk_delete_documents.delay(document_ids, meta)`
   - **Single:** Behalte synchron für einzelne Dokumente

3. **Theme-Views: API-First Strategy**
   - **Ziel:** Theme-Views rufen NUR Official API auf (via HTMX `hx-post`)
   - **Vorteil:** Konsistenz, Single Source of Truth, einfacher zu testen
   - **Migration:** Schrittweise (Chat → Web Search → Framework Analysis)

4. **RAG Query: Async Option**
   - **Erstelle:** `/v1/ai/rag/query/async/` Endpoint (gibt `task_id` zurück)
   - **Polling:** `/v1/ai/rag/query/status/<task_id>/` (analog zu Ingestion Status)
   - **Worker:** `run_business_graph.delay("rag.default", state, meta)`

---

## 8. Zusammenfassung: Architektur-Divergenz

| **Komponente** | **IST-Zustand** | **SOLL-Zustand** | **Priorität** |
|----------------|-----------------|------------------|---------------|
| **Crawler Submit** | Synchron (`run_crawler_runner`) | Async (`CrawlerManager`) | 🔴 Hoch |
| **Chat Submit** | Synchron (Graph direkt) | Async Worker ODER API-Reuse | 🔴 Hoch |
| **Web Search** | Synchron (Graph direkt) | Async Worker | 🟡 Mittel |
| **Framework Analysis** | Synchron, nur Dev-Workbench | Async Worker + Production API | 🟡 Mittel |
| **Context Building** | Dupliziert in jeder View | Shared Helper | 🟡 Mittel |
| **Collection Search** | Kein Auto-Ingest | Trigger Ingestion Node | 🟢 Niedrig |
| **Web Acquisition** | Kein Select Best | Implementieren ODER entfernen | 🟢 Niedrig |
| **Rerank Workflow** | 120s Timeout (blockiert) | Async + Polling | 🟡 Mittel |
| **Document Delete** | Synchron (könnte langsam sein) | Optional Async Bulk-Mode | 🟢 Niedrig |

---

## 9. Next Steps

1. **Review mit Team:** Diese Analyse mit Entwicklern besprechen
2. **Priorisierung:** Welche Quick Wins zuerst (Empfehlung: Crawler Submit + Context Helper)
3. **ADR erstellen:** Architektur-Entscheidungen dokumentieren (Theme-Views: API-First vs. Service-Reuse)
4. **Refactoring-Tickets:** Jira-Tickets für Quick Wins + Mittelfristig
5. **Migration-Plan:** Schrittweise Migration zu Async Worker-Pattern

---

**Autor:** Claude Code
**Review:** Pending
**Status:** Draft v1.0
