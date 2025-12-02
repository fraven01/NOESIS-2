# ID-Propagation und End-to-End-Flows

Diese Referenz beschreibt, wie Kontext-IDs von HTTP über Django und Celery bis in LangGraph und Tools weitergereicht werden. Sie basiert auf den aktuellen Graph- und Worker-Implementierungen.

## Graph- und Worker-Inventur

### LangGraph-Definitionen

- `ai_core/graphs/collection_search.py` – Graph für Collection- und Websuche; erwartet `GraphContextPayload` mit `tenant_id`, `workflow_id`, `case_id` sowie optional `trace_id`, `run_id`, `ingestion_run_id`. Validiert Input strikt (`extra=forbid`). Tools: Websuche (`WebSearchWorker`), Hybrid-Scoring (`HybridScoreExecutor`), Ingestion-Trigger.
- `ai_core/graphs/external_knowledge_graph.py` – Websuche mit optionalem HITL und Crawler-Weiterleitung; derselbe Kontext wie oben, erzwingt `tenant_id`, `workflow_id`, `case_id`.
- `ai_core/graphs/crawler_ingestion_graph.py` – LangGraph-Orchestrierung der Crawling/Ingestion-Kette; nutzt `run_ingestion_graph` Task als Worker-Einstieg.
- `ai_core/graphs/upload_ingestion_graph.py` – steuert Upload-Ingestion-Pfade mit denselben Kontextfeldern.
- `ai_core/graphs/retrieval_augmented_generation.py`, `framework_analysis_graph.py`, `info_intake.py`, `rag_demo.py`, `document_service.py`, `cost_tracking.py`, `transition_contracts.py` – weitere Graphen; alle konsumieren vorbereitete Kontext-Metas aus dem Dispatcher und propagieren `tenant_id`, `trace_id`, `workflow_id`, `case_id` plus Laufzeit-ID.
- `llm_worker/graphs/hybrid_search_and_score.py` & `score_results.py` – Worker-seitige Graphen für Hybrid-Scoring; erhalten `tenant_id`, `case_id`, `trace_id` und Laufzeit-IDs über Task-Meta.

### Celery-Tasks

- `llm_worker.tasks.run_graph` (Queue `agents`) – führt LangGraph-Runner aus und gibt `state`, `result`, `cost_summary` zurück; nimmt optionale `tenant_id`, `case_id`, `trace_id` entgegen und re-emittiert Case-Lifecycle-Events.
- `ai_core.tasks.ingest_raw` / `extract_text` / `pii_mask` / `_split_sentences` – Hilfstasks im Ingestion-Pfad; propagieren `meta` (enthält `tenant_id`, `case_id`, `run_id` oder `ingestion_run_id`).
- `ai_core.tasks.embed` – erzeugt Embeddings; nutzt `meta` für `tenant_id`, `case_id`, `workflow_id`, `trace_id`, `run_id`/`ingestion_run_id`.
- `ai_core.tasks.upsert` – schreibt Vektoren; validiert `ChunkMeta` inkl. `tenant_id`, `case_id`, `workflow_id`, `vector_space_id` und Laufzeit-ID.
- `ai_core.tasks.ingestion_run` – Platzhalter-Dispatcher für Ingestion Runs; loggt `tenant_id`, `case_id`, `document_ids`, `trace_id`.
- `ai_core.tasks.run_ingestion_graph` (Queue `ingestion`) – orchestriert den Crawler-Ingestion-Graph; extrahiert Kontext mit `IngestionContextBuilder`, startet Observability mit `trace_id`/`run_id`/`ingestion_run_id`.
- `ai_core.ingestion.process_document` (Queue `ingestion`) – verarbeitet einzelne Dokumente; erwartet `tenant`, `case`, `document_id`, `embedding_profile`, optional `trace_id`; erzeugt Vector-Space-Kontext.
- `ai_core.ingestion.run_ingestion` (Queue `ingestion`) – orchestriert dokumentbasierte Ingestion; verlangt `tenant`, `case`, `document_ids`, `embedding_profile`, Pflichtfelder `run_id`, `trace_id`.

### Tools

- `ai_core.tools.web_search.WebSearchWorker` – Input `WebSearchInput`, Kontext `WebSearchContext` mit Pflichtfeldern `tenant_id`, `trace_id`, `workflow_id`, `case_id`, `run_id`, optional `worker_call_id`; Output `WebSearchResponse` mit `ToolOutcome` und Suchergebnissen.
- `ai_core.tools.search_adapters.google.GoogleSearchAdapter` – Provider-spezifischer Adapter; erzeugt `SearchAdapterResponse` und normalisiert Links, übernimmt Kontext aus aufrufendem Worker.
- Framework-/Shared-Tools (`ai_core.tools.framework_contracts`, `ai_core.tools.shared_workers`) folgen demselben `ToolContext`-Vertrag: `tenant_id`, `trace_id`, genau eine Laufzeit-ID (`run_id` XOR `ingestion_run_id`), optional `case_id`, `workflow_id`, `invocation_id`.

## ID-Propagation je Szenario

### HTTP → Django → Celery → Graph (Collection Search)

1. HTTP-Request liefert `X-Tenant-ID`, optional `X-Case-ID`, `X-Trace-ID`.
2. Middleware (`RequestContextMiddleware`) ruft `normalize_request` auf, erstellt `ScopeContext` und validiert IDs (z.B. Tenant-Existenz).
3. View nutzt `request.scope_context`.
4. Dispatcher serialisiert `ScopeContext` und legt Celery-Task `llm_worker.tasks.run_graph` in Queue `agents`.
5. Graph erhält `ScopeContext` (als Dict/Model), validiert Pflichtfelder, propagiert IDs in jeden Tool-Call (`WebSearchWorker`, Hybrid-Score) via `ToolContext`.
6. Tools erzeugen `invocation_id` und geben Outcomes mit identischen Kontextfeldern an Langfuse/Logs weiter.

### Websearch + Auto-Ingest

1. Collection- oder External-Knowledge-Graph entscheidet über Ingestion (`auto_ingest` oder HITL approved).
2. Graph erzeugt `ingestion_run_id` am Trigger-Punkt, übergibt `tenant_id`, `case_id`, `workflow_id`, `trace_id`, `ingestion_run_id`, `collection_id` an Ingestion-Trigger.
3. Celery-Task `ai_core.tasks.run_ingestion_graph` startet mit diesem Kontext; Phase-Spans (`crawler.ingestion.*`) führen `trace_id`, `ingestion_run_id`, `workflow_id`.
4. Nach Abschluss wird `CaseEvent`/`ingestion.end` mit denselben IDs emittiert.

### Cron-Crawler ohne Case

1. Scheduler ruft `run_ingestion_graph` direkt mit `tenant_id`, `trace_id`, `collection_id` und **ohne** `case_id` auf.
2. Validation-Policy erlaubt fehlende `case_id` für System-Tasks; Graph setzt `case_id=None` und markiert Events als systemisch.
3. Tools erhalten `ingestion_run_id` + `tenant_id` + `trace_id`; `workflow_id` beschreibt den Crawler-Flow.

### Multi-Step Graph → Graph

1. Ein Graph ruft einen nachgelagerten Graph (z. B. `collection_search` → `hybrid_search_and_score`).
2. Aufrufer gibt `tenant_id`, `case_id`, `workflow_id` (des Kind-Graphs) und **denselben** `trace_id` weiter.
3. `run_id` wird pro Ausführung neu generiert; Kind-Graph darf `ingestion_run_id` nur setzen, wenn er Ingestion startet.

## Validierung und Fehlerpfade

- **Pflicht**: `tenant_id` immer; `case_id` Pflicht für fachliche Workloads; `trace_id` Pflicht für Observability; genau eine Laufzeit-ID.
- **Fail-fast**: Graph-Inputs (`GraphContextPayload`, `WebSearchContext`, Tool-Inputs) verwenden `extra=forbid` und Feldvalidatoren. Fehlende oder leere Felder brechen den Start ab.
- **System-Tasks**: `case_id` optional; Events markieren `system_task=true` im Metadata-Feld.
- **Fallback**: Kein Default-Case; fehlende Cases müssen explizit erstellt werden (Auto-Create-Flag) oder führen zu 4xx.

## Observability

- `trace_id` bleibt identisch über alle Layer; `workflow_id`, `case_id`, `run_id`/`ingestion_run_id` werden als Span-Attribute geführt.
- LangGraph-Transitions mappen auf Span-Namen (`crawler.ingestion.update_status`, `ingest`, `document_pipeline`, etc.).
- Logs/Events spiegeln ID-Payloads (`case_lifecycle`, `ingestion.end`) und dienen als Auditquelle.
