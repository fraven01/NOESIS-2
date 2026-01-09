# ID-Propagation und End-to-End-Flows

Diese Referenz beschreibt, wie Kontext-IDs von HTTP über Django und Celery bis in LangGraph und Tools weitergereicht werden. Sie basiert auf den aktuellen Graph- und Worker-Implementierungen.

## Graph- und Worker-Inventur

### LangGraph-Definitionen

- `ai_core/graphs/collection_search.py` – Graph für Collection- und Websuche; erwartet ToolContext in `meta` (ScopeContext + BusinessContext). Business-IDs (z.B. `case_id`, `workflow_id`) liegen in `business_context`, Infrastruktur-IDs in `scope_context`.
- `ai_core/graphs/technical/universal_ingestion_graph.py` – Orchestriert Ingestion für `source=search`, `upload` und `crawler`. Validiert benötigte Business-IDs graph-spezifisch; `case_id` ist optional im BusinessContext.
- `ai_core/graphs/retrieval_augmented_generation.py`, `framework_analysis_graph.py`, `info_intake.py`, `document_service.py`, `cost_tracking.py` – weitere Graphen; konsumieren `meta` mit ToolContext und propagieren Scope/Business-IDs über ToolContext.
- `llm_worker/graphs/hybrid_search_and_score.py` & `score_results.py` – Worker-seitige Graphen für Hybrid-Scoring; erhalten ToolContext über Task-Meta.

### Celery-Tasks

- `llm_worker.tasks.run_graph` (Queue `agents`) – führt LangGraph-Runner aus und gibt `state`, `result`, `cost_summary` zurück; nimmt `state`/`meta` entgegen (ToolContext), re-emittiert Case-Lifecycle-Events.
- `ai_core.tasks.ingest_raw` / `extract_text` / `pii_mask` / `_split_sentences` – Hilfstasks im Ingestion-Pfad; nutzen `meta` mit ToolContext.
- `ai_core.tasks.embed` – erzeugt Embeddings; nutzt `meta` (ToolContext) für Scope/Business-IDs.
- `ai_core.tasks.upsert` – schreibt Vektoren; nutzt `ChunkMeta` und `meta` (ToolContext) inkl. `tenant_id`, `vector_space_id`, Laufzeit-IDs.
- `ai_core.tasks.ingestion_run` – Platzhalter-Dispatcher für Ingestion Runs; nimmt `state`/`meta`, loggt `tenant_id`, `case_id`, `document_ids`, `trace_id`.
- `ai_core.tasks.run_ingestion_graph` (Queue `ingestion`) – orchestriert den Crawler-Ingestion-Graph; extrahiert Kontext mit `IngestionContextBuilder`, startet Observability mit `trace_id`/`run_id`/`ingestion_run_id`.
- `ai_core.ingestion.process_document` (Queue `ingestion`) – verarbeitet einzelne Dokumente; nimmt `state`/`meta` (inkl. `document_id`, `embedding_profile`), erzeugt Vector-Space-Kontext.
- `ai_core.ingestion.run_ingestion` (Queue `ingestion`) – orchestriert dokumentbasierte Ingestion; nimmt `state`/`meta`, verlangt `document_ids`, `embedding_profile`, Pflichtfelder `run_id`, `trace_id`.

### Tools

- `ai_core.tools.web_search.WebSearchWorker` – Input `WebSearchInput`, Kontext über ToolContext (ScopeContext + BusinessContext) mit `tenant_id`, `trace_id`, Laufzeit-ID und optionalen Business-IDs; Output `WebSearchResponse` mit `ToolOutcome` und Suchergebnissen.
- `ai_core.tools.search_adapters.google.GoogleSearchAdapter` – Provider-spezifischer Adapter; erzeugt `SearchAdapterResponse` und normalisiert Links, übernimmt Kontext aus aufrufendem Worker.
- Framework-/Shared-Tools (`ai_core.tools.framework_contracts`, `ai_core.tools.shared_workers`) folgen dem `ToolContext`-Vertrag: `tenant_id`, `trace_id`, Laufzeit-ID (`run_id` und/oder `ingestion_run_id`), Business-IDs über `context.business`.

## ID-Propagation je Szenario

### HTTP ? Django ? Celery ? Graph (Collection Search)

1. HTTP-Request liefert `X-Tenant-ID`, optional `X-Case-ID`, `X-Trace-ID`.
2. Middleware (`RequestContextMiddleware`) ruft `normalize_request` auf, erstellt `ScopeContext` und validiert IDs (z.B. Tenant-Existenz); Business-IDs kommen aus Headern.
3. View nutzt `request.scope_context` plus `business_context`.
4. Dispatcher serialisiert `ScopeContext` + `BusinessContext` + `ToolContext` und legt Celery-Task `llm_worker.tasks.run_graph` in Queue `agents`.
5. Graph baut `ToolContext` aus `meta`, validiert Pflichtfelder, propagiert IDs in jeden Tool-Call (`WebSearchWorker`, Hybrid-Score) via `ToolContext`.
6. Tools erzeugen `invocation_id` und geben Outcomes mit identischen Kontextfeldern an Langfuse/Logs weiter.

### Websearch + Auto-Ingest

1. Collection- oder External-Knowledge-Graph entscheidet ?ber Ingestion (`auto_ingest` oder HITL approved).
2. Graph erzeugt `ingestion_run_id` am Trigger-Punkt und ?bergibt Scope-/Business-IDs in `meta` (Business: `case_id`, `workflow_id`, `collection_id`).
3. Celery-Task `ai_core.tasks.run_ingestion_graph` startet mit `state/meta`; Phase-Spans (`crawler.ingestion.*`) f?hren `trace_id`, `ingestion_run_id`, `workflow_id`.
4. Nach Abschluss wird `CaseEvent`/`ingestion.end` mit denselben IDs emittiert.

### Cron-Crawler ohne Case

1. Scheduler ruft `run_ingestion_graph` mit `state/meta` auf; `business_context` kann **ohne** `case_id` gebaut werden.
2. Validation-Policy erlaubt fehlende `case_id` f?r System-Tasks; Graph markiert Events als systemisch.
3. Tools erhalten ToolContext mit `ingestion_run_id`, `tenant_id`, `trace_id`; `workflow_id` beschreibt den Crawler-Flow.

### Multi-Step Graph ? Graph

1. Ein Graph ruft einen nachgelagerten Graph (z.B. `collection_search` ? `hybrid_search_and_score`).
2. Aufrufer gibt `tenant_id`, `workflow_id` und **denselben** `trace_id` weiter; `case_id` liegt in `business_context`.
3. `run_id` wird pro Ausf?hrung neu generiert; Kind-Graph darf `ingestion_run_id` nur setzen, wenn er Ingestion startet.

### 1.2 OpenTelemetry & W3C Trace Context

To support distributed tracing across process boundaries (e.g. Django -> Celery), we adopt the **W3C Trace Context** standard.

- **`traceparent` Header**: Carries the `version-trace_id-parent_id-flags`.
- **Alignment**: The `trace_id` in our `ScopeContext` MUST match the `trace_id` in the W3C header.
- **Propagation**:
  - **HTTP**: `headers` (standard).
  - **Celery**: `task.request.headers` (standard) or `task_kwargs["headers"]`.
  - **Langfuse**: Mapped from OTel span context automatically.

### 1.3 Context Priority

When resolving IDs, the priority is:

1. **ToolContext** (ScopeContext + BusinessContext)
2. **OpenTelemetry Span** (Infrastructure source of truth)
3. **HTTP Headers / Task Metadata** (Transport)

For `trace_id`, if an active OTel span exists, it takes precedence for generation/alignment.
## Validierung und Fehlerpfade

- **Pflicht**: `tenant_id` immer; `case_id` Pflicht für fachliche Workloads; `trace_id` Pflicht für Observability; genau eine Laufzeit-ID.
- **Fail-fast**: Graph-Inputs (GraphIOSpec-boundary models, Tool-Inputs) verwenden `extra=forbid` und Feldvalidatoren. Fehlende oder leere Felder brechen den Start ab.
- **System-Tasks**: `case_id` optional; Events markieren `system_task=true` im Metadata-Feld.
- **Fallback**: Kein Default-Case; fehlende Cases müssen explizit erstellt werden (Auto-Create-Flag) oder führen zu 4xx.

## Observability

- `trace_id` bleibt identisch über alle Layer; `workflow_id`, `case_id`, `run_id`/`ingestion_run_id` werden als Span-Attribute geführt.
- LangGraph-Transitions mappen auf Span-Namen (`crawler.ingestion.update_status`, `ingest`, `document_pipeline`, etc.).
- Logs/Events spiegeln ID-Payloads (`case_lifecycle`, `ingestion.end`) und dienen als Auditquelle.
- Telemetry uses `user_id` for the user and `tenant_id` for the tenant; never swap them.
