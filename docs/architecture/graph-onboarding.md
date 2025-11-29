# Onboarding Guide für neue Graph-Entwickler:innen

Dieses Handbuch stellt sicher, dass neue Graphen und Worker die ID- und Lifecycle-Regeln einhalten.

## 1. Grundlagen verstehen
- Lies `docs/ids.md` und `docs/architecture/id-semantics.md` für Begriffe & Lebenszyklen.
- Prüfe bestehende Graphen (`ai_core/graphs/collection_search.py`, `external_knowledge_graph.py`) als Referenz für Kontext-Validierung (`GraphContextPayload`).
- Tools müssen den Vertrag aus `ai_core/tools/framework_contracts.py` respektieren (`ToolContext`, `ToolError`).

## 2. Kontextmodelle definieren
- Definiere `GraphContextPayload` mit `tenant_id`, `workflow_id`, `case_id` (pflicht), `trace_id`, `run_id`/`ingestion_run_id` (XOR), `extra="forbid"`, `frozen=True`.
- Lege ein `GraphInput`-Modell mit Validatoren und Beispielen an.

## 3. ID-Propagation im Graph
- Beim Start: generiere `run_id` falls nicht vorhanden, **nicht** `workflow_id`.
- Reiche Kontext an jeden Node/Tool weiter (`tenant_id`, `trace_id`, `workflow_id`, `case_id`, Laufzeit-ID, `collection_id`/`document_id` falls relevant).
- Bei Ingestion-Triggern: erzeuge `ingestion_run_id` und protokolliere Übergabe.

## 4. Case-Lifecycle & Events
- Nutze standardisierte Events (`search.completed`, `document.ingested`, `hitl.pending`).
- Phasen-Mapping erfolgt per Tenant-Config; Graph kodiert nur neutrale Events.
- Für HITL: speichere `trace_id`, `run_id`, `workflow_id`, `case_id` im Review-Payload.

## 5. Validierung & Fehlerbehandlung
- Verwende `assert_case_active` im HTTP-Einstieg oder entsprechendes Guarding im Scheduler.
- Tools werfen `InputError` bei fehlenden IDs; Graph bricht vor externen Calls ab.
- System-Tasks ohne `case_id` kennzeichnen `system_task=true` in Logs/Events.

## 6. Observability
- Nutze `observe_span`/`record_span` mit Attributen: `tenant_id`, `trace_id`, `workflow_id`, `case_id`, Laufzeit-ID, `collection_id`.
- Leitfaden: `docs/observability/langfuse.md` und Code in `ai_core.infra.observability`.

## 7. Übergabe an Worker/Tasks
- Celery-Tasks deklarieren Kontext-Parameter explizit (`tenant_id`, `case_id`, `trace_id`, `workflow_id`, Laufzeit-ID, `collection_id`).
- Verwende `ScopedTask`/`with_scope_apply_async`, damit Maskierung/Headers erhalten bleiben.
- Child-Graph-Aufrufe behalten `trace_id`, setzen neue `run_id`.

## 8. Abschluss-Check
- Prüfe die [ID-Sync-Checklist](./id-sync-checklist.md).
- Ergänze Beispiele in den Pydantic-Modellen und aktualisiere Dokumentation, wenn neue Events/Phasen eingeführt werden.

