# Case Domain Overview

Cases bilden den geschäftlichen Kontext einer Anfrage (z. B. Anzeige, Vertragsverhandlung,
Monitoring). Das Domänenmodell ergänzt die technischen Ingestion-Runs und Vektor-
Collections um eine stabile Business-ID.

## Modelle

- **`Case`**
  - Primärschlüssel: `id` (UUID).
  - `tenant` ForeignKey auf `customers.Tenant`.
  - `external_id` entspricht dem bisherigen `case_id`/`X-Case-ID` Header und bleibt
    stabil über alle APIs.
  - Optionale Attribute: `title`, `status` (`open`/`closed`), `phase`, `metadata`,
    `created_at`/`updated_at`/`closed_at`.
- **`CaseEvent`**
  - Referenziert `Case` und `Tenant`.
  - `event_type`, `source`, optional `graph_name`.
  - Optionale Referenzen: `ingestion_run`, `workflow_id`, `collection_id`, `trace_id`.
  - `payload` kapselt weitere Details (z. B. Counters eines Ingestion-Laufs).
- **`DocumentCollection`**
  - Ordnet menschlich lesbare Keys (`key`, `name`) einem technischen `collection_id`
    zu und kann optional einem `Case` zugewiesen sein.

## Beziehungen

```
Case ──< CaseEvent
  │         │
  │         └── (optional) DocumentIngestionRun
  │
  └── (optional) DocumentCollection ──> vector collection_id
```

- `DocumentIngestionRun.case` speichert weiterhin den String (`external_id`).
  Backfill-Migrationen sorgen dafür, dass für jede Kombination aus
  `(tenant_id, case)` ein `Case` existiert.
- `case_id` wird in allen Graphen, Worker-Tasks und ToolContext-Feldern
  in `Case.external_id` aufgelöst. `assert_case_active` blockt Requests,
  wenn der Status `closed` ist.
- RAG-Metadaten enthalten `case_id` neben `tenant_id` und `collection_id`,
  sodass Filter in `vector_client` entlang aller drei Achsen möglich sind.

## Lifecycle und Events

- Ingestion:
  - `DocumentIngestionRun` Statusänderungen erzeugen
    `CaseEvent`-Typen wie `ingestion_run_queued`, `…_started`, `…_completed`,
    `…_failed`.
  - Die Events werden als `case.lifecycle.ingestion` Observability-Events
    nach Langfuse gespiegelt (Tags: `case_id`, `case_status`, `case_phase`,
    `ingestion_run_id`, `collection_scope`).
- CollectionSearchGraph:
  - Jeder Übergang (`strategy_generated`, `ingest_triggered`, `hitl_pending`,
    `verified`, …) erzeugt ein `CaseEvent` mit Präfix `collection_search:`.
  - Die Lifecycle-Heuristik setzt `Case.phase` abhängig von den Events und
    schreibt strukturierte Logs (Tenant, Case-ID, neue Phase, Eventtyp).
- Tenant-spezifische Lifecycle-Definitionen (JSON auf `Tenant`) bestimmen,
  welche Events welche Phasen auslösen (z. B. Anzeige → Verhandlung → Unterschrift).

## Beispielablauf

1. API-Request mit `X-Tenant-ID` und `X-Case-ID=crm-42` erstellt einen Ingestion-Run.
2. Beim Anlegen des `DocumentIngestionRun` entsteht (falls nötig) ein `Case`
   `external_id="crm-42"`.
3. Der Worker aktualisiert den Run (`running → completed`) und erzeugt passende
   `CaseEvent`-Einträge. Die Lifecycle-Definition setzt `Case.phase` auf
   `evidence_collection` bzw. `search_completed`.
4. `CollectionSearchGraph` wird gestartet; Transitionen schreiben weitere
   Events und aktualisieren `Case.phase` (z. B. `external_review`).
5. Observability:
   - Traces/Events erhalten Tags `case_id`, `case_status`, `case_phase`.
   - Langfuse kann anhand `case.lifecycle.*` Events den kompletten Verlauf eines
     Cases (HTTP → Ingestion → Graph → RAG) nachvollziehen.
