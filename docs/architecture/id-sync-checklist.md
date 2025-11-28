# ID-Sync-Checklist

Verbindliche Prüfliste für neue Graphen, Tools und Worker.

## Kontextaufnahme
- [ ] `tenant_id` vorhanden und aus Header/Token gelesen
- [ ] `trace_id` gesetzt (bei Bedarf generieren) und unverändert weitergereicht
- [ ] `case_id` gesetzt oder bewusst als System-Task dokumentiert
- [ ] `workflow_id` gesetzt vom Dispatcher (nicht im Graph generieren)
- [ ] Laufzeit-ID gewählt: `run_id` **oder** `ingestion_run_id` (XOR)

## Input-Modelle
- [ ] Pydantic `BaseModel` mit `frozen=True`, `extra="forbid"`
- [ ] Feldvalidatoren trimmen Strings und lehnen leere Werte ab
- [ ] Beispiele (`model_config['json_schema_extra']` oder `examples`) gepflegt

## Propagation
- [ ] Kontext an jeden Tool-Call weitergeben (`tenant_id`, `trace_id`, `workflow_id`, `case_id`, Laufzeit-ID)
- [ ] Sub-Graph-Aufrufe behalten `trace_id`, wählen neue `run_id`
- [ ] Ingestion-Trigger erzeugen `ingestion_run_id` und propagieren `collection_id`

## Observability
- [ ] Spans/Logs enthalten konsistente Attribute (`tenant_id`, `trace_id`, `workflow_id`, `case_id`, Laufzeit-ID, `collection_id`)
- [ ] Langfuse/LangGraph Hooks aktiviert, Sampling-Regeln beachtet
- [ ] Events (z. B. `case_lifecycle`, `ingestion.end`) mit ID-Payload emittiert

## Validation & Fehlerpfade
- [ ] `assert_case_active` oder äquivalent genutzt (sofern fachlicher Flow)
- [ ] Fehlende IDs führen zu ValidationError/4xx bevor externe Calls passieren
- [ ] System-Tasks ohne `case_id` loggen Warnung + `system_task=true`

## Tenancy & Schema
- [ ] `tenant_schema` nur optional; nie als Ersatz für `tenant_id`
- [ ] Vector-/DB-Clients per Tenant geroutet
- [ ] Keine Ableitung von `tenant_id` aus Dokument-/Case-IDs

