# ADR-004: Case-Lifecycle und Phasen-Konfiguration

## Status
Accepted

## Kontext
Cases besitzen `status` und `phase`, aktuell mit Beispielphasen. Events (`CaseEvent`) verfolgen Workflow-, Trace- und Collection-IDs. Tenants benötigen eigene Phasen- und Event-Mappings.

## Entscheidung
- Phasen sind mandantenspezifisch konfigurierbar über eine zentrale Registry (z. B. Tenant-Config), nicht hardcodiert im Graph.
- Event-Namen werden standardisiert und tenant-übergreifend verwendet (`document.ingested`, `search.completed`, `ingestion.failed`, `hitl.pending`). Mapping auf Mandantenphasen erfolgt in der Registry.
- Graphen emittieren nur Events mit neutralen Namen und ID-Payload (`tenant_id`, `case_id`, `workflow_id`, `trace_id`, Laufzeit-ID, `collection_id`), keine UI-spezifischen Payloads.

## Konsequenzen
- Neue Tenants können Phasen hinzufügen, ohne Graph-Code zu ändern; nur Mapping/Config wird erweitert.
- Case-Lifecycle-Auswertungen bleiben vergleichbar, weil Event-Namen stabil sind.
- HITL-Gates (z. B. in External Knowledge) publizieren `hitl.pending` Events mit Review-Metadaten und überlassen die Phasen-Übersetzung dem Tenant.

