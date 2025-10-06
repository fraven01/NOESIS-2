# Dokumentlebenszyklus im RAG-Store

Dieser Leitfaden beschreibt, wie Dokumente nach dem Upload durch Ingestion, Retrieval und Pflege laufen. Der Fokus liegt auf Löschpfaden sowie Verantwortlichkeiten für Entwickler- und Administrator-Rollen.

## Überblick über den Lebenszyklus

1. **Upload** – Dateien gelangen über UI, API oder Batch-Import in die Ingestion-Warteschlange. Metadaten enthalten mindestens `tenant_id`, `source`, `hash` und optionale Klassifizierungen (z. B. Prozess, Dokumentklasse).
2. **Ingestion** – Loader, Chunker und Embedder schreiben Chunks mit Embeddings nach `pgvector`. Konsistenzprüfungen (Hash, Dimension, Tenant-Isolation) sind in [Ingestion](ingestion.md) dokumentiert.
3. **Retrieval** – Der `VectorStoreRouter` liest nur aktive Dokumente (`deleted_at IS NULL`). Reranker und Agenten sehen dadurch ausschließlich gültige Inhalte pro Tenant.
4. **Pflege & Löschpfade** – Routinejobs prüfen Staleness, führen Re-Embeddings durch und setzen Lösch-Flags. Soft- und Hard-Delete unterscheiden Verantwortlichkeiten sowie Audit-Anforderungen.

## Soft-Delete

- **Flag `deleted_at`**: Soft-Delete setzt einen Zeitstempel auf Dokument- und Chunk-Ebene. Der Datensatz bleibt erhalten, ist aber für Retrieval gesperrt.
- **Router-Verhalten**: Der `VectorStoreRouter` und alle zugehörigen `search`-Queries filtern auf `deleted_at IS NULL`. Smoke-Tests validieren diese Filter, bevor neue Router-Regeln produktiv gehen.
- **Empfohlene Nutzung**: Entwickler:innen nutzen Soft-Delete für Korrekturen (z. B. fehlerhafte Ingestion, temporäre Deaktivierung). Wiederherstellungen erfolgen über das Entfernen des Flags oder erneute Ingestion.
- **Monitoring**: Soft-gelöschte Datensätze erscheinen in Pflege-Dashboards (Langfuse/BI). Cleanup-Schwellenwerte (z. B. >30 Tage) lösen Hard-Delete-Review aus.

## Hard-Delete

- **Admin-Checks**: Vor Hard-Delete prüft ein*e Administrator:in den Kontext (Tenant-Zuordnung, Audit-Trail, rechtliche Vorgaben). Entscheidung wird im Ticketing-System dokumentiert.
- **Audit & Nacharbeiten**:
  - Entfernt Datensätze aus `documents`, `chunks`, `embeddings`, ggf. S3/Blob-Quellen und Trace-Anreicherungen.
  - Erzwingt Reindexing oder Cache-Invalidierung in Downstream-Systemen (Suche, Reporting).
  - Aktualisiert Audit-Log (z. B. `rag_document_audit`) mit Referenz auf Ticket und verantwortliche Person.
- **Verantwortung**: Hard-Delete ist auf Administrator:innen bzw. Data-Ops beschränkt. Entwickler:innen lösen Hard-Delete nur nach Freigabe durch das Betriebs-Team aus.

## API-Endpunkt `POST /api/v1/documents/delete/`

| Parameter | Typ | Beschreibung |
| --- | --- | --- |
| `mode` | `soft_delete` \| `hard_delete` | Steuert, ob `deleted_at` gesetzt oder die Daten physisch entfernt werden. Default: `soft_delete`. |
| `dry_run` | bool | Standard `true`. Bei `true` werden betroffene Dokumente lediglich aufgelistet und Validierungen ausgeführt, ohne Änderungen zu persistieren. |
| `document_ids` | Array[UUID] | Pflichtfeld. Enthält die Ziel-Dokumente innerhalb des Tenants. |

### Ablauf

1. Authentifizierte Anfrage mit Tenant-Kontext (`X-Tenant-ID`).
2. Service prüft Berechtigungen: Soft-Delete erfordert Entwickler:innen- oder höherwertige Rolle; Hard-Delete benötigt Administrator:innenrechte plus Audit-Ticket.
3. Bei `dry_run=true` liefert die Antwort eine Preview (Dokumente, Chunks, Embeddings, letzte `retrieved_at`).
4. Bei `dry_run=false` führt der Service je nach `mode` das Setzen von `deleted_at` bzw. die physische Löschung inklusive Audit-Log durch.

## Workflows & Verantwortlichkeiten

| Rolle | Aufgaben | Tools/Artefakte |
| --- | --- | --- |
| Entwickler:innen | Soft-Delete für fehlerhafte Uploads, Re-Upload nach Fix, Dokumentation im Issue-Tracker. | Endpoint mit `mode=soft_delete`, `dry_run=true` → Review → `dry_run=false`. |
| Administrator:innen/Data-Ops | Freigabe & Durchführung von Hard-Deletes, Pflege von Audit-Logs, Nacharbeiten in Downstream-Systemen. | Endpoint mit `mode=hard_delete`, obligatorisches `dry_run`-Review, Ticket-Verlinkung, Post-Delete-Checks. |
| QA/Support | Meldet betroffene Dokumente, initiiert Soft-Delete-Requests, prüft, ob Retrieval wieder korrekt arbeitet. | Support-Playbooks, Monitoring-Dashboards (Langfuse, BI). |

**Best Practices**

- Kombiniere Soft-Delete und erneute Ingestion für Rollbacks statt sofortem Hard-Delete.
- Halte `dry_run=true` als Pflichtschritt in allen Skripten und CLI-Tools; automatisierte Pipelines failen, wenn `dry_run` übersprungen wird.
- Dokumentiere Löschentscheidungen in zentralen Tickets und referenziere Audit-Logs sowie beteiligte Personen.
- Plane Hard-Deletes in Wartungsfenstern, damit Reindexing und Cache-Invalidierung ohne Laufzeiteinfluss erfolgen.
