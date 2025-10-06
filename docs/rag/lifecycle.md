# Dokumentlebenszyklus im RAG-Store

Dieser Leitfaden beschreibt, wie Dokumente nach dem Upload durch Ingestion, Retrieval und Pflege laufen. Der Fokus liegt auf Löschpfaden sowie Verantwortlichkeiten für Entwickler- und Administrator-Rollen.

## Überblick über den Lebenszyklus

1. **Upload** – Dateien gelangen über UI, API oder Batch-Import in die Ingestion-Warteschlange. Metadaten enthalten mindestens `tenant_id`, `source`, `hash` und optionale Klassifizierungen (z. B. Prozess, Dokumentklasse).
2. **Ingestion** – Loader, Chunker und Embedder schreiben Chunks mit Embeddings nach `pgvector`. Konsistenzprüfungen (Hash, Dimension, Tenant-Isolation) sind in [Ingestion](ingestion.md) dokumentiert.
3. **Retrieval** – Der `VectorStoreRouter` kennt aktuell keinen Filter auf `deleted_at`. Soft-gelöschte Einträge bleiben für Suchanfragen sichtbar, bis nachgelagerte Komponenten oder Operator:innen sie explizit ausschließen.
4. **Pflege & Löschpfade** – Routinejobs prüfen Staleness, führen Re-Embeddings durch und setzen Lösch-Flags. Soft- und Hard-Delete unterscheiden Verantwortlichkeiten sowie Audit-Anforderungen.

## Soft-Delete

- **Flag `deleted_at`**: Soft-Delete setzt einen Zeitstempel auf Dokument- und Chunk-Ebene. Der Datensatz bleibt erhalten und dient als Markierung – ohne zusätzliche Filter erscheint er weiterhin im Retrieval.
- **Router-Verhalten**: Der `VectorStoreRouter` reicht Soft-Delete-Flags unverändert durch. Wer Soft-Deletes produktiv nutzen will, muss die Filterung in der eigenen Retrieval-Logik ergänzen oder auf Hard-Delete ausweichen.
- **Empfohlene Nutzung**: Entwickler:innen nutzen Soft-Delete für Korrekturen (z. B. fehlerhafte Ingestion, temporäre Deaktivierung). Wiederherstellungen erfolgen über das Entfernen des Flags oder erneute Ingestion.
- **Monitoring**: Soft-gelöschte Datensätze erscheinen in Pflege-Dashboards (Langfuse/BI). Cleanup-Schwellenwerte (z. B. >30 Tage) lösen Hard-Delete-Review aus.

## Hard-Delete

- **Admin-Checks**: Vor Hard-Delete prüft ein*e Administrator:in den Kontext (Tenant-Zuordnung, Audit-Trail, rechtliche Vorgaben). Entscheidung wird im Ticketing-System dokumentiert.
- **Audit & Nacharbeiten**:
  - Entfernt Datensätze aus `documents`, `chunks`, `embeddings`, ggf. S3/Blob-Quellen und Trace-Anreicherungen.
  - Erzwingt Reindexing oder Cache-Invalidierung in Downstream-Systemen (Suche, Reporting).
  - Aktualisiert Audit-Log (z. B. `rag_document_audit`) mit Referenz auf Ticket und verantwortliche Person.
- **Verantwortung**: Hard-Delete ist auf Administrator:innen bzw. Data-Ops beschränkt. Entwickler:innen lösen Hard-Delete nur nach Freigabe durch das Betriebs-Team aus.

## Operative Durchführung

Der beschriebene HTTP-Endpunkt für Löschvorgänge ist noch nicht implementiert. Bis zur Bereitstellung eines Services erfolgt die Durchführung über das [Runbook „RAG-Dokumente löschen & pflegen“](../runbooks/rag_delete.md). Dieses Runbook beschreibt sowohl das Setzen von `deleted_at` per SQL als auch die physischen Löschschritte inkl. Audit-Logging.

## Workflows & Verantwortlichkeiten

| Rolle | Aufgaben | Tools/Artefakte |
| --- | --- | --- |
| Entwickler:innen | Soft-Delete für fehlerhafte Uploads, Re-Upload nach Fix, Dokumentation im Issue-Tracker. | SQL-Skript aus E1 des Runbooks, danach Router-Invalidierung & Smoke-Test. |
| Administrator:innen/Data-Ops | Freigabe & Durchführung von Hard-Deletes, Pflege von Audit-Logs, Nacharbeiten in Downstream-Systemen. | SQL-Skript aus E2 des Runbooks, Audit-Log-Eintrag, Post-Delete-Checks. |
| QA/Support | Meldet betroffene Dokumente, initiiert Soft-Delete-Requests, prüft, ob Retrieval wieder korrekt arbeitet. | Support-Playbooks, Monitoring-Dashboards (Langfuse, BI). |

**Best Practices**

- Kombiniere Soft-Delete und erneute Ingestion für Rollbacks statt sofortem Hard-Delete.
- Halte `dry_run=true` als Pflichtschritt in allen Skripten und CLI-Tools; automatisierte Pipelines failen, wenn `dry_run` übersprungen wird.
- Dokumentiere Löschentscheidungen in zentralen Tickets und referenziere Audit-Logs sowie beteiligte Personen.
- Plane Hard-Deletes in Wartungsfenstern, damit Reindexing und Cache-Invalidierung ohne Laufzeiteinfluss erfolgen.
