# Dokumentlebenszyklus im RAG-Store

Dieser Leitfaden beschreibt, wie Dokumente nach dem Upload durch Ingestion, Retrieval und Pflege laufen. Der Fokus liegt auf Löschpfaden sowie Verantwortlichkeiten für Entwickler- und Administrator-Rollen.

## Überblick über den Lebenszyklus

1. **Upload** – Dateien gelangen über UI, API oder Batch-Import in die Ingestion-Warteschlange. Metadaten enthalten mindestens `tenant_id`, `source`, `hash` und optionale Klassifizierungen (z. B. Prozess, Dokumentklasse).
2. **Ingestion** – Loader, Chunker und Embedder schreiben Chunks mit Embeddings nach `pgvector`. Konsistenzprüfungen (Hash, Dimension, Tenant-Isolation) sind in [Ingestion](ingestion.md) dokumentiert.
3. **Retrieval** – Der `VectorStoreRouter` ruft den Retriever auf, der `deleted_at` auf Dokument- (`documents.deleted_at IS NULL`) und Chunk-Ebene (`chunks.metadata->>'deleted_at' IS NULL`) standardmäßig filtert. Damit ist die in dieser Notiz zuvor dokumentierte Lücke geschlossen: Soft-gelöschte Einträge bleiben ohne Admin-Override unsichtbar, Smoke-Tests können direkt über die Suche validieren, dass Löschungen greifen. Die Sichtbarkeitsregel wird explizit über das Feld `visibility` gesteuert (siehe Abschnitt „Soft-Delete“).
4. **Pflege & Löschpfade** – Routinejobs prüfen Staleness, führen Re-Embeddings durch und setzen Lösch-Flags. Soft- und Hard-Delete unterscheiden Verantwortlichkeiten sowie Audit-Anforderungen.

## Soft-Delete

- **Flag `deleted_at`**: Soft-Delete setzt einen Zeitstempel auf Dokument- und Chunk-Ebene. Der Datensatz bleibt erhalten und dient als Markierung – der Retriever blendet ihn jedoch per Default aus, bis Admin-Kontexte explizit eine erweiterte Sicht aktivieren.
- **Router-Verhalten**: Der `VectorStoreRouter` injiziert standardmäßig `visibility="active"` und delegiert mit `meta.visibility_effective` an den Retriever, der `deleted_at` filtert. Admin- oder Service-Kontexte, die gelöschte Inhalte sehen müssen, aktivieren dies explizit über den Guard (siehe Roadmap für Auth).
- **Sichtbarkeitsoptionen**: Clients können optional `visibility` setzen:
  - `"active"` (Default) blendet Soft-Deletes aus und entspricht dem bisherigen Verhalten der Suche.
  - `"all"` zeigt aktive und Soft-Delete-Dokumente gleichzeitig an, sofern der Guard die Anfrage autorisiert.
  - `"deleted"` liefert ausschließlich Soft-Delete-Dokumente für Administrations- oder Reviewzwecke.
  Der Guard erzwingt `"active"`, wenn keine Berechtigung vorliegt; die gewählte bzw. effektive Einstellung wird immer als `meta.visibility_effective` zurückgegeben.
- **Empfohlene Nutzung**: Entwickler:innen nutzen Soft-Delete für Korrekturen (z. B. fehlerhafte Ingestion, temporäre Deaktivierung). Wiederherstellungen erfolgen über das Entfernen des Flags oder erneute Ingestion.
- **Monitoring**: Soft-gelöschte Datensätze erscheinen in Pflege-Dashboards (Langfuse/BI). Cleanup-Schwellenwerte (z. B. >30 Tage) lösen Hard-Delete-Review aus. Trefferkontrollen müssen auf SQL-Queries oder angepasste Retrieval-Pfade zurückgreifen.

## Hard-Delete

- **Admin-Checks**: Vor Hard-Delete prüft ein*e Administrator:in den Kontext (Tenant-Zuordnung, Audit-Trail, rechtliche Vorgaben). Entscheidung wird im Ticketing-System dokumentiert.
- **Audit & Nacharbeiten**:
  - Entfernt Datensätze aus `documents`, `chunks`, `embeddings`, ggf. S3/Blob-Quellen und Trace-Anreicherungen.
  - Erzwingt Reindexing oder Cache-Invalidierung in Downstream-Systemen (Suche, Reporting).
  - Aktualisiert Audit-Log (z. B. `rag_document_audit`) mit Referenz auf Ticket und verantwortliche Person. Der Celery-Task `rag.hard_delete` übernimmt diese Schritte inklusive VACUUM/Reindex-Triggern automatisch und protokolliert ein `rag.hard_delete.audit`-Event.
- **Verantwortung**: Hard-Delete ist auf Administrator:innen bzw. Data-Ops beschränkt. Entwickler:innen lösen Hard-Delete nur nach Freigabe durch das Betriebs-Team aus.

## Operative Durchführung

Hard-Delete-Aufträge werden bevorzugt über den Admin-Endpunkt [`POST /ai/rag/admin/hard-delete/`](../api/reference.md#post-airagadminhard-delete) ausgelöst. Der Endpoint kapselt Autorisierung (Service-Key oder aktive Admin-Session), erzeugt einen Trace (`trace_id`) und reicht den Task `rag.hard_delete` mit Audit-Metadaten an die Worker weiter. Das [Runbook „RAG-Dokumente löschen & pflegen“](../runbooks/rag_delete.md) beschreibt zusätzlich den manuellen Fallback (direkter Task-Aufruf oder SQL), falls der Endpoint temporär nicht verfügbar ist.

## Workflows & Verantwortlichkeiten

| Rolle | Aufgaben | Tools/Artefakte |
| --- | --- | --- |
| Entwickler:innen | Soft-Delete für fehlerhafte Uploads, Re-Upload nach Fix, Dokumentation im Issue-Tracker. | SQL-Skript aus E1 des Runbooks, danach Router-Invalidierung & SQL-Abgleich (`deleted_at IS NULL`). |
| Administrator:innen/Data-Ops | Freigabe & Durchführung von Hard-Deletes, Pflege von Audit-Logs, Nacharbeiten in Downstream-Systemen. | Celery-Task `rag.hard_delete` (oder Fallback-SQL), Audit-Log-Eintrag, Post-Delete-Checks. |
| QA/Support | Meldet betroffene Dokumente, initiiert Soft-Delete-Requests, prüft, ob Retrieval (mit explizitem Filter) wieder korrekt arbeitet. | Support-Playbooks, Monitoring-Dashboards (Langfuse, BI). |

**Best Practices**

- Kombiniere Soft-Delete und erneute Ingestion für Rollbacks statt sofortem Hard-Delete.
- Dokumentiere Löschentscheidungen in zentralen Tickets und referenziere Audit-Logs sowie beteiligte Personen.
- Plane Hard-Deletes in Wartungsfenstern, damit Reindexing und Cache-Invalidierung ohne Laufzeiteinfluss erfolgen.
