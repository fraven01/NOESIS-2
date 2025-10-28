# Dokumentlebenszyklus im RAG-Store

Dieser Leitfaden beschreibt, wie Dokumente nach dem Upload durch Ingestion, Retrieval und Pflege laufen. Der Fokus liegt auf Löschpfaden sowie Verantwortlichkeiten für Entwickler- und Administrator-Rollen.

## Überblick über den Lebenszyklus

1. **Upload** – Dateien gelangen über UI, API oder Batch-Import in die Ingestion-Warteschlange. Metadaten enthalten mindestens `tenant_id`, `source`, `hash` und optionale Klassifizierungen (z. B. Prozess, Dokumentklasse).
2. **Ingestion** – Loader, Chunker und Embedder schreiben Chunks mit Embeddings nach `pgvector`. Konsistenzprüfungen (Hash, Dimension, Tenant-Isolation) sind in [Ingestion](ingestion.md) dokumentiert.
3. **Retrieval** – Der `VectorStoreRouter` ruft den Retriever auf, der den Lifecycle auf Dokument- (`documents.lifecycle = 'active'`) und Chunk-Ebene (`chunks.metadata->>'lifecycle_state' = 'active'`) standardmäßig filtert. Soft-gelöschte Datensätze sind dadurch in allen Standard-Retrievalpfaden unsichtbar, bis ein autorisierter Kontext eine andere Sichtbarkeit erzwingt. Smoke-Tests können direkt über die Suche validieren, dass Löschungen greifen. Die Sichtbarkeitsregel wird explizit über das Feld `visibility` gesteuert (siehe Abschnitt „Soft-Delete“).
4. **Pflege & Löschpfade** – Routinejobs prüfen Staleness, führen Re-Embeddings durch und setzen Lösch-Flags. Soft- und Hard-Delete unterscheiden Verantwortlichkeiten sowie Audit-Anforderungen.

## Soft-Delete

- **Lifecycle-Status `retired`**: Soft-Delete setzt den Lifecycle auf Dokument- und Chunk-Ebene auf `retired`. Der Datensatz bleibt erhalten und dient als Markierung – der Retriever blendet ihn jedoch per Default aus, bis Admin-Kontexte explizit eine erweiterte Sicht aktivieren. Der bisherige Zeitstempel `deleted_at` bleibt aus Kompatibilitätsgründen optional erhalten und wird beim Wechsel in einen nicht-aktiven Lifecycle gesetzt.
- **Router-Verhalten**: Der `VectorStoreRouter` injiziert standardmäßig `visibility="active"` und delegiert mit `meta.visibility_effective` an den Retriever, der auf `lifecycle='active'` filtert. Admin- oder Service-Kontexte, die gelöschte Inhalte sehen müssen, aktivieren dies explizit über den Guard (siehe Roadmap für Auth).
- **Sichtbarkeitsoptionen**: Clients können optional `visibility` setzen:
  - `"active"` (Default) blendet Soft-Deletes aus und entspricht dem bisherigen Verhalten der Suche.
  - `"all"` zeigt aktive und Soft-Delete-Dokumente gleichzeitig an, sofern der Guard die Anfrage autorisiert.
  - `"deleted"` liefert ausschließlich Soft-Delete-Dokumente für Administrations- oder Reviewzwecke.
  Der Guard erzwingt `"active"`, wenn keine Berechtigung vorliegt; die gewählte bzw. effektive Einstellung wird immer als `meta.visibility_effective` zurückgegeben.
- **Empfohlene Nutzung**: Entwickler:innen nutzen Soft-Delete für Korrekturen (z. B. fehlerhafte Ingestion, temporäre Deaktivierung). Wiederherstellungen erfolgen über das Entfernen des Flags oder erneute Ingestion.
- **Monitoring**: Soft-gelöschte Datensätze erscheinen in Pflege-Dashboards (Langfuse/BI). Cleanup-Schwellenwerte (z. B. >30 Tage) lösen Hard-Delete-Review aus. Trefferkontrollen müssen auf SQL-Queries oder angepasste Retrieval-Pfade zurückgreifen.

## Hard-Delete

- **Admin-Checks**: Vor Hard-Delete prüft ein*e Administrator:in den Kontext (Tenant-Zuordnung, Audit-Trail, rechtliche Vorgaben). Entscheidung wird im Ticketing-System dokumentiert.
- **Aktueller Ablauf**: Hard-Delete bleibt vorerst ein manueller Prozess gemäß [Runbook „RAG-Dokumente löschen & pflegen“](../runbooks/rag_delete.md). Administrator:innen nutzen die dort beschriebenen SQL- und Wartungsschritte, inklusive Dokumentation der Audit-Events und anschließender Router-Invalidierung.
- **Ausblick (TODO)**: In einer späteren Ausbaustufe wird ein eigener Celery-Task die Runbook-Schritte automatisieren: physisches Entfernen der Einträge aus `documents`, `chunks` und `embeddings`, Protokollierung eines Audit-Events, Invalidation des Router-Caches sowie optionale Wartung wie `VACUUM` oder `REINDEX`.
- **Verantwortung**: Hard-Delete ist auf Administrator:innen bzw. Data-Ops beschränkt. Entwickler:innen lösen Hard-Delete nur nach Freigabe durch das Betriebs-Team aus.

## Operative Durchführung

Hard-Delete-Aufträge werden bevorzugt über den Admin-Endpunkt [`POST /ai/rag/admin/hard-delete/`](../api/reference.md#post-airagadminhard-delete) ausgelöst. Der Endpoint kapselt Autorisierung (Service-Key oder aktive Admin-Session), erzeugt einen Trace (`trace_id`) und reicht den Task `rag.hard_delete` mit Audit-Metadaten an die Worker weiter. Die Verarbeitung läuft auf der Celery-Queue `rag_delete`, die von den Standard-Workern konsumiert wird (`celery -A noesis2 worker -l info -Q celery,rag_delete`). Das [Runbook „RAG-Dokumente löschen & pflegen“](../runbooks/rag_delete.md) beschreibt zusätzlich den manuellen Fallback (direkter Task-Aufruf oder SQL), falls der Endpoint temporär nicht verfügbar ist.

## Workflows & Verantwortlichkeiten

| Rolle | Aufgaben | Tools/Artefakte |
| --- | --- | --- |
| Entwickler:innen | Soft-Delete für fehlerhafte Uploads, Re-Upload nach Fix, Dokumentation im Issue-Tracker. | SQL-Skript aus E1 des Runbooks, danach Router-Invalidierung & SQL-Abgleich (`lifecycle = 'active'`). |
| Administrator:innen/Data-Ops | Freigabe & Durchführung von Hard-Deletes, Pflege von Audit-Logs, Nacharbeiten in Downstream-Systemen. | Celery-Task `rag.hard_delete` (oder Fallback-SQL), Audit-Log-Eintrag, Post-Delete-Checks. |
| QA/Support | Meldet betroffene Dokumente, initiiert Soft-Delete-Requests, prüft, ob Retrieval (mit explizitem Filter) wieder korrekt arbeitet. | Support-Playbooks, Monitoring-Dashboards (Langfuse, BI). |

**Best Practices**

- Kombiniere Soft-Delete und erneute Ingestion für Rollbacks statt sofortem Hard-Delete.
- Dokumentiere Löschentscheidungen in zentralen Tickets und referenziere Audit-Logs sowie beteiligte Personen.
- Plane Hard-Deletes in Wartungsfenstern, damit Reindexing und Cache-Invalidierung ohne Laufzeiteinfluss erfolgen.
