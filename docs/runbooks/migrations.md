# Warum
Schemaänderungen sind kritisch für Tenant-Daten. Dieses Runbook stellt sicher, dass Migrationen reproduzierbar und ohne Downtime laufen.

# Wie
## Vorbereitung
- Prüfe das Migration-Diff im Repo und stelle sicher, dass `manage.py migrate_schemas` idempotent ist (kein Datenverlust bei erneutem Lauf).
- Kontrolliere, dass aktuelle Backups verfügbar sind: in Staging ein manuelles Snapshot, in Prod PITR + letztes tägliches Backup.
- Verifiziere, dass keine Web/Worker-Container automatische Migrationen ausführen ([Docker-Konventionen](../docker/conventions.md)).

## Durchführung
- Verwende ausschließlich den Cloud Run Job `noesis2-migrate`. Er nutzt das Web-Image und führt `python manage.py migrate_schemas --noinput` aus.
- Job-Konfiguration: Timeout 600 s (Staging) bzw. 900 s (Prod), `maxRetries` 0 (Staging) bzw. 1 (Prod).
- Stelle sicher, dass `DJANGO_SETTINGS_MODULE=noesis2.settings.production` gesetzt ist und dass Cloud SQL/Redis-Verbindungen vorhanden sind.
- Vector-spezifische DDL (Tabellen, Indizes) laufen separat in der Pipeline-Stufe „Vector-Schema-Migrations“. Skript liegt in [`docs/rag/schema.sql`](../rag/schema.sql) und wird per `psql` oder Cloud SQL Proxy angewendet.

## Validierung
- Überwache Job-Logs in Cloud Logging: Erfolgsnachricht `Job completed successfully` und Exit-Code 0 bestätigen den Abschluss.
- Führe direkt im Anschluss Readiness-Checks aus: HTTP 200 auf `/` und `/tenant-demo/`, Celery-Queue leer (`redis` Keys stabil), LiteLLM `/health` grün.
- Dokumentiere Schema-Versionen (`django_migrations` Tabelle) und vergleiche sie mit erwarteten Migrationen.

## Rollback
- Bricht der Job, bleibt Schema unverändert. Analysiere Logs, fixiere Migration und wiederhole den Job.
- Bei teilweisen Änderungen nutze Staging-Backup bzw. Prod-PITR, um in ein konsistentes Zeitfenster vor dem Lauf zurückzuspringen.
- Nach einem Rollback wird der Job mit derselben Image-Version erneut gestartet, nachdem das Problem behoben wurde.

## pgvector-Schema
- Tabellenlayout laut [`docs/rag/schema.sql`](../rag/schema.sql): `documents`, `chunks`, `embeddings` mit Foreign Keys und `metadata` als `jsonb`.
- Indizes: Primärschlüssel je Tabelle, GIN auf `metadata`, Vektorindex IVFFLAT oder HNSW auf `embeddings.embedding` (`cosine`).
- DDL ist idempotent: `CREATE TABLE IF NOT EXISTS`, `CREATE INDEX IF NOT EXISTS` und `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`.
- Führe `ANALYZE` nach großen Uploads aus und plane `REINDEX CONCURRENTLY` bei Index-Wechseln.
- Rollback bei fehlerhaften Index-Updates: `DROP INDEX CONCURRENTLY`, danach vorherige Definition erneut ausführen; Daten bleiben unverändert.

# Schritte
1. Sichere, dass Backups aktuell sind und Approval für Migration vorliegt.
2. Trigger den Cloud Run Migrate-Job (Staging automatisch via [Pipeline](../cicd/pipeline.md), Prod nach Approval manuell/CI).
3. Überwache Logs und Validierungschecks laut Abschnitt „Validierung“.
4. Melde den Status im Release-Channel und dokumentiere Schema-Versionen.
5. Bei Fehlern: Stoppe weiteren Traffic-Shift, führe Rollback laut Abschnitt „Rollback“ durch und wiederhole Schritt 2 nach Fix.
