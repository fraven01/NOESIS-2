# Runbook: Django/Tenant Migrationen (Dev)

Ziel: Sichere, reproduzierbare Schritte für Schema‑Änderungen in der lokalen Entwicklungsumgebung mit `django-tenants`.

## Begriffe
- Shared Apps: laufen im `public`‑Schema (z. B. `customers`).
- Tenant Apps: laufen in je eigenem Tenant‑Schema (z. B. `users`, `projects`, …).
- Befehlsfamilie: `migrate_schemas --shared` vs. `migrate_schemas --tenant`.

## Voraussetzungen
- Stack läuft: `docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d`
- `.env` gesetzt (DB/Redis/Keys); Services erreichbar: `npm run win:dev:check` (Windows) oder `npm run dev:check` (Bash)

## Typische Workflows

1) Modelle geändert → Migrationen erzeugen
- Im Web‑Container ausführen (Dateien landen via Bind‑Mount im Repo):
  - Windows: `npm run win:dev:manage makemigrations <app>`
  - Bash: `npm run dev:manage makemigrations <app>`

### Legacy-Workflow-Tabellen abbauen

- Vor dem Ausführen von `migrate_schemas` die ehemaligen Workflow-Tabellen prüfen:
  ```sql
  SELECT 'workflows_workflowtemplate' AS table_name, COUNT(*) AS rows
  FROM {{SCHEMA_NAME}}.workflows_workflowtemplate
  UNION ALL
  SELECT 'workflows_workflowstep', COUNT(*)
  FROM {{SCHEMA_NAME}}.workflows_workflowstep
  UNION ALL
  SELECT 'workflows_workflowinstance', COUNT(*)
  FROM {{SCHEMA_NAME}}.workflows_workflowinstance;
  ```
- Wenn eine der Tabellen noch Daten enthält, sichere sie vor der Migration, z. B. via:
  ```sql
  \copy (
    SELECT *
    FROM {{SCHEMA_NAME}}.workflows_workflowinstance
    ORDER BY created_at
  ) TO 'legacy_workflowinstance.csv' WITH (FORMAT csv, HEADER true);
  ```
- Erst nach dem Backup `npm run dev:manage migrate_schemas --tenant` ausführen. Migration `common.0001_drop_legacy_workflows` löscht die Tabellen.

2) Shared‑Migrationen anwenden (falls Shared App betroffen)
- `npm run dev:manage migrate_schemas --shared`

3) Public Tenant sicherstellen (idempotent)
- `npm run dev:manage bootstrap_public_tenant --domain=localhost`

4) Tenant‑Migrationen anwenden
- `npm run dev:manage migrate_schemas --tenant`

5) Neuen Tenant anlegen (optional)
- Anlegen: `npm run dev:manage create_tenant --schema=dev2 --name="Dev 2" --domain=dev2.localhost`
- Superuser (nicht interaktiv):
  - Windows: `npm run win:dev:manage DJANGO_SUPERUSER_PASSWORD=admin123 create_tenant_superuser --schema=dev2 --username=admin --email=admin@example.com --noinput`
  - Bash:   `npm run dev:manage DJANGO_SUPERUSER_PASSWORD=admin123 create_tenant_superuser --schema=dev2 --username=admin --email=admin@example.com --noinput`

## Smoke‑Checks (nach Migrationen)
- API‑Ping (Header‑Routing):
  - `curl -i -H "X-Tenant-Schema: dev" -H "X-Tenant-ID: dev-tenant" -H "X-Case-ID: local" -H "Idempotency-Key: smoke-ping" http://localhost:8000/ai/ping/`
- Beispiel‑Graph:
  - `curl -s -X POST http://localhost:8000/v1/ai/rag/query/ -H "Content-Type: application/json" -H "X-Tenant-Schema: dev" -H "X-Tenant-ID: dev-tenant" -H "X-Case-ID: local" -H "Idempotency-Key: smoke-rag" --data '{"question":"Ping?"}'`
- Komplett: `npm run win:dev:check` (Windows) bzw. `npm run dev:check` (Bash)

## RAG / pgvector (optional)
- Schema anwenden (idempotent, nur einmal nötig):
  - Windows: `Get-Content docs/rag/schema.sql | docker compose -f docker-compose.yml -f docker-compose.dev.yml exec -T db psql -U $env:DB_USER -d $env:DB_NAME -v ON_ERROR_STOP=1 -f /dev/stdin`
- Dienste nach Schemaänderungen neu starten: `npm run dev:restart` (Windows: `npm run win:dev:restart`).
- Hinweis: Alte Einträge bleiben ohne Scope valide; optionaler Backfill nur bei Bedarf.
- **Normalisierung alter Embeddings:** Nach dem Rollout der Einheitsvektor-Normalisierung müssen bestehende Einträge neu berechnet werden. Plane unmittelbar im Anschluss einen Re-Embed-Lauf (Ingestion-Batch oder dediziertes Maintenance-Skript), der alle aktiven Dokumente erneut embedden und überschreiben darf. Ohne diesen Schritt mischen sich alte, nicht normierte Vektoren in die Suche und verfälschen Cosine-Distanzen.

### Collection-Scope Migration Checklist
1) **Collections-Tabelle prüfen:**
   ```sql
   SELECT column_name
   FROM information_schema.columns
   WHERE table_schema = '{{SCHEMA_NAME}}'
     AND table_name = 'collections';
   ```
   Erwartet werden `tenant_id`, `id`, optionale Labels sowie der zusammengesetzte Primärschlüssel `(tenant_id, id)`.
2) **Foreign Keys & Spalten verifizieren:**
   ```sql
   SELECT attrelid::regclass AS table_name, attname AS column_name
   FROM pg_attribute
   WHERE attrelid IN ('{{SCHEMA_NAME}}.documents'::regclass,
                      '{{SCHEMA_NAME}}.chunks'::regclass,
                      '{{SCHEMA_NAME}}.embeddings'::regclass)
     AND attname = 'collection_id';
   ```
   Ergänzend stellt der folgende Check sicher, dass alle Tabellen auf `collections` referenzieren:
   ```sql
   SELECT conrelid::regclass, confrelid::regclass
   FROM pg_constraint
   WHERE confrelid = '{{SCHEMA_NAME}}.collections'::regclass;
   ```
   Erwartet werden drei Zeilen (`documents`, `chunks`, `embeddings`).
3) **Duplicate-Guards nach Scope testen:**
   ```sql
   EXPLAIN SELECT 1
   FROM {{SCHEMA_NAME}}.documents
   WHERE tenant_id = :tenant
     AND collection_id IS NULL
     AND hash = :hash;

   EXPLAIN SELECT 1
   FROM {{SCHEMA_NAME}}.documents
   WHERE tenant_id = :tenant
     AND collection_id = :collection
     AND hash = :hash;
   ```
   Beide Abfragen müssen Index-Scans verwenden (`documents_tenant_hash_null_collection_idx` bzw. `documents_tenant_collection_hash_idx`).
4) **Header-Bridge Smoke-Test:** Sende einen Upload mit `X-Collection-ID` ohne Body-Feld und prüfe, ob `.meta.json` sowie `documents.collection_id` denselben Wert tragen. Anschließend einen Query-Aufruf mit `filters.collection_ids=[...]` ausführen, um zu bestätigen, dass der Header ignoriert wird (Scope-Priorität Liste > Body > Header).

### Pgvector-Versionen vereinheitlichen (HNSW + Cosine)
Ziel: `vector_cosine_ops` ist für HNSW-Indizes in allen Umgebungen verfügbar, damit Reindex-Läufe ohne Fallback (L2/IP) funktionieren.

1) Vorab-Prüfungen (alle Umgebungen)
- Version ermitteln: `SELECT extversion FROM pg_extension WHERE extname = 'vector';`
- Operator-Klasse vorhanden: `SELECT 1 FROM pg_catalog.pg_opclass opc JOIN pg_catalog.pg_am am ON am.oid = opc.opcmethod WHERE opc.opcname = 'vector_cosine_ops' AND am.amname = 'hnsw';`
- Erwartung: pgvector >= `0.5.0` (HNSW-Unterstützung) und `vector_cosine_ops` für `hnsw` verfügbar.

2) Deployment-Plan
- Dev/Local
  - Stelle sicher, dass dein Postgres die Extension enthält. Bei Docker bevorzugt ein Image mit vorinstalliertem pgvector (z. B. `pgvector/pgvector:pg16`).
  - Schema anwenden: `docs/rag/schema.sql` (enthält Version-Guard und HNSW-Index mit `vector_cosine_ops`).
  - Reindex nach Bedarf: `python manage.py rebuild_rag_index` (nutzt HNSW und bevorzugt `vector_cosine_ops`).
- Staging
  - Cloud SQL: `CREATE EXTENSION IF NOT EXISTS vector;` einmalig, danach `ALTER EXTENSION vector UPDATE;` (bringt die aktuell in Cloud SQL verfügbare Version).
  - Pipeline-Stufe „Vector-Schema-Migrations“ ausführen (führt `docs/rag/schema.sql` aus; bricht ab, wenn Version < 0.5.0).
  - Reindex-Job ausführen: `python manage.py rebuild_rag_index` mit `RAG_INDEX_KIND=HNSW` (Defaults: `m=32`, `ef_construction=200`).
- Prod
  - Approval-gesteuert über die CI/CD-Pipeline. Reihenfolge: Datenbank-Backup/PITR prüfen → „Vector-Schema-Migrations“ → Reindex-Job.
  - Hinweis: Aktueller Reindex ist nicht „CONCURRENTLY“. Zeitfenster bei niedriger Last wählen oder vorheriges IVFFLAT temporär bestehen lassen und HNSW im Wartungsfenster umschalten.

3) Validierung (nach Deployment)
- Version ok: `SELECT extversion FROM pg_extension WHERE extname='vector';` (>= 0.5.0 erwartet; in Cloud SQL kann höher sein, z. B. 0.6.x)
- Operator-Klasse vorhanden (HNSW/Cosine):
  ```sql
  SELECT opc.opcname, am.amname
  FROM pg_catalog.pg_opclass opc
  JOIN pg_catalog.pg_am am ON am.oid = opc.opcmethod
  WHERE opc.opcname = 'vector_cosine_ops' AND am.amname = 'hnsw';
  ```
- Index-Definition prüfbar über:
  ```sql
  SELECT indexdef
  FROM pg_indexes
  WHERE schemaname = 'rag'
    AND tablename = 'embeddings'
    AND indexname = 'embeddings_embedding_hnsw';
  -- Erwartet: "USING hnsw (embedding vector_cosine_ops)"
  ```
- Management-Command Feedback (sollte HNSW nennen, kein Fallback):
  - `python manage.py rebuild_rag_index` → Ausgabe enthält „using HNSW (scope: rag.embeddings, m=.. ef_construction=..)”

4) Rollback-Strategie
- Falls `vector_cosine_ops` unerwartet fehlt, Extension-Upgrade nachholen (`ALTER EXTENSION vector UPDATE;`) und Reindex erneut ausführen.
- Temporärer Fallback (nur wenn nötig): IVFFLAT neu erstellen (`RAG_INDEX_KIND=IVFFLAT`), später auf HNSW/Cosine zurückschalten.

## Fehlerbilder & Hinweise
- Container restarten ständig → Logs prüfen:
  - `docker compose -f docker-compose.yml -f docker-compose.dev.yml logs -n 120 web` (z. B. CRLF im `entrypoint.sh`)
- 403 bei API‑POSTs → AI‑Endpoints sind CSRF‑exempt; sicherstellen, dass Code aktualisiert wurde.
- LiteLLM „unhealthy“ → Healthcheck läuft Python‑Probe; bei Erststart dauert Prisma‑Migration etwas. Danach `web/worker` ggf. per `npm run dev:restart` (Windows: `npm run win:dev:restart`) neu starten.
  Sollte der Fehler nach Code-Abhängigkeitsänderungen bestehen bleiben, `npm run dev:rebuild` / `npm run win:dev:rebuild` ausführen.

## Best Practices
- Dev: ein DB‑User für App & LiteLLM verwenden (aus `.env`).
- Prod: getrennte Rollen/DSNs je Dienst (Least Privilege), Migrationen über Pipeline‑Stufen ausführen.

