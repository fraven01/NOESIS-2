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
  - Windows: `docker compose -f docker-compose.yml -f docker-compose.dev.yml exec web python manage.py makemigrations <app>`
  - Bash: `docker compose -f docker-compose.yml -f docker-compose.dev.yml exec web python manage.py makemigrations <app>`

2) Shared‑Migrationen anwenden (falls Shared App betroffen)
- `docker compose -f docker-compose.yml -f docker-compose.dev.yml exec web python manage.py migrate_schemas --shared`

3) Public Tenant sicherstellen (idempotent)
- `docker compose -f docker-compose.yml -f docker-compose.dev.yml exec web python manage.py bootstrap_public_tenant --domain=localhost`

4) Tenant‑Migrationen anwenden
- `docker compose -f docker-compose.yml -f docker-compose.dev.yml exec web python manage.py migrate_schemas --tenant`

5) Neuen Tenant anlegen (optional)
- Anlegen: `docker compose -f docker-compose.yml -f docker-compose.dev.yml exec web python manage.py create_tenant --schema=dev2 --name="Dev 2" --domain=dev2.localhost`
- Superuser (nicht interaktiv):
  - Windows: `docker compose -f docker-compose.yml -f docker-compose.dev.yml exec -T -e DJANGO_SUPERUSER_PASSWORD=admin123 web python manage.py create_tenant_superuser --schema=dev2 --username=admin --email=admin@example.com --noinput`
  - Bash:   `docker compose -f docker-compose.yml -f docker-compose.dev.yml exec -T -e DJANGO_SUPERUSER_PASSWORD=admin123 web python manage.py create_tenant_superuser --schema=dev2 --username=admin --email=admin@example.com --noinput`

## Smoke‑Checks (nach Migrationen)
- API‑Ping (Header‑Routing):
  - `curl -i -H "X-Tenant-Schema: dev" -H "X-Tenant-ID: dev-tenant" -H "X-Case-ID: local" -H "Idempotency-Key: smoke-ping" http://localhost:8000/ai/ping/`
- Beispiel‑Graph:
  - `curl -s -X POST http://localhost:8000/ai/scope/ -H "Content-Type: application/json" -H "X-Tenant-Schema: dev" -H "X-Tenant-ID: dev-tenant" -H "X-Case-ID: local" -H "Idempotency-Key: smoke-scope" --data '{"hello":"world"}'`
- Komplett: `npm run win:dev:check` (Windows) bzw. `npm run dev:check` (Bash)

## RAG / pgvector (optional)
- Schema anwenden (idempotent, nur einmal nötig):
  - Windows: `Get-Content docs/rag/schema.sql | docker compose -f docker-compose.yml -f docker-compose.dev.yml exec -T db psql -U $env:DB_USER -d $env:DB_NAME -v ON_ERROR_STOP=1 -f /dev/stdin`
- Feature aktivieren: `.env → RAG_ENABLED=true`, danach `npm run dev:restart` (Windows: `npm run win:dev:restart`)
  (reicht für Flag-Änderungen; `dev:rebuild` nur bei neuen Abhängigkeiten notwendig).

## Fehlerbilder & Hinweise
- Container restarten ständig → Logs prüfen:
  - `docker compose -f docker-compose.yml -f docker-compose.dev.yml logs -n 120 web` (z. B. CRLF im `entrypoint.sh`)
- 403 bei API‑POSTs → AI‑Endpoints sind CSRF‑exempt; sicherstellen, dass Code aktualisiert wurde.
- LiteLLM „unhealthy“ → Healthcheck läuft Python‑Probe; bei Erststart dauert Prisma‑Migration etwas. Danach `web/worker` ggf. per `npm run dev:restart` (Windows: `npm run win:dev:restart`) neu starten.
  Sollte der Fehler nach Code-Abhängigkeitsänderungen bestehen bleiben, `npm run dev:rebuild` / `npm run win:dev:rebuild` ausführen.

## Best Practices
- Dev: ein DB‑User für App & LiteLLM verwenden (aus `.env`).
- Prod: getrennte Rollen/DSNs je Dienst (Least Privilege), Migrationen über Pipeline‑Stufen ausführen.

