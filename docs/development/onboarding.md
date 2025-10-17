# Onboarding-Leitfaden für Entwickler:innen

Dieser Leitfaden bündelt die notwendigen Voraussetzungen und Schritte, um eine lokale NOESIS 2 Instanz lauffähig zu machen. Er ersetzt nicht die Detaildokumentation, verweist aber auf die wichtigsten Quellen für Architektur, Betrieb und weiterführende Themen.

## 1. Voraussetzungen
- **Git** und Zugriff auf das Repository
- **Docker** & **Docker Compose** (empfohlenes Setup)
- **Node.js** (18+) & **npm** – werden für PostCSS/Tailwind und Utility-Skripte benötigt
- **Python 3.12+** – nur erforderlich, wenn ohne Docker gearbeitet wird
- Optional für das manuelle Setup: Lokale **PostgreSQL**- und **Redis**-Instanzen
- Ein `.env` basierend auf [`./.env.example`](../../.env.example) mit gültigen Secrets (z. B. `GEMINI_API_KEY`, `LANGFUSE_*`)

> ℹ️ Die zentrale Navigationsübersicht inklusive Rollen und Verantwortlichkeiten befindet sich in [`AGENTS.md`](../../AGENTS.md).

## 2. Repository klonen
```bash
git clone https://github.com/fraven01/NOESIS-2.git
cd NOESIS-2
```

## 3. Empfohlenes Setup mit Docker
1. `.env.example` nach `.env` kopieren und Werte für Datenbank, LiteLLM & API-Keys setzen.
2. Container bauen und starten:
   ```bash
   docker compose -f docker-compose.yml -f docker-compose.dev.yml build
   docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d
   ```
3. Idempotente Bootstrap-Skripte ausführen oder den Komplettstack hochfahren:
   ```bash

   npm run dev:stack         # App + ELK + Migrationen + Demo/Heavy-Seeding
   # Windows-Variante:
   npm run win:dev:stack
   # Alternativ (nur App-Stack):
   npm run dev:up      # Migrationen, Public-Tenant, Demo-Daten & Superuser
   npm run dev:check   # Smoke-Checks (LiteLLM, /ai/ping, /ai/scope)
   ```
4. Für tägliche Entwicklungszyklen:
   - `npm run dev:init` führt nur die Jobs `migrate` & `bootstrap` aus (nach einem `up -d`).
   - `npm run dev:reset` setzt die Umgebung vollständig neu auf (inkl. Volumes löschen).
   - `npm run dev:down` stoppt und entfernt alle Container samt Volumes.
   - `npm run dev:rebuild` baut Web- und Worker-Images neu, um Python-/Node-Abhängigkeiten aufzufrischen, ohne Daten-Volumes zu
     löschen. Optional `npm run dev:rebuild -- --with-frontend`, falls auch das Frontend-Image aktualisiert werden soll.
   - `npm run dev:restart -- [services…]` startet gezielt Dienste neu (Default: `web worker ingestion-worker`). Fällt bei Bedarf auf `up -d --no-deps --no-build` zurück.
   - `npm run dev:prune` räumt dangling Images und Build-Cache auf; mit `-- --all` zusätzlich Netzwerke/Volumes (destruktiv).

### 3.1 ELK-Smoke-Test nach dem Start
- Übernimm die Standard-Credentials aus [`../../.env.dev-elk`](../../.env.dev-elk) in deine `.env`, falls noch nicht geschehen (`elastic`/`changeme`).
- Öffne [http://localhost:5601](http://localhost:5601) und melde dich mit `elastic` + `ELASTIC_PASSWORD` an.
- Navigiere zu **Discover** und setze den Filter `test_suite:chaos`, um die strukturierten Chaos-Logs zu validieren.
- Prüfe in der Trace-Ansicht, dass Felder wie `X-Tenant-ID`, `X-Case-ID` und `Idempotency-Key` in den Log-Dokumenten auftauchen.

Nach erfolgreichem Bootstrap ist der Django-Server unter `http://localhost:8000/` erreichbar. Die AI-Core-Endpunkte laufen unter `http://localhost:8000/ai/` und erwarten die Header `X-Tenant-ID`, `X-Case-ID` und `Idempotency-Key`.

## 4. Manuelles Setup ohne Docker (Fallback)
1. Python-Umgebung anlegen:
   ```bash
   python -m venv .venv
   # Linux/macOS
   source .venv/bin/activate
   # Windows PowerShell
   .\.venv\Scripts\Activate.ps1

   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```
2. Node-Abhängigkeiten installieren:
   ```bash
   npm install
   ```
3. Datenbank und Redis bereitstellen, `.env` anpassen und Migrationen ausführen:
   ```bash
   python manage.py migrate_schemas --noinput
   python manage.py bootstrap_public_tenant --domain localhost
   python manage.py create_tenant --schema=demo --name="Demo Tenant" --domain=demo.localhost
   python manage.py create_tenant_superuser --schema=demo --username=demo --password=<PASSWORT>
   ```
4. Entwicklungsserver starten (Django + Tailwind Watcher):
   ```bash
   npm run dev
   ```

## 5. Tests, Linting & Qualitätssicherung
- Backend-Tests: `pytest -q`
- Mit Coverage: `pytest -q --cov=noesis2 --cov-report=term-missing`
- Linting (ruff + black): `npm run lint`
- Auto-Fixes anwenden: `npm run lint:fix`
- Tailwind/PostCSS-Build: `npm run build:css`

> Hinweis Build-Kontext: `.dockerignore` schließt große Ordner (z. B. `docs/`, `e2e/`, `playwright/`, `logs/`) aus. Unit-/Integrations-Tests bleiben im Kontext, damit `pytest` im Container/Image läuft.

## 6. Optionale Jobs & RAG-Checks
- RAG-Schema anwenden: `npm run jobs:rag`
- RAG-Gesundheit prüfen: `npm run jobs:rag:health`

Diese Kommandos greifen auf das `docs/rag/schema.sql` zurück und benötigen eine PostgreSQL-Instanz mit `pgvector`-Extension.

## 7. Weiterführende Dokumentation
Nach dem ersten Setup sollten folgende Dokumente gelesen werden:
- **Architekturüberblick:** [`docs/architektur/overview.md`](../architektur/overview.md)
- **Agenten & LangGraph:** [`docs/agents/overview.md`](../agents/overview.md)
- **RAG & Ingestion:** [`docs/rag/overview.md`](../rag/overview.md) und [`docs/rag/ingestion.md`](../rag/ingestion.md)
- **Mandantenfähigkeit:** [`docs/multi-tenancy.md`](../multi-tenancy.md) & [`docs/tenant-management.md`](../tenant-management.md)
- **CI/CD Pipeline & Deployments:** [`docs/cicd/pipeline.md`](../cicd/pipeline.md)
- **Security & Secrets:** [`docs/security/secrets.md`](../security/secrets.md)
- **Observability:** [`docs/observability/langfuse.md`](../observability/langfuse.md)

Diese Quellen vertiefen Architekturentscheidungen, Betriebsprozesse sowie die Erwartungen an Guardrails, Monitoring und Kostenkontrolle.

## 8. Nächste Schritte
- Richte dir Zugriff auf die Langfuse- und LiteLLM-Oberflächen gemäß [`docs/litellm/admin-gui.md`](../litellm/admin-gui.md).
- Prüfe die Runbooks in [`docs/runbooks/`](../runbooks) für Incident- und Migration-Szenarien.
- Stimme dich bei Fragen zum Mandanten-Setup mit dem Platform-Team ab (siehe Kontaktpunkte in [`docs/multi-tenancy.md`](../multi-tenancy.md)).

Mit diesem Leitfaden solltest du eine konsistente lokale Umgebung bereitstellen und weißt, an welche Stellen der Dokumentation du für vertiefende Informationen anknüpfen kannst.
