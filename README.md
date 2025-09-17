# NOESIS 2

## Vision
KI-gestützte SaaS-Plattform zur prozessualen Unterstützung der betrieblichen Mitbestimmung nach § 87 Abs. 1 Nr. 6 BetrVG.

## Kernfunktionen (Geplant)
- Flexible, workflow-basierte Analyse von Dokumenten (z. B. Systembeschreibungen, Betriebsvereinbarungen)
- Mandantenfähigkeit zur Trennung von Daten verschiedener Parteien (Arbeitgeber, Betriebsräte, Anwälte)
- Wissensgenerierung und -abfrage durch angebundene Large Language Models (LLMs)
- Asynchrone Verarbeitung von rechenintensiven Analyse-Aufgaben

---

## Technologie-Stack
- Backend: Python 3.12+ mit Django 5.x
- Asynchrone Tasks: Celery & Redis
- Datenbank: PostgreSQL
- Frontend: Tailwind CSS v4 (PostCSS)
- Entwicklungsumgebung: Node.js, npm
- CI/CD & Testing: GitHub Actions, pytest

## AI Core

### API-Endpunkte
Alle Pfade erfordern die Header `X-Tenant-ID` und `X-Case-ID`. Antworten enthalten Standard-Trace-Header und optionale `gaps` oder `citations`.

- `GET /ai/ping/` – einfacher Health-Check
- `POST /ai/intake/` – Metadaten speichern und Eingangsbestätigung liefern
- `POST /ai/scope/` – Auftragsumfang prüfen und fehlende Angaben melden
- `POST /ai/needs/` – Informationen dem Tenant-Profil zuordnen, Abbruch bei Lücken
- `POST /ai/sysdesc/` – Systembeschreibung nur wenn keine Informationen fehlen

### Graphen
Die Views orchestrieren reine Python-Graphen. Jeder Graph erhält `state: dict` und `meta: {tenant, case, trace_id}` und gibt `(new_state, result)` zurück. Der Zustand wird nach jedem Schritt in `.ai_core_store/{tenant}/{case}/state.json` persistiert. Gates wie `needs_mapping` oder `scope_check` brechen früh ab, statt unvollständige Drafts zu erzeugen.

### Lokale Nutzung
Das bestehende `docker compose`-Setup startet Web-App und Redis. Ein externer LiteLLM-Proxy kann über `LITELLM_BASE_URL` angebunden werden. Nach dem Start (`docker compose ... up`) können die Endpunkte lokal unter `http://localhost:8000/ai/` getestet werden.

---

## Docker Quickstart
```bash
copy .env.example .env   # Linux/macOS: cp .env.example .env
docker compose -f docker-compose.yml -f docker-compose.dev.yml build
docker compose -f docker-compose.yml -f docker-compose.dev.yml up
```

Der Entwicklungscontainer führt `python manage.py collectstatic` beim Start automatisch aus.
Falls du ein älteres Compose-Setup verwendest, führe stattdessen manuell aus:
`docker-compose exec web python manage.py collectstatic`.
Dies ist erforderlich, da `CompressedManifestStaticFilesStorage` aktiviert ist.

---

## Lokales Setup (Alternative ohne Docker)

Docker Compose ist die bevorzugte Methode für ein konsistentes, schnelles Setup.
Die folgenden Schritte sind ein manueller Fallback, falls Docker nicht genutzt wird.

### Voraussetzungen
- Python 3.12+
- Node.js und npm
- PostgreSQL-Server
- Redis-Server

### Installations-Schritte
1. Repository klonen
   ```bash
   git clone https://github.com/fraven01/NOESIS-2.git
   cd NOESIS-2
   ```
2. Python-Umgebung einrichten
   ```bash
   python -m venv .venv
   # Linux/macOS
   source .venv/bin/activate
   # Windows PowerShell
   .\.venv\Scripts\Activate.ps1

   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```
3. Frontend-Abhängigkeiten installieren
   ```bash
   npm install
   ```
4. Datenbank einrichten
   - Leere PostgreSQL-Datenbank erstellen (z. B. `CREATE DATABASE noesis2_db;`).
   - `.env.example` nach `.env` kopieren und Zugangsdaten anpassen.
   - Migrationen ausführen:
     ```bash
     python manage.py migrate
     ```
5. Superuser anlegen
   ```bash
   python manage.py createsuperuser
   ```

### Entwicklungsserver starten
```bash
npm run dev
```

## Anwendung ausführen mit Docker

- `docker compose -f docker-compose.yml -f docker-compose.dev.yml up`: Startet die gesamte Anwendung im Vordergrund (Logs im Terminal).
- `docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d`: Startet die Anwendung im Hintergrund (detached mode).
- `docker compose -f docker-compose.yml -f docker-compose.dev.yml down`: Stoppt und entfernt die Container (Volumes wie Datenbankdaten bleiben erhalten).
- `docker compose -f docker-compose.yml -f docker-compose.dev.yml exec web python manage.py <befehl>`: Führt einen `manage.py`-Befehl (z. B. `createsuperuser`) im laufenden `web`-Container aus.
- Der Entwicklungscontainer führt `python manage.py collectstatic` beim Start automatisch aus.
  Bei älteren Compose-Setups muss dieser Schritt manuell erfolgen:
  `docker-compose exec web python manage.py collectstatic`.
  Notwendig, da `CompressedManifestStaticFilesStorage` aktiv ist.
- Datenbankmigrationen laufen nicht automatisch mit. Führe sie bei Bedarf manuell aus, z. B. mit
  `docker compose -f docker-compose.yml -f docker-compose.dev.yml exec web python manage.py migrate`.

---
 
## Konfiguration (.env)
Benötigte Variablen (siehe `.env.example`):

- SECRET_KEY: geheimer Schlüssel für Django
- DEBUG: `true`/`false`
- DATABASE_URL: Verbindungs-URL zur PostgreSQL-Datenbank
- REDIS_URL: Redis-Endpoint (z. B. für Celery)
- RAG_ENABLED: `true`/`false` zur Aktivierung des Retrieval-Augmented-Generation-Workflows.
  Aktiviere das Flag nur, wenn das `rag`-Schema inklusive `pgvector`-Extension (`CREATE EXTENSION IF NOT EXISTS vector;`)
  anhand von [`docs/rag/schema.sql`](docs/rag/schema.sql) bereitsteht und über `DATABASE_URL` oder `RAG_DATABASE_URL`
  erreichbar ist. Für den Datenbankzugriff wird `psycopg2` (oder das Binär-Pendant `psycopg2-binary`) benötigt.
  Achtung: Mandanten-IDs müssen UUIDs sein. Ältere nicht-UUID-Werte werden deterministisch gemappt, sollten aber per Migration
  bereinigt werden, bevor Deployments `RAG_ENABLED=true` setzen.

AI Core:
- LITELLM_BASE_URL: Basis-URL des LiteLLM-Proxys
- LITELLM_API_KEY: API-Key für den Proxy
- LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY: Schlüssel für Langfuse-Tracing

Die Settings lesen `.env` via `django-environ`. `DATABASE_URL` muss eine vollständige URL enthalten (inkl. URL-Encoding für Sonderzeichen); `REDIS_URL` wird sowohl für Celery als auch für Redis-basierte Integrationen verwendet.

## Settings-Profile
Das alte `noesis2/settings.py` wurde entfernt; verwende ausschließlich das modulare Paket `noesis2/settings/`.

- Standard: `noesis2.settings.development` (in `manage.py`, `asgi.py`, `wsgi.py` vorkonfiguriert)
- Production: `noesis2.settings.production`
- Umstellung per Env-Var: `DJANGO_SETTINGS_MODULE=noesis2.settings.production`

## Datenmigration und Standard-Organisation
Beim Upgrade auf die mandantenfähige Struktur muss jedem bestehenden Projekt
eine Organisation zugeordnet werden. Die Migration
`projects/migrations/0002_project_organization.py` legt für vorhandene Projekte
automatisch eine Organisation an. Sollten nach dem Deploy noch Projekte ohne
Organisation existieren (z. B. nach einem manuellen Datenimport), kann der
Management-Befehl `assign_default_org` genutzt werden:

```bash
python manage.py assign_default_org
```

Der Befehl erzeugt für jedes betroffene Projekt eine neue Organisation und
aktualisiert das Projekt entsprechend.

## Tenant-Verwaltung

Eine ausführliche Anleitung zur Einrichtung und Pflege von Mandanten (inkl. lokalem Setup, Admin/Operator‑Rollen und X‑Tenant‑Schema) befindet sich im Dokument [docs/multi-tenancy.md](docs/multi-tenancy.md).

## LiteLLM Proxy (lokal)
- `.env.example` → `.env` kopieren und `GOOGLE_API_KEY` setzen
- Start: `docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d litellm`
- Healthcheck: `curl -s http://localhost:4000/health`

## Frontend-Build (Tailwind v4 via PostCSS)
- Build/Watch: `npm run build:css` (wird in `npm run dev` automatisch gestartet)
- Konfiguration: `postcss.config.js` mit `@tailwindcss/postcss` und `autoprefixer`
- Eingabe/Ausgabe: `theme/static_src/input.css` → `theme/static/css/output.css`

## Frontend-Richtlinien
- [Frontend-Überblick](docs/frontend-ueberblick.md)
- Der vollständige Rahmen für React/TypeScript-Komponenten ist im [Frontend Master Prompt](docs/frontend-master-prompt.md) beschrieben.

## Testing
- Ausführen: `pytest -q`
- Mit Coverage: `pytest -q --cov=noesis2 --cov-report=term-missing`
- Pytest ist via `pytest.ini` auf `noesis2.settings.development` konfiguriert

## Linting & Formatierung
- Prüfen: `npm run lint` (ruff + black --check)
- Fixen: `npm run lint:fix` (ruff --fix + black)

## Abhängigkeitsmanagement (pip-tools)
- Produktion: `pip-compile requirements.in` → `requirements.txt`
- Entwicklung: `pip-compile requirements-dev.in` → `requirements-dev.txt`
- Installation: `pip install -r requirements*.txt`

## Troubleshooting (Windows)
- Falls `pytest`, `black`, `ruff` oder `pip-compile` nicht gefunden werden: `%APPDATA%\Python\Python313\Scripts` zum PATH hinzufügen.
- `.env` sollte UTF‑8 ohne BOM sein (bei Parsen-Fehlern Datei neu speichern).
