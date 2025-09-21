# NOESIS 2

## Vision
KI-gestützte SaaS-Plattform zur prozessualen Unterstützung der betrieblichen Mitbestimmung nach § 87 Abs. 1 Nr. 6 BetrVG.

> 📘 **Zentrale Leitplanken & Navigation:** [AGENTS.md](AGENTS.md) bündelt Rollen, Trigger und Links zu allen Primärquellen.
>
> 🚀 **Neu im Projekt?** Der [Onboarding-Leitfaden](docs/development/onboarding.md) führt Schritt für Schritt durch Setup, Skripte und weiterführende Dokumentation.

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
Alle Pfade erfordern die Header `X-Tenant-ID`, `X-Case-ID` sowie `Idempotency-Key`. Antworten enthalten Standard-Trace-Header und optionale `gaps` oder `citations`. Der `Idempotency-Key` muss pro Mandant und Vorgang eindeutig sein, damit wiederholte POST-Aufrufe dedupliziert werden können.

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
# Falls noch keine .env vorhanden ist:
# Windows: copy .env.example .env   |   Linux/macOS: cp .env.example .env

docker compose -f docker-compose.yml -f docker-compose.dev.yml build
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# Bootstrap & Checks (idempotent):
npm run dev:up
npm run dev:check
```

Hinweise:
- Wenn bereits eine lokale `.env` mit gültigen Schlüsseln existiert, Kopierschritt überspringen.
- Der Entwicklungscontainer führt `python manage.py collectstatic` beim Start automatisch aus.
- Bei älteren Compose-Setups ggf. manuell:
  `docker-compose exec web python manage.py collectstatic` (notwendig wegen `CompressedManifestStaticFilesStorage`).

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


## Lokales Testen

Die Tests greifen auf die in `docker-compose.dev.yml` definierten Service-Hosts (u. a. `db`) zu. Ausserhalb der Compose-Umgebung fehlen diese Namensaufloesungen und Django bricht mit Verbindungsfehlern ab. Zudem enthaelt das `web`-Image nur Produktionsabhaengigkeiten, daher muessen vor dem Testlauf sowohl Basis- als auch Dev-Abhaengigkeiten installiert werden.

```bash
docker compose -f docker-compose.dev.yml run --rm web sh -c "pip install -r requirements.txt -r requirements-dev.txt && python -m pytest"
```

Der Befehl laedt die benoetigten Pakete in den temporaeren Container, fuehrt die Tests aus und entfernt den Container anschliessend wieder. Fuehre ihn bei jedem frischen Containerlauf erneut aus, da Installationen nicht persistent sind.

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

### Dev-Skripte (idempotent)
- `npm run dev:up`: Startet/initialisiert die Umgebung, führt `migrate_schemas`, bootstrapped den Public‑Tenant, legt `dev`‑Tenant und Superuser an.
- `npm run dev:check`: Führt Smoke‑Checks aus (LiteLLM Health/Chat, `GET /ai/ping/`, `POST /ai/scope/`). Chat erfordert gültige `GEMINI_API_KEY`.
- `npm run dev:down`: Stoppt und entfernt alle Container inkl. Volumes.
- `npm run dev:rebuild`: Baut Web- und Worker-Images neu und aktualisiert Code-Abhängigkeiten, ohne Volumes zu löschen. Mit `npm run dev:rebuild -- --with-frontend` lässt sich optional auch das Frontend-Image aktualisieren.
- `npm run dev:restart`: Startet Web- und Worker-Container schnell neu (Compose `restart`).
- `npm run dev:init`: Führt die Compose‑Jobs `migrate` und `bootstrap` aus (nach `up -d`).
- `npm run dev:reset`: Full reset (down -v → build --no-cache → up -d → init → checks).
- `npm run jobs:rag`: Führt `docs/rag/schema.sql` gegen die DB aus (idempotent).
- `npm run rag:enable`: Setzt `RAG_ENABLED=true` in `.env` und startet Web/Worker neu.
 - `npm run jobs:rag:health`: Prüft RAG-Gesundheit (pgvector installiert, Tabellen vorhanden).

**Wann welches Skript?**
- Konfigurationswerte, Feature-Flags oder Django-Code geändert und die laufenden Container sollen die Änderungen aufnehmen → `npm run dev:restart` (Windows: `npm run win:dev:restart`).
- Neue Python-/Node-Abhängigkeiten installiert oder das Dockerfile angepasst → `npm run dev:rebuild` (Windows: `npm run win:dev:rebuild`).
- Komplettes Setup inkl. Datenbanken zurücksetzen (z. B. nach kaputten Fixtures) → `npm run dev:reset` (Windows: `npm run win:dev:reset`).
- Nur das RAG-Schema aktualisieren bzw. die Installation verifizieren → `npm run jobs:rag` bzw. `npm run jobs:rag:health`.
- Demo-Tenant & Beispieldaten neu befüllen → `docker compose -f docker-compose.yml -f docker-compose.dev.yml exec web python manage.py create_demo_data` (Windows: gleicher Befehl in PowerShell).

Hinweis (Windows): PowerShell‑Varianten sind enthalten:
- `npm run win:dev:up`
- `npm run win:dev:check`
- `npm run win:dev:down`
- `npm run win:dev:rebuild` (optional mit `npm run win:dev:rebuild -- -WithFrontend`)
- `npm run win:dev:restart`
- `npm run win:dev:reset`
- `npm run win:rag:enable`
- `npm run win:jobs:rag`
- `npm run win:jobs:rag:health`

### Git Hooks (Lint vor Push)
- Installieren: 
  - macOS/Linux: `npm run hooks:install`
  - Windows: `npm run win:hooks:install`
- Wirkung: Vor jedem `git push` läuft automatisch `npm run lint` (ruff + black). Zum Überspringen einmalig `SKIP_LINT=1 git push` setzen.

---
 
## Konfiguration (.env)
Benötigte Variablen (siehe `.env.example` oder `.env.dev.sample`):

- SECRET_KEY: geheimer Schlüssel für Django
- DEBUG: `true`/`false`
- DB_USER / DB_PASSWORD / DB_NAME: gemeinsame Dev‑Credentials; werden für den Container‑Init und DSNs genutzt.
- DATABASE_URL: Verbindungs-URL zur PostgreSQL-Datenbank (App‑DB, default: `postgresql://noesis2:noesis2@db:5432/noesis2`)
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
- LITELLM_DATABASE_URL (optional): separates DB‑DSN für die LiteLLM Admin‑DB (default in Compose aus `DB_*` + `LITELLM_DB_NAME`)
- LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY: Schlüssel für Langfuse-Tracing
- LITELLM_SALT_KEY: Salt für Verschlüsselung gespeicherter Provider‑Keys in der Admin‑DB (empfohlen für Admin UI)

Die Settings lesen `.env` via `django-environ`. `DATABASE_URL` muss eine vollständige URL enthalten (inkl. URL-Encoding für Sonderzeichen); `REDIS_URL` wird sowohl für Celery als auch für Redis-basierte Integrationen verwendet.

Best Practice: In Dev einen gemeinsamen DB‑User für App und LiteLLM verwenden (einfachere Einrichtung). In Produktion getrennte Rollen/DSNs je Dienst (Least Privilege).

## LiteLLM & Modelle (Prod vs. Lokal)
- Produktion (Cloud Run)
  - LiteLLM authentifiziert gegen Vertex AI per Service Account (ADC), kein API‑Key nötig.
  - Regionensplit: Cloud Run `europe-west3`, Vertex `us-central1` (siehe `scripts/init_litellm_gcloud.sh` und CI).
  - Routing: `MODEL_ROUTING.yaml` zeigt auf `vertex_ai/*` Modelle (z. B. `vertex_ai/gemini-2.5-flash`).
  - CI setzt `VERTEXAI_PROJECT` und `VERTEXAI_LOCATION` und pinned die Cloud‑Run Service‑Account‑Identität.

- Lokal (Docker Compose)
  - Vertex ADC steht lokal i. d. R. nicht zur Verfügung. Nutze AI‑Studio (Gemini) über LiteLLM‑Config.
  - Datei `config/litellm-config.yaml` ist für AI‑Studio konfiguriert (`gemini/*` + `GEMINI_API_KEY`).
  - Erzeuge eine lokale Routing‑Override, damit Django lokale Modelle anspricht:
    ```bash
    cp MODEL_ROUTING.local.yaml.sample MODEL_ROUTING.local.yaml
    ```
  - Setze in `.env` mindestens:
    - `GEMINI_API_KEY=<dein_ai_studio_key>`
    - `LITELLM_MASTER_KEY=<beliebiger_dev_key>`
  - Starte LiteLLM via Compose und teste:
    ```bash
    docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d litellm
    bash scripts/smoke_litellm.sh
    ```

Hinweis: `MODEL_ROUTING.local.yaml` ist git‑ignored und überschreibt nur lokal. In Prod wird ausschließlich `MODEL_ROUTING.yaml` verwendet.

## GCloud Bootstrap (Windows)
Mit dem Skript `scripts/gcloud-bootstrap.ps1` kannst du dich per `gcloud` anmelden, ein Projekt/Region wählen und häufige Laufzeitwerte aus GCP sammeln (Redis, Cloud SQL, Cloud Run URLs). Die Werte werden sicher in `.env.gcloud` geschrieben (Git-ignored) und können selektiv in deine `.env` übernommen werden.

PowerShell (Windows):

```powershell
pwsh -File scripts/gcloud-bootstrap.ps1
# oder nicht-interaktiv mit Vorgaben:
pwsh -File scripts/gcloud-bootstrap.ps1 -Region europe-west3 -Zone europe-west3-a

# Optional: Secrets explizit ziehen (schreibt sensible Werte in .env.gcloud!)
pwsh -File scripts/gcloud-bootstrap.ps1 -FetchSecrets -Secrets SECRET_KEY,LANGFUSE_KEY
```

Hinweise:
- Secrets werden standardmäßig nicht geladen. Mit `-FetchSecrets` + `-Secrets` kannst du gezielt Secret-Names laden, sofern dein Account berechtigt ist.
- Für Cloud SQL im lokalen Setup bevorzugen wir weiterhin die lokale DB (Docker). Alternativ Cloud SQL Auth Proxy verwenden und `DB_HOST`/`DATABASE_URL` passend setzen.

## GCloud Bootstrap (Bash/WSL/Linux)
Das Bash-Pendant läuft unter WSL oder Linux. Es sammelt dieselben Werte und schreibt `.env.gcloud` (Git-ignored).

```bash
# Interaktiv
bash scripts/gcloud-bootstrap.sh

# Mit Parametern
bash scripts/gcloud-bootstrap.sh --region europe-west3 --zone europe-west3-a

# Optional Secrets (vorsichtig)
bash scripts/gcloud-bootstrap.sh --fetch-secrets \
  --secret SECRET_KEY --secret LANGFUSE_KEY
```

Voraussetzungen: `gcloud` im PATH. `jq` ist nicht erforderlich.

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

## Operator-Kommandos (Makefile)

Im Projektwurzelverzeichnis stehen Makefile-Targets zur Verfügung, um wiederkehrende Operator-Aufgaben auszuführen:

- `make jobs:migrate` – führt `python manage.py migrate_schemas --noinput` aus.
- `make jobs:bootstrap` – legt den öffentlichen Tenant per `bootstrap_public_tenant` an (`DOMAIN` erforderlich).
- `make tenant-new` – erstellt ein neues Schema und die zugehörige Domain (`SCHEMA`, `NAME`, `DOMAIN`).
- `make tenant-superuser` – erzeugt einen Superuser in einem Schema (`SCHEMA`, `USERNAME`, `PASSWORD`, optional `EMAIL`).
- `make jobs:rag` – spielt [`docs/rag/schema.sql`](docs/rag/schema.sql) gegen den RAG-Store ein (`RAG_DATABASE_URL` oder fallback `DATABASE_URL`).
- `make jobs:rag:health` – prüft Schema, Tabellen und `vector`-Extension im RAG-Store (`RAG_DATABASE_URL` oder fallback `DATABASE_URL`).

Setze die benötigten Umgebungsvariablen vor dem Aufruf, z. B.:

```bash
export DOMAIN=demo.localhost
export SCHEMA=demo
export NAME="Demo GmbH"
export USERNAME=admin
export PASSWORD=changeme
export RAG_DATABASE_URL=postgresql://user:pass@host:5432/rag
```

`RAG_DATABASE_URL` kann leer bleiben, sofern `DATABASE_URL` auf dieselbe Instanz zeigt. Für alternative Python-Binaries lässt sich `PYTHON` überschreiben (`make PYTHON=python3 jobs:migrate`).

## LiteLLM Proxy (lokal)
- `.env.example` → `.env` kopieren und `GOOGLE_API_KEY` setzen
- Start: `docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d litellm`
- Healthcheck: `curl -s http://localhost:4000/health`
- Admin UI Login: Standardmäßig via Master Key (Wahl im UI). Alternativ User/Pass über `UI_USERNAME`/`UI_PASSWORD` setzen (in dev/staging per Compose auf `DB_USER`/`DB_PASSWORD` voreingestellt).

## Frontend-Build (Tailwind v4 via PostCSS)
- Build/Watch: `npm run build:css` (wird in `npm run dev` automatisch gestartet)
- Konfiguration: `postcss.config.js` mit `@tailwindcss/postcss` und `autoprefixer`
- Eingabe/Ausgabe: `theme/static_src/input.css` → `theme/static/css/output.css`

## Frontend-Richtlinien
- [Frontend-Überblick](docs/frontend-ueberblick.md)
- Der vollständige Rahmen für React/TypeScript-Komponenten ist im [Frontend Master Prompt](docs/frontend-master-prompt.md) beschrieben.

## Testing
- Bevorzugt: in Docker ausführen, siehe Abschnitt "Lokales Testen".
- Schnelllauf: `docker compose -f docker-compose.dev.yml run --rm web sh -c "pip install -r requirements.txt -r requirements-dev.txt && python -m pytest -q"`
- Mit Coverage: `docker compose -f docker-compose.dev.yml run --rm web sh -c "pip install -r requirements.txt -r requirements-dev.txt && python -m pytest -q --cov=noesis2 --cov-report=term-missing"`
- Kurzbefehle: `npm run test:py` bzw. `npm run test:py:cov`
- Hinweis: Direktes `pytest` auf dem Host führt häufig zu DB-/Hostname-Fehlern (kein `db` im Compose-Netz). Nur nativ ausführen, wenn Postgres/Redis lokal verfügbar und korrekt konfiguriert sind.

## Linting & Formatierung
- Prüfen: `npm run lint` (ruff + black --check)
- Fixen: `npm run lint:fix` (ruff --fix + black)

## Abhängigkeitsmanagement (pip-tools)
- Produktion: `pip-compile requirements.in` → `requirements.txt`
- Entwicklung: `pip-compile requirements-dev.in` → `requirements-dev.txt`
- Installation: `pip install -r requirements*.txt`

## Troubleshooting (Windows)
- Nur bei nativer Ausführung ohne Docker relevant: Falls `pytest`, `black`, `ruff` oder `pip-compile` nicht gefunden werden, `%APPDATA%\Python\Python313\Scripts` zum PATH hinzufügen.
- `.env` sollte UTF‑8 ohne BOM sein (bei Parsen-Fehlern Datei neu speichern).
