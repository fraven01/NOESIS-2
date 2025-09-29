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

### API-Auth
Die Django REST Framework Authentifizierung ist standardmäßig deaktiviert, damit öffentliche Mandanten-Endpunkte keinen Bearer-Token erfordern. Admin- oder LiteLLM-Routen binden den Master-Key explizit per View-Decorator. Weil keine SessionAuth aktiv ist, gelten auch keine CSRF-Cookies – Token-Clients können ohne CSRF-Header arbeiten.

### Graphen
Die Views orchestrieren reine Python-Graphen. Jeder Graph erhält `state: dict` und `meta: {tenant, case, trace_id}` und gibt `(new_state, result)` zurück. Der Zustand wird nach jedem Schritt in `.ai_core_store/{tenant}/{case}/state.json` persistiert. Gates wie `needs_mapping` oder `scope_check` brechen früh ab, statt unvollständige Drafts zu erzeugen.

### Lokale Nutzung
Das bestehende `docker compose`-Setup startet Web-App und Redis. Ein externer LiteLLM-Proxy kann über `LITELLM_BASE_URL` angebunden werden. Nach dem Start (`docker compose ... up`) können die Endpunkte lokal unter `http://localhost:8000/ai/` getestet werden.

### PII-Scope Playbook
Der Session-Scope sorgt dafür, dass dieselben deterministischen Platzhalter in Requests, LLM-Aufrufen, Logs und Tasks genutzt werden. Das Playbook [docs/pii-scope.md](docs/pii-scope.md) beschreibt die Reihenfolge (Middleware → Masking → Logging → Tasks → Egress), enthält eine Review-Checkliste und eine FastAPI-Referenz für Microservices.

---

## Entwicklungsworkflow mit Docker

### 1️⃣ Vorbereitung
- `.env.example` nach `.env` kopieren (Windows: `copy`, Linux/macOS: `cp`).
- Optional: vorhandene Secrets und API-Keys ergänzen (LiteLLM, Gemini, Langfuse …).
- Für den ELK-Stack die Defaults aus `.env.dev-elk` übernehmen (z. B. `cat .env.dev-elk >> .env`), damit Passwörter für `elastic` und `kibana_system` gesetzt sind.

### 2️⃣ Build & Start
```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml build
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d
```


> 💡 **Alles in einem Schritt?** `npm run dev:stack` (Windows: `npm run win:dev:stack`) baut App- und ELK-Images, startet beide Compose-Stacks, führt Migrationen/Bootstrap aus und seedet Demo- sowie Heavy-Datensätze.


### 3️⃣ Bootstrap & Smoke-Checks
```bash
npm run dev:up
npm run dev:check
```

Die Skripte sind idempotent: Sie legen fehlende Tenants/Superuser an, führen `migrate_schemas` aus und prüfen LiteLLM sowie die AI-Endpunkte (`/ai/ping`, `/ai/scope`).

> ℹ️ **Compose-Notizen**
> - Der `web`-Container führt `collectstatic` automatisch aus (Storage: `CompressedManifestStaticFilesStorage`).
> - Volumes bleiben bei `up -d` erhalten. Für einen vollständigen Reset siehe `npm run dev:reset`.
> - Container lesen `.env.docker`. Host-Tools nutzen weiterhin `.env`.

### Häufige Docker-Kommandos

| Kommando | Zweck |
| --- | --- |
| `docker compose -f docker-compose.yml -f docker-compose.dev.yml up` | Start im Vordergrund (Logs im Terminal) |
| `docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d` | Start im Hintergrund |
| `docker compose -f docker-compose.yml -f docker-compose.dev.yml down` | Stoppen ohne Volumes zu löschen |
| `docker compose -f docker-compose.yml -f docker-compose.dev.yml down -v` | Stoppen inkl. Entfernen der Volumes |
| `docker compose -f docker-compose.yml -f docker-compose.dev.yml ps` | Statusübersicht der laufenden Container |

### Quality-of-Life Skripte (npm)

| Script | Beschreibung |
| --- | --- |
| `npm run dev:up` | Initialisiert Datenbank & Tenants im Compose-Stack, erstellt Superuser |
| `npm run dev:check` | Führt Health-Checks (LiteLLM, `/ai/ping`, `/ai/scope`) aus |
| `npm run dev:init` | Führt `jobs:migrate` und `jobs:bootstrap` aus (nach `up -d`) |
| `npm run dev:stack` / `npm run win:dev:stack` | Startet App + ELK, Migrationen, Bootstrap, Demo- & Heavy-Seeding |
| `npm run dev:down` | Stoppt alle Container inkl. Volumes (`down -v`) |
| `npm run dev:restart` | Neustart von Web- und Worker-Containern |
| `npm run dev:rebuild` | Rebuild von Web-/Worker-Images (`-- --with-frontend` für Tailwind) |
| `npm run dev:reset` | Komplettreset (down -v → build --no-cache → up -d → init → check) |
| `npm run dev:manage <cmd>` | Führt `python manage.py <cmd>` im `web`-Container aus |
| `npm run jobs:migrate` | Compose-Job `migrate` für `migrate_schemas` |
| `npm run jobs:bootstrap` | Compose-Job `bootstrap_public_tenant` |
| `npm run jobs:rag` | Führt `docs/rag/schema.sql` gegen das RAG-Schema aus |
| `npm run jobs:rag:health` | Prüft pgvector/RAG-Schema |

Windows-Varianten (PowerShell) stehen als `npm run win:<script>` zur Verfügung (z. B. `win:dev:up`, `win:dev:stack`, `win:dev:rebuild`).

### API-Schema & SDKs

| Kommando | Ergebnis |
| --- | --- |
| `npm run api:schema` | Exportiert das OpenAPI-Schema in `docs/api/openapi.yaml` (lokal, ohne Docker-Compose) |
| `make schema` | Baut `docs/api/openapi.yaml` via `manage.py spectacular` |
| `make sdk` | Generiert TypeScript- (`clients/typescript`) und Python-SDKs (`clients/python`) auf Basis des aktuellen Schemas |

`make sdk` ruft `make schema` implizit auf. Für die SDK-Generierung müssen Node.js (für `openapi-typescript-codegen`) und die Python-CLI `openapi-python-client` verfügbar sein – beide sind in den Projektabhängigkeiten enthalten.

### Frontend & Tooling

| Script | Beschreibung |
| --- | --- |
| `npm run dev` | Lokaler Django-Server + Tailwind-Watcher (nur ohne Docker sinnvoll) |
| `npm run build:css` | Einmaliger Tailwind-Build |
| `npm run build:css:watch` | Tailwind-Watcher |
| `npm run storybook` | Startet Storybook (Port 6006) |
| `npm run storybook:build` | Erzeugt statischen Storybook-Build |
| `npm run e2e` | Playwright E2E-Tests |
| `npm run test` | Vitest-Unit-Tests für Frontend |
| `npm run test:py` | Python-Tests innerhalb des Web-Containers |
| `npm run test:py:cov` | Python-Tests inkl. Coverage für `ai_core`, `common`, `customers`, `documents`, `organizations`, `profiles`, `projects`, `users`, `theme` und `noesis2` |
| `npm run lint` | Ruff + Black (Check-Modus) |
| `npm run lint:fix` | Ruff (Fix) + Black Formatierung |
| `npm run format` | Prettier für JS/TS/CSS/MD/JSON |
| `npm run hooks:install` | Git-Hooks (pre-push) für macOS/Linux |
| `npm run win:hooks:install` | Git-Hooks Installation für Windows |

### Make Targets

| Target | Beschreibung |
| --- | --- |
| `make jobs:migrate` | Führt `migrate_schemas --noinput` aus |
| `make jobs:bootstrap` | Erstellt den Public-Tenant (`DOMAIN` erforderlich) |
| `make tenant-new` | Legt einen neuen Tenant an (`SCHEMA`, `NAME`, `DOMAIN`) |
| `make tenant-superuser` | Erstellt einen Tenant-Superuser (`SCHEMA`, `USERNAME`, `PASSWORD`, optional `EMAIL`) |
| `make jobs:rag` | Spielt `docs/rag/schema.sql` gegen `RAG_DATABASE_URL`/`DATABASE_URL` ein |
| `make jobs:rag:health` | Validiert RAG-Schema & `vector`-Extension |
| `make schema` | Exportiert das OpenAPI-Schema nach `docs/api/openapi.yaml` |
| `make sdk` | Generiert SDKs unter `clients/` auf Basis des aktuellen Schemas |

Alle Make-Targets greifen auf lokale Tools (`psql`, `python`). Innerhalb des Compose-Stacks empfiehlt sich die Nutzung der äquivalenten npm-Skripte (`npm run jobs:*`).

## Tests & Qualitätssicherung

```bash
npm run test:py        # Django/Celery Tests im Container
npm run test           # Frontend Tests (Vitest)
npm run lint           # Ruff + Black Checks
npm run e2e            # Playwright E2E
```

Der Python-Test-Runner installiert die benötigten Abhängigkeiten on-the-fly in einem temporären Container und räumt nach dem Lauf automatisch auf.

`npm run test:py:cov` ruft `pytest` mit gezielten `--cov`-Parametern für alle relevanten Django- und Service-Module auf. Die `.coveragerc` blendet Migrations- und Settings-Pfade aus, sodass der Bericht nur produktiven Quellcode berücksichtigt.

### Chaos-Tests (Fault Injection)

- `pytest -m chaos` führt nur die gezielt markierten Fault-Injection-Szenarien aus. Die Tests nutzen den Fixture `chaos_env`, um die Runtime-Schalter `REDIS_DOWN`, `SQL_DOWN` und `SLOW_NET` konsistent zu setzen.
- In GitHub Actions existiert der optionale Stage-Job `tests-chaos`, der über einen manuellen Workflow-Dispatch mit `run_chaos=true` gestartet wird. Der Job läuft parallelisiert via `pytest -m chaos -q -n auto`, erzeugt JUnit- und JSON-Artefakte sowie optionale k6-/Locust-Summaries und ist nach erfolgreichem Staging-Smoke-Test (Pipeline Stufen 8–10) als zusätzlicher QA-Gate vorgesehen. Weitere Details siehe [Pipeline-Dokumentation](docs/cicd/pipeline.md).
  - Für das Eingabefeld `run_chaos` akzeptiert der Workflow sowohl Boolean- als auch String-Werte. Die interne Normalisierung ergibt `RUN_CHAOS=true` ausschließlich für folgende Eingaben:
    - `true` (Boolean in der UI, Standard bei Häkchen)
    - `"true"` (String via `gh workflow run` oder API)
  - Jede andere Eingabe (`false`, `"false"`, leer, nicht gesetzt) führt zu `RUN_CHAOS=false`. Dieser Mini-Wahrheitstisch hilft, versehentliche Chaos-Runs bei benutzerdefinierten Dispatches oder Automatisierungen zu vermeiden.
- Freigaben erfolgen nur, wenn die zugehörigen [QA-Checklisten](docs/qa/checklists.md) als Gate dokumentiert und abgehakt sind.

#### Netzwerkchaos via Toxiproxy

- `docker compose up -d toxiproxy` startet den Proxy-Container lokal und richtet feste Listener ein (`localhost:15432` → PostgreSQL, `localhost:16379` → Redis, Admin-API unter `localhost:8474`). Die Web-/Worker-Container sprechen standardmäßig weiterhin `db`/`redis` an; zum Testen über den Proxy setze `COMPOSE_DATABASE_URL=postgresql://noesis2:noesis2@toxiproxy:15432/noesis2` bzw. `COMPOSE_REDIS_URL=redis://toxiproxy:16379/0`.
- Das Skript `scripts/chaos/toxiproxy.sh` verwaltet die benötigten Toxics. Mit `SLOW_NET=true scripts/chaos/toxiproxy.sh enable` werden Latenz, Bandbreitenlimit und Reset-Peers über die CLI injiziert; `scripts/chaos/toxiproxy.sh disable` räumt alles wieder auf. `status` zeigt den aktuellen Proxy-Zustand.
- Jeder Start/Stop wird samt Parametern in `logs/chaos/toxiproxy.log` protokolliert. Das File dient als Marker im ELK-Stack (Filebeat-Pickup oder manuelles Hochladen), damit sich Chaosphasen eindeutig mit Applikationslogs und Langfuse-Traces korrelieren lassen.

#### Chaos-Reporting & ELK-Verzahnung

- Chaos-Tests erzeugen pro Testlauf strukturierte Artefakte unter `logs/app/chaos/*.json`. Die Dateien enthalten u. a. `test_suite: "chaos"`, das `nodeid`, den Ausgang sowie die aktivierten Schalter aus `chaos_env`.
- Der lokale ELK-Stack liest die JSON-Artefakte automatisch ein. Starte ihn bei Bedarf mit `docker compose -f docker/elk/docker-compose.yml up -d` und stoppe ihn anschließend wieder mit `docker compose -f docker/elk/docker-compose.yml down`.
- In Kibana genügt eine Discover-Abfrage `test_suite:chaos`, um ausschließlich Chaos-Reports zu filtern und die Laufzeiten/Fehler direkt mit Applikationslogs zu korrelieren. Weitere Details siehe [docs/observability/elk.md](docs/observability/elk.md).

### Load-Testing Setup (k6 & Locust)

Ingestion läuft jetzt via `/ai/rag/ingestion/run/` → triggert Worker-Tasks, Queue ingestion.
Stelle sicher, dass ein Celery-Worker die Queue konsumiert, z. B. via `celery -A config worker -Q ingestion,celery -l INFO`.
Für performante RAG-Abfragen lohnt sich der optionale GIN-/B-Tree-Index aus [`docs/rag/schema.sql`](docs/rag/schema.sql), wenn
du häufig nach Metadaten (z. B. `case`) filterst. Der GIN-Index verwendet dabei `jsonb_path_ops`, um Gleichheitsfilter auf `metadata` effizient abzudecken.
Setze dabei auch den eindeutigen Index auf `(tenant_id, external_id)` und führe alle Statements schema-qualifiziert oder mit passendem `search_path` aus, wenn mehrere RAG-Schemata im Einsatz sind.

- **Skripte:**
  - `npm run load:k6` bzw. `make load:k6` startet das Spike+Soak-Szenario aus `load/k6/script.js`. Setze dafür die Staging-Parameter (`STAGING_WEB_URL`, `STAGING_TENANT_SCHEMA`, `STAGING_TENANT_ID`, `STAGING_CASE_ID`, optional `STAGING_BEARER_TOKEN`, `STAGING_KEY_ALIAS`). Zusätzliche Parameter wie `SCOPE_SOAK_DURATION` können via ENV angepasst werden.
  - `npm run load:locust` bzw. `make load:locust` lädt die User-Klassen aus `load/locust/locustfile.py`. Übergib weitere Flags nach `--`, z. B. `npm run load:locust -- --headless -u 30 -r 10 --run-time 5m`.
- **Matrix & Scaling:** Für Staging orientiert sich die Grundlast an der Web-Concurrency (~30 Worker). Nutze für Locust `-u 30` (gleichzeitige Nutzer) als Basis und erhöhe schrittweise (z. B. 30 → 60 → 90) je nach QA-Plan. Für k6 beschreibt das Script ein kurzes Spike+Soak-Profil mit Ramp-Up/-Down und optionalen Overrides (`SCOPE_SPIKE_RPS`, `SCOPE_SOAK_RPS`).
- **Tenancy & Idempotency:** Beide Skripte injizieren die Header `X-Tenant-Schema`, `X-Tenant-ID`, `X-Case-ID` sowie eindeutige `Idempotency-Key`s. Standardpayloads folgen den Beispielen aus [docs/api/reference.md](docs/api/reference.md) und lassen sich über ENV-Overrides (`LOCUST_SCOPE_PAYLOAD`, `LOCUST_INGESTION_PAYLOAD`, …) anpassen.
- **Ausführung:** Die Load-Skripte sind nicht in CI eingebunden. Führe sie manuell lokal oder gegen Staging aus (siehe [docs/cloud/gcp-staging.md](docs/cloud/gcp-staging.md) für URLs und Credentials) und archiviere Metriken/Artefakte als Bestandteil der QA-Gates.

## Manuelles Setup ohne Docker

Für Systeme ohne Docker-Unterstützung gibt es einen dokumentierten Fallback:
[docs/development/manual-setup.md](docs/development/manual-setup.md).

## Konfiguration (.env)
Benötigte Variablen (siehe `.env.example` oder `.env.dev.sample`):

- SECRET_KEY: geheimer Schlüssel für Django
- DEBUG: `true`/`false`
- DB_USER / DB_PASSWORD / DB_NAME: gemeinsame Dev‑Credentials; werden für den Container‑Init und DSNs genutzt.
- DATABASE_URL: Verbindungs-URL zur PostgreSQL-Datenbank (App‑DB, default: `postgresql://noesis2:noesis2@db:5432/noesis2`)
- REDIS_URL: Redis-Endpoint (z. B. für Celery)
- RAG_DATABASE_URL (optional): separates DSN für das pgvector-Schema. Ohne Angabe nutzt der Stack `DATABASE_URL`.
  Stelle sicher, dass [`docs/rag/schema.sql`](docs/rag/schema.sql) angewendet wurde und die `vector`-Extension aktiv ist.
  Mandanten-IDs müssen UUIDs sein; vorhandene Legacy-IDs werden deterministisch gemappt, sollten aber per Migration bereinigt
  werden, bevor produktive Daten geladen werden.
- RAG_STATEMENT_TIMEOUT_MS (optional, default `15000`): maximale Laufzeit für SQL-Statements des pgvector-Clients (Upsert, Suche,
  Health-Checks). Höhere Werte erhöhen das Timeout, niedrigere brechen Abfragen früher ab.
- RAG_RETRY_ATTEMPTS (optional, default `3`): Anzahl der Wiederholungsversuche für fehlgeschlagene pgvector-Operationen. Jeder
  Versuch wird protokolliert und nutzt dieselbe Verbindung.
- RAG_RETRY_BASE_DELAY_MS (optional, default `50`): Basiswartezeit zwischen Wiederholungen (linearer Backoff pro Versuch in
  Millisekunden).

Hinweis für RAG-Workloads: Lege die optionalen Indizes aus [`docs/rag/schema.sql`](docs/rag/schema.sql) an, wenn Metadatenfilter
häufig verwendet werden (GIN-Index auf `metadata` sowie B-Tree auf `metadata->>'case'`).

> ⚠️ **Vector-Backend Auswahl:** `RAG_VECTOR_STORES` unterstützt aktuell nur
> `pgvector`. Abweichende `backend`-Werte führen beim Start zu einem
> `ValueError` aus `get_default_router()`, sodass Fehlkonfigurationen früh
> auffallen.

Beispielkonfiguration für getrennte Scopes (z. B. isolierte Großmandanten):

```python
RAG_VECTOR_STORES = {
    "global": {
        "backend": "pgvector",
        "dsn_env": "RAG_DATABASE_URL",
        "default": True,
    },
    "enterprise": {
        "backend": "pgvector",
        "schema": "rag_enterprise",
        "tenants": ["f1d8f7af-4d4a-4f13-9d5b-a1c46b0d5b61"],
        "schemas": ["acme_prod"],
    },
}
```

Der Router mappt automatisch alle aufgeführten Tenant-IDs oder Schema-Namen auf
den jeweiligen Scope und fällt ansonsten auf `global` zurück.

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
  - Labels → Modelle (Vertex AI):
    - `default`, `fast`, `simple-query`, `synthesize`, `extract`, `classify`, `analyze` → `vertex_ai/gemini-2.5-flash`
    - `reasoning`, `draft` → `vertex_ai/gemini-2.5-pro`
    - `embedding` → `vertex_ai/text-embedding-004`
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

  - Labels → Modelle (AI Studio):
    - `default`, `fast`, `simple-query`, `synthesize`, `extract`, `classify`, `analyze` → `gemini-2.5-flash`
    - `reasoning`, `draft` → `gemini-2.5-pro`
    - `embedding` → `google/text-embedding-004`

Hinweis: `MODEL_ROUTING.local.yaml` ist git‑ignored und überschreibt nur lokal. In Prod wird ausschließlich `MODEL_ROUTING.yaml` verwendet.

## Observability

- [Langfuse Guide](docs/observability/langfuse.md)
- [ELK Stack für lokale Entwicklung](docs/observability/elk.md) — Oneshot-Bootstrap: `bash scripts/dev-up-all.sh`

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
