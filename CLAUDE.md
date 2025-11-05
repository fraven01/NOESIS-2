# CLAUDE Leitfaden

Zentrale Navigationsdatei für Claude Code bei der Arbeit mit NOESIS 2. Dieses Dokument ergänzt die [`AGENTS.md`](AGENTS.md) um Claude-spezifische Arbeitsweisen und Kontextverweise.

*Hinweis: Der Begriff „Pipeline" ist eine historische Bezeichnung für die heute als „Graph" (LangGraph) bezeichneten Orchestrierungs-Flows.*

## Zweck & Geltungsbereich
- Gilt für alle Arbeiten mit Claude Code in diesem Repository
- Ergänzt die [`AGENTS.md`](AGENTS.md) um Claude-spezifische Arbeitsweisen
- Vor Änderungen prüfe den Verzeichnispfad auf spezifischere Leitfäden (z. B. `theme/AGENTS.md`)
- Alle technischen Details und Primärquellen sind in [`AGENTS.md`](AGENTS.md) verlinkt

## Systemverständnis (Kurzfassung)

### Architektur-Überblick
NOESIS 2 ist eine mandantenfähige Django-Plattform (Python 3.12+) mit folgenden Hauptkomponenten:
- **Web-Service**: Django/Gunicorn für HTTP & Admin
- **Worker-Services**: Celery mit zwei Queues (`agents`, `ingestion`)
- **Agenten**: LangGraph-Orchestrierung für RAG-Flows
- **RAG-Store**: pgvector mit Multi-Tenant-Isolation
- **LiteLLM**: Proxy für LLM-Zugriff (Gemini, Vertex AI)
- **Observability**: Langfuse (Traces) + ELK (Logs)

Detaillierte Systemlandschaft und Diagramme: [docs/architektur/overview.md](docs/architektur/overview.md)

### Technologie-Stack
- Backend: Django 5.x, Python 3.12+
- Async: Celery, Redis
- DB: PostgreSQL mit `pgvector`, `django-tenants`
- Frontend: Tailwind CSS v4, React/TypeScript
- Entwicklung: Docker Compose, npm-Skripte
- CI/CD: GitHub Actions

## Wichtige Verträge & Pflichtfelder

### Tool-Verträge (Layer 2)
Alle Tools verwenden: `ToolContext`, `*Input`, `*Output`, `ToolError`.

**Pflicht-Tags**:
- `tenant_id` (UUID)
- `trace_id` (string)
- `invocation_id` (UUID)
- Genau **eine** Laufzeit-ID: `run_id` **XOR** `ingestion_run_id`

**Optional**:
- `idempotency_key`, `case_id`, `workflow_id`, `collection_id`, `document_id`

Siehe: [docs/agents/tool-contracts.md](docs/agents/tool-contracts.md)

### HTTP-Header
Jeder API-Aufruf erfordert:
- `X-Tenant-ID` (Pflicht)
- `X-Trace-ID` (Pflicht)
- `X-Case-ID` (Optional)
- `Idempotency-Key` (Optional, für POST)

### Fehlertypen
Typed Errors aus `ai_core/tools/errors.py`:
- `InputError` - Validierungsfehler
- `NotFound` - Ressource nicht gefunden
- `RateLimited` - Rate-Limit erreicht
- `Timeout` - Zeitbudget überschritten
- `Upstream` - Externer Dienst fehlgeschlagen
- `Internal` - Interner Fehler

## Paketstruktur & Import-Regeln

### Layer-Hierarchie (nur nach unten importieren)
```
tenant_logic → ai_core/graphs → tools → services → shared
Frontend (getrennt, keine Rückimporte)
```

### Wichtige Pakete
- `ai_core/graphs/` - LangGraph-Orchestrierung (Layer 3)
- `ai_core/nodes/` - Wiederverwendbare Graph-Knoten
- `ai_core/rag/` - Retrieval & Embedding-Logik
- `ai_core/tools/` - Tool-Implementierungen
- `ai_core/llm/` - Modellanbindung & Routing
- `ai_core/infra/` - Infrastruktur-Adapter
- `ai_core/middleware/` - Telemetrie & Caching

## Entwicklungsworkflow

### Lokales Setup (Docker)
```bash
# Setup
cp .env.example .env
npm run dev:stack  # Startet alles inkl. ELK

# Häufige Befehle
npm run dev:up       # Init DB & Tenants
npm run dev:check    # Health-Checks
npm run dev:restart  # Neustart Services
npm run dev:reset    # Komplett-Reset
```

Windows-Varianten: `npm run win:dev:*`

### Tests
```bash
npm run test:py        # Python-Tests (Docker)
npm run test:py:cov    # Mit Coverage
npm run test           # Frontend (Vitest)
npm run e2e            # Playwright E2E
```

Test-DB: `noesis2_test` (isoliert von Dev-Daten)

### Linting & Formatierung
```bash
npm run lint         # Ruff + Black (Check)
npm run lint:fix     # Auto-Fix
npm run format       # Prettier
```

### API & SDKs
```bash
npm run api:schema   # Export OpenAPI
make sdk             # Generate TS/Python SDKs
```

## RAG & Embedding-Profile

### Vector Spaces
Konfiguriert über `RAG_VECTOR_STORES`:
- `global` (default): 1536-dimensional, pgvector
- Optional: `enterprise` mit separatem Schema/Dimension

### Embedding-Profile
Konfiguriert über `RAG_EMBEDDING_PROFILES`:
- `standard`: oai-embed-large, 1536D, global space
- `premium`: vertex_ai/text-embedding-004, 3072D, enterprise space

### Routing-Regeln
In `config/rag_routing_rules.yaml`:
- Default-Profil (Pflicht)
- Optionale Overrides nach: `tenant`, `process`, `doc_class`
- Case-insensitive, Spezifität nach Anzahl gesetzter Felder

Management-Befehle:
```bash
python manage.py rag_routing_rules [--tenant=X --process=Y --doc-class=Z]
python manage.py rag_schema_smoke --space=global
python manage.py sync_rag_schemas    # Alle Spaces
python manage.py check_rag_schemas   # Health-Check
```

## Glossar (Häufige Begriffe)

| Begriff | Bedeutung | Status |
|---------|-----------|--------|
| `tenant_id` | Mandanten-ID (UUID) | Pflicht |
| `trace_id` | End-to-End-Korrelations-ID | Pflicht |
| `invocation_id` | Tool-Aufruf-ID | Pflicht |
| `case_id` | Geschäftsvorfall-ID | Optional |
| `workflow_id` | Graph-/Prozess-ID | Optional |
| `run_id` | Graph-Lauf-ID | Pflicht (XOR) |
| `ingestion_run_id` | Ingestion-Lauf-ID | Pflicht (XOR) |
| `collection_id` | Dokument-Scope für Filter | Optional |
| Graph | LangGraph-Orchestrierung | - |
| Pipeline | Veraltet, nutze "Graph" | ⚠️ Legacy |

## Arbeitsweise mit Claude Code

### Vor dem Start
1. Lies [`AGENTS.md`](AGENTS.md) für vollständige Verträge
2. Prüfe spezifischere `AGENTS.md` im Arbeitsverzeichnis
3. Konsultiere verlinkte Primärquellen für Details

### Bei Änderungen
1. **Contracts zuerst**: Tool-Inputs/-Outputs definieren
2. **Tests schreiben**: Unit → Integration → E2E
3. **Dokumentation**: README in betroffenen Paketen aktualisieren
4. **Tracing**: Langfuse-Tags für neue Features setzen
5. **Linting**: `npm run lint:fix` vor Commit

### Bei Fehlern
1. Prüfe Langfuse-Traces (`trace_id` suchen)
2. Konsultiere ELK-Logs für Chaos/Performance
3. Siehe Runbooks: [docs/runbooks/](docs/runbooks/)
4. Error-Codes in `ai_core/tools/errors.py`

### Bei RAG-Arbeiten
1. Lies [docs/rag/overview.md](docs/rag/overview.md)
2. Verstehe Vector Spaces & Profile
3. Prüfe Routing-Regeln vor Änderungen
4. Teste mit `rag_schema_smoke`
5. Dimensionen dürfen **nie** gemischt werden

### Bei Graph-Entwicklung
1. Lies [ai_core/graphs/README.md](ai_core/graphs/README.md)
2. Nutze `GraphNode` & `GraphTransition`
3. Teste mit Fake-Services/Retrievers
4. Emittiere strukturierte Transitions
5. Dokumentiere Guardrails

## Multi-Tenancy

### Tenant-Setup
```bash
# Neuer Tenant
make tenant-new SCHEMA=demo NAME="Demo" DOMAIN=demo.local

# Superuser anlegen
make tenant-superuser SCHEMA=demo USERNAME=admin PASSWORD=secret

# Migrationen
make jobs:migrate
```

### Tenant-Isolation
- DB: Schema pro Tenant (`django-tenants`)
- RAG: `tenant_id`-Filter in pgvector
- API: `X-Tenant-ID` Header verpflichtend
- Traces: `tenant_id` Tag in Langfuse

Details: [docs/multi-tenancy.md](docs/multi-tenancy.md)

## Observability

### Langfuse (Traces & Metrics)
- Host: `LANGFUSE_HOST`
- Keys: `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`
- Tags: `tenant_id`, `trace_id`, `case_id`, `workflow_id`
- Spans: Automatisch für Tools & Graphs

Siehe: [docs/observability/langfuse.md](docs/observability/langfuse.md)

### ELK-Stack (Logs)
```bash
# Lokal starten
docker compose -f docker/elk/docker-compose.yml up -d

# Logs ansehen: http://localhost:5601 (Kibana)
# Filter: test_suite:chaos für Chaos-Tests
```

Siehe: [docs/observability/elk.md](docs/observability/elk.md)

## Häufige Aufgaben

### Neues Tool erstellen
1. Definiere Input/Output-Modelle in `ai_core/tools/`
2. Verwende `ToolContext`, `ToolOutput[IT, OT]`
3. Implementiere Error-Handling mit `ToolError`
4. Registriere in LangChain-Tools
5. Teste mit Fake-Retrievers
6. Dokumentiere in Tool-Contract-Docs

### Neuen Graph hinzufügen
1. Erstelle Graph in `ai_core/graphs/`
2. Definiere Knoten mit `GraphNode`
3. Nutze `GraphTransition` für Übergänge
4. Implementiere Runner-Funktionen
5. Registriere in Graph-Registry
6. Teste alle Pfade & Transitions
7. Dokumentiere in `ai_core/graphs/README.md`

### Embedding-Profil ändern
1. **WARNUNG**: Dimensionswechsel = Migration!
2. Prüfe `RAG_EMBEDDING_PROFILES` in Settings
3. Validiere mit `validate_embedding_configuration()`
4. Teste Schema-Render mit `rag_schema_smoke`
5. Aktualisiere Routing-Regeln
6. Führe Re-Embedding durch (Runbook!)
7. Dokumentiere in [docs/rag/configuration.md](docs/rag/configuration.md)

### Deployment
1. Merge in `main` triggert CI/CD
2. Pipeline: Lint → Test → Build → Deploy Staging
3. Smoke-Checks in Staging
4. Manuelle Freigabe für Prod
5. Traffic-Split (10% → 100%)
6. Monitoring in Langfuse/ELK

Details: [docs/cicd/pipeline.md](docs/cicd/pipeline.md)

## Qualitätsregeln

### Code-Qualität
- Ruff + Black für Python
- TypeScript strict mode
- 100% Type-Hints in Tool-Contracts
- Pydantic für alle Datenmodelle
- Frozen Dataclasses für Immutability

### Test-Coverage
- Unit-Tests für alle Tools & Nodes
- Integration-Tests für Graphs
- E2E für kritische Pfade
- Chaos-Tests für Fault-Injection
- Load-Tests (k6, Locust) manuell

### Dokumentation
- README in jedem Paket
- Docstrings für öffentliche APIs
- JSON-Schema-Export für Tools
- Runbooks für Betrieb
- Changelog in Pull Requests

### Security
- PII-Maskierung verpflichtend
- Secrets via `.env` oder Secret Manager
- Keine API-Keys in Code/Logs
- Rate-Limiting für LLM-Calls
- Tenant-Isolation durchgängig

Siehe: [docs/security/secrets.md](docs/security/secrets.md)

## Nützliche Kommandos

```bash
# Entwicklung
npm run dev:stack           # Alles starten
npm run dev:check           # Health-Checks
npm run dev:manage <cmd>    # Django-Management

# Tests
npm run test:py            # Python
npm run test:py:cov        # Mit Coverage
npm run e2e                # E2E

# Linting
npm run lint               # Check
npm run lint:fix           # Auto-Fix

# RAG
python manage.py rag_routing_rules
python manage.py rag_schema_smoke --space=global
python manage.py sync_rag_schemas
python manage.py check_rag_schemas

# Tenants
make tenant-new SCHEMA=demo NAME="Demo" DOMAIN=demo.local
make tenant-superuser SCHEMA=demo USERNAME=admin PASSWORD=secret
make jobs:migrate
make jobs:bootstrap DOMAIN=public.localhost

# API
npm run api:schema         # Export OpenAPI
make sdk                   # Generate SDKs
```

## Navigationsverzeichnis (Primärquellen)

### Architektur & Konzepte
1. [AGENTS.md](AGENTS.md) - Hauptleitfaden
2. [docs/architektur/overview.md](docs/architektur/overview.md) - Systemarchitektur
3. [docs/multi-tenancy.md](docs/multi-tenancy.md) - Mandantenfähigkeit
4. [docs/contracts.md](docs/contracts.md) - Datenverträge

### AI Core
5. [docs/agents/overview.md](docs/agents/overview.md) - LangGraph-Agenten
6. [docs/agents/tool-contracts.md](docs/agents/tool-contracts.md) - Tool-Verträge
7. [docs/rag/overview.md](docs/rag/overview.md) - RAG-Architektur
8. [docs/rag/ingestion.md](docs/rag/ingestion.md) - Ingestion-Pipeline
9. [docs/rag/configuration.md](docs/rag/configuration.md) - RAG-Konfiguration

### Entwicklung
10. [docs/development/onboarding.md](docs/development/onboarding.md) - Einstieg
11. [docs/development/manual-setup.md](docs/development/manual-setup.md) - Setup ohne Docker
12. [README.md](README.md) - Projekt-README

### Betrieb
13. [docs/operations/scaling.md](docs/operations/scaling.md) - Skalierung
14. [docs/runbooks/migrations.md](docs/runbooks/migrations.md) - Migrationen
15. [docs/runbooks/incidents.md](docs/runbooks/incidents.md) - Incident-Handling

### Observability
16. [docs/observability/langfuse.md](docs/observability/langfuse.md) - Langfuse
17. [docs/observability/elk.md](docs/observability/elk.md) - ELK-Stack

### CI/CD & Qualität
18. [docs/cicd/pipeline.md](docs/cicd/pipeline.md) - CI/CD-Pipeline
19. [docs/qa/checklists.md](docs/qa/checklists.md) - QA-Checklisten
20. [docs/security/secrets.md](docs/security/secrets.md) - Security

## LLM-Kurzreferenz (für Claude Code)

- `trace_id` ist die verbindliche End-to-End-Korrelations-ID
- Jeder Tool-Aufruf benötigt: `tenant_id`, `trace_id`, `invocation_id` + genau **eine** Laufzeit-ID
- HTTP-APIs erfordern immer `X-Tenant-ID` Header
- `Graph` ist der aktuelle Begriff, `Pipeline` ist veraltet
- Graphen liegen in `ai_core/graphs/`
- Ingestion-Queue: `ingestion`, Task: `run_ingestion_graph`
- Dimensionen dürfen **niemals** gemischt werden (Migration erforderlich!)
- Tool-Errors nutzen typisierte `ToolErrorType` aus `ai_core/tools/errors.py`
- Routing-Regeln sind case-insensitive
- PII-Maskierung ist verpflichtend (Session-Scope)
- Idempotenz über `Idempotency-Key` für POST-Requests

## Governance & Änderungen

- Architektur-Änderungen zuerst in Primärquellen dokumentieren
- Pull Requests verlinken auf aktualisierte Docs
- Runbooks haben eigene Changelogs
- Breaking Changes erfordern Migration-Runbook
- Idempotenz bewahren: Nur bei neuen Links/Begriffen/Widersprüchen aktualisieren

## Fragen oder Probleme?

1. Konsultiere [`AGENTS.md`](AGENTS.md) für vollständige Verweise
2. Suche in Primärquellen unter [docs/](docs/)
3. Prüfe Langfuse-Traces für Laufzeitfehler
4. Konsultiere ELK-Logs für Performance/Chaos
5. Siehe Runbooks für Betriebsprobleme
6. Bei Unsicherheit: Frage nach oder erstelle Issue

---

**Version**: 1.0
**Zuletzt aktualisiert**: 2025-11-05
**Gilt für**: NOESIS 2, Branch `main`
