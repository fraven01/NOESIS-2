# CLAUDE Leitfaden

Zentrale Navigationsdatei für Claude Code bei der Arbeit mit NOESIS 2. Dieses Dokument fokussiert sich auf **Claude-spezifische Workflows** und verweist für alle Contracts, Architektur und Glossar auf [`AGENTS.md`](AGENTS.md).

*Hinweis: Der Begriff „Pipeline" ist eine historische Bezeichnung für die heute als „Graph" (LangGraph) bezeichneten Orchestrierungs-Flows.*

## Zweck & Geltungsbereich

- **Operational Guide** für Claude Code in diesem Repository
- Gleicher Wissensstand wie Gemini Code und Codex: alle drei nutzen dieselben Contracts & Workflows
- Alle Contracts, Glossar, Architektur → siehe [`AGENTS.md`](AGENTS.md)
- Vor Änderungen prüfe den Verzeichnispfad auf spezifischere Leitfäden (z. B. `theme/AGENTS.md`)
- Dieser Guide enthält nur Claude-spezifische Arbeitsweisen und Workflows

## Systemverständnis (Kurzfassung)

### Architektur-Überblick

NOESIS 2 ist eine mandantenfähige Django-Plattform (Python 3.12+) mit folgenden Hauptkomponenten:

- **Web-Service**: Django/Gunicorn für HTTP & Admin
- **Worker-Services**: Celery mit zwei Queues (`agents`, `ingestion`)
- **Agenten**: LangGraph-Orchestrierung für RAG-Flows
- **RAG-Store**: pgvector mit Multi-Tenant-Isolation
- **LiteLLM**: Proxy für LLM-Zugriff (Gemini, Vertex AI)
- **Observability**: Langfuse (Traces) + ELK (Logs)

**Vollständige Architektur & Contracts**: [AGENTS.md](AGENTS.md) → [docs/architecture/overview.md](docs/architecture/overview.md)

### Technologie-Stack

- Backend: Django 5.x, Python 3.12+
- Async: Celery, Redis
- DB: PostgreSQL mit `pgvector`, `django-tenants`
- Frontend: Tailwind CSS v4, React/TypeScript
- Entwicklung: Docker Compose, npm-Skripte
- CI/CD: GitHub Actions

## Contracts & Pflichtfelder (Kurzreferenz)

**Vollständige Contracts**: [AGENTS.md#Tool-Verträge](AGENTS.md#tool-verträge-layer-2--norm)
**Implementierungs-Guide**: [docs/architecture/id-guide-for-agents.md](docs/architecture/id-guide-for-agents.md)

### Pflicht-Tags für alle Tool-Aufrufe

- `tenant_id` (UUID)
- `trace_id` (string)
- `invocation_id` (UUID)
- **Mindestens eine** Laufzeit-ID: `run_id` und/oder `ingestion_run_id` (beide können koexistieren, z.B. wenn Workflow Ingestion triggert)

### HTTP-Header (Pflicht)

- `X-Tenant-ID` (UUID)
- `X-Trace-ID` (string)

**Glossar & Feld-Matrix**: [AGENTS.md#glossar--feld-matrix](AGENTS.md#glossar--feld-matrix)

## Entwicklungsworkflow (Claude-spezifisch)

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

**Windows-Varianten**: `npm run win:dev:*`

### Tests

**Python-Tests (Plattformunabhängig via Docker)**:
```bash
npm run test:py              # Vollständig (inkl. @pytest.mark.slow)
npm run test:py:parallel     # EMPFOHLEN: Fast parallel + slow serial (optimiert)
npm run test:py:single -- <path>  # Einzelner Test oder spezifische Funktion
npm run test:py:fast         # Schnell (ohne @pytest.mark.slow)
npm run test:py:unit         # Unit-Tests (ohne DB, super schnell)
npm run test:py:cov          # Coverage über alles
npm run test:py:clean        # Clean (pip --no-cache-dir) + vollständig
```

**Hinweis**: `test:py:parallel` ist die **empfohlene** Methode für lokale Entwicklung:
- **Phase 1**: Fast tests (`-m 'not slow'`) parallel mit xdist (`-n auto`)
- **Phase 2**: Slow tests (`-m 'slow'`) serial für Tenant-Operationen
- Isolierte Test-DBs pro Worker (`test_noesis2_gw0`, etc.)
- ~5-10x schneller als `test:py` (je nach CPU-Kernen)

**Beispiele für einzelne Tests**:
```bash
# Einzelne Datei
npm run test:py:single -- ai_core/tests/graphs/test_universal_ingestion_graph.py

# Spezifische Testfunktion
npm run test:py:single -- ai_core/tests/graphs/test_universal_ingestion_graph.py::test_function_name

# Testklasse und -methode
npm run test:py:single -- ai_core/tests/graphs/test_universal_ingestion_graph.py::TestClass::test_method
```

**Windows-Aliase** (verweisen auf plattformunabhängige Docker-Befehle):
```bash
npm run win:test:py              # → test:py
npm run win:test:py:parallel     # → test:py:parallel (EMPFOHLEN)
npm run win:test:py:fast         # → test:py:fast
npm run win:test:py:unit         # → test:py:unit
npm run win:test:py:cov          # → test:py:cov
npm run win:test:py:clean        # → test:py:clean

# Platform-spezifische Befehle (PowerShell):
npm run win:test:py:single -- <path>  # Einzelner Test via PowerShell-Skript
npm run win:test:py:rag               # RAG-spezifische Tests
```

**Test-DB-Isolation (wichtig!)**:
- **Test-Settings**: `noesis2.settings.test_parallel` konfiguriert worker-spezifische Test-DBs
- **Produktions-DBs werden NIEMALS für Tests verwendet** (seit 2026-01-02 Fix)
- **Parallele Workers**: Jeder xdist-Worker (`gw0`, `gw1`, etc.) bekommt eigene DB (`test_noesis2_gw0`, etc.)
- **Slow Tests**: Laufen gegen dedizierte Test-DB mit eigenem Tenant-Schema (`autotest_gw0`)
- **Schema-Namen**: MÜSSEN gequotet werden (via `connection.ops.quote_name()`) um Bindestriche zu unterstützen

**Frontend-Tests**:
```bash
npm run test           # Vitest
npm run e2e            # Playwright E2E
```

**Chaos Tests** (2026-01-05 Contract Migration):

Fault-injection tests für Redis, SQL, Rate Limits und Netzwerk-Issues. Validiert System-Resilience und Observability unter Fehlerbedingungen.

```bash
# Alle Chaos-Tests
npm run test:py -- tests/chaos/ -v

# Spezifische Test-Suite
npm run test:py:single -- tests/chaos/test_tool_context_contracts.py

# Mit Fault-Injection
REDIS_DOWN=1 npm run test:py -- tests/chaos/redis_faults.py -v
SQL_DOWN=1 npm run test:py -- tests/chaos/sql_faults.py -v
```

**Neue Meta-Struktur** (ScopeContext + BusinessContext):
```python
from tests.chaos.conftest import _build_chaos_meta

meta = _build_chaos_meta(
    tenant_id="tenant-001",
    trace_id="trace-001",
    case_id="case-001",      # Optional business context
    run_id="run-001",        # Und/oder ingestion_run_id (beide können koexistieren)
)
```

**Test-Dateien**:
- `ingestion_faults.py` - Rate Limits, Deduplication, Dead Letter
- `sql_faults.py` - SQL Downtime, Idempotency
- `redis_faults.py` - Broker Downtime, Task Backoff
- `test_tool_context_contracts.py` - ✨ **NEU**: ToolContext Validierung
- `test_graph_io_contracts.py` - ✨ **NEU**: Graph I/O Specs (`schema_id`/`schema_version`)
- `test_chunker_routing.py` - ✨ **NEU**: Chunker Routing
- `test_observability_contracts.py` - ✨ **NEU**: Langfuse Tag Propagation

**Siehe**: [tests/chaos/README.md](tests/chaos/README.md) für vollständige Migration-Guide.

**Test-Marker**:
- `@pytest.mark.slow`: DB-intensive Tests (Tenant-Operationen, Schema-Erstellung)
- `@pytest.mark.gold`: PII-Tests mit `PII_MODE=gold`
- `@pytest.mark.chaos`: Fault-Injection-Tests
- `@pytest.mark.perf_smoke`: High-Concurrency-Tests
- `@pytest.mark.tenant_write`: Tests die Tenant-Schemas erstellen/modifizieren (gruppiert für xdist)

**Häufige Test-Probleme & Lösungen**:

1. **"relation does not exist" in slow tests**:
   - **Ursache**: Fehlende `TEST["NAME"]` Konfiguration oder nicht-gequotete Schema-Namen
   - **Lösung**: Bereits gefixt in `noesis2/settings/test_parallel.py` (2026-01-02)
   - **Verifikation**: `TEST["NAME"]` muss auf `test_noesis2_*` gesetzt sein

2. **"current transaction is aborted" Fehler**:
   - **Ursache**: Vorheriger SQL-Fehler (oft durch nicht-gequotete Schema-Namen mit `-`)
   - **Lösung**: Immer `connection.ops.quote_name()` für Schema-/Tabellen-Namen verwenden
   - **Beispiel**: `cursor.execute(f"SET search_path TO {connection.ops.quote_name(schema)}")`

3. **Tests laufen gegen Production-DB**:
   - **Ursache**: `TEST["NAME"]` nicht gesetzt, pytest-django verwendet `NAME` statt `TEST["NAME"]`
   - **Symptom**: DB-Name in Logs zeigt `noesis2` statt `test_noesis2`
   - **Fix**: Siehe `noesis2/settings/test_parallel.py` Zeilen 30-39

4. **Slow tests überspringen Schemas von fast tests**:
   - **Ursache**: `_MIGRATED_SCHEMAS` Set ist prozess-global, aber Tests laufen in separaten Prozessen
   - **Lösung**: `test_tenant_schema_name` Fixture prüft fehlende Tabellen und migriert nach
   - **Siehe**: `conftest.py` Zeilen 537-622

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

## RAG & Embedding-Profile (Praktische Hinweise)

**Vollständige RAG-Architektur**: [docs/rag/overview.md](docs/rag/overview.md)

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

## Arbeitsweise mit Claude Code

### Vor dem Start

1. **Lies [`AGENTS.md`](AGENTS.md)** für vollständige Verträge & Architektur
2. Prüfe spezifischere `AGENTS.md` im Arbeitsverzeichnis
3. Konsultiere verlinkte Primärquellen für Details

### Bei Änderungen

1. **Contracts zuerst**: Tool-Inputs/-Outputs definieren (siehe [AGENTS.md#Tool-Verträge](AGENTS.md#tool-verträge-layer-2--norm))
2. **Layer-Regeln beachten**: Import-Hierarchie in [AGENTS.md#Paketgrenzen](AGENTS.md#paketgrenzen-import-regeln)
3. **Tests schreiben**: Unit → Integration → E2E
4. **Dokumentation**: README in betroffenen Paketen aktualisieren
5. **Tracing**: Langfuse-Tags für neue Features setzen
6. **Linting**: `npm run lint:fix` vor Commit

### Bei Fehlern

1. Prüfe Langfuse-Traces (`trace_id` suchen)
2. Konsultiere ELK-Logs für Chaos/Performance
3. Siehe Runbooks: [docs/runbooks/](docs/runbooks/)
4. Error-Codes in `ai_core/tools/errors.py` (Typed Errors: `InputError`, `NotFound`, `RateLimited`, `Timeout`, `Upstream`, `Internal`)

### Bei RAG-Arbeiten

1. Lies [docs/rag/overview.md](docs/rag/overview.md)
2. Verstehe Vector Spaces & Profile
3. Prüfe Routing-Regeln vor Änderungen
4. Teste mit `rag_schema_smoke`
5. **WARNUNG**: Dimensionen dürfen **nie** gemischt werden (Migration erforderlich!)

### Bei Graph-Entwicklung

1. Lies [ai_core/graphs/README.md](ai_core/graphs/README.md)
2. Definiere Graph I/O Spezifikationen: versionierte Pydantic Input/Output Modelle mit `schema_id`/`schema_version` und `io_spec` (siehe `ai_core/graph/io.py`)
3. Nutze `GraphNode` & `GraphTransition`
4. Teste mit Fake-Services/Retrievers
5. Emittiere strukturierte Transitions
6. Dokumentiere Guardrails

## Multi-Tenancy (Setup & Commands)

**Vollständige Multi-Tenancy-Architektur**: [docs/multi-tenancy.md](docs/multi-tenancy.md)

### Tenant-Setup

```bash
# Neuer Tenant
make tenant-new SCHEMA=demo NAME="Demo" DOMAIN=demo.local

# Superuser anlegen
make tenant-superuser SCHEMA=demo USERNAME=admin PASSWORD=secret

# Migrationen
make jobs:migrate
```

### Tenant-Isolation (Kurzfassung)

- DB: Schema pro Tenant (`django-tenants`)
- RAG: `tenant_id`-Filter in pgvector
- API: `X-Tenant-ID` Header verpflichtend
- Traces: `tenant_id` Tag in Langfuse

## Observability (Praktische Hinweise)

**Vollständige Observability-Guides**: [docs/observability/langfuse.md](docs/observability/langfuse.md), [docs/observability/elk.md](docs/observability/elk.md)

### Langfuse (Traces & Metrics)

- Host: `LANGFUSE_HOST`
- Keys: `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`
- Tags: `tenant_id`, `trace_id`, `case_id`, `workflow_id`
- Spans: Automatisch für Tools & Graphs

### ELK-Stack (Logs)

```bash
# Lokal starten
docker compose -f docker/elk/docker-compose.yml up -d

# Logs ansehen: http://localhost:5601 (Kibana)
# Filter: test_suite:chaos für Chaos-Tests
```

## Häufige Aufgaben (Step-by-Step)

### Neues Tool erstellen

1. Definiere Input/Output-Modelle in `ai_core/tools/`
2. Verwende `ToolContext`, `ToolOutput[IT, OT]` (siehe [AGENTS.md#Tool-Verträge](AGENTS.md#tool-verträge-layer-2--norm))
3. Implementiere Error-Handling mit `ToolError` (Typed Errors)
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
7. Dokumentiere in [ai_core/graphs/README.md](ai_core/graphs/README.md)

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

## Qualitätsregeln (Claude-Checkliste)

### Code-Qualität

- Ruff + Black für Python (immer `npm run lint:fix` vor Commit!)
- TypeScript strict mode
- 100% Type-Hints in Tool-Contracts
- Pydantic für alle Datenmodelle mit `frozen=True`
- `model_json_schema()` ist kanonische Quelle für Schemas

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

- PII-Maskierung verpflichtend (siehe [docs/pii-scope.md](docs/pii-scope.md))
- Secrets via `.env` oder Secret Manager
- Keine API-Keys in Code/Logs
- Rate-Limiting für LLM-Calls
- Tenant-Isolation durchgängig

Details: [docs/security/secrets.md](docs/security/secrets.md)

## Nützliche Kommandos (Kurzreferenz)

```bash
# Entwicklung
npm run dev:stack           # Alles starten
npm run dev:check           # Health-Checks
npm run dev:manage <cmd>    # Django-Management

# Tests
npm run test:py            # Vollständig (inkl. slow)
npm run test:py:single -- path/to/test.py  # Einzelner Test
npm run test:py:fast       # Schnell (ohne slow)
npm run test:py:unit       # Unit-Tests (ohne DB)
npm run test:py:cov        # Coverage
npm run test:py:clean      # Clean + vollständig
npm run e2e                # E2E
# Windows: npm run win:test:py, win:test:py:single -- path/to/test.py, etc.

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

## Navigation (Primärquellen)

### Contracts & Architektur (AGENTS.md)

1. **[AGENTS.md](AGENTS.md)** - Hauptleitfaden (Contracts, Glossar, Schnittstellen)
2. [docs/architecture/overview.md](docs/architecture/overview.md) - Systemarchitektur
3. [docs/multi-tenancy.md](docs/multi-tenancy.md) - Mandantenfähigkeit
4. [docs/architecture/id-semantics.md](docs/architecture/id-semantics.md) & [docs/architecture/id-propagation.md](docs/architecture/id-propagation.md) - ID-Semantik & End-to-End-Propagation
5. [docs/architecture/id-guide-for-agents.md](docs/architecture/id-guide-for-agents.md) - Praktischer Implementierungs-Guide für Agenten

### AI Core

5. [docs/agents/overview.md](docs/agents/overview.md) - LangGraph-Agenten
6. [docs/agents/tool-contracts.md](docs/agents/tool-contracts.md) - Tool-Verträge
7. [docs/rag/overview.md](docs/rag/overview.md) - RAG-Architektur
8. [docs/rag/ingestion.md](docs/rag/ingestion.md) - Ingestion-Pipeline

### Entwicklung & Betrieb

9. [docs/development/onboarding.md](docs/development/onboarding.md) - Einstieg
10. [README.md](README.md) - Projekt-README
11. [docs/operations/scaling.md](docs/operations/scaling.md) - Skalierung
12. [docs/runbooks/migrations.md](docs/runbooks/migrations.md) - Migrationen

### Observability & CI/CD

13. [docs/observability/langfuse.md](docs/observability/langfuse.md) - Langfuse
14. [docs/observability/elk.md](docs/observability/elk.md) - ELK-Stack
15. [docs/cicd/pipeline.md](docs/cicd/pipeline.md) - CI/CD-Pipeline
16. [docs/security/secrets.md](docs/security/secrets.md) - Security

## Quick Reference (für schnelle Lookups)

- **Contracts & Glossar**: [AGENTS.md#Glossar & Feld-Matrix](AGENTS.md#glossar--feld-matrix)
- **Layer-Hierarchie**: [AGENTS.md#Paketgrenzen](AGENTS.md#paketgrenzen-import-regeln)
- **Tool-Verträge**: [AGENTS.md#Tool-Verträge](AGENTS.md#tool-verträge-layer-2--norm)
- **Ingestion**: Queue `ingestion`, Task `run_ingestion_graph`
- **Graphs**: Liegen in `ai_core/graphs/`
- **Fehler**: Typed Errors in `ai_core/tools/errors.py`

## Governance & Änderungen

- Architektur-Änderungen → zuerst in [AGENTS.md](AGENTS.md) + Primärquellen
- Pull Requests → verlinken auf aktualisierte Docs
- Breaking Changes → erfordern Migration-Runbook
- Idempotenz bewahren: Nur bei neuen Workflows/Commands aktualisieren

## Fragen oder Probleme?

1. **Contracts/Architektur**: Konsultiere [`AGENTS.md`](AGENTS.md)
2. **Workflows/Commands**: Diese Datei (CLAUDE.md)
3. **Details**: Primärquellen unter [docs/](docs/)
4. **Laufzeitfehler**: Langfuse-Traces (`trace_id` suchen)
5. **Performance/Chaos**: ELK-Logs
6. **Betrieb**: Runbooks unter [docs/runbooks/](docs/runbooks/)

---

**Version**: 2.0
**Zuletzt aktualisiert**: 2025-11-24
**Gilt für**: NOESIS 2, Branch `main`
**Master Reference**: [AGENTS.md](AGENTS.md)
