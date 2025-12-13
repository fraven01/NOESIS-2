# Gemini Leitfaden

Operativer Leitfaden für den Gemini-Code-Agenten in NOESIS 2. **Alle Contracts, Glossar und Architektur gelten identisch wie für Codex und Claude** und werden zentral in [`AGENTS.md`](AGENTS.md) gepflegt.

## Zweck & Geltungsbereich

- Gleicher Wissensstand wie Claude Code und Codex: folgt denselben Contracts, Workflows und Paketschranken.
- Primärquelle für Architektur & IDs: [`AGENTS.md`](AGENTS.md) und die Referenzen unter `docs/`.
- Vor Änderungen auf spezifischere Leitfäden im Verzeichnispfad prüfen (z. B. `theme/AGENTS.md`).

## ID-Architektur & Lifecycle (Kurzreferenz)

- Pflichtfelder für Tool-Aufrufe: `tenant_id`, `trace_id`, `invocation_id` **und genau eine** Laufzeit-ID (`run_id` XOR `ingestion_run_id`).
- HTTP-Header: `X-Tenant-ID` (immer), `X-Trace-ID` (immer), `X-Case-ID` (fachlich, wenn Case-Kontext besteht).
- ID-Semantik & Propagation: siehe [`docs/architecture/id-semantics.md`](docs/architecture/id-semantics.md) und [`docs/architecture/id-propagation.md`](docs/architecture/id-propagation.md).
- **Implementierungs-Guide**: [`docs/architecture/id-guide-for-agents.md`](docs/architecture/id-guide-for-agents.md) (Pflichtlektüre für Coding).
- Lifecycle-Events & Checkliste in [`docs/architecture/id-sync-checklist.md`](docs/architecture/id-sync-checklist.md) und ADRs unter `docs/architecture/adrs/`.

## Architektur & Systemverständnis (Kurzfassung)

- Mandantenfähige Django-Plattform mit LangGraph-Agenten und Celery-Workern (`agents`, `ingestion`).
- RAG-Store: PostgreSQL + `pgvector`; LLM-Zugriff über LiteLLM (Gemini, Vertex AI, weitere Modelle).
- Observability: Langfuse Traces + ELK Logs; korrelieren immer über `trace_id`.
- Vollständige Architektur: [docs/architektur/overview.md](docs/architektur/overview.md).

## Arbeitsweise & Commands (Pointer)

- Entwicklung & Tests: Siehe Befehle in [`CLAUDE.md`](CLAUDE.md) (Docker-Setup, Linting, Tests, RAG-Management); dieselben Kommandos gelten für Gemini.
- API & SDK: `npm run api:schema`, `make sdk`.
- RAG-Werkzeuge: `python manage.py rag_routing_rules`, `python manage.py sync_rag_schemas`, `python manage.py check_rag_schemas`.

## Architektur-Heuristiken & Coding-Vibe

Um Reibungsverluste zu vermeiden, befolge strikt diese Design-Patterns:

1. **Layer-Integrität (View vs. Service):**
   - **Views sind Endpunkte**, keine wiederverwendbaren Bibliotheken.
   - Rufe **niemals** eine View-Funktion aus einer anderen View auf.
   - Wenn Logik geteilt werden muss, extrahiere sie in einen **Manager** oder **Service** (z.B. `CrawlerManager`).
   - *Flow:* `UI -> View (L2) -> Manager/Service (L3) -> Domain`.

2. **Explizite Verträge (HTML vs. JSON):**
   - Vermeide hybride Views, die mal HTML und mal JSON zurückgeben, basierend auf implizitem Kontext.
   - Nutze Pydantic-Modelle für den Datenaustausch zwischen Layern, um Typ-Sicherheit zu garantieren.

3. **Verifikation ("Smoke Tests"):**
   - Wenn Unit-Tests zu stark gemockt sind (z.B. Celery-Calls), erstelle **sofort** ein manuelles Skript in `scripts/debug_{issue}.py`.
   - Nutze dieses Skript als "Source of Truth" für die Fehlerbehebung, statt dich auf Unit-Test-Mocks zu verlassen.

## Navigation (Primärquellen)

1. **Master-Referenz**: [`AGENTS.md`](AGENTS.md) – Contracts, Glossar, Schnittstellen, Paketgrenzen.
2. **Agenten & Tools**: [docs/agents/overview.md](docs/agents/overview.md), [docs/agents/tool-contracts.md](docs/agents/tool-contracts.md).
3. **IDs & Lifecycle**: [docs/architecture/id-guide-for-agents.md](docs/architecture/id-guide-for-agents.md) (Guide), [docs/architecture/id-semantics.md](docs/architecture/id-semantics.md) (Theorie), [docs/architecture/id-propagation.md](docs/architecture/id-propagation.md).
4. **Multi-Tenancy**: [docs/multi-tenancy.md](docs/multi-tenancy.md).
5. **Entwicklung & Betrieb**: [docs/development/onboarding.md](docs/development/onboarding.md), [docs/cicd/pipeline.md](docs/cicd/pipeline.md), [docs/operations/scaling.md](docs/operations/scaling.md).

## Governance

- Änderungen an Contracts/IDs zuerst in [`AGENTS.md`](AGENTS.md) und den Architektur-Docs dokumentieren; Gemini.md bleibt referenzierend.
- Pull Requests sollen aktualisierte Quellen verlinken und Idempotenz sicherstellen.
- Bei Fragen: zuerst [`AGENTS.md`](AGENTS.md), dann Primärquellen unter `docs/`.
