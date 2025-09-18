# AGENTS Leitfaden

Zentrale Navigations- und Vertragsdatei für NOESIS 2. Dieses Dokument fasst die verbindlichen Leitplanken zusammen und verweist
auf die maßgeblichen Quellen unter `docs/` sowie ergänzende Hinweise aus der `README.md`.

## Zweck & Geltungsbereich
- Gilt für alle Beiträge in diesem Repository. Entscheidungen und Detailregeln werden ausschließlich in den Primärdokumenten unter `docs/` gepflegt und hier verlinkt.
- Vor Änderungen prüfe den Verzeichnispfad auf spezifischere `AGENTS.md`-Dateien (z. B. `theme/AGENTS.md`, `theme/components/AGENTS.md`) und befolge stets die tiefste Anweisung.
- Dieses Dokument dient als Einstieg und Navigationshilfe; es enthält keine sich wiederholenden Inhalte aus den Primärquellen.

## Systemkontext
- Die Systemlandschaft (Web, Worker, Ingestion, LiteLLM, Datenpfade) und die zugehörigen Diagramme sind in der [Architekturübersicht](docs/architektur/overview.md) beschrieben.
- Laufzeitpfade pro Umgebung sowie die Sequenz von Deploy- zu RAG-Flows befinden sich in den Mermaid-Diagrammen derselben Quelle.

## Rollen & Funktionsblöcke
- **Web- & Worker-Services** verantworten HTTP-Verarbeitung und Celery-Queues laut [Architekturübersicht](docs/architektur/overview.md).
- **Agenten (LangGraph)** orchestrieren Retrieval und Guardrails nach [Agenten-Übersicht](docs/agents/overview.md).
- **RAG & Ingestion** decken Loader→Embedding→pgvector gemäß [RAG-Overview](docs/rag/overview.md) und [Ingestion-Leitfaden](docs/rag/ingestion.md) ab.
- **LiteLLM Betrieb** folgt den Betriebs- und Auth-Regeln aus [LiteLLM Admin GUI](docs/litellm/admin-gui.md).
- **Observability & Kostenkontrolle** wird über [Langfuse Guide](docs/observability/langfuse.md) geführt.
- **Multi-Tenancy & Tenant-CLI** inklusive Rollen findest du im [Multi-Tenancy Leitfaden](docs/multi-tenancy.md).
- **CI/CD & Releases** werden über die [Pipeline-Dokumentation](docs/cicd/pipeline.md) gesteuert.
- **Security & Secrets** verwalten Plattform- und AI-Schlüssel gemäß [Security Guide](docs/security/secrets.md).

## Ereignisse & Trigger
| Quelle | Trigger | Beschreibung | Primärquelle |
| --- | --- | --- | --- |
| GitHub Actions | Pull Request oder Merge nach `main` | Startet alle Pipeline-Stufen von Lint bis Deploy. | [docs/cicd/pipeline.md#pipeline-stufen](docs/cicd/pipeline.md#pipeline-stufen) |
| Cloud Run Jobs | Freigabe nach Staging-Checks | Führt `noesis2-migrate` sowie Vector-Schema-Migrationen aus. | [docs/cicd/pipeline.md#pipeline-stufen](docs/cicd/pipeline.md#pipeline-stufen) |
| Django Web-Service | HTTP-Request mit `X-Tenant-ID` & `X-Case-ID` | Legt Celery Task `agents.run` an und startet den LangGraph-Flow. | [docs/agents/overview.md#kontrollfluss](docs/agents/overview.md#kontrollfluss) |
| Ingestion Worker | Pipeline-Stufe „Ingestion-Smoke“ oder manueller Batch | Lädt Daten, chunked sie und schreibt Embeddings in pgvector. | [docs/rag/ingestion.md#pipeline](docs/rag/ingestion.md#pipeline) |
| Langfuse Observability | Fehlerquote > 5 % oder Kosten > 80 % Budget | Löst Alerts für Agenten, Ingestion und LiteLLM aus. | [docs/observability/langfuse.md#felder-und-sampling](docs/observability/langfuse.md#felder-und-sampling) |

## Inputs & Outputs
| Datenobjekt | Schema/Ort | Besitzer | Primärquelle |
| --- | --- | --- | --- |
| Tenant-Schemata | PostgreSQL Public- & Tenant-Schemas (`django-tenants`) | Platform Engineering | [docs/multi-tenancy.md#architektur](docs/multi-tenancy.md#architektur) |
| Organisations- & Projektmigrationen | Django Migrationen (`migrate_schemas`, `assign_default_org`) | Backend & Operatoren | [docs/multi-tenancy.md#lokales-setup-nach-pull](docs/multi-tenancy.md#lokales-setup-nach-pull) |
| RAG-Dokumente & Embeddings | `pgvector`-Tabellen inkl. Hash- und Metadata-Feldern | AI Platform & Data Ops | [docs/rag/schema.sql](docs/rag/schema.sql) |
| Langfuse Traces & Kostenmetriken | Langfuse Store (Trace-, Span-, Metric-Records) | Observability Team | [docs/observability/langfuse.md#datenfluss](docs/observability/langfuse.md#datenfluss) |
| Secrets & Konfigurationswerte | `.env`, GitHub Secrets, Secret Manager Versionen | Security & Platform | [docs/security/secrets.md#env-verträge](docs/security/secrets.md#env-verträge) |

## Schnittstellen & Contracts
| Interface | Endpunkt/Topic | Kurzbeschreibung | Primärquelle |
| --- | --- | --- | --- |
| AI Core REST | `/ai/ping/`, `/ai/intake/`, `/ai/scope/`, `/ai/needs/`, `/ai/sysdesc/` | HTTP-Endpunkte mit Tenant-Headern, orchestriert durch LangGraph. | [docs/agents/overview.md#kontrollfluss](docs/agents/overview.md#kontrollfluss) |
| Agenten Queue | Celery Queue `agents` | Startet LangGraph-Nodes, setzt Guardrails & Cancellation. | [docs/agents/overview.md#knoten-und-guardrails](docs/agents/overview.md#knoten-und-guardrails) |
| Ingestion Queue | Celery Queue `ingestion` | Führt Loader→Chunk→Embedding→Upsert Schritte aus. | [docs/rag/ingestion.md#pipeline](docs/rag/ingestion.md#pipeline) |
| Vector Schema Migration | Cloud SQL Verbindung via Pipeline-Stufe | Führt `docs/rag/schema.sql` gegen das RAG-Schema aus. | [docs/cicd/pipeline.md#pipeline-stufen](docs/cicd/pipeline.md#pipeline-stufen) |
| LiteLLM Admin GUI | Cloud Run Service `litellm` (`/health`, GUI) | Verwaltung von Modellen, Rate-Limits und Master Keys. | [docs/litellm/admin-gui.md](docs/litellm/admin-gui.md) |
| Langfuse API | `LANGFUSE_HOST` (`/api/public`, `/api/ingest`) | Erfasst Traces, Metrics, Alerts und Sampling-Konfiguration. | [docs/observability/langfuse.md#datenfluss](docs/observability/langfuse.md#datenfluss) |

## Laufzeit & Betrieb
- Skalierungs- und Ressourcenregeln pro Dienst stehen in den [Operations Guidelines](docs/operations/scaling.md).
- Deploy- und Traffic-Shift-Abläufe folgen der [CI/CD Pipeline](docs/cicd/pipeline.md) inklusive Approval-Stufen und Smoke-Checks.
- Runbooks zu Migrationen und Incidents liegen unter [docs/runbooks/](docs/runbooks) und ergänzen diese Übersicht.
- Migrations-Runbook (Django/Tenants): [docs/runbooks/migrations.md](docs/runbooks/migrations.md)

## Sicherheit
- ENV-Verträge, Secret-Rotation und Log-Scopes werden im [Security Guide](docs/security/secrets.md) definiert.
- LiteLLM Master-Key Verwaltung, Auth und Rate-Limits folgen [LiteLLM Admin GUI](docs/litellm/admin-gui.md).
- PII-Redaction und Zugriffskontrolle für Traces werden in [Langfuse Guide](docs/observability/langfuse.md) erläutert.

## Qualität & KPIs
- Überwachung der Fehlerraten, Kosten und Queue-Längen erfolgt über Langfuse-Dashboards laut [Observability Guide](docs/observability/langfuse.md).
- Skalierungs- und Kostenlimits sind in [Operations](docs/operations/scaling.md) dokumentiert.
- QA-Abbruchkriterien und Smoke-Checklisten liegen in [docs/qa/checklists.md](docs/qa/checklists.md).

## Teststrategie
- Pipeline-Stufen für Lint, Unit, Build, E2E und Migrationsprüfungen sind in [CI/CD Pipeline](docs/cicd/pipeline.md#pipeline-stufen) beschrieben.
- Lokale Kommandos (`pytest`, `npm run lint`, `npm run build:css`) werden in der [README.md](README.md#testing) und [README.md → Linting & Formatierung](README.md#linting--formatierung) dokumentiert.

## Governance & Änderungen
- Architektur-, Security-, RAG- oder Betriebsanpassungen werden zuerst in den jeweiligen Primärquellen unter `docs/` dokumentiert.
- Runbooks besitzen ihre eigenen Changelogs (siehe [docs/runbooks/](docs/runbooks)); verweise in Pull Requests auf aktualisierte Quellen.
- Bewahre Idempotenz: aktualisiere diese Datei nur bei neuen Links, Begriffsklärungen oder widersprüchlichen Aussagen.

## Navigationsverzeichnis
1. [Architekturübersicht](docs/architektur/overview.md)
2. [Multi-Tenancy Leitfaden](docs/multi-tenancy.md) & [Tenant-Management](docs/tenant-management.md)
3. [RAG Overview](docs/rag/overview.md), [Ingestion](docs/rag/ingestion.md) & [Schema](docs/rag/schema.sql)
4. [Agenten-Übersicht](docs/agents/overview.md)
5. [LiteLLM Admin GUI](docs/litellm/admin-gui.md)
6. [Observability Langfuse](docs/observability/langfuse.md)
7. [Operations & Scaling](docs/operations/scaling.md)
8. [Security & Secrets](docs/security/secrets.md) 
9. [CI/CD Pipeline](docs/cicd/pipeline.md)
10. [Runbooks](docs/runbooks) & [QA Checklisten](docs/qa/checklists.md)
11. [README Einstieg & Kommandos](README.md)

