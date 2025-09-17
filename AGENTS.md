# AGENTS Leitfaden

Zentrale Navigations- und Vertragsdatei für NOESIS 2. Dieses Dokument fasst die verbindlichen Leitplanken zusammen und verweist auf die maßgeblichen Quellen unter `docs/` sowie ergänzende Hinweise aus der `README.md`.

## Zweck & Geltung
- Alle Beiträge orientieren sich an den Architektur-, Betriebs- und Sicherheitszielen aus den NOESIS 2-Dokumenten.
- Änderungen an Prozessen oder Richtlinien werden zuerst in den Primärdokumenten gepflegt und anschließend hier referenziert.

## Primärdokumente & Rollen
- **Architektur, Infrastruktur & Cloud-Pfade** – [docs/architektur/overview.md](docs/architektur/overview.md), [docs/cloud/gcp-staging.md](docs/cloud/gcp-staging.md), [docs/cloud/gcp-prod.md](docs/cloud/gcp-prod.md), [docs/environments/matrix.md](docs/environments/matrix.md) · Verantwortlich: Platform Engineering & Cloud Ops.
- **Container & Laufzeitkonventionen** – [docs/docker/conventions.md](docs/docker/conventions.md) · Verantwortlich: Platform Engineering.
- **Multi-Tenancy & Tenant-Betrieb** – [docs/multi-tenancy.md](docs/multi-tenancy.md), [docs/tenant-management.md](docs/tenant-management.md) · Verantwortlich: Backend & Tenant Operations.
- **RAG, Ingestion & Vector Store** – [docs/rag/overview.md](docs/rag/overview.md), [docs/rag/ingestion.md](docs/rag/ingestion.md), [docs/rag/schema.sql](docs/rag/schema.sql) · Verantwortlich: AI Platform & Data Ops.
- **Agenten & Guardrails** – [docs/agents/overview.md](docs/agents/overview.md) · Verantwortlich: AI Platform.
- **LiteLLM Betrieb** – [docs/litellm/admin-gui.md](docs/litellm/admin-gui.md) · Verantwortlich: AI Platform Owner.
- **Observability & Langfuse** – [docs/observability/langfuse.md](docs/observability/langfuse.md) · Verantwortlich: Observability Team.
- **Skalierung & Betrieb** – [docs/operations/scaling.md](docs/operations/scaling.md) · Verantwortlich: Platform Engineering.
- **Runbooks & QA** – [docs/runbooks/migrations.md](docs/runbooks/migrations.md), [docs/runbooks/incidents.md](docs/runbooks/incidents.md), [docs/qa/checklists.md](docs/qa/checklists.md) · Verantwortlich: On-Call & QA.
- **Sicherheit & Secrets** – [docs/security/secrets.md](docs/security/secrets.md) · Verantwortlich: Security & Platform.
- **CI/CD & Releases** – [docs/cicd/pipeline.md](docs/cicd/pipeline.md) · Verantwortlich: DevEx & Release Management.
- **Frontend Guidelines** – [docs/frontend-ueberblick.md](docs/frontend-ueberblick.md), [theme/AGENTS.md](theme/AGENTS.md), [theme/components/AGENTS.md](theme/components/AGENTS.md), [docs/frontend-master-prompt.md](docs/frontend-master-prompt.md) · Verantwortlich: Frontend.

## Einstieg & Entwicklungsumgebungen
- Quickstart, lokale Alternativen und Docker-Setup stehen in der [README.md](README.md). Nutze sie als Einstiegspunkt; Detailleitfäden sind in diesem Dokument verlinkt.
- `.env` Werte, Installationsschritte und Mandanten-Demos folgen den Vorgaben in [docs/multi-tenancy.md](docs/multi-tenancy.md) und [docs/tenant-management.md](docs/tenant-management.md).
- Für Umfeld-spezifische Konfigurationen (Dev/Staging/Prod) gilt die [Environment-Matrix](docs/environments/matrix.md).

## Architektur & Infrastruktur
- Halte Dich an die System- und Deploy-Pfade aus der [Architekturübersicht](docs/architektur/overview.md). Dort sind Komponenten, Queues und Netzpfade beschrieben.
- Container-Builds und Startkommandos richten sich nach den [Docker-Konventionen](docs/docker/conventions.md); Änderungen am `Dockerfile` müssen die Multi-Stage- und Non-Root-Regeln respektieren.
- Infrastruktur- und Bereitstellungsdetails unterscheiden sich je Stage. Beachte insbesondere die Leitfäden für [GCP Staging](docs/cloud/gcp-staging.md) und [GCP Prod](docs/cloud/gcp-prod.md).

## Daten, Tenancy & RAG
- Mandantenführung und Header-Governance folgen [docs/multi-tenancy.md](docs/multi-tenancy.md); CLI-Kommandos und Admin-Workflows stehen zusätzlich in [docs/tenant-management.md](docs/tenant-management.md).
- RAG-Architektur, Ingestion-Parameter und Schema-Pflege sind in [docs/rag/overview.md](docs/rag/overview.md), [docs/rag/ingestion.md](docs/rag/ingestion.md) und [docs/rag/schema.sql](docs/rag/schema.sql) verbindlich dokumentiert.
- Lösch- und Backfill-Strategien orientieren sich am RAG-Overview sowie den Migrationsempfehlungen im [Migrations-Runbook](docs/runbooks/migrations.md).

## Agenten, AI Core & LiteLLM
- Kontrollfluss, Node-Verantwortlichkeiten und Guardrails für LangGraph stehen in der [Agenten-Übersicht](docs/agents/overview.md). Setze `prompt_version`, PII-Maskierung und Cancellation wie beschrieben um.
- API-Verträge, Graph-State und lokale Nutzung werden in der [README.md](README.md#ai-core) beschrieben; konsolidiere Änderungen dort.
- LiteLLM Betrieb, Authentifizierung und Rate-Limits folgen [docs/litellm/admin-gui.md](docs/litellm/admin-gui.md). Halte Secrets synchron mit den Quellen aus [docs/security/secrets.md](docs/security/secrets.md).

## Betrieb, Observability & Skalierung
- Skalierungsgrenzen, Queue-Aufteilung und Kosten-Guards sind in [docs/operations/scaling.md](docs/operations/scaling.md) festgelegt.
- Langfuse-Integration, Sampling und Trace-Felder sind in [docs/observability/langfuse.md](docs/observability/langfuse.md) definiert. Verbinde Agenten-, Ingestion- und LiteLLM-Traces entsprechend.
- Für Releases und Störungen gelten die Schritte aus den Runbooks zu [Migrationen](docs/runbooks/migrations.md) und [Incidents](docs/runbooks/incidents.md) sowie den QA-Checklisten [docs/qa/checklists.md](docs/qa/checklists.md).

## Sicherheit & Secrets
- ENV-Verträge, Rotationen und Log-Scopes werden ausschließlich über [docs/security/secrets.md](docs/security/secrets.md) gepflegt. Keine Secrets im Code oder in Container-Images.
- LiteLLM-Key- und Auth-Regeln folgen den Abschnitten „Berechtigungen“ und „Rate-Limits“ im [LiteLLM Guide](docs/litellm/admin-gui.md).
- PII-Redaction vor jedem LLM-Aufruf ist Pflicht (siehe [Agenten-Übersicht](docs/agents/overview.md) und Security-Leitfaden).

## Entwicklungs- & Reviewleitlinien
- Python: `ruff` + `black` verpflichtend; halte Requirements via `pip-compile` aktuell (`requirements*.in` → `.txt`).
- Frontend: Tailwind CSS v4/PostCSS laut [Frontend-Überblick](docs/frontend-ueberblick.md) und Detailregeln in [theme/AGENTS.md](theme/AGENTS.md) bzw. [theme/components/AGENTS.md](theme/components/AGENTS.md).
- Tests orientieren sich an der in der README beschriebenen Testpyramide; nutze `pytest` mit Tenant-Support und `factory-boy` für Daten. E2E-Pfade folgen der Pipeline ([docs/cicd/pipeline.md](docs/cicd/pipeline.md)).
- Vor jeder Änderung prüfe, ob betroffene Runbooks oder Leitfäden aktualisiert werden müssen, und verlinke die Primärquelle im PR.

## Arbeitsroutine
1. Relevante AGENTS- und Bereichsdokumente prüfen – Einstieg über [AGENTS.md → Primärdokumente & Rollen](#primärdokumente--rollen) sowie Detailleitfäden wie [Frontend-Überblick](docs/frontend-ueberblick.md) oder [docs/multi-tenancy.md](docs/multi-tenancy.md) je nach Scope.
2. Linting-, Test- und Build-Kommandos ausführen: `npm run lint` ([README.md → Linting & Formatierung](README.md#linting--formatierung)), `pytest -q` ([README.md → Testing](README.md#testing)), `npm test` ([docs/frontend-ueberblick.md → Tests und Storybook](docs/frontend-ueberblick.md#tests-und-storybook)), `npm run build:css` ([README.md → Frontend-Build](README.md#frontend-build-tailwind-v4-via-postcss)).
3. Betroffene Runbooks/Dokumentationen angleichen (z. B. [docs/runbooks/migrations.md](docs/runbooks/migrations.md), [docs/runbooks/incidents.md](docs/runbooks/incidents.md), [docs/qa/checklists.md](docs/qa/checklists.md)) und die aktualisierte Quelle im PR direkt verlinken.

## CI/CD & Releases
- Die GitHub Actions Pipeline, Gates und Workload Identity Rollen sind in [docs/cicd/pipeline.md](docs/cicd/pipeline.md) verbindlich geregelt.
- Releases durchlaufen die dort beschriebenen Gates (Lint → Tests → Build → Deploy). Ohne bestandene QA-/Smoke-Schritte kein Prod-Traffic.
- Migrationen und Vector-DLLs laufen ausschließlich über die dedizierten Pipeline-Stufen und Runbooks.

## On-Call & Incident-Verhalten
- Folge bei Störungen den Szenarien im [Incident-Runbook](docs/runbooks/incidents.md). Dokumentiere Maßnahmen und aktualisiere Checklisten nach Post-Mortems.
- QA-Abbruchkriterien und Rollback-Regeln sind in [docs/qa/checklists.md](docs/qa/checklists.md) definiert; Traffic-Split-Anpassungen laufen gemäß [Pipeline](docs/cicd/pipeline.md).

## Dokumentationspflicht
- Änderungen an Architektur, Security, RAG, Agenten oder Betriebsprozessen werden zuerst in den jeweiligen `docs/`-Quellen aktualisiert.
- Dieser Leitfaden bleibt idempotent: passe ihn nur an, um neue oder geänderte Primärquellen zu verlinken oder widersprüchliche Aussagen zu korrigieren.
