# Warum
Die Staging-Umgebung bildet die Produktionsarchitektur nach, erlaubt aber schnelle Iteration. Dieses Dokument zeigt, welche GCP-Ressourcen nötig sind und wie sie zusammenspielen.

# Wie
## Ressourcenübersicht
| Dienst | Typ | Wichtige Parameter |
| --- | --- | --- |
| Web | Cloud Run Service | Region `europe-west4`, CPU 1, RAM 1 GiB, Concurrency 30, minInstances 0, maxInstances 3, Cloud SQL Connector (Public IP) |
| Worker | Cloud Run Service | Gleiches Image wie Web, Concurrency 1, CPU 1, maxInstances 3, verbindet sich zu Redis über Serverless VPC Access |
| Beat (optional) | Cloud Run Service | Startbefehl `celery -A noesis2 beat -l info`, 1 Instanz fix, Zugriff auf Redis |
| Migrate-Job | Cloud Run Job | Timeout 600 s, maxRetries 0, Cloud SQL Connector, führt `migrate_schemas --noinput` aus |
| Seed-Job | Cloud Run Job | Re-usable Job aus CI für Demo-Daten, gleiche Env wie Migrate-Job |
| LiteLLM | Cloud Run Service | Image `ghcr.io/berriai/litellm:main-stable`, Concurrency 10, minInstances 0, benötigt Cloud SQL Connector und Redis |
| Cloud SQL | PostgreSQL 16 | Public IP aktiviert, Cloud SQL Auth Proxy/Connector, Datenbank `noesis2_staging`, zweites Schema für LiteLLM |
| RAG Store | Cloud SQL Schema | Schema `rag` in derselben Instanz, `pgvector` Extension aktiviert, Tabellen laut [Schema](../rag/schema.sql) |
| Memorystore | Redis 7 | Private Service, Verbindung über Serverless VPC Connector, Standardgröße 1 GiB |
| Artifact Registry | Docker Repository | Region wie Cloud Run, Reponame `noesis2`, nutzt Semver+SHA Tags |
| Langfuse | Cloud Run Service oder SaaS | Optional als separater Dienst, empfängt Traces aus Worker und LiteLLM, API Keys aus CI |

## Netzwerk und Konnektivität
- Cloud Run Dienste greifen auf Cloud SQL über den Cloud SQL Connector (Public IP) zu; kein Öffnen von Firewalls nötig.
- Memorystore liegt in einer dedizierten VPC. Alle Cloud Run Dienste erhalten einen Serverless VPC Access Connector (`/28` Subnetz) für Redis-Zugriffe.
- Ausgehender Traffic nutzt standardmäßig egress über das Internet; sensibler Traffic geht über den Connector.

## Konfiguration
- GitHub Actions setzt alle Variablen per `--set-env-vars`, siehe [Pipeline](../cicd/pipeline.md). Secret Manager wird in Staging nicht verwendet.
- Wichtige Variablen: `EMBEDDINGS_MODEL`, `EMBEDDINGS_DIM`, `EMBEDDINGS_API_BASE`, `LANGFUSE_KEY`, `LANGFUSE_HOST`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LITELLM_URL`.
- Basissecrets folgen dem Schema aus [Security](../security/secrets.md); Rotation erfolgt durch Aktualisierung der CI-Secrets und anschließendes Redeploy.
- LiteLLM verlangt `require_auth: true` laut [`config/litellm-config.yaml`](../../config/litellm-config.yaml); nur Benutzer mit gültigem Master Key oder API Key erhalten Zugang.

## pgvector aktivieren
- Aktiviere die Extension nach Provisionierung: `CREATE EXTENSION IF NOT EXISTS vector;` im Schema `rag`.
- Optionale Flags für Performance: `work_mem=128MB` und `effective_io_concurrency=200`. Änderungen erfolgen über das Cloud SQL Flag-Set.
- Halte Tabellen `documents`, `chunks`, `embeddings` konsistent zum [RAG-Schema](../rag/schema.sql); Migrationen laufen über die Stufe „Vector-Schema-Migrations“ in der [Pipeline](../cicd/pipeline.md).
- Mindestversion: pgvector >= `0.5.0` (HNSW-Unterstützung). Führe nach Provisionierung `ALTER EXTENSION vector UPDATE;` aus, um die in Cloud SQL verfügbare Version zu aktivieren und `vector_cosine_ops` für HNSW sicherzustellen.

## Service-Konten
- CI-Service-Konto benötigt `roles/cloudsql.client`, `roles/run.developer` und `roles/iam.serviceAccountUser` für Deployments.
- Worker-Identität (Workload Identity) benötigt `roles/cloudsql.client`, um Embeddings schreiben und lesen zu können.
- Zugriff auf Langfuse (Self-Hosted) erfolgt über denselben Service-Account mit `roles/run.invoker`, falls Langfuse in GCP läuft.

## Ingestion-Limits
- Plane pro Worker maximal zwei gleichzeitige Ingestion-Batches à 128 Chunks. Größere Batches erhöhen die Speicherauslastung.
- Setze Cloud Run Timeout auf 900 Sekunden für Ingestion-Worker, um lange Embedding-Läufe abzudecken.
- Verwende `BATCH_SIZE` und `MAX_TOKENS` laut [RAG-Ingestion](../rag/ingestion.md) und dokumentiere Anpassungen im Release-Plan.

## Zugriffsmodell LiteLLM
- Web- und Worker-Container nutzen `LITELLM_URL` intern.
- Administrativer Zugriff auf die LiteLLM GUI erfolgt über das Cloud Run Auth-Gateway; Service-Konten erhalten den `roles/run.invoker`.
- Audit-Logs landen in Cloud Logging und werden mindestens 30 Tage aufbewahrt.

# Schritte
1. Lege das Artifact-Registry-Repository im Zielprojekt an, damit CI Push-Rechte konfigurieren kann.
2. Provisioniere Cloud SQL (Public IP) und Memorystore; dokumentiere Verbindungsnamen für die [Pipeline](../cicd/pipeline.md) und aktiviere `pgvector` im Schema `rag`.
3. Erstelle den Serverless VPC Access Connector und verbinde Web, Worker, Beat, LiteLLM sowie optionale Langfuse-Instanz damit.
4. Deploye die Cloud Run Dienste mit dem aktuellen Image und setze Umgebungsvariablen ausschließlich via CI (`--set-env-vars`), inklusive `EMBEDDINGS_*` und `LANGFUSE_*`.
5. Richte die Cloud Run Jobs (Migration, Seed, Ingestion-Testbatch) ein und teste sie gemäß [Migrations-Runbook](../runbooks/migrations.md) und [RAG-Ingestion](../rag/ingestion.md).
6. Aktiviere Logging-Dashboards, Langfuse-Workspaces und Smoke-Checks laut [QA-Checklisten](../qa/checklists.md).
