# Warum
Die Umgebungsmatrix verhindert Fehlkonfigurationen und schafft Klarheit über Infrastruktur, Sicherheit und Kosten pro Stage. Junior-Entwickler sehen sofort, welche Dienste wo laufen und welche Regeln gelten.

# Wie
Die drei Stufen folgen denselben Container-Artefakten, unterscheiden sich aber bei Netzwerk, Geheimnissen, RAG-Bausteinen und Governance. Wichtige Leitplanken:
- Automatische Migrationen laufen nur lokal; Cloud-Deployments folgen dem [Migrations-Runbook](../runbooks/migrations.md).
- Staging erhält Konfiguration ausschließlich über `--set-env-vars` aus der CI ([Pipeline](../cicd/pipeline.md)); kein Secret Manager.
- Prod liest Secrets zwingend aus dem [Secret-Management](../security/secrets.md) via Secret Manager Versionen.
 - Der `pgvector`-Speicher folgt den Tabellen aus [RAG-Schema](../rag/schema.sql); in Prod gelten Private-IP und PITR.
 - Ingestion- und Agenten-Queues bleiben getrennt, damit Last sauber skaliert werden kann (siehe [Scaling](../operations/scaling.md)).

## Vergleich
| Umgebung | Container | Netz | Datenbanken | RAG Store | Ingestion | Agenten | Secrets | Ingress | Observability (Langfuse) | Skalierung | Kostenhebel |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Dev lokal | Docker Compose (Web, Worker, LiteLLM, Postgres, Redis) | Docker-Bridge, lokale Ports | Postgres 16 Container, Redis 7 Container, LiteLLM nutzt `DATABASE_URL` | Gleiches Postgres-Container-Schema, `pgvector` lokal aktiviert. Start ohne Initial-Import, Datenbestand beginnt leer. | Compose-Worker mit Queue `ingestion`, Batchgröße klein halten. Start ohne Initial-Import, Datenbestand beginnt leer. | Agenten laufen im selben Worker, Logging auf Konsole. Start ohne Initial-Import, Datenbestand beginnt leer. | `.env` aus Repo-Wurzel | `http://localhost:{8000,4000}` | Langfuse optional lokal mit SQLite; Logs primär Konsole | Manuell via Compose (Start/Stop) | Container pausieren, Datenbank-Volume löschen |
| Staging GCP | Cloud Run Services: Web, Worker, optional Beat, LiteLLM; Cloud Run Job für Migration/Seed | Öffentliches Cloud Run + Serverless VPC Access zu Memorystore; Cloud SQL Connector (Public IP) | Cloud SQL Postgres mit Public IP, Memorystore Redis Private Service; LiteLLM nutzt Cloud SQL Schema | Cloud SQL Schema `rag` mit `pgvector` Extension, Index-Tests mit 1k Vektoren. Start ohne Initial-Import, Datenbestand beginnt leer. | Cloud Run Worker mit Queue `ingestion`, Limits laut [Scaling](../operations/scaling.md) (max 2 gleichzeitige Batches). Start ohne Initial-Import, Datenbestand beginnt leer. | Agenten-Queue `agents` mit LangGraph, Output nach Langfuse. Start ohne Initial-Import, Datenbestand beginnt leer. | CI setzt Umgebungsvariablen je Deploy (`--set-env-vars`) | Cloud Run URL (authentifiziert für Admin), temporär öffentlich für QA | Langfuse Cloud/Container über CI-Variablen (`LANGFUSE_*`), Sampling 25% | Auto-Scaling pro Dienst (min 0, max 3), Worker nach Queue-Metriken | Min-Instances = 0, regionale Ressourcen klein halten |
| Prod GCP | Cloud Run Services identisch zu Staging plus dedizierter Beat falls benötigt | Private VPC + Serverless Connector, Cloud SQL Private IP, interne Dienste | Cloud SQL Postgres mit PITR, Memorystore Redis, optional getrennte DB für LiteLLM | Cloud SQL Private IP, `pgvector` mit IVFFLAT/HNSW Indizes, PITR aktiv. Start ohne Initial-Import, Datenbestand beginnt leer. | Cloud Run Worker Queue `ingestion`, maxInstances abgestimmt auf Kostenlimit; Batches 512 Chunks. Start ohne Initial-Import, Datenbestand beginnt leer. | Agenten-Queue `agents` mit RLS-sicherem Zugriff; Guardrails aktiv. Start ohne Initial-Import, Datenbestand beginnt leer. | Secret Manager + Runtime Secrets ([Security](../security/secrets.md)) | Interne Cloud Run URLs; externer Zugriff nur via HTTPS Load Balancer & Managed SSL | Langfuse via Secret Manager (`LANGFUSE_*`), Sampling 5%, PII-Redaction aktiv | Min-Instances > 0 für Web, Worker horizontal via Traffic-Split, Beat fix 1 | Traffic-Split für Canary, Skalierungsgrenzen optimieren, Speicherklassen wählen |

Agenten werden plattformweit betrieben; Tenancy greift ausschließlich im Retriever.

# Schritte
1. Nutze die Matrix als Vorlage, wenn eine neue Stage entsteht oder Einstellungen abweichen sollen.
2. Prüfe pro Spalte, ob notwendige Ressourcen bereits im passenden Cloud-Projekt existieren.
3. Gleiche Secret-Quellen mit dem [Security-Guide](../security/secrets.md) ab, bevor Deployments freigegeben werden.
4. Plane Observability- und Skalierungsmaßnahmen gemeinsam mit [Operations](../operations/scaling.md), damit Kosten und Stabilität zusammenpassen.
