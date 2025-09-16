# Warum
Die Produktionsumgebung schützt Kundendaten und stellt Verfügbarkeit sicher. Dieses Dokument beschreibt alle Abweichungen zu Staging und führt durch die Härtungsmaßnahmen.

# Wie
## Architekturunterschiede
- **Cloud SQL** nutzt ausschließlich Private IP und liegt in einer dedizierten VPC. Alle Dienste greifen über Serverless VPC Access und Private Service Connect zu.
- **Ingress**: Cloud Run Web & LiteLLM sind intern; ein globaler HTTPS Load Balancer mit Managed SSL-Zertifikat und DNS-Eintrag veröffentlicht die Anwendung. Nur der Load Balancer ist extern erreichbar.
- **Secrets**: Laufzeitkonfiguration wird aus Secret Manager geladen. Deployments referenzieren Secret-Versionen gemäß [Security-Leitfaden](../security/secrets.md).
- **LiteLLM GUI**: Optional mit Cloud IAP abgesichert, zusätzlich zur Master-Key-Authentifizierung aus [`config/litellm-config.yaml`](../../config/litellm-config.yaml).
- **RAG Store**: `pgvector` läuft im Cloud SQL-Instance mit Private IP; Indexe (IVFFLAT/HNSW) und Wartungsjobs sichern Latenzen.

## Ressourcen und Parameter
| Dienst | Produktionsspezifikum |
| --- | --- |
| Web | minInstances 2, maxInstances 10, Concurrency 30, CPU 2, Memory 2 GiB, Traffic-Split für Releases (z.B. 10/90, 50/50, 100%) |
| Worker | Separater Cloud Run Service mit Concurrency 1, minInstances 1, maxInstances 20; Queue-Länge aus Monitoring steuert manuelle Anpassungen |
| Beat | Eine Instanz fix, keine Skalierung, Private Ingress |
| Migrate-Job | Cloud Run Job mit Timeout 900 s, maxRetries 1, startet nur nach Freigabe laut [Pipeline](../cicd/pipeline.md) |
| LiteLLM | Interner Cloud Run Service, Zugriff via IAP-Gruppe oder VPC-Only, Logging aktiv; Rate-Limits via `AI_CORE_RATE_LIMIT_QUOTA` |
| Cloud SQL | PITR aktiv (min. 7 Tage), Backups täglich, Wartungsfenster definiert; getrennte DB für App und LiteLLM empfohlen |
| RAG Store | Schema `rag` oder `tenant_rag_*` im selben Instance; `pgvector` Extension aktiv, Indizes IVFFLAT (k=100) oder HNSW (ef_search=64) |
| Memorystore | Hochverfügbarer Redis (Standard Tier), Auth-Token aktiviert |
| Artifact Registry | Gleiches Repo wie Staging, nur signierte Tags (`cosign` optional) |
| Secret Manager | Versioniertes Speichern aller produktiven ENV-Variablen, Rotation automatisiert |
| Langfuse | SaaS oder selbst gehostet; Secrets ausschließlich über Secret Manager, Sampling 5% (Default), PII-Redaction aktiv |

## Betrieb und Sicherheit
- **Traffic-Split & Rollback**: Releases laufen über Cloud Run Revisions. Nach Smoke-Tests (siehe [QA](../qa/checklists.md)) wird Traffic sukzessive erhöht. Rollback bedeutet Traffic zurück auf die letzte Revision und ggf. Restore aus PITR.
- **Backups**: Cloud SQL automatisierte Backups + PITR, Memorystore Snapshots wöchentlich exportieren.
- **IAM**: Workload Identity Federation nutzt ein Service-Konto mit Rollen `roles/run.admin`, `roles/artifactregistry.writer`, `roles/cloudsql.client`, `roles/secretmanager.secretAccessor`, `roles/redis.admin`, `roles/compute.networkUser`.
- **Log-Retention**: Cloud Logging auf 90 Tage einstellen, Audit-Logs nicht kürzen. Export wichtiger Logtypen nach BigQuery zur Langzeitaufbewahrung.
- **Langfuse**: Secret-Verteilung ausschließlich via Secret Manager (`LANGFUSE_*`), Sampling-Regeln pro Mandant dokumentieren, PII-Redaction (`masking_rules`) aktiv.

## pgvector Betrieb
- Verwende IVFFLAT-Indizes (`lists` nach Datensatzanzahl skalieren) für Standard-Suche; für hochwertige Re-ranking-Pfade optional HNSW mit `ef_search=64`.
- Plane Wartungsjobs: `REINDEX CONCURRENTLY` und `ANALYZE` monatlich, `VACUUM` nach größeren Löschläufen laut [RAG-Übersicht](../rag/overview.md).
- Alle schemaändernden Migrationen laufen getrennt in der Pipeline-Stufe „Vector-Schema-Migrations“. Rollbacks nutzen PITR oder Index-Drop + Rebuild (siehe [Migrations-Runbook](../runbooks/migrations.md)).

## Tenancy-Modell
- Option 1: Schema pro Mandant (`tenant_<slug>`), getrennte `rag`-Schemas spiegeln diese Struktur. Vorteil: natürliche Isolation, Nachteil: mehr Verwaltungsaufwand.
- Option 2: Gemeinsames Schema mit Spalte `tenant_id` und aktivem Row-Level-Security (`USING tenant_id = current_setting('app.tenant_id')::uuid`).
- Entscheide pro Produktphase; dokumentiere Wahl in [Security](../security/secrets.md) unter Allowed Hosts und `DATABASE_URL`.

# Schritte
1. Erstelle oder erweitere die Produktions-VPC mit Subnetzen für Private Service Connect und Serverless VPC Access.
2. Provisioniere Cloud SQL mit Private IP, aktiviere PITR und definiere Backups sowie Wartungsfenster.
3. Richte Memorystore (Standard Tier) ein und sichere den Zugriff über die VPC.
4. Hinterlege alle produktiven Secrets im Secret Manager und setze Zugriffspfade gemäß [Security](../security/secrets.md).
5. Konfiguriere den globalen HTTPS Load Balancer (Managed SSL, DNS, optional Cloud Armor) und verknüpfe Cloud Run über interne Ingress-Einstellungen.
6. Binde Workload Identity Federation mit den genannten IAM-Rollen an die CI/CD-Pipeline ([Pipeline](../cicd/pipeline.md)).
7. Plane Release-Runbooks: Migrationen gemäß [Runbook](../runbooks/migrations.md), Incident-Reaktionen via [Incidents](../runbooks/incidents.md), Skalierung über [Operations](../operations/scaling.md).
