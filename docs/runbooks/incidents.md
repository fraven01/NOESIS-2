# Warum
Incidents passieren. Dieses Runbook liefert schnelle Orientierung für die häufigsten Störungen und verhindert Ad-hoc-Entscheidungen.

# Wie
## Szenarien
| Szenario | Messpunkte | Typische Ursachen | Erstmaßnahmen | Eskalation |
| --- | --- | --- | --- | --- |
| Cloud SQL nicht erreichbar | Cloud Monitoring Uptime, `django.db.utils` Fehler in Logs, Pipeline-Alerts | Wartung, Verbindungslimit erreicht, VPC-Problem | Traffic einfrieren (kein weiteres Rollout), Statusseite aktualisieren, Verbindungslimits prüfen | Datenbank-Team / GCP Support, falls länger als 15 Minuten |
| Memorystore/Redis down | Celery `ConnectionError`, Queue-Stau, LiteLLM Timeout | Memorystore Wartung, VPC Connector ausgefallen, Auth-Token abgelaufen | Worker pausieren, Connector-Status prüfen, Failover-Redis aktivieren falls vorhanden | Platform-Team, ggf. Redis-Snapshot wiederherstellen |
| HTTP 5xx Spike | Cloud Run Error Rate > 5%, Sentry/Logging Anstieg, Load Balancer Fehler | Regression in neuem Release, erschöpfte DB-Verbindungen, externe API langsam | Sofortiger Traffic-Rollback laut [Pipeline](../cicd/pipeline.md), letzte stabile Revision aktivieren, Ressourcen-Auslastung prüfen | Produkt & Engineering Management, ggf. Feature-Flag deaktivieren |
| LiteLLM nicht erreichbar | `/health/liveliness` schlägt fehl, Admin-GUI 503, AI Requests timeouts | Master Key falsch, Cloud SQL Schema gesperrt, Rate-Limit überschritten | Schlüssel prüfen ([Security](../security/secrets.md)), Cloud Run Logs analysieren, Dienst neu starten | AI Platform Owner, ggf. API-Provider kontaktieren |
| Retrieval liefert 0 Treffer | Langfuse Trace „retrieval_results=0“, Django Response `fallback=true`, Queue-Metriken stabil | Fehlende Vektor-Daten, falscher `tenant_id`, Filter in LangChain falsch | Prüfe `documents`/`chunks` für Tenant, starte Ingestion-Mini-Batch laut [RAG-Ingestion](../rag/ingestion.md) | Data Ops für Content-Gaps, ggf. Produktteam |
| Embedding-Rate-Limit | LiteLLM Logs `429`, Langfuse Alert „embedding_rate_limit“, Queue Stau in `ingestion` | Modellprovider-Limit erreicht, fehlendes Backoff, parallele Batches zu hoch | Reduziere `BATCH_SIZE`, setze Retry Delay (siehe [Scaling](../operations/scaling.md)), informiere Stakeholder über Verzögerung | Provider-Support bei anhaltender Limitierung |
| Vektorindex korrupt/langsam | Query-Latenz > 1s, PG Logs `ERROR: index corrupted`, Monitoring Alarme | Abgebrochene `REINDEX`, veraltete Statistiken, Storage-Bottleneck | Leite Verkehr auf Fallback (`fallback=true`), führe `REINDEX CONCURRENTLY`/`ANALYZE` in Staging-Test durch, bevor Prod wieder Traffic erhält | Datenbank-Team, ggf. PITR-Restore |

Hinweis: Detaillierte Qualitätstests zu falschen Treffern oder Halluzinationen folgen nach der Implementierung.

# Schritte
1. Identifiziere das Szenario anhand der Messpunkte in der Tabelle.
2. Setze die aufgeführten Erstmaßnahmen um und dokumentiere jede Aktion im Incident-Channel.
3. Entscheide nach fünf Minuten, ob Eskalation nötig ist, und informiere die angegebenen Teams.
4. Nach Stabilisierung: Erfasse Post-Mortem-Notizen und aktualisiere Checklisten in [QA](../qa/checklists.md), falls Anpassungen nötig sind.
