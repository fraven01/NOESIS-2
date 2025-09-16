# Warum
Skalierung stellt sicher, dass Anfragen rechtzeitig beantwortet werden und Ressourcen effizient genutzt werden. Dieses Dokument definiert Regeln für Web-, Worker- und Scheduler-Dienste.

# Wie
## Web-Service
- Gunicorn startet mit drei Sync-Workern (`--workers 3`). Setze Cloud Run Concurrency auf höchstens 30, damit nicht mehr als zehn Anfragen pro Worker parallel ankommen.
- Prod benötigt mindestens zwei Instanzen, damit Releases mit Traffic-Split möglich sind; Staging darf auf null skalieren.
- CPU 1 in Staging, CPU 2 in Prod. Memory 1 GiB (Staging) bzw. 2 GiB (Prod) deckt Tenant-Wechsel und Caching.
- Beobachte `requests_per_second` und `latency` aus Cloud Monitoring. Bei dauerhaft >70% CPU erhöhe Worker-Anzahl im Container oder Concurrency in kleinen Schritten.

## Worker-Service
- Celery Worker laufen mit Concurrency = CPU-Kerne. Für Cloud Run Services bedeutet das: CPU 1 → 1 gleichzeitige Task. Passe `--concurrency` explizit an, wenn Tasks I/O-bound sind.
- Skalierung richtet sich nach Queue-Tiefe (`redis` Keys). Cloud Monitoring Alarm bei >100 offenen Tasks löst manuelles Hochskalieren (maxInstances) aus.
- Tasks müssen idempotent sein (siehe `ai_core` Rate-Limit Tests), damit Wiederholungen keine doppelten Effekte auslösen.
- Timeout-Grenzen: Standard 10 Minuten; Tasks, die länger laufen, sollten in dedizierte Worker ausgelagert werden.
- **Ingestion-Queue**: Concurrency pro Instanz maximal 2; `BATCH_SIZE` 128, `CHUNK_SIZE` 800 Token, `CHUNK_OVERLAP` 80. Bei Rate-Limit-Fehlern Exponential-Backoff mit Start 30 Sekunden, Maximum 5 Minuten.
- **Agenten-Queue**: Concurrency = 1 pro Instanz für deterministisches State-Handling. LangGraph-Knoten müssen idempotent sein und Timeouts auf 120 Sekunden setzen; darüber hinaus abgebrochene Tasks loggen.
- Kostengrenzen: Staging < 5 EUR/Tag, Prod < 30 EUR/Stunde für Embeddings. Überwachung via Langfuse Usage-Dashboard, Alerts im [Langfuse Guide](../observability/langfuse.md).

## Beat vs Scheduler
- Aktivere Beat nur, wenn periodische Jobs existieren. Eine einzelne Instanz reicht; hohe Verfügbarkeit wird durch schnelle Restart-Zeiten erreicht.
- Alternative Scheduler (Cloud Scheduler -> Pub/Sub -> Worker) kommen zum Einsatz, wenn Jobs selten sind oder exakte Cron-Zeiten erfordern.
- Beat schreibt in Redis; stelle sicher, dass Beat und Worker dieselbe Redis-Instanz nutzen, sonst laufen Tasks doppelt.

# Schritte
1. Prüfe Metriken (CPU, Latenz, Queue-Länge) vor jeder Skalierungsentscheidung und dokumentiere Baseline-Werte.
2. Passe Concurrency, min/max Instances und ggf. Worker-Flags an und deploye neue Revisionen über die [Pipeline](../cicd/pipeline.md).
3. Überwache Effekte mindestens einen Tag lang; setze Alerts bei erwarteten Grenzwerten.
4. Aktualisiere das [Incident-Runbook](../runbooks/incidents.md) und die [QA-Checklisten](../qa/checklists.md), falls neue Schwellenwerte gelten.
