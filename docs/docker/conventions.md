# Warum
Saubere Container-Images sind die Grundlage für reproduzierbare Deployments und sichere Cloud-Workloads. Dieses Dokument fasst verbindliche Bau- und Laufzeitregeln zusammen.

# Wie
## Build-Prinzipien
- **Multi-Stage**: Das bestehende [`Dockerfile`](../../Dockerfile) trennt Node-Asset-Build, Python-Build und Runner. Diese Struktur bleibt Pflicht, damit das Runtime-Image klein bleibt.
- **Non-root**: Der Runner legt `appuser` (UID 10001) an und wechselt dauerhaft zu ihm. Neue Images dürfen keine Root-Prozesse starten.
- **Read-only Filesystem**: Für Staging und Prod wird das Root-Filesystem als read-only konfiguriert; Schreibzugriff erfolgt über `/tmp` und Cloud-Volumes. Django-Collectstatic liefert fertige Assets, daher sind keine Schreiboperationen nötig.
- **Deterministische Versionierung**: Images erhalten Semver+SHA (z.B. `v1.4.0-<commit>`) laut [Pipeline](../cicd/pipeline.md); keine Floating-Tags.
- **Bibliotheksabhängigkeiten**: Python-Requirements enthalten LangChain, LangGraph, `pgvector`-Client (`pgvector`/`psycopg`) und das Langfuse SDK. Versionen werden in `requirements*.txt` eingefroren und mit dem Image ausgeliefert.

## Startkommandos
- **Web-Service**: Startet Gunicorn über das `CMD` im Image (`gunicorn noesis2.wsgi:application --bind 0.0.0.0:${PORT} --workers 3`). Cloud Run `PORT` steuert den Listener.
- **Worker-Service**: Startbefehl `celery -A noesis2 worker -l info`. In Prod zusätzliche `--concurrency`-Flags gemäß [Scaling-Guide](../operations/scaling.md).
- **Ingestion-Worker**: Separate Queue `celery -A noesis2 worker -l info -Q ingestion` für Embedding-Läufe. Deployment siehe [RAG-Ingestion](../rag/ingestion.md).
- **Agenten-Worker**: Verwenden dieselbe Binary wie Worker, aber Queue-Flag `-Q agents`. Guardrails stehen in der [Agenten-Übersicht](../agents/overview.md).
- **Beat/Scheduler**: Separater Container mit `celery -A noesis2 beat -l info`; niemals im Worker-Prozess kombinieren.
- **Migrate-Job**: Führt `python manage.py migrate_schemas --noinput` aus und endet bei Erfolg. Siehe [Runbook](../runbooks/migrations.md).
- **LiteLLM**: Nutzt `litellm --config /config/config.yaml --host 0.0.0.0 --port 4000 --num_workers 1` laut `docker-compose.dev.yml`.

## Gesundheitsendpunkte
- **Web**: `/` beantwortet HTTP 200 und dient als Liveness; `/admin/login/` prüft Session-Stack. Für Readiness wird ein leichtgewichtiger JSON-Endpunkt ergänzt (z.B. `GET /tenant-demo/` liefert `{"status": "ok"}` via `DemoView`).
- **Worker**: Health basiert auf Celery Inspect (`celery --app noesis2 inspect ping` in separatem Kontrolljob). Cloud Run skaliert Worker-Container bei fehlgeschlagenen Pings herunter.
- **Ingestion-Queue**: Zusätzliche Probe `celery --app noesis2 inspect active -d ingestion@*` prüft, dass die Queue verarbeitet wird und Batch-Laufzeiten innerhalb der Grenzwerte aus [Scaling](../operations/scaling.md) bleiben.
- **Agenten-Queue**: Heartbeat-Metriken (`celery events`) melden Fehlerhäufigkeit und Antwortzeit. Langfuse-Traces dienen als ergänzende Readiness, siehe [Langfuse Observability](../observability/langfuse.md).
- **LiteLLM**: `/health/liveliness` liefert `alive`, `/health` gibt Aggregatszustand zurück (siehe `docker-compose.dev.yml`).

## Was nicht passiert
- Web- und Worker-Container führen **keine** automatischen Migrationen aus; der [Migrate-Job](../runbooks/migrations.md) übernimmt das exklusiv. Für lokale Entwicklung darf `entrypoint.sh` angepasst werden, Staging/Prod nutzen eine Variante ohne `migrate`.
- In Staging und Prod werden **keine lokalen Volumes** eingehängt; persistente Daten liegen ausschließlich in Cloud SQL und Memorystore.

# Schritte
1. Halte dich beim Erstellen neuer Images an die vorhandene Multi-Stage-Struktur und überprüfe nach jedem Build die Dateigröße.
2. Prüfe Startkommandos und Health-Ziele jedes Dienstes, bevor du Deployment-Templates aktualisierst.
3. Entferne alle automatischen Migrationen aus Laufzeit-Entrypoints für Cloud-Umgebungen und verweise auf das dedizierte Runbook.
4. Verifiziere bei Reviews, dass neue Container ohne lokale Volumes und mit read-only Filesystem spezifiziert sind.
