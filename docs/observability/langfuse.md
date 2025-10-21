# Warum
Langfuse bündelt Tracing und Kostenkontrolle für Agenten, Ingestion und LiteLLM. Dieses Kapitel zeigt Integration, Datenflüsse und Alarmierung.

# Wie
## Datenfluss
```mermaid
flowchart LR
    subgraph Worker
        CW[Celery Worker]
        AG[LangGraph Agent]
        IN[Ingestion Pipeline]
    end
    subgraph LiteLLM
        LL[LiteLLM Service]
    end
    subgraph Langfuse
        LF[Langfuse API]
        DB[(Langfuse Store)]
    end
    CW -->|Callbacks| LF
    AG -->|trace/span| LF
    IN -->|metrics| LF
    LL -->|model trace| LF
    LF --> DB
```

- Worker binden das Langfuse Python SDK (`langfuse.decorators`, Async-Callbacks) ein. Jeder Task sendet `trace_id`, `tenant_id`, `operation`.
- LangGraph nutzt das integrierte Observability-Plugin; Nodes melden `span`-Informationen inklusive Input/Output-Metadaten.
- Ingestion sendet Batch-Statistiken (`batch_size`, `duration`, `cost_estimate`) über Custom Metrics.
- LiteLLM aktiviert Model-Trace per ENV `LANGFUSE_HOST`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_KEY`; Sampling wird über `LANGFUSE_SAMPLE_RATE` (0.05 Prod, 0.25 Staging) gesteuert.

## Felder und Sampling
| Feld | Quelle | Beschreibung |
| --- | --- | --- |
| `trace_id` | Django Request ID | Verbindet Web, Celery, LangGraph |
| `tenant_id` | Celery Task Meta | Steuert RLS und Filter |
| `operation` | Task- oder Node-Name | `ingestion.embed`, `agent.reason`, `liteLLM.call` |
| `cost_total` | LiteLLM | Aggregierte Kosten pro Request |
| `error_type` | Worker/LiteLLM | Klassifiziert Fehlertyp (z.B. `rate_limit`) |
| `visibility_effective` | VectorStoreRouter | Wirksame Soft-Delete-Sichtbarkeit (`active`, `all`, `deleted`) pro Retrieval-Span |
| `deleted_matches_blocked` | VectorStoreRouter | Anzahl der durch `visibility=active` weggefilterten Kandidaten |

- Sampling-Richtwerte: Prod 5%, Staging 25%, Dev 100%. Anpassungen erfolgen über `LANGFUSE_SAMPLE_RATE` und werden im [Security-Guide](../security/secrets.md) dokumentiert.
- Retrieval-Spans laufen als `rag.hybrid.search` ein und tragen `visibility_effective` sowie `deleted_matches_blocked`, damit Guard-Entscheidungen zu Soft-Deletes sichtbar werden.
- Hard-Delete-Läufe erzeugen einen Span `rag.hard_delete` mit Zählern (`documents_deleted`, `not_found`, `vacuum`, `reindex`, `operator`, `actor_mode`), sodass Audit und Bereinigungsschritte nachvollziehbar bleiben.
- PII-Redaction: Die Anwendung maskiert PII vor jedem LLM-Aufruf; Langfuse erhält bereits redaktierte Daten. Zusätzliche Langfuse-Regeln sind optional.
- Dashboards: Standard-Dashboards `Agent Erfolg`, `Ingestion Durchsatz`, `Kosten pro Tenant`. Alerts lösen bei Fehlerquote >5% oder Kosten >80% Budget aus.

# Schritte
1. Installiere und initialisiere das Langfuse SDK in Worker und Agenten (siehe [Docker-Konventionen](../docker/conventions.md)); setze ENV `LANGFUSE_*` laut [Security](../security/secrets.md).
2. Aktiviere LiteLLM Model-Trace (`LANGFUSE_*`, `LANGFUSE_SAMPLE_RATE`) und validiere Traces über die GUI.
3. Richte Dashboards und Alerts ein, dokumentiere Schwellenwerte im Incident-Runbook und überprüfe sie nach jedem Release.

## Lokal (Self-hosted)
- Compose-Stack: `docker/docker-compose.langfuse.yml` startet `langfuse` (GUI+API, Port 3100), eine isolierte Postgres-DB, ClickHouse (erforderlich ab Langfuse v3), Redis und Minio (S3‑kompatibel) für Uploads/Queues.
  - ClickHouse Credentials (lokal): `CLICKHOUSE_USER=langfuse`, `CLICKHOUSE_PASSWORD=langfuse` (im Compose voreingestellt); `CLICKHOUSE_URL` ist entsprechend gesetzt.
- Start/Stop per NPM:
  - `npm run dev:langfuse:up`
  - `npm run dev:langfuse:down`
  - `npm run dev:langfuse:logs`
- ENV (lokal):
  - `LANGFUSE_HOST=http://localhost:3100` (aus Host-Sicht)
  - Für Container (Web/Worker/LiteLLM) nutze standardmäßig `http://host.docker.internal:3100` auf Docker Desktop; unter Linux ggf. Bridge-Gateway exportieren (z. B. `172.17.0.1`).
  - `LANGFUSE_PUBLIC_KEY=pk_dev_...`, `LANGFUSE_SECRET_KEY=sk_dev_...`
  - `LANGFUSE_SAMPLE_RATE=1.0` (Dev), `0.25` (Staging), `0.05` (Prod)
- Sicherheit: In Dev/Staging blockiert ein Settings-Guard versehentliche Prod-Keys.
- ELK: Leite Container-Logs optional in ELK (`docs/observability/elk.md`); Traces/Kosten verbleiben in Langfuse.

### Troubleshooting
- Meldung „CLICKHOUSE_URL is not configured“: Stelle sicher, dass der ClickHouse-Container läuft (im Compose enthalten) und `CLICKHOUSE_URL` für den `langfuse`-Service gesetzt ist. In unserem Compose ist dies bereits auf `http://default:@clickhouse:8123` vorkonfiguriert.
- Meldung „CLICKHOUSE_MIGRATION_URL is not configured“: Ab Langfuse v3 muss zusätzlich `CLICKHOUSE_MIGRATION_URL` gesetzt sein (kann identisch zu `CLICKHOUSE_URL` sein). In unserem Compose ist dies entsprechend ergänzt.
- Fehler „unknown driver http (forgotten import?)“ bei ClickHouse-Migrationen: Setze `CLICKHOUSE_MIGRATION_URL` mit dem nativen ClickHouse‑Schema (`clickhouse://user:pass@clickhouse:9000/<db>`). Die Runtime‑URL bleibt HTTP (`http://...:8123/<db>`), die Migrationen benötigen jedoch den nativen Treiber.
- Unerwarteter `ZodError` beim Start (ENV-Validation): Stelle sicher, dass Redis- und S3/Minio-Variablen gesetzt sind. In unserem Compose sind Defaults hinterlegt (`REDIS_*`, `LANGFUSE_S3_*`, `LANGFUSE_USE_AZURE_BLOB=false`).
- ClickHouse bleibt im Zustand „waiting“: Der Healthcheck nutzt `clickhouse-client SELECT 1`. Stelle sicher, dass der Container diese Binary enthält (im offiziellen Image vorhanden) und dass die Credentials mit der Compose-Umgebung übereinstimmen.
- Fehler „CLICKHOUSE_USER is not set“: Abhilfe durch Setzen von `CLICKHOUSE_USER`, `CLICKHOUSE_PASSWORD`, `CLICKHOUSE_DATABASE` auch für den `langfuse`-Service (bereits in unserem Compose ergänzt).

## SDK Tracing (App)
- Aktivierung: Installiere optional `langfuse` im Python-Env. Ohne SDK laufen alle Dekoratoren als No-Op.
- Abdeckung (out of the box):
  - Graph-Ausführung: `ai_core/services.py:execute_graph` (Span `graph.execute` + Tags `graph:<name>`, `version:<v>`)
  - LLM-Calls: `ai_core/llm/client.py:call` (Span `llm.call` + Tags `label:<label>`, `model:<id>`)
  - Ingestion-Embeddings: `ai_core/tasks.py:embed` (Span `ingestion.embed`)
- Kontextfelder: `user_id=tenant_id`, `session_id=case_id`, `metadata.trace_id=<X-Trace-ID>`; keine PII-Payloads werden an Langfuse gesendet.
- LangChain (bei Einsatz): Verwende `ai_core/infra/observability.get_langchain_callbacks()` um den Langfuse-Callback-Handler an LangChain-LLMs zu übergeben.
- Fehlermeldung „Invalid environment variables: SALT“: Setze `SALT` (beliebige, aber starke Zeichenkette) und `ENCRYPTION_KEY` (64 Hex-Zeichen, z. B. via `openssl rand -hex 32`). In unserem Compose ist beides mit Dev-Defaults hinterlegt; überschreibe sie in deiner `.env`.
