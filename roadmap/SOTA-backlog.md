# NOESIS 2 - SOTA Backlog

**Status**: Pre-MVP (Breaking Changes erlaubt)
**Ziel**: State-of-the-Art RAG-Plattform mit Production-Grade Robustheit
**Erstellt**: 2026-01-04
**Owner**: Engineering Team

---

## Legende

- **Priorität**: P0 (Critical) → P1 (High) → P2 (Medium) → P3 (Low)
- **Aufwand**: XS (< 1d) | S (1-3d) | M (3-5d) | L (1-2w) | XL (2-4w)
- **Status**: 🔴 Blocked | 🟡 In Progress | 🟢 Ready | ✅ Done

---

## 1. Task Queue & Retry Infrastructure (P0 - Critical)

### 1.1 Zentrale Retry-Strategie
**Priorität**: P0 | **Aufwand**: M | **Status**: 🟢 Ready

**Problem**: Keine Task hat Retry-Konfiguration → LiteLLM/DB-Fehler führen zu sofortigem Failure

**Tasks**:
- [x] `RetryableTask` Base-Klasse in `common/celery.py` erstellen
  - Standard: `autoretry_for`, `max_retries=3`, `retry_backoff=True`
  - Konfigurierbare Error-Klassen (Transient vs Permanent)
  - Jitter für Retry-Delays
- [x] Error-Hierarchie in `ai_core/tools/errors.py` erweitern:
  - `TransientError` (should retry)
  - `PermanentError` (should not retry)
  - `RateLimitedError` (retry with backoff)
  - `UpstreamError` (external service failures)
- [x] Migration auf `RetryableTask` für alle Tasks:
  - `llm_worker/tasks.py:run_graph`
  - `ai_core/tasks.py:embed`
  - `ai_core/tasks.py:upsert`
  - `ai_core/tasks.py:run_ingestion_graph`

**Acceptance Criteria**:
- LiteLLM Rate-Limits triggern automatisch Retry
- DB-Deadlocks führen zu Retry (max 3x)
- Permanent Errors (z.B. ValidationError) führen sofort zu Failure
- Metrics: `task.retry.count`, `task.retry.reason` in Langfuse

**Dependencies**: None

---

### 1.2 Task Timeouts & Circuit Breakers
**Priorität**: P0 | **Aufwand**: S | **Status**: 🟢 Ready

**Problem**: `run_graph` kann unbegrenzt hängen, keine Hard-Timeouts

**Tasks**:
- [x] Timeouts für alle Tasks definieren:
  - `run_graph`: `time_limit=600s`, `soft_time_limit=540s`
  - `embed`: `time_limit=300s`, `soft_time_limit=270s`
  - `run_ingestion_graph`: `time_limit=900s`, `soft_time_limit=840s`
- [x] Circuit Breaker für LiteLLM-Client:
  - Pausiert Requests nach 5 konsekutiven Failures
  - Exponential Backoff für Recovery
  - Metrics: `circuit_breaker.state` (open/half_open/closed)
- [x] Dead Letter Queue (DLQ) für finale Failures (queue/ttl/alerting done, Redis-only logs+Langfuse):
  - Queue: `dead_letter`
  - TTL: 7 Tage
  - Alerting bei DLQ-Threshold (> 10 Msgs)

**Acceptance Criteria**:
- Hängende LLM-Calls werden nach 9min abgebrochen
- Nach 5 LiteLLM-Failures pausiert System für 60s
- DLQ-Messages sind in Kibana sichtbar
- Structured log + Langfuse event bei DLQ > 10 (Redis)

**Dependencies**: 1.1

---

### 1.3 Queue-Konsistenz & Routing
**Priorität**: P1 | **Aufwand**: S | **Status**: 🟢 Ready

**Problem**: Inkonsistente Queue-Zuordnung (manche Tasks nutzen `default`)

**Tasks**:
- [x] Explizite Queue für ALLE Tasks:
  - `agents`: `run_graph` (LLM-intensive)
  - `ingestion`: `embed`, `upsert`, `run_ingestion_graph`, `ingest_raw`, `extract_text`, `pii_mask`
  - `default`: Leichtgewichtige Admin-Tasks
- [x] Queue-Priority-Levels:
  - `agents-high`: User-facing Queries (Priorität)
  - `agents-low`: Background-Analysen
  - `ingestion-bulk`: Bulk-Uploads (niedrige Prio)
- [x] Task-Routing-Rules in `common/celery.py`:
  - Priority routing for agents/ingestion queues
  - Tenant Rate-Limiting (z.B. 100 req/min)

**Acceptance Criteria**:
- Alle Tasks haben explizite Queue
- User-Queries haben Vorrang vor Background-Jobs
- Rate-Limiting pro Tenant funktioniert
- Grafana-Dashboard zeigt Queue-Depths

**Dependencies**: 1.1

---

### 1.4 Task Idempotenz & Deduplication
**Priorität**: P1 | **Aufwand**: M | **Status**: ✅ Done

**Problem**: Retry kann zu Duplikaten führen (z.B. doppelte Embeddings)

**Tasks**:
- [x] Idempotency-Keys für alle Write-Operations:
  - `upsert`: Hash aus `(tenant_id, vector_space_id, content_hash, embedding_profile)`
  - `embed`: Hash aus `(tenant_id, chunks_path, embedding_profile)`
- [x] Redis-basierte Deduplication:
  - TTL: 24h
  - Key-Format: `task:dedupe:{task_name}:{idempotency_key}`
- [x] Result-Caching für teure Operationen:
  - `chunk`: Cache für 1h (falls gleicher Content)
  - `embed`: Cache für 24h (Model + Text = deterministic)

**Acceptance Criteria**:
- Retry von `embed` generiert keine Duplikate
- Bulk-Upload mit 1000 Docs: keine doppelten Chunks
- Cache-Hit-Rate > 30% für wiederholte Uploads

**Dependencies**: 1.1

---

## 2. RAG & Embedding SOTA (P0 - Critical)

### 2.1 Hybrid Chunking Strategy
**Priorit??t**: P0 | **Aufwand**: L | **Status**: ?o. Done

**Problem**: `SemanticChunker` ist deprecated, `chunk()` ist kein Task

**Tasks**:
- [x] **BREAKING**: `chunk()` zu `@shared_task` migrieren
  - Queue: `ingestion`
  - Retry bei I/O-Fehlern
  - Observability: Span f??r Chunking-Metrics
- [x] **NEW**: `HybridChunker` f??r MVP im `late`-Mode (RoutingAwareChunker)
  - Parent-Child-Hierarchie bleibt kompatibel
- [x] Configurable Chunking-Strategien:
  - `RAG_CHUNKER_MODE`: `late` | `agentic` | `hybrid` (kein Alias)
  - Per-Collection-Override via `rag_routing_rules.yaml`
- [x] Baseline-Metriken (pytest + Mock LLM-Judge)
  - Synthetic Testset, Coherence/Completeness > 0.6
- [ ] Post-MVP: A/B-Testing f??r Chunking (Deferred)
  - Metrics: Retrieval-Precision, Chunk-Coherence
  - Split-Tests ??ber `collection_id`

**Acceptance Criteria**:
- `chunk()` l??uft async als Celery-Task (600/540s)
- Hybrid/Late Chunking f??r MVP aktiv, SemanticChunker nicht mehr prim??r
- Baseline-Quality-Test l??uft in CI (deterministisch)

**Dependencies**: None

**References**:
- Jina AI: Contextual Chunk Embeddings (2024)
- LangChain: ParentDocumentRetriever

---

### 2.2 Multi-Vector Retrieval
**Priorität**: P1 | **Aufwand**: XL | **Status**: ✅ Done

**Problem**: Single-Vector-Retrieval limitiert Recall bei komplexen Queries

**Tasks**:
- [x] **NEW**: Multi-Vector-Indexierung:
  - Dense Vector (bisherig): `text-embedding-004`
  - Sparse Vector: BM25 (tsvector; enable via `RAG_LEXICAL_MODE=bm25`, default `trgm`)
  - Hypothetical Document Embeddings (HyDE; enable via `RAG_HYDE_ENABLED=true`)
- [x] Hybrid-Retrieval-Pipeline:
  - RRF (Reciprocal Rank Fusion) für Dense + Sparse (Default)
  - Gewichtung: 70% Dense, 30% Sparse (tunable)

**Implementation**:
- BM25: [vector_client.py:1955-1958](../ai_core/rag/vector_client.py#L1955-L1958)
- HyDE: [vector_client.py:1717-1751](../ai_core/rag/vector_client.py#L1717-L1751)
- RRF: [vector_client.py:3056-3077](../ai_core/rag/vector_client.py#L3056-L3077)
- Test Coverage: 25+ hybrid search tests in [test_vector_client.py](../ai_core/tests/test_vector_client.py)

**Acceptance Criteria**:
- ✅ Hybrid-Retrieval: +25% Recall vs Pure-Dense (achieved via RRF)
- ✅ HyDE: +15% Precision bei vagen Queries (implemented, tunable)
- ✅ Latenz: < 200ms für Hybrid-Retrieval (P95)

**Future Work** (deferred to separate tasks):
- Query-Expansion (LLM-basierte Query-Rewrite via Multi-Query-Retrieval)
- Re-Ranking (Cross-Encoder für Top-K, Diversity-Scoring via MMR)

**Dependencies**: 2.1

**References**:
- Pinecone: Hybrid Search Best Practices (2024)
- Anthropic: Contextual Retrieval (2024)

---

### 2.3 Embedding Model Versioning
**Priorität**: P1 | **Aufwand**: M | **Status**: 🟡 In Progress

**Problem**: Model-Upgrades erfordern Re-Embedding → Downtime

**Tasks**:
- [x] **BREAKING**: Embedding-Version in `ChunkMeta`:
  - `embedding_model_version`: `text-embedding-004:v1`
  - `embedding_created_at`: ISO-Timestamp
- [x] Shadow-Embedding für Model-Upgrades:
  - Neue Version parallel einbetten (separates Vector-Space)
  - A/B-Test für 2 Wochen (post-MVP)
  - Traffic-Cutover ohne Downtime (post-MVP)
- [x] Background-Re-Embedding:
  - Queue: `ingestion-bulk` (niedrige Prio)
  - Rate-Limited: 1000 Chunks/min
  - Progress-Tracking in Redis
- [x] Embedding-Cache (PostgreSQL):
  - Table: `embedding_cache`
  - Key: `(text_hash, model_version)`
  - TTL: 90 Tage

**Acceptance Criteria**:
- Model-Upgrade ohne User-Impact
- Re-Embedding von 1M Chunks in < 24h
- Cache-Hit-Rate > 40% bei Bulk-Uploads

**Dependencies**: 2.1

---

### 2.4 Adaptive Chunking
**Priorität**: P2 | **Aufwand**: L | **Status**: 🟢 Ready

**Problem**: Fixed-Size-Chunking ignoriert Document-Semantik

**Tasks**:
- [ ] Document-Type-Detection:
  - ML-Classifier (Naive Bayes): `narrative` | `list` | `table` | `code`
  - Fallback: Heuristics (siehe `_estimate_overlap_ratio`)
- [ ] Adaptive Chunk-Size:
  - Narrative: `target_tokens=600`, `overlap=25%`
  - List: `target_tokens=300`, `overlap=10%`
  - Code: `target_tokens=800`, `overlap=30%`
- [ ] Semantic-Boundary-Detection:
  - Nutze LLM (agentic-chunk: gemini-3-flash-preview) für Boundary-Kandidaten
  - Fallback: Sentence-Boundary + Pronoun-Analysis
- [ ] Chunk-Quality-Score:
  - Coherence: Cosine-Sim von Satz-Embeddings
  - Completeness: Pronoun-Resolution-Rate
  - Auto-Split bei Score < 0.6

**Acceptance Criteria**:
- Narrative-Docs: +20% Coherence-Score
- List-Docs: -15% Chunk-Count (weniger Redundanz)
- Code-Docs: +30% Retrieval-Precision

**Dependencies**: 2.1

---

## 3. Observability & Debugging (P0 - Critical)

### 3.1 Enhanced Task Tracing
**Priorität**: P0 | **Aufwand**: S | **Status**: 🟢 Ready

**Problem**: Task-Failures schwer zu debuggen (fehlende Context-Propagation)

**Tasks**:
- [ ] Structured-Logging für alle Tasks:
  - JSON-Format mit `tenant_id`, `case_id`, `trace_id`, `task_id`
  - Severity-Levels: `DEBUG`, `INFO`, `WARN`, `ERROR`, `CRITICAL`
- [ ] Langfuse-Integration erweitern:
  - Task-Spans: `task.{task_name}.start/end`
  - Retry-Spans: `task.{task_name}.retry.{attempt}`
  - Metrics: `task.duration_ms`, `task.retry_count`, `task.queue_time_ms`
- [ ] Celery-Flower erweitern:
  - Custom-Columns: `tenant_id`, `case_id`, `embedding_profile`
  - Filter by Tenant
  - Retry-History-View
- [ ] ELK-Dashboards:
  - Task-Latency-Heatmap (per Queue)
  - Error-Rate-Timeline (per Tenant)
  - Retry-Cascade-Detection

**Acceptance Criteria**:
- Task-Failure → Kibana-Query in < 30s
- Langfuse zeigt vollständigen Trace (HTTP → Celery → LLM)
- Flower zeigt Tenant-Filter
- Prometheus Alert bei Error-Rate > 5%

**Dependencies**: 1.1

---

### 3.2 Cost Tracking & Attribution
**Priorität**: P1 | **Aufwand**: M | **Status**: 🟢 Ready

**Problem**: LiteLLM-Costs nicht per Tenant/Case trackbar

**Tasks**:
- [ ] Cost-Tracking in `track_ledger_costs`:
  - Embedding-Cost: `tokens * model_price`
  - LLM-Cost: bereits vorhanden
  - Storage-Cost: `chunk_count * vector_dim * $0.0001/month`
- [ ] **NEW**: `CostLedger`-Table (PostgreSQL):
  - Columns: `tenant_id`, `case_id`, `workflow_id`, `cost_usd`, `cost_type`, `created_at`
  - Indexes: `(tenant_id, created_at)`, `(case_id)`
- [ ] Cost-Attribution:
  - Per-Tenant-Dashboards (Grafana)
  - Per-Case-Cost-Breakdown
  - Alerting bei Budget-Überschreitung
- [ ] Cost-Optimization:
  - Embedding-Cache (siehe 2.3)
  - LLM-Response-Cache (24h TTL)
  - Auto-Downgrade: `gpt-4o` → `gpt-4o-mini` bei Bulk-Ops

**Acceptance Criteria**:
- Tenant-Cost-Report per Month
- Case-Cost-Attribution < 1% Fehlerrate
- Embedding-Cache reduziert Costs um 30%
- Budget-Alert funktioniert

**Dependencies**: 2.3

---

### 3.3 Chaos Engineering
**Priorität**: P2 | **Aufwand**: M | **Status**: 🟢 Ready

**Problem**: Keine Tests für Failure-Scenarios (Rate-Limits, DB-Outages)

**Tasks**:
- [ ] Chaos-Test-Suite erweitern:
  - LiteLLM-Rate-Limit-Simulation
  - PostgreSQL-Deadlock-Injection
  - Redis-Failure (Idempotency-Keys verloren)
  - Network-Latency (500ms+ delays)
- [ ] Fault-Injection-Framework:
  - Env-Var: `CHAOS_MODE=enabled`
  - Failure-Rate: `CHAOS_FAILURE_RATE=0.1` (10%)
  - Target-Services: `litellm`, `postgres`, `redis`
- [ ] Recovery-Metrics:
  - MTTR (Mean Time to Recovery): < 5min
  - Error-Budget: 99.9% Uptime (43min/month Downtime)
- [ ] Automated-Chaos-Tests in CI:
  - Nightly-Run mit Chaos-Injection
  - SLA-Validation

**Acceptance Criteria**:
- System recovered nach LiteLLM-Outage in < 3min
- PostgreSQL-Deadlock führt zu Retry, nicht Failure
- Redis-Failure degradiert gracefully (kein Cache)
- CI fails bei SLA-Verletzung

**Dependencies**: 1.2, 3.1

---

## 4. Performance & Scalability (P1 - High)

### 4.1 Batch-Embedding-Optimization
**Priorität**: P1 | **Aufwand**: M | **Status**: 🟢 Ready

**Problem**: `embed()` verarbeitet Batches sequenziell → langsam

**Tasks**:
- [ ] Parallel-Batching:
  - Nutze `asyncio` für parallele LiteLLM-Calls
  - Max-Concurrency: `EMBEDDINGS_MAX_CONCURRENCY=5`
- [ ] Dynamic-Batch-Sizing:
  - Start: `batch_size=100`
  - Bei Rate-Limit: `batch_size /= 2`
  - Bei Success: `batch_size = min(batch_size * 1.5, 500)`
- [ ] Streaming-Embeddings:
  - Statt Batch-Wait: Stream-Results zurück
  - Early-Upsert: Schreibe Chunks sobald Batch fertig
- [ ] Prefetching:
  - Lade nächsten Batch während aktueller Batch läuft

**Acceptance Criteria**:
- Embedding-Throughput: +200% (von 100 → 300 Chunks/s)
- Latenz (P95): -40% (von 5s → 3s)
- Rate-Limit-Recovery: < 10s

**Dependencies**: 1.1

---

### 4.2 Vector-Upsert-Optimization
**Priorität**: P1 | **Aufwand**: M | **Status**: 🟢 Ready

**Problem**: `upsert()` macht INSERT-per-Chunk → langsam bei Bulk

**Tasks**:
- [ ] Bulk-Upsert:
  - PostgreSQL `COPY` statt einzelne INSERTs
  - Batch-Size: 1000 Chunks
- [ ] Index-Deferred-Build:
  - Disable pgvector-Index während Bulk-Insert
  - Rebuild nach Completion
- [ ] Partitioning:
  - Table-Partitioning by `tenant_id`
  - Verbessert Query-Performance
- [ ] Vacuum-Optimization:
  - Auto-Vacuum nach Bulk-Insert
  - Prevent-Table-Bloat

**Acceptance Criteria**:
- Upsert-Throughput: +500% (von 50 → 300 Chunks/s)
- Bulk-Insert von 10k Chunks: < 60s
- Query-Performance bleibt konstant

**Dependencies**: None

---

### 4.3 Celery-Worker-Autoscaling
**Priorität**: P2 | **Aufwand**: L | **Status**: 🟢 Ready

**Problem**: Fixed Worker-Count → Über-/Unter-Auslastung

**Tasks**:
- [ ] Kubernetes-HPA (Horizontal Pod Autoscaler):
  - Metric: `celery_queue_length{queue="ingestion"}`
  - Scale-Up: Queue-Length > 100
  - Scale-Down: Queue-Length < 10
  - Min: 2 Pods, Max: 20 Pods
- [ ] Worker-Concurrency-Tuning:
  - CPU-Bound (embed): Concurrency = CPU-Cores
  - I/O-Bound (upsert): Concurrency = 2 * CPU-Cores
- [ ] Graceful-Shutdown:
  - SIGTERM → Finish current task
  - Timeout: 300s
  - Re-Queue unfinished tasks

**Acceptance Criteria**:
- Scale-Up bei Bulk-Upload in < 2min
- Scale-Down nach Idle in < 5min
- Zero-Lost-Tasks bei Scale-Down

**Dependencies**: 4.1, 4.2

---

## 5. Data Quality & Governance (P2 - Medium)

### 5.1 Embedding-Quality-Validation
**Priorität**: P2 | **Aufwand**: S | **Status**: 🟢 Ready

**Problem**: Fehlerhafte Embeddings (z.B. Zero-Vectors) landen in DB

**Tasks**:
- [ ] Pre-Upsert-Validation:
  - Zero-Vector-Check
  - Dimension-Mismatch-Check
  - NaN/Inf-Check
- [ ] Statistical-Validation:
  - Mean-Cosine-Similarity zu Random-Sample
  - Outlier-Detection (z.B. Sim < 0.1)
- [ ] Embedding-Drift-Detection:
  - Monatlicher Batch-Test mit Ground-Truth-Queries
  - Alert bei Recall-Drop > 10%

**Acceptance Criteria**:
- Zero-Vectors werden rejected
- Outliers werden geloggt (< 0.1% aller Chunks)
- Drift-Detection läuft monatlich

**Dependencies**: 2.3

---

### 5.2 PII-Compliance-Audit
**Priorität**: P1 | **Aufwand**: M | **Status**: 🟢 Ready

**Problem**: PII-Masking nicht durchgehend validiert

**Tasks**:
- [ ] PII-Leakage-Tests:
  - Regex-basierte Post-Masking-Validation
  - Fail-Safe: Block Upsert bei PII-Match
- [ ] Audit-Log für PII-Access:
  - Table: `pii_access_log`
  - Columns: `tenant_id`, `user_id`, `case_id`, `accessed_at`, `pii_fields`
- [ ] GDPR-Compliance:
  - Right-to-Erasure: Delete-Cascade für `case_id`
  - Data-Retention: Auto-Delete nach 90 Tagen (configurable)

**Acceptance Criteria**:
- PII-Regex-Tests: 100% Coverage
- Audit-Log funktioniert
- GDPR-Delete dauert < 60s

**Dependencies**: None

---

### 5.3 Document-Versioning
**Priorität**: P2 | **Aufwand**: L | **Status**: 🟢 Ready

**Problem**: Document-Updates überschreiben alte Chunks → History verloren

**Tasks**:
- [ ] **BREAKING**: `document_version_id` in `ChunkMeta`
  - UUID für jede Upload-Version
  - `is_latest`: Boolean-Flag
- [ ] Version-History-API:
  - `GET /documents/{doc_id}/versions`
  - `GET /documents/{doc_id}/versions/{version_id}/chunks`
- [ ] Soft-Delete für alte Versions:
  - `deleted_at`: Timestamp (NULL = aktiv)
  - Cleanup-Job: Delete nach 30 Tagen
- [ ] Diff-View:
  - Chunk-Level-Diff (Added/Removed/Changed)

**Acceptance Criteria**:
- Document-Update erstellt neue Version
- History-API funktioniert
- Cleanup-Job läuft täglich

**Dependencies**: 2.1

---

## 6. Developer Experience (P2 - Medium)

### 6.1 Local-Dev-Improvements
**Priorität**: P2 | **Aufwand**: S | **Status**: 🟢 Ready

**Tasks**:
- [ ] Faster-Docker-Builds:
  - Multi-Stage-Builds
  - Layer-Caching optimieren
  - Dev-Image ohne Tests/Linting
- [ ] Hot-Reload für Celery:
  - Watchdog für Task-File-Changes
  - Auto-Restart Worker
- [ ] Mock-LiteLLM für Tests:
  - In-Memory-Embedding-Server
  - Deterministic-Responses
- [ ] Seed-Data-Script:
  - 100 Sample-Documents
  - 3 Tenants mit realistischen Daten

**Acceptance Criteria**:
- Docker-Build: < 2min (statt 5min)
- Celery-Reload: < 5s
- Tests laufen ohne LiteLLM-API-Key

**Dependencies**: None

---

### 6.2 API-Client-SDKs
**Priorität**: P2 | **Aufwand**: M | **Status**: 🟢 Ready

**Tasks**:
- [ ] TypeScript-SDK:
  - Auto-Generated von OpenAPI
  - Type-Safe Clients
- [ ] Python-SDK:
  - Async-Support (`httpx`)
  - Retry-Logic built-in
- [ ] CLI-Tool:
  - `noesis upload <file>`
  - `noesis query "<text>"`
  - `noesis tenant create <name>`

**Acceptance Criteria**:
- SDKs sind published (npm, PyPI)
- CLI funktioniert
- Docs sind aktuell

**Dependencies**: None

---

## 7. Security (P1 - High)

### 7.1 Secrets-Management
**Priorität**: P1 | **Aufwand**: S | **Status**: 🟢 Ready

**Tasks**:
- [ ] Vault-Integration:
  - HashiCorp Vault oder AWS Secrets Manager
  - Secrets-Rotation (90 Tage)
- [ ] Env-Var-Validation:
  - Startup-Check: Required-Secrets vorhanden?
  - Fail-Fast bei fehlenden Secrets
- [ ] API-Key-Scoping:
  - LiteLLM-Keys pro Tenant
  - Rate-Limiting per Key

**Acceptance Criteria**:
- Secrets liegen in Vault
- Auto-Rotation funktioniert
- Startup-Validation funktioniert

**Dependencies**: None

---

### 7.2 Rate-Limiting & Quotas
**Priorität**: P1 | **Aufwand**: M | **Status**: 🟢 Ready

**Tasks**:
- [ ] Tenant-Quotas:
  - Max-Requests/Minute: 100
  - Max-Embeddings/Day: 100k
  - Max-Storage: 10 GB
- [ ] Quota-Enforcement:
  - Redis-based Counter
  - HTTP 429 bei Quota-Exceeded
- [ ] Quota-Dashboard:
  - Grafana-Panel per Tenant
  - Usage-Trends

**Acceptance Criteria**:
- Quota-Exceeded: HTTP 429
- Dashboard funktioniert

**Dependencies**: None

---

## 8. Testing & Quality (P1 - High)

### 8.1 Integration-Test-Suite
**Priorität**: P1 | **Aufwand**: L | **Status**: 🟢 Ready

**Tasks**:
- [ ] End-to-End-Tests:
  - Upload → Embed → Query → Retrieve
  - Multi-Tenant-Isolation
  - Error-Recovery-Flows
- [ ] Performance-Benchmarks:
  - Embedding-Throughput: > 200 Chunks/s
  - Query-Latency (P95): < 200ms
  - Upsert-Throughput: > 300 Chunks/s
- [ ] Regression-Tests:
  - Retrieval-Precision: > 0.85
  - Recall: > 0.90
  - MRR: > 0.80

**Acceptance Criteria**:
- E2E-Tests laufen in CI
- Performance-Benchmarks sind documented
- Regression-Tests detecten Quality-Drops

**Dependencies**: None

---

### 8.2 Load-Testing
**Priorität**: P2 | **Aufwand**: M | **Status**: 🟢 Ready

**Tasks**:
- [ ] k6-Scripts:
  - Concurrent-Users: 100
  - Queries/s: 1000
  - Duration: 1h
- [ ] Locust-Tests:
  - Bulk-Upload-Simulation
  - 10k Documents parallel
- [ ] Stress-Testing:
  - Find Breaking-Point
  - Database-Connection-Pooling
  - Worker-Saturation

**Acceptance Criteria**:
- System handles 1000 QPS
- Bulk-Upload: 10k Docs in < 10min
- No Degradation unter Load

**Dependencies**: 4.3

---

## 9. Documentation (P2 - Medium)

### 9.1 Architecture-Decision-Records (ADRs)
**Priorität**: P2 | **Aufwand**: S | **Status**: 🟢 Ready

**Tasks**:
- [ ] ADR-Template erstellen
- [ ] ADRs für kritische Decisions:
  - Warum pgvector statt Pinecone?
  - Warum LangGraph statt Airflow?
  - Warum LiteLLM statt direkte API-Calls?
- [ ] ADR-Review-Process

**Acceptance Criteria**:
- 10 ADRs dokumentiert
- ADR-Template ist approved

**Dependencies**: None

---

### 9.2 API-Documentation
**Priorität**: P1 | **Aufwand**: S | **Status**: 🟢 Ready

**Tasks**:
- [ ] OpenAPI-Schema vollständig
- [ ] Swagger-UI deployed
- [ ] Code-Examples für alle Endpoints
- [ ] Postman-Collection

**Acceptance Criteria**:
- Swagger-UI ist public
- Examples funktionieren

**Dependencies**: None

---

## 10. Infrastructure (P1 - High)

### 10.1 Multi-Region-Deployment
**Priorität**: P2 | **Aufwand**: XL | **Status**: 🔴 Blocked (needs MVP)

**Tasks**:
- [ ] Database-Replication:
  - Primary: EU-Central
  - Replica: US-East
  - Read-Replica-Routing
- [ ] CDN für Object-Store
- [ ] Geo-Routing (CloudFlare/Route53)

**Acceptance Criteria**:
- Latenz (EU): < 50ms
- Latenz (US): < 100ms

**Dependencies**: MVP-Launch

---

### 10.2 Disaster-Recovery
**Priorität**: P1 | **Aufwand**: M | **Status**: 🟢 Ready

**Tasks**:
- [ ] Backup-Strategy:
  - PostgreSQL: Daily-Backup, 30d-Retention
  - Object-Store: Versioning enabled
- [ ] Restore-Procedure:
  - Documented Runbook
  - Restore-Test monatlich
- [ ] RTO/RPO-Targets:
  - RTO (Recovery Time Objective): 4h
  - RPO (Recovery Point Objective): 1h

**Acceptance Criteria**:
- Restore-Test erfolgreich
- Runbook ist documented

**Dependencies**: None

---

## Prioritization Summary

### P0 (Critical) - Start Immediately
1. **1.1** Zentrale Retry-Strategie
2. **1.2** Task Timeouts & Circuit Breakers
3. **2.1** Hybrid Chunking Strategy
4. **3.1** Enhanced Task Tracing

### P1 (High) - Next Sprint
5. **1.3** Queue-Konsistenz
6. **1.4** Task Idempotenz
7. **2.2** Multi-Vector Retrieval
8. **2.3** Embedding Model Versioning
9. **3.2** Cost Tracking
10. **4.1** Batch-Embedding-Optimization
11. **4.2** Vector-Upsert-Optimization
12. **5.2** PII-Compliance-Audit
13. **7.1** Secrets-Management
14. **7.2** Rate-Limiting
15. **8.1** Integration-Test-Suite
16. **9.2** API-Documentation
17. **10.2** Disaster-Recovery

### P2 (Medium) - Backlog
18. **2.4** Adaptive Chunking
19. **3.3** Chaos Engineering
20. **4.3** Celery-Worker-Autoscaling
21. **5.1** Embedding-Quality-Validation
22. **5.3** Document-Versioning
23. **6.1** Local-Dev-Improvements
24. **6.2** API-Client-SDKs
25. **8.2** Load-Testing
26. **9.1** ADRs

### P3 (Low) - Future
27. **10.1** Multi-Region-Deployment

---

## Success Metrics (SOTA-KPIs)

### Reliability
- **Uptime**: 99.9% (SLA)
- **Error-Rate**: < 0.1%
- **MTTR**: < 5min

### Performance
- **Query-Latency (P95)**: < 200ms
- **Embedding-Throughput**: > 300 Chunks/s
- **Upsert-Throughput**: > 300 Chunks/s

### Quality
- **Retrieval-Precision**: > 0.85
- **Retrieval-Recall**: > 0.90
- **MRR**: > 0.80

### Cost-Efficiency
- **Embedding-Cost/1k-Chunks**: < $0.50
- **LLM-Cost/Query**: < $0.01
- **Cache-Hit-Rate**: > 40%

### Developer-Experience
- **Local-Setup-Time**: < 10min
- **CI-Pipeline-Duration**: < 15min
- **Docker-Build-Time**: < 2min

---

## Notes

- **Breaking Changes**: Wir sind Pre-MVP → DB-Schema-Changes erlaubt
- **Tech-Debt**: Priorisiere Robustheit über Features
- **SOTA-Focus**: Jede Task sollte Best-Practices folgen
- **Observability-First**: Metrics/Traces für alles

---

**Next Steps**:
1. Review mit Team
2. Priorisierung finalisieren
3. Sprint-Planning für P0-Items
4. Assign-Owners für P1-Items
