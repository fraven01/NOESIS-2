# Crawler – Überblick

Der Crawler übernimmt die kontinuierliche Synchronisation externer Quellen und
liefert normalisierte Dokumente an die RAG-Ingestion. Dieses Dokument fasst
Pipeline, Kernverträge und Betriebsschalter zusammen und orientiert sich am
Aufbau der übrigen App-Dokumentationen.

## Zweck
- Erstellt deterministische Frontier-Entscheidungen auf Basis von Robots-,
  Politeness- und Failure-Signalen, damit Quellen nur dann besucht werden, wenn
  es Policies erlauben und Recrawl-Intervalle eingehalten werden.
- Streamt Inhalte über den HTTP-Fetcher ein, erzwingt Sicherheits-Limits und
  mappt Response-Metadaten in strukturierte Fetch-Ergebnisse.
- Normalisiert Parser-Ausgaben zu konsistenten Dokument-Artefakten, berechnet
  Delta- und Near-Duplicate-Signaturen und liefert Basis-Metadaten an den
  zentralen Dedup-Service. Skip- oder Replace-Entscheidungen erfolgen erst im
  `ai_core.rag.vector_client`.
- Übergibt finale Payloads an die Ingestion, aktualisiert Lifecycle-Timelines
  und emittiert Telemetrie laut [Crawler Observability](../observability/crawler-langfuse.md).

## Pipeline
```mermaid
flowchart TD
    F[Frontier] --> G[Guardrails]
    G --> H[HTTP Fetch]
    H --> P[Parser]
    P --> N[Normalizer]
    N --> D[Delta]
    D --> I[Ingestion]
    I --> L[Lifecycle]
```

- **Frontier (`crawler.frontier.decide_frontier_action`)** bewertet Robots,
  Host-Politeness und Recrawl-Signale. Rückgaben wie `enqueue`, `defer`, `skip`
  oder `retire` enthalten Gründe und optionale Policy-Events, um Scheduling und
  Blacklisting nachvollziehbar zu machen.【F:crawler/frontier.py†L14-L122】【F:crawler/frontier.py†L164-L218】
- **Guardrails (`crawler.guardrails.enforce_guardrails`)** prüfen Host- und
  Tenant-Quoten, MIME-Blacklists sowie Dokumentgrößen, bevor Fetch und Parser
  Ressourcen verbrauchen. Deny-Entscheidungen liefern einen `CrawlerError` mit
  konsistenter Error-Class.【F:crawler/guardrails.py†L7-L140】【F:crawler/guardrails.py†L170-L232】
- **HTTP Fetch (`crawler.http_fetcher.HttpFetcher`)** kapselt Streaming-Fetches
  mit Retry-Policy, Politeness-Delays und Limit-Prüfungen für Bytes, Timeout und
  MIME-Whitelist. Response-Details werden in `FetchResult` und
  `FetchTelemetry` gespiegelt.【F:crawler/http_fetcher.py†L1-L126】【F:crawler/fetcher.py†L1-L119】
- **Parser & Normalizer (`crawler.normalizer.build_normalized_document`)**
  erzeugen `NormalizedDocument` inklusive Meta-, Content- und
  External-Referenzen. Parser-Statistiken werden für Downstream-Systeme
  konserviert.【F:crawler/normalizer.py†L1-L129】【F:crawler/normalizer.py†L153-L229】
- **Delta (`crawler.delta.evaluate_delta`)** berechnet Content-Hashes und
  Near-Duplicate-Signaturen, vergleicht Vorgängerversionen und liefert die
  Metadaten an den gemeinsamen Dedup-Service. Skip- oder Replace-Aktionen
  passieren downstream im `vector_client`.【F:crawler/delta.py†L1-L114】【F:ai_core/rag/vector_client.py†L60-L220】
- **Ingestion (`crawler.ingestion.build_ingestion_decision`)** erzeugt strukturierte
  Payloads mit Tenant-, Workflow- und Content-Metadaten. Ergebnisse werden als
  `upsert`, `skip` oder `retire` markiert und enthalten Lifecycle-State sowie
  Policy-Events.【F:crawler/ingestion.py†L1-L129】【F:crawler/ingestion.py†L137-L215】
- **Lifecycle (`crawler.lifecycle_model.LifecycleTimeline`)** stellt sicher,
  dass Status-Übergänge nur entlang der erlaubten Sequenzen erfolgen und bietet
  Telemetrie-Attribute pro Ereignis.【F:crawler/lifecycle_model.py†L1-L123】【F:crawler/lifecycle_model.py†L129-L208】

## Kernverträge & Artefakte
| Modul | Verantwortung | Schlüsselklassen |
| --- | --- | --- |
| `crawler.frontier` | Robots-Compliance, Recrawl-Intervalle, Failure-Backoff | `FrontierDecision`, `RobotsPolicy`, `HostPolitenessPolicy` |
| `crawler.guardrails` | Tenant/Host-Quoten, MIME- und Host-Blocklisten | `GuardrailLimits`, `GuardrailSignals`, `GuardrailDecision` |
| `crawler.fetcher` | Kanonischer Fetch-Contract inkl. Limits und Telemetrie | `FetchRequest`, `FetchResult`, `FetcherLimits` |
| `crawler.http_fetcher` | Streaming-HTTP-Client mit Retries und User-Agent-Steuerung | `HttpFetcher`, `HttpFetcherConfig`, `FetchRetryPolicy` |
| `crawler.normalizer` | Normalisierte Dokumente und Provider-Referenzen | `NormalizedDocument` (mit `documents.contracts.DocumentMeta`, `ProviderReference`) |
| `crawler.delta` | Hashing & Near-Duplicate-Detektion | `DeltaDecision`, `DeltaSignatures`, `NearDuplicateSignature` |
| `crawler.ingestion` | Übergabe an RAG-Ingestion & Lifecycle | `IngestionStatus`, `ChunkMeta`, `DefaultCrawlerIngestionAdapter` |
| `crawler.errors` | Vereinheitlichtes Fehler-Vokabular | `CrawlerError`, `ErrorClass` |

## Normalisierung & Delta
- Parser-Ergebnisse müssen `ParserContent` liefern; ohne Text wird ein
  `binary_payload_ref` erwartet, ansonsten schlägt der Normalizer fehl. Das
  garantiert, dass entweder Text oder Binärdaten für Hashing und Ingestion
  vorhanden sind.【F:crawler/normalizer.py†L73-L125】
- Parser-Statistiken werden in `NormalizedDocument.parser_stats`
  übernommen. Der Normalizer ergänzt eigene Kennzahlen wie
  `normalizer.bytes_in`, damit Langfuse und Dead-Letter-Payloads denselben
  Zahlenraum teilen.【F:crawler/normalizer.py†L153-L214】
- Delta-Bewertungen nutzen `evaluate_delta` und speichern Content-Hashes sowie
  Near-Duplicate-Signaturen für spätere Vergleiche. Die tatsächliche
  Skip/Replace-Logik liegt im gemeinsamen Dedup-Service (`match_near_duplicate`)
  des Vector-Clients.【F:crawler/delta.py†L70-L114】【F:ai_core/rag/vector_client.py†L60-L220】

## Ingestion, Retire & Lifecycle
- `build_ingestion_decision` kombiniert Normalizer-Output, Delta-Status und
  optionale Lifecycle-Regeln. Statt eigener Payload-Klassen liefert die
  Entscheidung heute ein generisches `Decision`-Objekt mit validiertem
  `ChunkMeta` und einem Adapter-Mapping für Provider-spezifische Felder.
  Retire-Entscheidungen referenzieren dieselben Metadaten, sodass
  Downstream-Systeme ohne Sonderpfad auf ai_core-Ingestion-Services
  zugreifen können.【F:crawler/ingestion.py†L130-L219】
- Lifecycle-Regeln stammen aus `crawler.retire` und `crawler.lifecycle_model`.
  `LifecycleTimeline` erzwingt Sequenzen von `seeded` bis zu terminalen
  Statuswerten (`ingested`, `skipped`, `retired`) und verhindert unzulässige
  Sprünge.【F:crawler/lifecycle_model.py†L1-L123】【F:crawler/lifecycle_model.py†L129-L208】
- Fehler oder Policy-Denies werden über `CrawlerError` in Events gespiegelt und
  nutzen die gemeinsame Error-Class-Taxonomie (`timeout`, `rate_limit`,
  `policy_deny`, …). Das stellt sicher, dass Langfuse und Dead-Letter-Queues
  dieselbe Semantik verwenden.【F:crawler/errors.py†L1-L41】

## Konfiguration & Betriebsschalter
- **User Agent**: `CRAWLER_HTTP_USER_AGENT` kann in Django-Settings oder via
  Environment überschrieben werden. Fallback ist `noesis-crawler/1.0`.【F:noesis2/settings/base.py†L202-L202】【F:crawler/http_fetcher.py†L13-L45】
- **Fetcher Limits**: `FetcherLimits` decken Bytes-Limits, Timeouts und
  MIME-Whitelists ab. Violations werden als Policy-Events zurückgegeben und
  führen zu `FetchStatus.POLICY_DENIED`.【F:crawler/fetcher.py†L69-L119】
- **Retry-Policy**: `FetchRetryPolicy` steuert Anzahl Versuche, Backoff und
  Fehlergründe (HTTP 429/5xx, Netzwerkfehler). Backoff und Jitter sind pro
  Versuch berechenbar und werden in Telemetrie gespiegelt.【F:crawler/http_fetcher.py†L47-L106】
- **Guardrails**: `GuardrailLimits` erlauben Quoten pro Tenant oder Host,
  blocken MIME-Typen/Hosts und begrenzen Prozessdauer sowie Dokumentgröße.
  Überschreitungen erzeugen deterministische Policy-Events.【F:crawler/guardrails.py†L33-L140】【F:crawler/guardrails.py†L170-L232】
- **Recrawl-Intervalle**: `RecrawlFrequency` und `RECRAWL_INTERVALS` definieren
  stündliche bis wöchentliche Frequenzen und berücksichtigen Observed-Change- und
  Manual-Override-Signale.【F:crawler/frontier.py†L55-L115】

## Telemetrie & Fehlerhandhabung
- Alle Stufen liefern `policy_events` und optionale `CrawlerError`-Payloads, die
  direkt in Langfuse-Traces und Dead-Letter-Events übernommen werden. Sie
  korrespondieren mit den Pflichtfeldern aus dem Observability-Leitfaden.【F:crawler/fetcher.py†L121-L152】【F:crawler/ingestion.py†L1-L129】
- `FetchTelemetry` speichert Latenz, Bytes und Retry-Gründe. Die Werte fließen in
  Metrics (`crawler_fetch_latency_ms`, `crawler_fetch_bytes_total`) ein und
  werden von Guardrails genutzt, um Backoff-Strategien zu begründen.【F:crawler/fetcher.py†L81-L119】【F:docs/observability/crawler-langfuse.md†L9-L41】
- Lifecycle-Events enthalten Zeitstempel und optionale Dauer in Millisekunden.
  Dadurch lassen sich SLA-Verstöße pro Stage erkennen und Alerts konfigurieren.【F:crawler/lifecycle_model.py†L129-L208】

## Erweiterungshinweise
- Neue Provider sollten `NormalizedSource.provider_tags` und `ExternalDocumentReference`
  nutzen, um kanonische IDs zu übertragen. Zusätzliche Tags werden unverändert an
  die Ingestion übergeben.【F:crawler/normalizer.py†L17-L63】【F:crawler/ingestion.py†L168-L215】
- Weitere Guardrails lassen sich über `GuardrailLimits` erweitern; bei neuen
  Violations immer einen passenden `CrawlerError` mit eindeutiger
  `ErrorClass`-Zuordnung ausgeben.【F:crawler/guardrails.py†L170-L232】【F:crawler/errors.py†L1-L41】
- Für spezialisierte Recrawl-Logik kann `CrawlSignals.override_recrawl_frequency`
  befüllt werden, ohne die Standardintervalle hart zu ändern.【F:crawler/frontier.py†L67-L114】
