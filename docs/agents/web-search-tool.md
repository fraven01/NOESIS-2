# Web Search Tool

## Überblick

Das Web Search Tool (`WebSearchWorker`) bietet eine abstrahierte Schnittstelle für externe Web-Suchprovider mit automatischer Retry-Logik, Deduplication, URL-Normalisierung und vollständiger Observability-Integration. Aktuell wird der Google Custom Search Adapter unterstützt.

**Standort**: [`ai_core/tools/web_search.py`](../../ai_core/tools/web_search.py)

**Adapter**: [`ai_core/tools/search_adapters/google.py`](../../ai_core/tools/search_adapters/google.py)

## Architektur

### Komponenten

```
WebSearchWorker
    ↓ verwendet
BaseSearchAdapter (Protocol)
    ↓ implementiert von
GoogleSearchAdapter
    ↓ ruft auf
Google Custom Search JSON API
```

### Layer-Aufteilung

1. **Worker-Layer** (`WebSearchWorker`): Orchestrierung, Validation, Retry, Telemetrie
2. **Adapter-Layer** (`BaseSearchAdapter`): Provider-spezifische HTTP-Logik
3. **Provider-Layer**: Externe API (Google Custom Search, Bing, etc.)

## Verträge

### ToolContext

Runtime metadata for each web search invocation (ToolContext).

Required fields:
- scope.tenant_id
- scope.trace_id
- scope.invocation_id
- run_id and/or ingestion_run_id

Optional fields:
- business.workflow_id, business.case_id
- metadata.worker_call_id (generated if missing)

Example:

```python
from ai_core.contracts import BusinessContext, ScopeContext
from ai_core.tool_contracts import ToolContext

scope = ScopeContext(
    tenant_id="acme",
    trace_id="trace-a1b2c3",
    invocation_id="123e4567-e89b-12d3-a456-426614174000",
    run_id="run_def456",
)
business = BusinessContext(workflow_id="universal_ingestion", case_id="crm-7421")
context = ToolContext(scope=scope, business=business)
```

### WebSearchInput

Validiertes Eingabe-Modell mit Query-Normalisierung.

| Feld | Typ | Pflicht | Beschreibung |
|------|-----|---------|--------------|
| `query` | `str` | ja | Suchanfrage (min. 1 Zeichen, wird normalisiert) |

**Validierung**:

- Entfernt Zero-Width-Zeichen (Unicode Category `Cf`)
- Trimmt Whitespace
- Normalisiert mehrfache Spaces zu einem
- Blockiert Operator-Only-Queries (z. B. `site:` ohne Wert)

**Beispiel**:

```python
search_input = WebSearchInput(query="aktuelle Entwicklungen KI-Regulierung")
```

### WebSearchResponse

Envelope-Modell für Suchergebnisse mit Outcome-Metadaten.

| Feld | Typ | Beschreibung |
|------|-----|--------------|
| `results` | `list[SearchResult]` | Deduplizierte, validierte Suchergebnisse |
| `outcome` | `ToolOutcome` | Decision (`ok`/`error`), Rationale, Metadaten |

#### SearchResult

| Feld | Typ | Beschreibung |
|------|-----|--------------|
| `url` | `HttpUrl` | Bereinigter URL (ohne Tracking-Parameter) |
| `title` | `str` | Ergebnis-Titel (nicht leer) |
| `snippet` | `str` | Text-Snippet (nicht leer) |
| `source` | `str` | Provider-Name (z. B. `google`) |
| `score` | `float \| None` | Relevanz-Score (optional) |
| `is_pdf` | `bool` | True, falls PDF-Dokument erkannt wurde |

#### ToolOutcome

| Feld | Typ | Beschreibung |
|------|-----|--------------|
| `decision` | `str` | `ok` oder `error` |
| `rationale` | `str` | Grund (z. B. `search_completed`, `provider_timeout`) |
| `meta` | `dict[str, object]` | Metadaten (Latenz, HTTP-Status, Quota, Error-Details) |

**Outcome Meta-Felder**:

- `tenant_id`, `trace_id`, `workflow_id`, `case_id`, `run_id`, `worker_call_id`
- `provider`: Provider-Name (z. B. `google`)
- `latency_ms`: Dauer in Millisekunden
- `http_status`: HTTP-Statuscode des Providers
- `result_count`, `raw_result_count`, `normalized_result_count`
- `quota_remaining`: Verbleibende API-Quota (falls vom Provider geliefert)
- `error`: Strukturierte Fehler-Details (bei `decision: "error"`)

**Beispiel (Erfolg)**:

```json
{
  "results": [
    {
      "url": "https://example.com/article",
      "title": "KI-Regulierung: Neue EU-Verordnung",
      "snippet": "Die EU hat neue Richtlinien...",
      "source": "google",
      "score": null,
      "is_pdf": false
    }
  ],
  "outcome": {
    "decision": "ok",
    "rationale": "search_completed",
    "meta": {
      "tenant_id": "acme",
      "trace_id": "trace-a1b2c3",
      "provider": "google",
      "latency_ms": 245,
      "http_status": 200,
      "result_count": 1,
      "raw_result_count": 10,
      "normalized_result_count": 1
    }
  }
}
```

**Beispiel (Fehler)**:

```json
{
  "results": [],
  "outcome": {
    "decision": "error",
    "rationale": "provider_rate_limited",
    "meta": {
      "tenant_id": "acme",
      "trace_id": "trace-a1b2c3",
      "provider": "google",
      "latency_ms": 102,
      "http_status": 429,
      "result_count": 0,
      "raw_result_count": 0,
      "normalized_result_count": 0,
      "error": {
        "kind": "SearchProviderQuotaExceeded",
        "message": "Rate limit exceeded",
        "retry_in_ms": 1200
      }
    }
  }
}
```

## Worker-Konfiguration

### WebSearchWorker Constructor

```python
WebSearchWorker(
    adapter: BaseSearchAdapter,
    *,
    max_results: int = 10,
    max_attempts: int = 3,
    backoff_factor: float = 0.6,
    oversample_factor: int = 2,
    sleep: Callable[[float], None] | None = None,
    timer: Callable[[], float] | None = None,
    logger: logging.Logger | None = None,
)
```

**Parameter**:

- `adapter`: Provider-Adapter (z. B. `GoogleSearchAdapter`)
- `max_results`: Maximale Anzahl zurückgegebener Ergebnisse (nach Deduplizierung)
- `max_attempts`: Anzahl Wiederholungsversuche bei transienten Fehlern
- `backoff_factor`: Exponential-Backoff-Faktor (in Sekunden)
- `oversample_factor`: Faktor für Oversampling vor Deduplication (Standard: 2x)
- `sleep`, `timer`, `logger`: Dependency-Injection für Tests

### Retry-Strategie

**Wiederholbare Fehler**:

- `SearchProviderQuotaExceeded` (Rate-Limit)
- `SearchProviderTimeout`

**Nicht wiederholbare Fehler**:

- `SearchProviderBadResponse` (Parsing-Fehler, ungültige Antwort)

**Backoff-Formel**:

```
delay = min(backoff_factor * (2 ** (attempt - 1)), 10.0)
```

Wenn der Provider ein `retry_in_ms` zurückgibt, wird dieser Wert bevorzugt.

**Beispiel**:

- Attempt 1: 0.6s
- Attempt 2: 1.2s
- Attempt 3: 2.4s

## URL-Normalisierung

Der Worker entfernt automatisch Tracking-Parameter und normalisiert URLs für Deduplication.

**Entfernte Parameter**:

- Präfix `utm_*` (Google Analytics)
- `gclid`, `fbclid`, `igshid`, `msclkid`, `mc_eid`, `vero_conv`, `vero_id`, `yclid`

**Normalisierung**:

- Scheme auf Lowercase (`HTTPS` → `https`)
- Host auf Lowercase (`Example.COM` → `example.com`)
- Query-Parameter sortiert
- Fragment entfernt

**Beispiel**:

```
Input:  https://Example.COM/page?utm_source=fb&b=2&a=1#section
Output: https://example.com/page?a=1&b=2
```

## PDF-Erkennung

PDFs werden automatisch erkannt via:

1. URL-Extension: `.pdf` (case-insensitive)
2. Content-Type: `application/pdf` oder `*pdf*`

**Verwendung**: Downstream-Flows können PDF-Treffer für spezielle Verarbeitung (OCR, Layout-Analyse) filtern.

## Adapter-Protokoll

### BaseSearchAdapter (Protocol)

Minimale Schnittstelle für Provider-Implementierungen.

```python
class BaseSearchAdapter(Protocol):
    provider_name: str

    def search(self, query: str, *, max_results: int) -> SearchAdapterResponse:
        """Execute search request against provider."""
```

**Hinweis**: Der Worker unterstützt sowohl `max_results` als auch `limit` als Keyword-Argument (automatische Erkennung via `inspect.signature`).

### SearchAdapterResponse

```python
@dataclass
class SearchAdapterResponse:
    results: Sequence[ProviderSearchResult]
    status_code: int
    raw_results: Sequence[RawSearchResult] = ()
    quota_remaining: int | None = None
```

### ProviderSearchResult

```python
@dataclass
class ProviderSearchResult:
    url: str
    title: str
    snippet: str
    source: str
    score: float | None = None
    content_type: str | None = None
```

## Verwendungsbeispiel

### Standalone

```python
from ai_core.contracts import BusinessContext, ScopeContext
from ai_core.tool_contracts import ToolContext
from ai_core.tools.web_search import WebSearchWorker
from ai_core.tools.search_adapters.google import GoogleSearchAdapter

# Setup
adapter = GoogleSearchAdapter(
    api_key="YOUR_API_KEY",
    search_engine_id="YOUR_ENGINE_ID",
)
worker = WebSearchWorker(
    adapter=adapter,
    max_results=5,
    max_attempts=3,
)

# Execute
scope = ScopeContext(
    tenant_id="acme",
    trace_id="trace-123",
    invocation_id="123e4567-e89b-12d3-a456-426614174000",
    run_id="run_abc",
)
business = BusinessContext(
    workflow_id="universal_ingestion",
    case_id="crm-7421",
)
context = ToolContext(scope=scope, business=business)
response = worker.run(
    query="aktuelle KI-Regulierung EU",
    context=context,
)

# Process results
if response.outcome.decision == "ok":
    for result in response.results:
        print(f"{result.title}: {result.url}")
else:
    print(f"Error: {response.outcome.rationale}")
    print(response.outcome.meta.get("error"))
```

### In LangGraph

Der Worker wird in LangGraph-Nodes verwendet (z. B. `universal_ingestion_graph`).

**Beispiel-Node**:

```python
from ai_core.tools.web_search import WebSearchWorker

def web_search_node(state: GraphState) -> dict:
    """Execute web search and return results."""
    context = extract_context_from_state(state)

    response = worker.run(
        query=state["search_query"],
        context=context,
    )

    if response.outcome.decision == "ok":
        return {"search_results": response.results}
    else:
        # Handle error
        return {"search_error": response.outcome.rationale}
```

## Observability

### Langfuse Spans

Jeder Aufruf emittiert einen `tool.web_search`-Span mit:

**Span Attributes**:

- `tenant_id`, `trace_id`, `workflow_id`, `case_id`, `run_id`, `worker_call_id`
- `provider`: Provider-Name
- `query`: Suchanfrage
- `http.status`: HTTP-Statuscode
- `result.count`, `raw_result_count`, `normalized_result_count`
- `quota.remaining`: Verbleibende API-Quota (falls vorhanden)
- `error.kind`, `error.message`: Bei Fehlern

**Query-Parameter**: Der volle Query-String wird geloggt (keine PII-Maskierung auf Tool-Ebene; erfolgt in übergeordneten Layern).

### Strukturierte Logs

Der Worker verwendet `logging.Logger` für Debug-Meldungen:

```python
logger.debug("dropping invalid search result", extra={"payload": payload})
logger.exception("web search worker failed", exc_info=exc)
```

**Best Practice**: Logger mit `structlog` konfigurieren für JSON-Logs.

## Fehler-Typen

### SearchProviderError (Basis)

```python
class SearchProviderError(RuntimeError):
    retry_in_ms: int | None
    http_status: int | None
```

### SearchProviderTimeout

Raised bei Timeout des Providers.

**HTTP-Mapping**: 504 Gateway Timeout

### SearchProviderQuotaExceeded

Raised bei Rate-Limit.

**HTTP-Mapping**: 429 Too Many Requests

**Hinweis**: `retry_in_ms` enthält die vom Provider empfohlene Wartezeit.

### SearchProviderBadResponse

Raised bei ungültiger/nicht-parsebarer Antwort.

**HTTP-Mapping**: 502 Bad Gateway

## Google Search Adapter

### Konfiguration

```python
from ai_core.tools.search_adapters.google import GoogleSearchAdapter

adapter = GoogleSearchAdapter(
    api_key=settings.GOOGLE_API_KEY,
    search_engine_id=settings.GOOGLE_SEARCH_ENGINE_ID,
    timeout=10.0,  # Sekunden
)
```

**Umgebungsvariablen**:

- `GOOGLE_API_KEY`: Google Cloud API Key
- `GOOGLE_SEARCH_ENGINE_ID`: Custom Search Engine ID

### Rate Limits

Google Custom Search API:

- **Free Tier**: 100 Queries/Tag
- **Paid Tier**: 10.000 Queries/Tag (erweiterbarer)

Der Adapter parst `quota_remaining` aus der Response, falls verfügbar.

### Response-Mapping

Google JSON → `ProviderSearchResult`:

```python
{
  "items": [
    {
      "title": "...",
      "link": "https://...",
      "snippet": "...",
      "displayLink": "example.com",
      "mime": "application/pdf"  # optional
    }
  ]
}
```

**Hinweis**: `displayLink` wird als `source` verwendet (Fallback auf `google`).

## Tests

### Unit-Tests

**Standort**: [`ai_core/tests/test_web_search_worker.py`](../../ai_core/tests/test_web_search_worker.py)

**Coverage**:

- Query-Validierung (Normalisierung, Operator-Only-Queries)
- Retry-Logik (Exponential Backoff, max_attempts)
- URL-Deduplication & Tracking-Parameter-Entfernung
- PDF-Erkennung
- Adapter-Fehlerbehandlung (Quota, Timeout, BadResponse)
- Observability (Spans, Outcome-Meta)

### Fake Adapter

Für Tests ohne externe API:

```python
from dataclasses import dataclass
from ai_core.tools.web_search import SearchAdapterResponse, ProviderSearchResult

@dataclass
class FakeSearchAdapter:
    provider_name: str = "fake"

    def search(self, query: str, *, max_results: int) -> SearchAdapterResponse:
        return SearchAdapterResponse(
            results=[
                ProviderSearchResult(
                    url="https://example.com/1",
                    title="Fake Result 1",
                    snippet="This is a test.",
                    source="fake",
                )
            ],
            status_code=200,
        )
```

## Best Practices

### 1. Context-Propagation

Immer vollständigen `ToolContext` übergeben:

```python
# ✅ Good
response = worker.run(query="...", context=full_context)

# ❌ Bad (fehlende Felder)
response = worker.run(query="...", context={"tenant_id": "acme"})
```

### 2. Error-Handling

Prüfe immer `outcome.decision`:

```python
response = worker.run(...)
if response.outcome.decision == "ok":
    # Process results
    pass
else:
    # Log error, emit metric, retry at higher level
    logger.error(
        "web_search.failed",
        rationale=response.outcome.rationale,
        meta=response.outcome.meta,
    )
```

### 3. Quota-Monitoring

Tracke `quota_remaining` für Alerts:

```python
remaining = response.outcome.meta.get("quota_remaining")
if remaining is not None and remaining < 100:
    alert("Google Search quota low", remaining=remaining)
```

### 4. Result-Limits

Setze `max_results` basierend auf Downstream-Kapazität:

```python
# Für LLM-Context: 5-10 Ergebnisse
worker = WebSearchWorker(adapter, max_results=5)

# Für UI-Display: 10-20 Ergebnisse
worker = WebSearchWorker(adapter, max_results=15)
```

### 5. Oversample für Deduplizierung

Standard `oversample_factor=2` ist konservativ. Bei hoher Duplikat-Rate erhöhen:

```python
# High-Duplication-Queries (z. B. News)
worker = WebSearchWorker(adapter, oversample_factor=3)
```

## Erweiterung: Neue Provider

### 1. Adapter implementieren

```python
from ai_core.tools.web_search import (
    BaseSearchAdapter,
    SearchAdapterResponse,
    ProviderSearchResult,
)

class BingSearchAdapter:
    provider_name = "bing"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def search(self, query: str, *, max_results: int) -> SearchAdapterResponse:
        # HTTP-Request an Bing API
        response = requests.get(
            "https://api.bing.microsoft.com/v7.0/search",
            params={"q": query, "count": max_results},
            headers={"Ocp-Apim-Subscription-Key": self.api_key},
        )

        # Parse Response
        data = response.json()
        results = [
            ProviderSearchResult(
                url=item["url"],
                title=item["name"],
                snippet=item["snippet"],
                source="bing",
            )
            for item in data.get("webPages", {}).get("value", [])
        ]

        return SearchAdapterResponse(
            results=results,
            status_code=response.status_code,
        )
```

### 2. Worker konfigurieren

```python
adapter = BingSearchAdapter(api_key=settings.BING_API_KEY)
worker = WebSearchWorker(adapter)
```

### 3. Tests schreiben

```python
def test_bing_adapter_success(monkeypatch):
    # Mock Bing API
    monkeypatch.setattr("requests.get", mock_bing_response)

    adapter = BingSearchAdapter(api_key="test")
    response = adapter.search("test query", max_results=5)

    assert response.status_code == 200
    assert len(response.results) == 5
```

## Migration & Deprecations

### Legacy Protocol-Name

`SearchAdapter` ist ein Alias für `BaseSearchAdapter`:

```python
# ✅ Neu
from ai_core.tools.web_search import BaseSearchAdapter

# ⚠️ Legacy (weiterhin unterstützt)
from ai_core.tools.web_search import SearchAdapter
```

**Grund**: Klarere Benennung (Basis-Protocol vs. konkrete Implementierung).

### Legacy Provider-Attribut

Alte Adapter können `provider()` als Methode haben:

```python
class LegacyAdapter:
    def provider(self) -> str:
        return "legacy"
```

Der Worker erkennt dies automatisch via `getattr(adapter, "provider", None)`.

**Migration**: Wechsle auf `provider_name`-Attribut.

## Verwandte Dokumentation

- [Tool Contracts](./tool-contracts.md) - Generische Tool-Schnittstellen
- [Universal Ingestion Graph](../../ai_core/graph/README.md) - LangGraph-Integration
- [Google Custom Search API](https://developers.google.com/custom-search/v1/overview) - Provider-Doku
- [Observability: Langfuse](../observability/langfuse.md) - Tracing-Instrumentation

## Changelog

| Datum | Version | Änderung |
|-------|---------|----------|
| 2025-11-07 | 1.0 | Initiale Dokumentation |
| 2025-12-19 | 1.1 | Integration in Universal Ingestion & Collection Search |

## Integration

### Universal Technical Graph (`source="search"`)

Der Tool-Worker ist direkt in den `UniversalIngestionGraph` integriert.

- **Mode `acquire_only`**: Führt Suche aus, dedupliziert und speichert Ergebnisse als "staged" Artefakte (Session & Candidates).
- **Mode `acquire_and_ingest`**: Führt Suche aus, selektiert (automatisch oder preselected) und triggert Ingestion für die Gewinner.

### Collection Search Delegation

Der `CollectionSearch` Graph nutzt den Worker indirekt über Delegation:

1. **Planung**: Collection Search führt komplexe Recherche/Scoring durch (Hybrid Search).
2. **Delegation**: Selektierte URLs werden als `preselected_results` an den `UniversalIngestionGraph` übergeben.
3. **Bypass**: Für diese expliziten URLs wird die Snippet-Längen-Prävention im Universal Graph umgangen.
