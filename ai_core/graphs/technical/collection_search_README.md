# Collection Search Graph - Dokumentation

## Überblick

Der `collection_search`-Graph koordiniert Query-Expansion, Web-Suche,
Embedding-basiertes Ranking und optionale automatische Ingestion für beliebige
Dokumentationstypen innerhalb eines Tenants. Er erzeugt deterministische Telemetrie
pro Knoten für Observability (Langfuse).

## Aktiver Execution-Flow

Die **aktuelle** Knotenfolge lautet:

### 1. `k_generate_strategy` - Query-Expansion

- **LLM-basiert**: Nutzt das `analyze`-Label für strukturierte JSON-Generierung
- **Output**: 3-5 Query-Varianten
- **Fallback**: Deterministischer Fallback bei LLM-Fehlern
- **Telemetrie**: Latenz, Token-Usage, Kosten

**Beispiel**:
```python
Input:  "TypeScript documentation"
Output: [
    "TypeScript documentation admin guide",
    "TypeScript documentation telemetry",
    "TypeScript documentation reporting"
]
```

### 2. `k_parallel_web_search` - Web-Suche

- **Parallel**: Führt alle generierten Queries parallel aus
- **Aggregation**: Dedupliziert Ergebnisse nach URL
- **Provider**: `WebSearchWorker` (konfigurierbar)
- **Telemetrie**: Query-Count, Result-Count, Latenz

### 3. `k_embedding_rank` - Embedding-basiertes Ranking

**Seit v2.1** - Ersetzt das teure LLM-basierte Hybrid-Reranking.

#### Score-Berechnung

```python
# Embedding-Similarity (0-100)
embedding_score = cosine_similarity(query_embedding, result_embedding) * 100

# Generische Heuristiken (0-100)
heuristic_score = (
    title_relevance_score +      # 0-30 Punkte
    snippet_quality_score +       # 0-25 Punkte
    query_coverage_score +        # 0-20 Punkte
    url_quality_penalty           # 0 to -20 Punkte
)

# Hybrid-Score
final_score = (0.6 * embedding_score) + (0.4 * heuristic_score)
```

#### Kostenersparnis

- **Alter Hybrid-Rerank**: ~$0.02-0.05 pro Query
- **Embedding-Rank**: ~$0.0004 pro Query
- **Ersparnis**: ~98%

#### Output

- Top-K Ergebnisse (Standard: 20)
- Jedes Ergebnis enthält:
  - `embedding_rank_score` (Hybrid)
  - `embedding_similarity` (Cosine)
  - `heuristic_score` (Heuristics)

### 4. `k_auto_ingest` - Automatische Ingestion (Optional)

**Aktivierung**: `auto_ingest=True` im GraphInput

#### Score-basierte Filterung

```python
# Primärer Threshold
filtered = [r for r in results if r.embedding_rank_score >= min_score]  # Default: 60

# Fallback: Wenn < 3 Ergebnisse
if len(filtered) < 3 and min_score > 50:
    filtered = [r for r in results if r.embedding_rank_score >= 50]

# Error: Wenn keine Ergebnisse >= 50
if not filtered:
    raise QualityThresholdError()
```

#### Top-K Selection

- Nimmt die besten `auto_ingest_top_k` Ergebnisse (Standard: 10)
- Triggert Crawler-Ingestion für diese URLs
- Outcome: `auto_ingest_triggered` oder `auto_ingest_failed_quality_threshold`

## Nicht verwendete HITL-Nodes

Die folgenden Nodes sind **implementiert aber aktuell nicht im Execution-Flow**:

### `_k_hitl_gate` - HITL-Approval-Gateway

- Erstellt Review-Payload für menschliche Freigabe
- Auto-Approve nach 24h Deadline
- Status: `pending` | `approved` | `rejected` | `partial`

### `_k_trigger_ingestion` - Manuelle Ingestion-Trigger

- Nimmt freigegebene URLs aus HITL-Decision
- Triggert Crawler für approved + manually added URLs
- **Ersetzt durch**: `k_auto_ingest` (automatisch)

### `_k_verify_coverage` - Coverage-Verifikation

- Pollt Coverage-Status nach Ingestion
- Timeout: 10 Minuten, Interval: 30 Sekunden
- Outcome: Success-Ratio, Failed-Count, Pending-Count

**Reaktivierung**: Diese Nodes können bei Bedarf in den Execution-Flow eingefügt
werden, wenn manuelle Review-Workflows (HITL = Human-in-the-Loop) erforderlich sind.

## GraphInput-Parameter

```python
class GraphInput(BaseModel):
    """Validated input payload for the collection search graph."""

    question: str                      # Suchanfrage (min. 1 Zeichen)
    collection_scope: str              # Target-Collection-ID
    quality_mode: str = "standard"     # Quality-Modus (standard | premium)
    max_candidates: int = 20           # Max. Suchergebnisse (5-40)
    purpose: str                       # Zweck (z.B. "collection_search")

    # Auto-Ingest (seit v2.1)
    auto_ingest: bool = False                 # Automatische Ingestion aktivieren
    auto_ingest_top_k: int = 10              # Max. URLs für Ingestion (1-20)
    auto_ingest_min_score: float = 60.0      # Mindest-Score (0-100, Fallback: 50)
```

## UI-Integration

Die RAG-Tools-UI (`theme/templates/theme/rag_tools.html`) bietet:

### Auto-Ingest-Checkbox

```html
<input type="checkbox" id="auto-ingest-toggle" />
Auto-Ingest
```

- **Disabled by default**: Verhindert unbeabsichtigte Ingestion
- **Toggle-Funktionalität**: Aktiviert/deaktiviert Optionen-Controls

### Konfigurationsoptionen

1. **Min. Score** (50-100, Schritte: 5)
   - Standard: 60
   - Beschreibung: "Score ≥60, Fallback ≥50"

2. **Top-K** (1-20)
   - Standard: 10
   - Beschreibung: "Max. URLs für Crawler"

### Pipeline-Visualisierung

Echtzeitanzeige aller Graph-Nodes:
- ✅ Strategy Generation → ⏱️ 1.2s, $0.003
- ✅ Web Search → ⏱️ 2.5s
- ✅ Embedding Rank → ⏱️ 1.9s, $0.0004
- ✅ Auto-Ingest → ⏱️ 0.2s

## Outcomes

| Outcome | Beschreibung | Ingestion |
|---------|--------------|-----------|
| `search_completed` | Standard-Flow ohne Auto-Ingest | ❌ |
| `auto_ingest_triggered` | Ingestion erfolgreich gestartet | ✅ |
| `auto_ingest_trigger_failed` | Crawler-API-Fehler | ❌ |
| `auto_ingest_failed_quality_threshold` | Keine Ergebnisse ≥ 50 | ❌ |
| `search_failed` | Web-Search-Fehler | ❌ |
| `no_candidates` | Keine Suchergebnisse | ❌ |

## Tests

Die Tests in `ai_core/tests/graphs/test_collection_search_graph.py` validieren:

### End-to-End-Tests
- ✅ `test_run_returns_search_results` - Standard-Flow
- ✅ `test_search_failure_aborts_flow` - Error-Handling
- ✅ `test_search_without_results_returns_no_candidates` - Empty-Results

### Auto-Ingest-Tests
- ✅ `test_auto_ingest_disabled_by_default` - Default-Verhalten
- ✅ `test_auto_ingest_triggers_with_high_scores` - High-Score-Szenario
- ✅ `test_auto_ingest_fallback_to_lower_threshold` - Fallback-Logik
- ✅ `test_auto_ingest_fails_with_insufficient_quality` - Quality-Threshold-Error

## Beispiel-Verwendung

### Manueller Workflow (HITL)

```python
graph_state = {
    "question": "TypeScript documentation",
    "collection_scope": "software_docs",
    "purpose": "collection_search",
    "auto_ingest": False,  # User wählt manuell aus
}

state, result = graph.run(graph_state)

# User sieht Ergebnisse mit Scores
# User wählt URLs aus
# User klickt "Crawl Selected"
```

### Automatischer Workflow (Auto-Ingest)

```python
graph_state = {
    "question": "TypeScript documentation",
    "collection_scope": "software_docs",
    "purpose": "collection_search",
    "auto_ingest": True,              # Automatisch
    "auto_ingest_top_k": 10,          # Max 10 URLs
    "auto_ingest_min_score": 60.0,    # Min Score 60
}

state, result = graph.run(graph_state)

# result["outcome"] == "auto_ingest_triggered"
# result["ingestion"]["count"] == 10
# Crawler läuft automatisch
```

## Migration von Hybrid-Reranking

**Vor v2.1** (teuer):
```
Strategy → Search → Hybrid-Rerank (LLM) → HITL → Ingestion
                    ^^^^^^^^^^^^^^
                    $0.02-0.05 pro Query
```

**Ab v2.1** (günstig):
```
Strategy → Search → Embedding-Rank → Auto-Ingest (optional)
                    ^^^^^^^^^^^^^^^^
                    $0.0004 pro Query (98% Ersparnis)
```

## Langfuse-Telemetrie

Jeder Node emittiert strukturierte Spans mit:

```python
{
    "node_name": "k_embedding_rank",
    "latency_ms": 1900,
    "input_count": 31,
    "ranked_count": 31,
    "top_k_count": 20,
    "avg_embedding_score": 45.7,
    "avg_heuristic_score": 30.6,
    "cost_usd": 0.0004,
}
```

## Fehlerbehandlung

### LLM-Fehler (Strategy-Generation)
- **Fallback**: Deterministischer Fallback ohne LLM
- **Queries**: Query + Purpose + Variations
- **Outcome**: `search_completed` (mit Fallback-Flag)

### Web-Search-Fehler
- **Outcome**: `search_failed`
- **Payload**: `search.errors` mit Details
- **Retry**: Keine automatischen Retries

### Embedding-Fehler
- **Fallback**: Score = 0 für betroffene Ergebnisse
- **Logging**: Warning mit Exception-Details
- **Outcome**: `search_completed` (degradiert)

### Auto-Ingest-Fehler
- **Crawler-Fehler**: `auto_ingest_trigger_failed`
- **Quality-Threshold**: `auto_ingest_failed_quality_threshold`
- **Logging**: Error-Level mit Context

## Performance-Optimierungen

1. **Parallel Web Search**: Alle Queries parallel
2. **Embedding Batch**: Ein API-Call für alle Ergebnisse
3. **Top-K Early Exit**: Stoppt nach K Ergebnissen
4. **Heuristic Caching**: Wiederverwendung von Scores
5. **Telemetrie**: Minimal-Overhead (<10ms)

## Zukünftige Erweiterungen

### Geplant
- [ ] Adaptive Score-Threshold (ML-basiert)
- [ ] Multi-Model-Embedding (Ensemble)
- [ ] Incremental Crawling (nur neue URLs)
- [ ] Coverage-Tracking Dashboard

### Optional
- [ ] HITL-Reaktivierung für regulierte Branchen
- [ ] Hybrid-Rerank als Premium-Option
- [ ] Custom Heuristics per Tenant
- [ ] A/B-Testing Framework

## Weitere Informationen

- **Architektur**: [docs/architektur/overview.md](../../docs/architektur/overview.md)
- **Tool-Contracts**: [docs/agents/tool-contracts.md](../../docs/agents/tool-contracts.md)
- **RAG-Architektur**: [docs/rag/overview.md](../../docs/rag/overview.md)
- **Embedding-Profile**: [docs/rag/configuration.md](../../docs/rag/configuration.md)
