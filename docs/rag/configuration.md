# RAG Konfiguration

Dieses Dokument beschreibt, wie die RAG-Komponenten (Retrieval-Augmented Generation) konfiguriert werden, insbesondere die Embedding-Modelle und die Vektor-Datenbank.

## Laufzeit-Konfiguration (Umgebungsvariablen)

Einige wichtige Parameter können zur Laufzeit über Umgebungsvariablen (z.B. in einer `.env`-Datei) gesteuert werden. Dies ermöglicht es dem Betrieb (Ops), die Konfiguration ohne Code-Änderungen anzupassen.

### Embedding-Modell & Dimension

Das für die Erstellung von Vektor-Embeddings verwendete Modell sowie dessen Dimensionen werden über Umgebungsvariablen definiert.

| Umgebungsvariable          | Beschreibung                                                                 | Standardwert                  |
| -------------------------- | ----------------------------------------------------------------------------- | ----------------------------- |
| `EMBEDDINGS_MODEL_PRIMARY` | Der Name/Alias des primären Embedding-Modells, das für das "standard" Profil verwendet wird. | `oai-embed-small`             |
| `EMBEDDINGS_DIM`           | Die Anzahl der Dimensionen für das primäre Embedding-Modell.                  | `1536`                        |
| `DEMO_EMBEDDINGS_MODEL`    | Der Modellname für das "demo" Profil.                                        | Wert von `EMBEDDINGS_MODEL_PRIMARY` |
| `DEMO_EMBEDDINGS_DIM`      | Die Anzahl der Dimensionen für das Demo-Modell.                              | Wert von `EMBEDDINGS_DIM`     |

### Vector-Schema Override

| Umgebungsvariable   | Beschreibung                                                                                 | Standardwert |
| ------------------- | --------------------------------------------------------------------------------------------- | ------------ |
| `RAG_VECTOR_SCHEMA` | Überschreibt das Schema des Standard-Vector-Spaces (`RAG_VECTOR_STORES['global']`).            | `rag` (oder Wert von `DEV_TENANT_SCHEMA`) |

### Kontext-Header für Retrieval

Kurze Kontext-Header werden als Präfix in Chunks geschrieben, um Retrieval zu stabilisieren.

| Umgebungsvariable | Beschreibung | Standardwert |
| ----------------- | ------------ | ------------ |
| `RAG_CONTEXT_HEADER_MODE` | Modus für Header-Erzeugung (`off`, `heuristic`, `llm`). | `heuristic` |
| `RAG_CONTEXT_HEADER_MODEL` | MODEL_ROUTING-Label für den LLM-Header (nur bei `llm`). | `fast` |
| `RAG_CONTEXT_HEADER_MAX_CHARS` | Maximale Zeichenlänge des Headers. | `140` |
| `RAG_CONTEXT_HEADER_MAX_WORDS` | Maximale Wortanzahl des Headers. | `14` |

### Contextual Enrichment (Chunk Prefix)

Optionaler LLM-Schritt, der pro Chunk einen semantischen Kontext erzeugt und als `contextual_prefix`
in den Chunk-Metadaten speichert. Der Prefix wird bei Embeddings vor den Chunk-Text gestellt.

| Umgebungsvariable | Beschreibung | Standardwert |
| ----------------- | ------------ | ------------ |
| `RAG_CONTEXTUAL_ENRICHMENT` | Aktiviert Contextual Enrichment nach dem Chunking. | `false` |
| `RAG_CONTEXTUAL_ENRICHMENT_MODEL` | MODEL_ROUTING-Label fuer den Enrichment-LLM-Call. | `fast` |
| `RAG_CONTEXTUAL_ENRICHMENT_MAX_DOC_CHARS` | Maximale Zeichen fuer das Dokument im Prompt. | `12000` |
| `RAG_CONTEXTUAL_ENRICHMENT_MAX_CHUNK_CHARS` | Maximale Zeichen fuer den Chunk im Prompt. | `2000` |
| `RAG_CONTEXTUAL_ENRICHMENT_MAX_CHUNKS` | Maximal anzureichernde Chunks pro Dokument. | `120` |
| `RAG_CONTEXTUAL_ENRICHMENT_MAX_PREFIX_CHARS` | Maximalzeichen fuer den generierten Prefix. | `800` |
| `RAG_CONTEXTUAL_ENRICHMENT_MAX_PREFIX_WORDS` | Maximale Wortanzahl fuer den Prefix. | `120` |

### Kontext-Budget für Antwortgenerierung

Die Anzahl der ausgewählten Snippets wird token-basiert an dieses Budget angepasst.

| Umgebungsvariable | Beschreibung | Standardwert |
| ----------------- | ------------ | ------------ |
| `RAG_CONTEXT_TOKEN_BUDGET` | Tokenbudget für die RAG-Kontextbefüllung. | `1800` |
| `RAG_CONTEXT_OVERSAMPLE_FACTOR` | Oversampling-Faktor für Retrieval vor dem Budget-Cut. | `4` |

**Beispiel:**
Um ein anderes Modell zu verwenden, kann die folgende Zeile in die `.env`-Datei eingetragen werden:

```
EMBEDDINGS_MODEL_PRIMARY=another-model-name
EMBEDDINGS_DIM=1024
```

## Statische Konfiguration (Code-Änderung erforderlich)

Bestimmte grundlegende Konfigurationen sind derzeit direkt im Code festgelegt und erfordern für eine Änderung einen Eingriff durch einen Entwickler.

### Vektor-Datenbank-Backend

Das Backend für die Vektor-Speicherung ist statisch in der Django-Einstellungsdatei konfiguriert.

- **Datei**: `noesis2/settings/base.py`
- **Einstellung**: `RAG_VECTOR_STORES`

In dieser Einstellung ist das `backend` für jeden Vektorraum fest auf `"pgvector"` gesetzt:

```python
# noesis2/settings/base.py

RAG_VECTOR_STORES = {
    "global": {
        "backend": "pgvector",  # <-- Statisch konfiguriert
        "schema": "rag",
        "dimension": DEFAULT_EMBEDDING_DIMENSION,
    },
    "demo": {
        "backend": "pgvector",  # <-- Statisch konfiguriert
        "schema": "rag_demo",   # Legacy / Demo
        "dimension": DEMO_EMBEDDING_DIMENSION,
    },
}
```

> Hinweis: Über die Umgebungsvariable `RAG_VECTOR_SCHEMA` kann das Schema des `global`-Spaces ohne Code-Änderung überschrieben werden. Ohne Override wird – falls gesetzt – automatisch `DEV_TENANT_SCHEMA` verwendet, ansonsten bleibt der Standard `rag`.

Ein Wechsel zu einem anderen Backend (z.B. Weaviate, Milvus) würde eine direkte Änderung dieses Python-Dictionarys erfordern.
