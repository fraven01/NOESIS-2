# RAG Konfiguration

Dieses Dokument beschreibt, wie die RAG-Komponenten (Retrieval-Augmented Generation) konfiguriert werden, insbesondere die Embedding-Modelle und die Vektor-Datenbank.

## Laufzeit-Konfiguration (Umgebungsvariablen)

Einige wichtige Parameter können zur Laufzeit über Umgebungsvariablen (z.B. in einer `.env`-Datei) gesteuert werden. Dies ermöglicht es dem Betrieb (Ops), die Konfiguration ohne Code-Änderungen anzupassen.

### Embedding-Modell & Dimension

Das für die Erstellung von Vektor-Embeddings verwendete Modell sowie dessen Dimensionen werden über Umgebungsvariablen definiert.

| Umgebungsvariable          | Beschreibung                                                                                             | Standardwert                  |
| -------------------------- | -------------------------------------------------------------------------------------------------------- | ----------------------------- |
| `EMBEDDINGS_MODEL_PRIMARY` | Der Name/Alias des primären Embedding-Modells, das für das "standard" Profil verwendet wird.             | `oai-embed-small`             |
| `EMBEDDINGS_DIM`           | Die Anzahl der Dimensionen für das primäre Embedding-Modell.                                             | `1536`                        |
| `DEMO_EMBEDDINGS_MODEL`    | Der Modellname für das "demo" Profil.                                                                    | Wert von `EMBEDDINGS_MODEL_PRIMARY` |
| `DEMO_EMBEDDINGS_DIM`      | Die Anzahl der Dimensionen für das Demo-Modell.                                                          | Wert von `EMBEDDINGS_DIM`     |

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
        "schema": "rag_demo",
        "dimension": DEMO_EMBEDDING_DIMENSION,
    },
}
```

Ein Wechsel zu einem anderen Backend (z.B. Weaviate, Milvus) würde eine direkte Änderung dieses Python-Dictionarys erfordern.
