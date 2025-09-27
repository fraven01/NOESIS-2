# Warum
Retrieval-Augmented Generation (RAG) verbindet strukturierte Tenant-Daten mit LLM-Antworten. Dieses Kapitel erklärt das Zielbild und zeigt, wie Ingestion, Speicherung und Agenten zusammenspielen.

# Wie
## Architektur
```mermaid
flowchart LR
    L[Loader] --> S[Splitter]
    S --> C[Chunking]
    C --> E[Embedding]
    E --> V[Vector Store (pgvector)]
    V --> R[Retriever]
    R --> RR[Re-Ranking (optional)]
    RR --> A[Agent (LangGraph)]
    A --> LLM[LiteLLM]
    LLM --> A
    A --> Out[Antwort]
```

- Loader ziehen Quelldaten aus Dokumenten, APIs oder Datenbanken.
- Splitter & Chunker erzeugen überlappende Textblöcke; Parameter stehen in [RAG-Ingestion](ingestion.md).
- Embeddings landen in `pgvector` (siehe [Schema](schema.sql)), inklusive Metadaten pro Tenant.
- Retriever filtert per `tenant_id`, optional Re-Ranking (z.B. Cross-Encoder) bevor Agenten reagieren.

## Mandantenfähigkeit
Standardweg: Embeddings und Metadaten liegen in einem gemeinsamen Schema, `tenant_id` trennt Zugriffe und wird vom Retriever gefiltert. Für wachsende Last kann optional ein Silo/Schemas je Tenant aufgebaut werden.

| Modell | Beschreibung | Einsatz |
| --- | --- | --- |
| Gemeinsames Schema mit `tenant_id` | Eine Tabelle pro Entität, `tenant_id` (UUID) erzwingt RLS-Regeln und sorgt für gemeinsames Embedding-Repository | Default für alle Umgebungen, solange Anforderungen <≈50 Tenants bleiben |
| Schema pro Mandant | Eigenes Schema `tenant_<slug>` für Tabellen `documents`, `chunks`, `embeddings` | Skalierungspfad für Großkunden oder erhöhte Isolation |
| Hybrid | Kern-Tabellen pro Schema, Embeddings global mit `tenant_id` | Wenn LiteLLM und Django gemeinsame Daten teilen müssen |

## VectorStore Router
Der VectorStore Router kapselt die Auswahl des Backends und erzwingt, dass jede Suche mit einer gültigen `tenant_id` ausgeführt wird. Er normalisiert Filter, deckelt `top_k` hart auf zehn Ergebnisse und schützt damit Agenten vor übermäßigen Resultatmengen. Silos pro Tenant lassen sich später über zusätzliche Router-Scopes konfigurieren, während der Standard weiterhin auf den globalen Store zeigt. Weil der Router immer aktiv ist, entfällt das frühere Feature-Flagging für RAG. Tests können dank Fake-Stores ohne PostgreSQL durchgeführt werden, während optionale Integrationsläufe weiterhin gegen pgvector laufen. Die Delegation sorgt zugleich dafür, dass PII-Redaktionen und Hash-Prüfungen im bestehenden `PgVectorClient` unverändert greifen.

```python
from ai_core.rag import get_default_router

router = get_default_router()
results = router.search("fallback instructions", tenant_id="tenant-uuid", top_k=5)
```

## Löschkonzept
- Dokumente erhalten Hashes (siehe [Schema](schema.sql)) und `metadata` mit Herkunft.
- Löschläufe laufen als Ingestion-Task mit Modus „delete“ und markieren `documents.deleted_at` (Soft Delete). Hard Delete optional über `DELETE ... WHERE tenant_id = ?`.
- Nach dem Löschen wird `VACUUM`/`ANALYZE` ausgeführt (Staging monatlich, Prod wöchentlich). Index-Rebuild via [Migrations-Runbook](../runbooks/migrations.md).

# Schritte
1. Plane Tenant-Strategie laut Tabelle und dokumentiere sie im Architektur-Overview.
2. Implementiere Ingestion-Pipelines mit Parametern aus [RAG-Ingestion](ingestion.md) und schreibe Embeddings in das Schema aus [schema.sql](schema.sql).
3. Aktiviere Observability für Agenten und Retriever über [Langfuse](../observability/langfuse.md), bevor Nutzer Zugriff erhalten.

> **Skalierung:** Bis zu 50 Tenants gilt die gemeinsame Ablage als ausreichend. Darüber evaluieren wir pro Tenant ein Silo-Schema.
