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

- **Validierungsstatus (Pydantic):**
  - **Abgedeckt:**
    - `RagUploadView` validiert optionale Metadaten mit `RagUploadMetadata` bevor Dateien persistiert werden.
    - `RagIngestionRunView` nutzt `RagIngestionRunRequest`, um Dokument-IDs zu trimmen, Priorität zu prüfen und ein nicht-leeres Embedding-Profil zu erzwingen.
    - `RagHardDeleteAdminView` übernimmt die Eingabeprüfung via `RagHardDeleteAdminRequest` und erzwingt UUID- sowie String-Kontrakte.
    - Die Graph-Endpoints (`IntakeViewV1`, `ScopeViewV1`, `NeedsViewV1`, `SysDescViewV1` und Legacy-Varianten) validieren eingehende States über `InfoIntakeRequest`, `ScopeCheckRequest`, `NeedsMappingRequest` bzw. `SystemDescriptionRequest` bevor sie in LangGraph laufen.

- Loader ziehen Quelldaten aus Dokumenten, APIs oder Datenbanken.
- Splitter & Chunker erzeugen überlappende Textblöcke; Parameter stehen in [RAG-Ingestion](ingestion.md).
- Embeddings landen in `pgvector` (siehe [Schema](schema.sql)), inklusive Metadaten pro Tenant.
- Retriever filtert per `tenant_id` und blendet Soft-Deletes (`deleted_at`) standardmäßig aus; optional folgt Re-Ranking (z.B. Cross-Encoder) bevor Agenten reagieren (siehe [Lifecycle-Notiz](lifecycle.md#soft-delete)).

## Mandantenfähigkeit
Standardweg: Embeddings und Metadaten liegen in einem gemeinsamen Schema, `tenant_id` trennt Zugriffe und wird vom Retriever gefiltert. Für wachsende Last kann optional ein Silo/Schemas je Tenant aufgebaut werden.

| Modell | Beschreibung | Einsatz |
| --- | --- | --- |
| Gemeinsames Schema mit `tenant_id` | Eine Tabelle pro Entität, `tenant_id` (UUID) erzwingt RLS-Regeln und sorgt für gemeinsames Embedding-Repository | Default für alle Umgebungen, solange Anforderungen <≈50 Tenants bleiben |
| Schema pro Mandant | Eigenes Schema `tenant_<slug>` für Tabellen `documents`, `chunks`, `embeddings` | Skalierungspfad für Großkunden oder erhöhte Isolation |
| Hybrid | Kern-Tabellen pro Schema, Embeddings global mit `tenant_id` | Wenn LiteLLM und Django gemeinsame Daten teilen müssen |

## Mehrdimensionale Profile
Wir unterscheiden künftig drei orthogonale Dimensionen, die Einfluss auf den Vektor-Speicher haben: (1) **Tenant** bzw. gebuchtes Service-Level, (2) **Prozesskontext** (z. B. Draft, Review, Final) und (3) **Dokumentklasse** (z. B. juristische Dokumente, technische Handbücher). Das aktuelle Setup unterstützt zwar Tenants über `tenant_id`, koppelt aber alle weiteren Dimensionen an ein einziges Embedding-Profil (`vector(1536)` + `oai-embed-large`).

> **Begriff Vector Space:** Ein Vector Space ist die kleinste persistente Einheit des RAG-Stores mit fester `embedding_dim`, eindeutigem Backend (z. B. pgvector, Faiss) und eigener Tabelle bzw. eigenem Schema. Jeder Vector Space korrespondiert mit einem Eintrag in `RAG_VECTOR_STORES` und den Staging-/Prod-Datenpfaden aus der [Architekturübersicht](../architektur/overview.md#datenpfade-und-tenancy). Damit verankern wir „Dimension ist physische Eigenschaft des Stores“ als Schnittstellenvertrag; Details zum Ingestionspfad stehen im [RAG-Store-Kapitel](ingestion.md).

Für ein skalierbares Zielbild dokumentieren wir folgende Anpassungen:

1. **Embedding-Profile definieren:** Führe eine Konfigurationsstruktur (z. B. `EMBEDDING_PROFILES`) ein, die pro Profil den LiteLLM-Alias, die erwartete Dimension und den Ziel-Vector-Space beschreibt. Beispiel:
   ```python
   EMBEDDING_PROFILES = {
       "standard": {"model": "oai-embed-large", "dimension": 1536, "vector_space": "global"},
       "premium": {"model": "vertex_ai/text-embedding-004", "dimension": 3072, "vector_space": "premium"},
       "fast": {"model": "oai-embed-small", "dimension": 1536, "vector_space": "fast"},
   }
   ```
   Jeder Vector Space verweist auf ein dediziertes Schema oder Backend mit passender `vector(n)`-Spalte.
   Startzeit-Gate: `ai_core.rag.embedding_config.validate_embedding_configuration()` prüft Dimensionen und referenzierte Spaces.

2. **Routing pro Dimension:** Ergänze den `VectorStoreRouter` um Regeln, die aus Tenant-Metadaten, Prozessschritt und Dokumentklasse ein Profil ermitteln. Tenants buchen Upgrades, indem sie im Admin-Backend einem anderen Profil zugeordnet werden; Prozessschritte und Dokumentklassen werden als Ingestion-Parameter übergeben und entscheiden, in welchen Vector Space geschrieben/gelesen wird.
   Konsistenz-Gate: `ai_core.rag.routing_rules.validate_routing_rules()` lädt das YAML und verhindert Mehrdeutigkeiten.

3. **Pipelines anreichern:** Der Ingestion-Task erhält einen verpflichtenden `embedding_profile`-Parameter. Vor dem Schreiben prüft `ai_core.rag.ensure_embedding_dimensions()` jeden Chunk gegen die konfigurierte Vector-Space-Dimension, erzwingt Konsistenz mit dem Zielprofil und führt bei Abweichungen einen Hard-Fail aus. Retrieval-Aufrufe (LangGraph, Reports) müssen denselben Profilschlüssel an den Router durchreichen, damit Query und Speicherung auf denselben Vector Space zeigen. Die Runtime bindet Profile über `ai_core.rag.resolve_ingestion_profile()`, wodurch `embedding_profile` und `vector_space_id` in Statusdateien, Chunk-Metadaten und Langfuse-Traces auftauchen. Verstöße (z. B. fehlendes Profil oder Dimensionsabweichung) werden als Dead-Letter markiert und gemäß [Ingestion-Runbook](ingestion.md#fehlertoleranz-und-deduplizierung) abgearbeitet.

4. **Fallback-Politik:** Fallback ist ausschließlich zu Anbietern oder Endpunkten mit identischer Ausgabelänge zulässig; Dimensionswechsel bedeuten Migration. Die Admin-GUI dokumentiert dieses „Do & Don’t“, damit Betriebs-Teams nicht spontan auf Modelle anderer Dimension umschalten.

5. **Observability:** `ai_core.rag.resolve_ingestion_profile()` und `ai_core.rag.resolve_vector_space_full()` emittieren `rag.vector_space.resolve`-Spans mit `embedding_profile`, `vector_space_id`, `vector_space_schema` und `vector_space_dimension`. Retrieval-Spans (`rag.hybrid.search`) liefern ergänzend `top_k_requested`, `top_k_effective`, `matches_returned`, `max_candidates_effective`, `visibility_*`, Kandidatenzahlen (`vector_candidates`, `lexical_candidates`, `fused_candidates`) sowie `deleted_matches_blocked`, damit Guard-Entscheidungen und Suchparameter transparent bleiben.

6. **Migration planen:** Profile mit abweichender Dimension (z. B. `premium` mit 3072) erfordern dedizierte Tabellen/Schemata sowie ein Re-Embedding der betroffenen Dokumente. Wechsel zwischen Profilen werden als Migration behandelt (Datenexport → Re-Embedding → Import in neues Schema) und nicht durch automatischen Fallback realisiert.

Mit dieser Schichtung bleibt LiteLLM weiterhin die abstrakte Schnittstelle zu den Modell-Anbietern, während Persistenz und Kostensteuerung pro Dimension kontrolliert werden können. Standard-Tenants nutzen weiterhin das globale Profil; Großkunden oder hochwertige Prozessschritte lassen sich gezielt auf erweiterte Vector Spaces routen, ohne Dimensionen zu mischen.

### Persistenz & Schema-Guards
Neben den Validierungen in Worker und Router muss auch das Datenbankschema Guardrails enthalten. `docs/rag/schema.sql` erhält je Vector Space einen expliziten Block – bevorzugt separate Tabellen oder Schemata. Beispiel für ein Premium-Profil:

```sql
CREATE TABLE rag_premium_embeddings (
  id uuid PRIMARY KEY,
  tenant_id uuid NOT NULL,
  doc_id uuid NOT NULL,
  chunk_id uuid NOT NULL,
  embedding vector(3072) NOT NULL,
  metadata jsonb NOT NULL,
  created_at timestamptz DEFAULT now()
);
```

Falls eine gemeinsame Tabelle unvermeidlich ist, erzwingt mindestens ein `CHECK` auf einer Meta-Spalte die erlaubte Dimension und der Upsert-Pfad prüft `len(embedding)` zur Laufzeit – erst nach erfolgreichem Guard wird geschrieben. Diese Guardrails sind Bestandteil der Migrations-Runbooks und Staging-Smoke-Tests.

### Betriebs- & Migrationspfad
Profilwechsel folgen einem festen Ablauf (Runbooks unter [docs/runbooks](../runbooks) beschreiben Details):

1. **Vector Space provisionieren:** Neues Schema/Backend samt Tabellen, Indexen und Guardrails erstellen.
2. **Backfill vorbereiten:** Re-Embedding via Queue-Ingestion, Batch-Limits und Backoff gemäß [Scaling-Leitfaden](../operations/scaling.md) konfigurieren.
3. **Dual-Read Smoke:** In Staging beide Vector Spaces abfragen, Ergebnisdifferenzen in Langfuse markieren und mit Trace-Tags versehen.
4. **Router-Switch:** Konfiguration/Feature-Flag umschalten, Monitoring beobachten und erst bei stabilen Zahlen den Altbestand dekommissionieren.

## VectorStore Router
Der VectorStore Router kapselt die Auswahl des Backends und erzwingt, dass jede Suche mit einer gültigen `tenant_id` ausgeführt wird. Er normalisiert Filter, deckelt `top_k` hart auf zehn Ergebnisse und schützt damit Agenten vor übermäßigen Resultatmengen. Silos pro Tenant lassen sich später über zusätzliche Router-Scopes konfigurieren, während der Standard weiterhin auf den globalen Store zeigt. Weil der Router immer aktiv ist, entfällt das frühere Feature-Flagging für RAG. Tests können dank Fake-Stores ohne PostgreSQL durchgeführt werden, während optionale Integrationsläufe weiterhin gegen pgvector laufen. Die Delegation sorgt zugleich dafür, dass PII-Redaktionen und Hash-Prüfungen im bestehenden `PgVectorClient` unverändert greifen.

Standardmäßig injiziert der Router `visibility="active"` in jede Anfrage, sodass Soft-Deletes ohne weitere Flags unsichtbar bleiben. Nur wenn der Guard (`visibility_override_allowed`) dies erlaubt, werden die Modi `all` oder `deleted` weitergegeben. Die effektive Einstellung wird im Retrieval-Ergebnis unter `meta.visibility_effective` gespiegelt und steht damit für Observability und Debugging bereit.

```python
from ai_core.rag import get_default_router, resolve_vector_space_full

router = get_default_router()
resolution = resolve_vector_space_full("standard")
vector_space = resolution.vector_space
profile = resolution.profile
embedding = embedder(
    text,
    model=profile.model,
)
assert len(embedding) == vector_space.dimension, "dim mismatch"
pg.upsert(
    schema=vector_space.schema,
    embedding=embedding,
    metadata=meta,
)
results = router.hybrid_search(
    "fallback instructions",
    tenant_id="tenant-uuid",
    process="review",
    doc_class="legal",
    top_k=5,
)
```

Der Router deckelt `top_k` weiterhin auf zehn Ergebnisse. Die Policy `RAG_CANDIDATE_POLICY` legt fest, wie Konflikte mit dem konfigurierbaren Kandidatenpool (`max_candidates`, `normalize_max_candidates`) behandelt werden: `error` (Default) bricht mit `ROUTER_MAX_CANDIDATES_LT_TOP_K` ab, `normalize` hebt den Pool deterministisch auf mindestens `top_k` an. Die Eingangsvalidierung schreibt den angehobenen Wert zurück und erzeugt genau eine `rag.hybrid.candidate_pool.normalized`-Warnung für die redundante Konfiguration.

Selektoren für Routing-Regeln (`tenant`, `process`, `doc_class`) werden beim Laden und zur Laufzeit getrimmt und in Kleinschreibung überführt. Die Spezifität bestimmt sich über die Anzahl gesetzter Felder; die Auflösung folgt „höchste Spezifität gewinnt, Gleichstand ⇒ Fehler“.

## Löschkonzept
- Dokumente erhalten Hashes (siehe [Schema](schema.sql)) und `metadata` mit Herkunft.
- Löschläufe laufen als Ingestion-Task mit Modus „delete“ und markieren `documents.deleted_at` (Soft Delete). Hard Delete optional über `DELETE ... WHERE tenant_id = ?`.
- Nach dem Löschen wird `VACUUM`/`ANALYZE` ausgeführt (Staging monatlich, Prod wöchentlich). Index-Rebuild via [Migrations-Runbook](../runbooks/migrations.md).

# Schritte
1. Plane Tenant-Strategie laut Tabelle und dokumentiere sie im Architektur-Overview.
2. Implementiere Ingestion-Pipelines mit Parametern aus [RAG-Ingestion](ingestion.md) und schreibe Embeddings in das Schema aus [schema.sql](schema.sql).
3. Aktiviere Observability für Agenten und Retriever über [Langfuse](../observability/langfuse.md), bevor Nutzer Zugriff erhalten.

> **Skalierung:** Bis zu 50 Tenants gilt die gemeinsame Ablage als ausreichend. Darüber evaluieren wir pro Tenant ein Silo-Schema.

## Offene Aufgaben

- [ ] Pgvector-Versionen vereinheitlichen, damit `vector_cosine_ops` auch für HNSW-Indizes in jeder Umgebung verfügbar ist und Reindex-Läufe ohne Fallback funktionieren. Deployment- und Validierungsplan im [Migrations-Runbook](../runbooks/migrations.md) dokumentieren.
