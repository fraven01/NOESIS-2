# NOESIS-2 API Referenz

## Überblick
Die NOESIS-2 API stellt eine mehrmandantenfähige Retrieval-Augmented-Generation-Plattform (RAG) mit vorgeschalteter LLM-Proxy-Schicht bereit. Über standardisierte HTTP-Endpunkte lassen sich Health-Checks, Agenten-Flows und Dokument-Ingestion steuern. Alle Antworten werden in Langfuse nachverfolgt und mit mandantenspezifischen Metadaten versehen. Für LiteLLM-verwaltete APIs existiert eine [separate Referenz](./litellm-admin.md), die dieselben Tenancy- und Authentifizierungsprinzipien übernimmt.

## Headers & Scopes
Alle Aufrufe erfordern HTTP/1.1 über TLS. Multi-Tenancy wird durch `django-tenants` realisiert; Requests müssen das aktive Schema explizit benennen. Fehlende oder inkonsistente Header führen zu `403 Forbidden`, ein nicht bekannter Tenant resultiert in `404 Not Found`.

| Header | Typ | Pflicht | Beschreibung |
| --- | --- | --- | --- |
| `X-Tenant-Schema` | string | required | Aktives PostgreSQL-Schema (z. B. `acme_prod`). Muss mit der Schemaauflösung via Subdomain/Hostname übereinstimmen. |
| `X-Tenant-Id` | string | required | Mandanteninterne Kennung. Wird für Rate-Limits, Object-Store-Pfade und Vektor-Indizes genutzt. |
| `X-Case-Id` | string | optional | Kontext-ID eines Workflows (z. B. CRM-Fall). Muss RFC3986-konforme Zeichen enthalten. |
| `X-Trace-Id` | string | optional | Client-seitig nutzbarer Trace-Identifier. Wird gespiegelt; fehlt er, erzeugt der Service eine neue ID und sendet sie zurück. |
| `X-Workflow-ID` | string | optional | Workflow type identifier (e.g., 'ingestion-2024'). Must match `[A-Za-z0-9._-]+`, max 128 chars. Case-sensitive. Defaults to 'ad-hoc' if not provided. |
| `Idempotency-Key` | string | optional | Empfohlen für POST-Endpunkte. Wiederholte Requests mit gleichem Schlüssel liefern denselben Response-Body und Statuscode. |
| `X-Key-Alias` | string | optional | Referenziert einen LiteLLM API-Key Alias. Wird für Rate-Limiting gebunden. |
| `X-Case-Scope` | string | optional | Zusätzliche Zugriffsscope für Agenten-Workflows. |
| `Authorization` | string | required für LiteLLM Admin | `Bearer <API-Key>` gemäß LiteLLM Master Key Policy. Weitere Details in der [LiteLLM Admin Referenz](./litellm-admin.md). |

**Tenancy-Hinweis:** Das Schema kann automatisch aus der anfragenden Domain (z. B. `tenant.example.com`) abgeleitet werden. Für interne Skripte ist der Header dennoch Pflicht, um versehentliche Schema-Wechsel zu verhindern.

**Trace-ID-Auflösung:** Die Middleware liest die Trace-ID in folgender Reihenfolge aus: `X-Trace-Id` Header, Query-Parameter `trace_id`, JSON-Body-Feld `trace_id` sowie W3C `traceparent`. Wird keine Quelle gefunden, generiert der Service eine neue ID und spiegelt sie in Header und Response-Body.

**Idempotente POST-Requests:** Setzen Sie einen stabilen `Idempotency-Key` pro fachlichem Vorgang (UUID oder Hash). Serverseitig wird der erste Abschluss pro Schlüssel persistiert; Wiederholungen liefern denselben `X-Trace-Id` sowie ein Flag `"idempotent": true` im Response-Body.

**PII-Handling:** Eingehende Prompts, Dokumentinhalte und Modellantworten werden automatisch via PII-Scope maskiert, bevor sie persistiert oder an Langfuse übermittelt werden.

### Beispiel-cURL

```bash
curl -X POST "https://api.noesis.example/v1/ai/rag/query/" \
  -H "X-Tenant-Schema: acme_prod" \
  -H "X-Tenant-Id: acme" \
  -H "X-Case-Id: crm-7421" \
  -H "Idempotency-Key: 6cdb89f6-8826-4f9b-8c82-1f14b3d4c21b" \
  -H "Content-Type: application/json" \
  -d '{"tenant_id": "acme", "trace_id": "trace-a12b3c4d5", "question": "Welche Reisekosten gelten für Consultants?"}'
```

## System Endpunkte

### GET `/ai/ping/`
Kurzer Health-Check der AI-Core-Anwendung.

**Headers**
- `X-Tenant-Schema` (required)
- `X-Tenant-Id` (required)
- `X-Case-Id` (optional)

**Response 200 Beispiel**
```json
{
  "ok": true,
  "trace_id": "6fd5882e9d7b4d7f9f5c3f5f53d6a1b2"
}
```

**Fehler**
- `400 Bad Request`: Ungültige oder fehlende Tenant-Header.
- `403 Forbidden`: Tenant konnte nicht aufgelöst werden.

### GET `/tenant-demo/`
Stellt eine mandantenabhängige Demo-Seite bereit. Dient zur Validierung der Schema-Auflösung per Domain/Subdomain.

**Headers**
- `X-Tenant-Schema` (required)
- `X-Tenant-Id` (required)

**Query Parameter**
- `lang` (optional, string): UI-Lokalisierung.

**Response 200 Beispiel**
```json
{
  "tenant": "acme",
  "schema": "acme_prod",
  "features": ["rag", "agents"],
  "trace_id": "0c953b4e5094436092c94bb6c51ab802"
}
```

### GET `/health/`
Aggregierter Health-Endpunkt für Web- und Worker-Dienste. Gibt auch Tenant-agnostische Statusinformationen aus.

**Headers**
- `X-Tenant-Schema` (optional)
- `X-Tenant-Id` (optional)

**Response 200 Beispiel**
```json
{
  "status": "ok",
  "services": {
    "web": "healthy",
    "worker": "healthy",
    "redis": "healthy"
  },
  "langfuse_trace": "https://app.langfuse.com/project/noesis/traces/7e92"
}
```

**Fehler**
- `503 Service Unavailable`: Mindestens eine Abhängigkeit ist gestört.

## RAG & Ingestion

### POST `/rag/documents/upload/`
Lädt Rohdokumente in den Object-Store hoch und macht sie für nachfolgende Ingestion-Läufe verfügbar.

**Headers**
- `X-Tenant-Schema` (required)
- `X-Tenant-Id` (required)
- `X-Case-Id` (optional)
- `Idempotency-Key` (optional, empfohlen)
- `Content-Type: multipart/form-data`

**Body (multipart/form-data)**
- `file` (required, binary): Quelldokument (TXT/PDF/etc.).
- `metadata` (optional, JSON string): Zusätzliche Tags.

**Response 202 Beispiel**
```json
{
  "status": "accepted",
  "document_id": "c5b406ad3e6f4a26a0f4f06ef8753d9e",
  "idempotent": false,
  "trace_id": "b1ca46f2191b44abbb74116bb6c1b724"
}
```

**Fehler**
- `400 Bad Request`: Kein File-Part oder ungültige Metadaten.
- `415 Unsupported Media Type`: Kein `multipart/form-data` Request.

### POST `/rag/ingestion/run/`
Startet einen Ingestion-Workflow für zuvor hochgeladene Dokumente. Der Prozess läuft asynchron über die Celery-Queue `ingestion`.

**Headers**
- `X-Tenant-Schema` (required)
- `X-Tenant-Id` (required)
- `X-Case-Id` (optional)
- `Idempotency-Key` (optional, empfohlen)
- `Content-Type: application/json`

**Body Schema**
```json
{
  "document_ids": ["doc_8fb6f3f4"],
  "priority": "normal"
}
```

**Response 202 Beispiel**
```json
{
  "status": "queued",
  "queued_at": "2024-05-17T12:02:33Z",
  "ingestion_run_id": "run_3c4e",
  "idempotent": true,
  "trace_id": "e74f50b6696642e5afecbaaf32cf0d9d"
}
```

**Fehler**
- `400 Bad Request`: Leere Dokumentliste.
- `429 Too Many Requests`: Rate-Limit auf Tenant-Level erreicht.

### POST `/ai/rag/admin/hard-delete/`
Interner Admin-Endpunkt, der den Celery-Task `rag.hard_delete` triggert. Zugriffe erfordern entweder einen erlaubten Service-Key (`X-Internal-Key`) oder eine aktive Admin-Session; alle anderen Requests werden mit `403 Forbidden` abgelehnt.

**Headers**
- `Content-Type: application/json`
- `X-Internal-Key` (required für Service-zu-Service-Aufrufe)

**Body Schema**
```json
{
  "tenant_id": "2f0955c2-21ce-4f38-bfb0-3b690cd57834",
  "document_ids": [
    "3fbb07d0-2a5b-4b75-8ad4-5c5e8f3e1d21",
    "986cf6d5-2d8c-4b6c-98eb-3ac80f8aa84f"
  ],
  "reason": "cleanup",
  "ticket_ref": "TCK-1234"
}
```
- Optionales Feld `operator_label` erlaubt die Angabe eines sprechenden Audit-Namens, der in den Task-Logs landet.

**Response 202 Beispiel**
```json
{
  "status": "queued",
  "job_id": "0d9f7ac1-0b07-4b7c-98b7-7237f8b9df5b",
  "trace_id": "c8b7e6c430864d6aa6c66de8f9ad6d47",
  "documents_requested": 2
}
```

**Fehler**
- `400 Bad Request`: Fehlende Pflichtfelder oder ungültige UUIDs in `tenant_id`/`document_ids`.
- `403 Forbidden`: Weder Service-Key noch berechtigter Admin-User vorhanden.

### POST `/ai/v1/rag-demo/`
Die „RAG Demo“ stellt einen rein retrieval-basierten Beispiel-Graphen bereit. Er beantwortet eine Query ohne LLM-Beteiligung, liest die Anfrage aus `{"query": ...}` (bzw. kompatiblen Alias-Feldern) und liefert die Top-K Treffer aus dem mandantenspezifisch gefilterten Vektor-Index. Ohne angebundenen Vektorstore werden deterministische Demo-Matches zurückgegeben.

**Headers**
- `X-Tenant-Schema` (required)
- `X-Tenant-Id` (required)
- `X-Case-Id` (optional)
- `Content-Type: application/json`

**Body Schema**
```json
{
  "query": "Wie konfiguriere ich Tenant-Filter?",
  "top_k": 5,
  "visibility": "active"
}
```

Das Feld `visibility` ist optional; wenn es fehlt oder nicht autorisiert ist, wird serverseitig auf `"active"` zurückgefallen.

**Response 200 Beispiel**
```json
{
  "ok": true,
  "query": "Wie konfiguriere ich Tenant-Filter?",
  "matches": [
    {
      "id": "demo-1",
      "score": 0.42,
      "text": "Tenant-scoped Retrieval Beispiel …",
      "metadata": {
        "tenant_id": "acme"
      }
    }
  ],
  "meta": {
    "visibility_effective": "active"
  }
}
```

`meta.visibility_effective` bestätigt die angewendete Sichtbarkeitsregel für Observability und nachgelagerte Auswertungen.
`meta.matches_returned` dokumentiert die Anzahl der nach Deduplizierung tatsächlich gelieferten Snippets.
`meta.deleted_matches_blocked` zeigt an, wie viele Treffer aufgrund der Sichtbarkeitsregeln entfernt wurden (Soft-Deleted Inhalte bei `"active"`).

#### Result-Metadaten

> ℹ️ **Response-Formate:** Graph-Nodes wie `retrieve` liefern Snippets mit
> flachen Feldern (`text`, `source`, `score`, `hash`, `id`) und einem optionalen
> `meta`-Dictionary, das zusätzliche Schlüssel aus dem Chunk (z. B.
> `doctype`, `published`) unverändert durchreicht. HTTP-Endpunkte wie
> `/ai/v1/rag-demo/` bündeln dieselben Informationen dagegen im Feld
> `metadata` und verwenden `snippets[].metadata.score` statt eines Top-Level
> Keys. Prüfen Sie daher stets den spezifischen Endpoint-Contract, bevor Sie
> Felder downstream weiterverarbeiten. Beide Varianten enthalten Hash und
> Dokument-ID zur nachträglichen Deduplication sowie den Similarity-Score.

> 📈 **Score-Interpretation:** Der im Graph zurückgegebene `score` ist der
> fusionierte Wert `α·semantisch + (1–α)·lexikalisch`, den die Hybrid-Suche des
> Routers liefert. Beide Komponenten werden auf einen Wertebereich von 0 bis 1
> normalisiert; höhere Werte sind relevanter. Die semantische Seite nutzt den
> Cosine-Vergleich über `vector_cosine_ops`, während die lexikalische Seite BM25
> beisteuert. Deklarieren Sie Scores im UI daher als „Similarity Score“ und
> vermeiden Sie Prozentangaben.

**cURL Beispiel**
```bash
curl -X POST "https://api.noesis.example/ai/v1/rag-demo/" \
  -H "X-Tenant-Schema: acme_prod" \
  -H "X-Tenant-Id: acme" \
  -H "X-Case-Id: crm-7421" \
  -H "Content-Type: application/json" \
  -d '{"query": "Wie konfiguriere ich Tenant-Filter?", "top_k": 3}'
```

#### Visibility

- **Breaking Default:** Ohne explizite Angabe (oder ohne Admin-Freigabe) wird
  immer `"active"` erzwungen – Soft-Deleted Inhalte tauchen somit nicht mehr in
  Standard-Abfragen auf.
- Requests akzeptieren optional das Feld `visibility` mit den Werten
  `"active"`, `"all"` oder `"deleted"`. Nur autorisierte Admin-Kontexte dürfen
  `"all"` oder `"deleted"` erzwingen; anderenfalls greift automatisch der
  Fallback `"active"`.
- Responses spiegeln die angewendete Policy unter
  `meta.visibility_effective`, damit Clients Debugging und Observability
  vereinheitlichen können.
- `meta.deleted_matches_blocked` quantifiziert, wie viele Treffer durch die
  Sichtbarkeitsregeln verworfen wurden (bei `"active"` also Soft-Deletes).

### RAG Umgebungsvariablen

| Variable | Default | Beschreibung |
| --- | --- | --- |
| `RAG_STATEMENT_TIMEOUT_MS` | `15000` | Maximale Ausführungszeit (in Millisekunden) für SQL-Statements des pgvector Clients. |
| `RAG_RETRY_ATTEMPTS` | `3` | Anzahl der Wiederholungsversuche für Datenbankoperationen, bevor der Fehler propagiert wird. |
| `RAG_RETRY_BASE_DELAY_MS` | `50` | Basiswartezeit zwischen Wiederholungsversuchen (linear skaliert mit dem Versuchszähler). |
| `RAG_INDEX_KIND` | `HNSW` | Auswahl des Vektorindex (HNSW oder IVFFLAT) für das Embedding-Backend. |
| `RAG_HNSW_M` | `32` | Kantenfaktor `m` für HNSW-Indizes; bestimmt die Graph-Konnektivität. |
| `RAG_HNSW_EF_CONSTRUCTION` | `200` | `ef_construction`-Parameter für HNSW beim Aufbau des Index. |
| `RAG_HNSW_EF_SEARCH` | `80` | Laufzeit-Parameter `ef_search` für HNSW-Abfragen (per Session gesetzt). |
| `RAG_IVF_LISTS` | `2048` | Anzahl der Listen (`lists`) für IVFFLAT-Indizes. |
| `RAG_IVF_PROBES` | `64` | Anzahl der `probes` für IVFFLAT-Suchen (per Session gesetzt). |
| `RAG_MIN_SIM` | `0.15` | Minimale Fused-Similarity für Treffer (Werte darunter werden verworfen). |
| `RAG_LEXICAL_MODE` | `trgm` | Lexikalischer Retrieval-Modus (`trgm` oder `bm25`). |
| `RAG_HYDE_ENABLED` | `false` | HyDE (Hypothetical Document Embeddings) für semantische Queries aktivieren. |
| `RAG_HYDE_MODEL_LABEL` | `simple-query` | LLM-Routing-Label für HyDE-Prompts. |
| `RAG_HYDE_MAX_CHARS` | `2000` | Maximalgröße des HyDE-Texts vor dem Embedding. |
| `RAG_HYBRID_ALPHA` | `0.7` | Gewichtung der semantischen Similarität in der Late-Fusion (0 = nur Lexikalik). |
| `RAG_CHUNK_TARGET_TOKENS` | `450` | Zielgröße pro Chunk in Tokens für die Vorverarbeitung. |
| `RAG_CHUNK_OVERLAP_TOKENS` | `80` | Token-Overlap zwischen aufeinanderfolgenden Chunks. |

Bei fehlenden `%`-Treffern wird automatisch auf eine reine Similarity-Sortierung
zurückgefallen.

## Dokumente

### GET `/documents/download/<document_id>/`
Streamt ein hochgeladenes Dokument mit produktionsreifen Caching- und Streaming-Features.

**Features**
- Tenant-isolierter Zugriff (prüft `X-Tenant-ID` gegen Dokument-Metadaten)
- HTTP-Caching: ETag (weak), Last-Modified, `304 Not Modified`
- Range-Requests: `206 Partial Content`, Suffix-Ranges (`bytes=-N`)
- RFC 5987: UTF-8-Filenames in `Content-Disposition`
- CRLF-Injection-Schutz für Dateinamen
- HEAD-Methode für Metadaten ohne Body
- FileResponse-Streaming (wsgi.file_wrapper/sendfile)

**URL-Parameter**
- `document_id` (UUID, required): Dokument-ID aus Upload/Ingestion

**Headers**
- `X-Tenant-Schema` (required)
- `X-Tenant-Id` (required)
- `If-None-Match` (optional): ETag(s) für Conditional Request (unterstützt mehrere ETags)
- `If-Modified-Since` (optional): Zeitstempel für Conditional Request
- `Range` (optional): Byte-Range (z. B. `bytes=0-999`, `bytes=100-`, `bytes=-500`)

**Response 200 (GET) Beispiel**
```
Content-Type: application/pdf
Content-Length: 245680
Content-Disposition: attachment; filename="document.pdf"; filename*=UTF-8''document.pdf
ETag: W/"3bf30-18a2f4c8e9a"
Last-Modified: Fri, 17 May 2024 14:23:45 GMT
Cache-Control: private, max-age=3600
Vary: Authorization, Cookie
Accept-Ranges: bytes
X-Content-Type-Options: nosniff

<binary content>
```

**Response 200 (HEAD) Beispiel**
```
Content-Type: application/pdf
Content-Length: 245680
Content-Disposition: attachment; filename="document.pdf"; filename*=UTF-8''document.pdf
ETag: W/"3bf30-18a2f4c8e9a"
Last-Modified: Fri, 17 May 2024 14:23:45 GMT
Cache-Control: private, max-age=3600
Vary: Authorization, Cookie
Accept-Ranges: bytes
X-Content-Type-Options: nosniff
```

**Response 206 (Partial Content) Beispiel**
```
Content-Type: application/pdf
Content-Length: 1000
Content-Range: bytes 0-999/245680
Content-Disposition: attachment; filename="document.pdf"; filename*=UTF-8''document.pdf
ETag: W/"3bf30-18a2f4c8e9a"
Last-Modified: Fri, 17 May 2024 14:23:45 GMT
Cache-Control: private, max-age=3600
Vary: Authorization, Cookie
Accept-Ranges: bytes
X-Content-Type-Options: nosniff

<partial binary content>
```

**Response 304 (Not Modified) Beispiel**
```
ETag: W/"3bf30-18a2f4c8e9a"
Last-Modified: Fri, 17 May 2024 14:23:45 GMT
Cache-Control: private, max-age=3600
Vary: Authorization, Cookie
Accept-Ranges: bytes
```

**Fehler**
- `400 Bad Request`: Ungültige UUID in `document_id`
```json
{
  "error": {
    "code": "InvalidDocumentId",
    "message": "document_id must be a valid UUID"
  }
}
```

- `403 Forbidden`: Tenant-Mismatch (Dokument gehört zu anderem Tenant)
```json
{
  "error": {
    "code": "TenantMismatch",
    "message": "Access denied"
  }
}
```

- `404 Not Found`: Dokument in Metadatenbank nicht gefunden
```json
{
  "error": {
    "code": "DocumentNotFound",
    "message": "Document 12345678-1234-1234-1234-123456789abc not found"
  }
}
```

- `404 Not Found`: Blob-Datei auf Disk fehlt
```json
{
  "error": {
    "code": "BlobNotFound",
    "message": "Document file not found on disk"
  }
}
```

- `416 Range Not Satisfiable`: Ungültiger Range-Request
```
Content-Range: bytes */245680
Cache-Control: private, max-age=3600
Vary: Authorization, Cookie
Accept-Ranges: bytes
```

- `500 Internal Server Error`: Unerwarteter Fehler
```json
{
  "error": {
    "code": "InternalError",
    "message": "<exception message>"
  }
}
```

**Filename-Extraction**

Der Dateiname wird mit 3-stufigem Fallback extrahiert:
1. `doc.meta.title` (Originalname vom Upload)
2. `doc.meta.external_ref["filename"]` (falls vorhanden)
3. `document_id` + Extension aus `Content-Type` (z. B. `123e4567.pdf`)

**Content-Type Detection**

Reihenfolge:
1. `blob.media_type` (aus Metadaten)
2. Magic-Number-Detection via `python-magic`
3. Fallback: `application/octet-stream`

**cURL Beispiel**
```bash
# Vollständiger Download
curl -X GET "https://api.noesis.example/documents/download/123e4567-e89b-12d3-a456-426614174000/" \
  -H "X-Tenant-Schema: acme_prod" \
  -H "X-Tenant-Id: acme" \
  -O -J

# HEAD Request (nur Metadaten)
curl -I "https://api.noesis.example/documents/download/123e4567-e89b-12d3-a456-426614174000/" \
  -H "X-Tenant-Schema: acme_prod" \
  -H "X-Tenant-Id: acme"

# Range Request (erste 1KB)
curl -X GET "https://api.noesis.example/documents/download/123e4567-e89b-12d3-a456-426614174000/" \
  -H "X-Tenant-Schema: acme_prod" \
  -H "X-Tenant-Id: acme" \
  -H "Range: bytes=0-1023"

# Conditional Request (nur wenn geändert)
curl -X GET "https://api.noesis.example/documents/download/123e4567-e89b-12d3-a456-426614174000/" \
  -H "X-Tenant-Schema: acme_prod" \
  -H "X-Tenant-Id: acme" \
  -H 'If-None-Match: W/"3bf30-18a2f4c8e9a"'
```

**Observability**

Alle Requests werden strukturiert geloggt mit:
- `documents.download.started`: Start mit Tenant-ID, Dokument-ID, HTTP-Methode
- `documents.download.streaming_started`: Erfolgreicher Stream mit File-Size, Content-Type, ETag, Dauer
- `documents.download.partial_content`: Range-Request mit Start/End/Dauer
- `documents.download.head_completed`: HEAD-Request mit File-Size, ETag, Dauer
- `documents.download.not_modified_etag`: 304 via ETag-Match
- `documents.download.not_modified_time`: 304 via If-Modified-Since
- `documents.download.not_found`: 404 mit Dauer
- `documents.download.tenant_mismatch`: 403 mit Tenant-IDs und Dauer
- `documents.download.blob_missing`: 404 für fehlende Disk-Datei
- `documents.download.range_invalid`: 416 mit ungültigem Range
- `documents.download.failed`: 500 mit Exception-Type und Dauer

## Agenten (Queue `agents`)

### POST `/ai/intake/`
Startet den Agenten-Flow `info_intake` zur Kontextanreicherung.

> **Deprecated:** `/ai/intake/` und `/v1/ai/intake/` sind veraltet. Bitte nutze
> `/v1/ai/rag/query/` fÇ¬r produktive RAG-Flows. Der Endpoint liefert einen
> `Deprecation`-Header und wird vor dem MVP entfernt.

> **Hinweis:** Der Graph persistiert den Zustand und führt das Request-JSON als
> shallow overwrite mit dem bestehenden State zusammen.

**Headers**
- `X-Tenant-Schema` (required)
- `X-Tenant-Id` (required)
- `X-Case-Id` (optional)
- `Idempotency-Key` (optional)
- `Content-Type: application/json`

**Body Schema**
```json
{
  "tenant_id": "acme",
  "trace_id": "trace-71f0a9c2",
  "prompt": "Fasse das Kundenfeedback zusammen",
  "metadata": {"channel": "email"}
}
```

`tenant_id` muss dem Header `X-Tenant-Id` entsprechen; `trace_id` kann aus Header oder Body geliefert werden und wird für Langfuse-Traces verwendet.

**Response 200 Beispiel**
```json
{
  "summary": "Kunde meldet Lieferverzug und wünscht Rückruf.",
  "suggested_entities": ["Bestellung-845"],
  "trace_id": "e6a3ff14f1a84154b92f0841a1ba46f9"
}
```

### POST `/v1/ai/rag/query/`
Startet den produktiven Retrieval-Augmented-Generation-Graphen (`retrieval_augmented_generation`). Die vollständigen Tool-Contracts (`RetrieveInput`, `RetrieveOutput`, Meta-Felder) sind unter [RAG → Retrieval Contracts](../rag/retrieval-contracts.md) dokumentiert.

**Headers**
- `X-Tenant-Schema` (required)
- `X-Tenant-Id` (required)
- `X-Case-Id` (optional)
- `Idempotency-Key` (optional, empfohlen)
- `Content-Type: application/json`

**Body Beispiel**
```json
{
  "tenant_id": "acme",
  "trace_id": "trace-b5d0c7a4",
  "question": "Welche Reisekosten gelten für Consultants?",
  "filters": {"doc_class": "policy", "process": "travel"},
  "visibility": "tenant"
}
```

Auch für Queries gilt: `tenant_id` ist ein Pflichtfeld im Body (muss dem Header entsprechen), `trace_id` wird entweder übernommen oder – falls nicht gesetzt – serverseitig ergänzt.

**Response 200 Beispiel**
```json
{
  "answer": "Consultants nutzen das Travel-Policy-Template.",
  "prompt_version": "2024-05-01",
  "retrieval": {
    "alpha": 0.7,
    "min_sim": 0.15,
    "top_k_effective": 1,
    "matches_returned": 1,
    "max_candidates_effective": 50,
    "vector_candidates": 37,
    "lexical_candidates": 41,
    "deleted_matches_blocked": 0,
    "visibility_effective": "active",
    "took_ms": 42,
    "routing": {
      "profile": "standard",
      "vector_space_id": "rag/standard@v1"
    }
  },
  "snippets": [
    {
      "id": "doc-871#p3",
      "text": "Rücksendungen sind innerhalb von 30 Tagen möglich, sofern das Produkt unbenutzt ist.",
      "score": 0.82,
      "source": "policies/returns.md",
      "hash": "7f3d6a2c",
      "meta": {"page": 3, "language": "de"}
    }
  ],
  "trace_id": "b5d0c7a4d5de4b89b9b7d0c14fd31b1b"
}
```
