# NOESIS-2 API Referenz

## Überblick
Die NOESIS-2 API stellt eine mehrmandantenfähige Retrieval-Augmented-Generation-Plattform (RAG) mit vorgeschalteter LLM-Proxy-Schicht bereit. Über standardisierte HTTP-Endpunkte lassen sich Health-Checks, Agenten-Flows und Dokument-Ingestion steuern. Alle Antworten werden in Langfuse nachverfolgt und mit mandantenspezifischen Metadaten versehen. Für LiteLLM-verwaltete APIs existiert eine [separate Referenz](./litellm-admin.md), die dieselben Tenancy- und Authentifizierungsprinzipien übernimmt.

## Headers & Scopes
Alle Aufrufe erfordern HTTP/1.1 über TLS. Multi-Tenancy wird durch `django-tenants` realisiert; Requests müssen das aktive Schema explizit benennen. Fehlende oder inkonsistente Header führen zu `403 Forbidden`, ein nicht bekannter Tenant resultiert in `404 Not Found`.

| Header | Typ | Pflicht | Beschreibung |
| --- | --- | --- | --- |
| `X-Tenant-Schema` | string | required | Aktives PostgreSQL-Schema (z. B. `acme_prod`). Muss mit der Schemaauflösung via Subdomain/Hostname übereinstimmen. |
| `X-Tenant-Id` | string | required | Mandanteninterne Kennung. Wird für Rate-Limits, Object-Store-Pfade und Vektor-Indizes genutzt. |
| `X-Case-Id` | string | required | Kontext-ID eines Workflows (z. B. CRM-Fall). Muss RFC3986-konforme Zeichen enthalten. |
| `X-Trace-Id` | string | optional (response) | Wird serverseitig generiert und als Echo-Header zurückgegeben. Dient zur Korrelation in Logs & Langfuse. |
| `Idempotency-Key` | string | optional | Empfohlen für POST-Endpunkte. Wiederholte Requests mit gleichem Schlüssel liefern denselben Response-Body und Statuscode. |
| `X-Key-Alias` | string | optional | Referenziert einen LiteLLM API-Key Alias. Wird für Rate-Limiting gebunden. |
| `X-Case-Scope` | string | optional | Zusätzliche Zugriffsscope für Agenten-Workflows. |
| `Authorization` | string | required für LiteLLM Admin | `Bearer <API-Key>` gemäß LiteLLM Master Key Policy. Weitere Details in der [LiteLLM Admin Referenz](./litellm-admin.md). |

**Tenancy-Hinweis:** Das Schema kann automatisch aus der anfragenden Domain (z. B. `tenant.example.com`) abgeleitet werden. Für interne Skripte ist der Header dennoch Pflicht, um versehentliche Schema-Wechsel zu verhindern.

**Idempotente POST-Requests:** Setzen Sie einen stabilen `Idempotency-Key` pro fachlichem Vorgang (UUID oder Hash). Serverseitig wird der erste Abschluss pro Schlüssel persistiert; Wiederholungen liefern denselben `X-Trace-Id` sowie ein Flag `"idempotent": true` im Response-Body.

**PII-Handling:** Eingehende Prompts, Dokumentinhalte und Modellantworten werden automatisch via PII-Scope maskiert, bevor sie persistiert oder an Langfuse übermittelt werden.

### Beispiel-cURL

```bash
curl -X POST "https://api.noesis.example/ai/scope/" \
  -H "X-Tenant-Schema: acme_prod" \
  -H "X-Tenant-Id: acme" \
  -H "X-Case-Id: crm-7421" \
  -H "Idempotency-Key: 1d1d8aa4-0f2e-4b94-8e41-44f96c42e01a" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Erstelle Meeting-Notizen"}'
```

## System Endpunkte

### GET `/ai/ping/`
Kurzer Health-Check der AI-Core-Anwendung.

**Headers**
- `X-Tenant-Schema` (required)
- `X-Tenant-Id` (required)
- `X-Case-Id` (required)

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
- `X-Case-Id` (required)
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
- `X-Case-Id` (required)
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
- `X-Case-Id` (required)
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

#### Soft-Delete Sichtbarkeit

- Requests akzeptieren optional das Feld `visibility` mit den Werten
  `"active"`, `"all"` oder `"deleted"`. Standard ist `"active"` und blendet
  Soft-Deletes vollständig aus.
- `"all"` bzw. `"deleted"` werden nur ausgeführt, wenn der Guard den Request als
  administrativ bestätigt (aktive Admin-Profile oder erlaubte
  Service-Keys). Andernfalls wird automatisch auf `"active"` zurückgefallen.
- Responses spiegeln die angewendete Policy unter
  `meta.visibility_effective`, damit Clients Debugging und Observability
  vereinheitlichen können.

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
| `RAG_HYBRID_ALPHA` | `0.7` | Gewichtung der semantischen Similarität in der Late-Fusion (0 = nur Lexikalik). |
| `RAG_CHUNK_TARGET_TOKENS` | `450` | Zielgröße pro Chunk in Tokens für die Vorverarbeitung. |
| `RAG_CHUNK_OVERLAP_TOKENS` | `80` | Token-Overlap zwischen aufeinanderfolgenden Chunks. |

Bei fehlenden `%`-Treffern wird automatisch auf eine reine Similarity-Sortierung
zurückgefallen.

## Agenten (Queue `agents`)

### POST `/ai/scope/`
Validiert Zugriffsscope und Kontextinformationen für eine Anfrage. Triggert den LangGraph-Knoten `scope_check`.

> **Hinweis:** Der Graph persistiert den Zustand und führt das Request-JSON als
> shallow overwrite mit dem bestehenden State zusammen.

**Headers**
- `X-Tenant-Schema` (required)
- `X-Tenant-Id` (required)
- `X-Case-Id` (required)
- `Idempotency-Key` (optional)
- `Content-Type: application/json`

**Body Schema**
```json
{
  "prompt": "Zeige den Projektstatus", 
  "conversation": [
    {"role": "user", "content": "Was ist der letzte Stand?"}
  ],
  "scope_overrides": {"projects": ["alpha"]}
}
```

**Response 200 Beispiel**
```json
{
  "scope": {
    "projects": ["alpha"],
    "pii_allowed": false
  },
  "idempotent": false,
  "trace_id": "4fcb5171d5264dd2919c91174f9bcf75"
}
```

**Fehler**
- `400 Bad Request`: Ungültige JSON-Struktur.
- `404 Not Found`: Tenant existiert nicht oder besitzt keinen aktiven Scope.

### POST `/ai/intake/`
Startet den Agenten-Flow `info_intake` zur Kontextanreicherung.

> **Hinweis:** Der Graph persistiert den Zustand und führt das Request-JSON als
> shallow overwrite mit dem bestehenden State zusammen.

**Headers**
- `X-Tenant-Schema` (required)
- `X-Tenant-Id` (required)
- `X-Case-Id` (required)
- `Idempotency-Key` (optional)
- `Content-Type: application/json`

**Body Schema**
```json
{
  "prompt": "Fasse das Kundenfeedback zusammen",
  "metadata": {"channel": "email"}
}
```

**Response 200 Beispiel**
```json
{
  "summary": "Kunde meldet Lieferverzug und wünscht Rückruf.",
  "suggested_entities": ["Bestellung-845"],
  "trace_id": "e6a3ff14f1a84154b92f0841a1ba46f9"
}
```

### POST `/ai/needs/`
Ermittelt offene Aufgaben und Anforderungen (`needs_mapping`).

> **Hinweis:** Der Graph persistiert den Zustand und führt das Request-JSON als
> shallow overwrite mit dem bestehenden State zusammen.

**Headers** wie `/ai/intake/`.

**Body Beispiel**
```json
{
  "conversation": [
    {"role": "user", "content": "Wir brauchen ein Angebot bis Freitag."}
  ]
}
```

**Response 200 Beispiel**
```json
{
  "needs": [
    {
      "type": "offer",
      "due": "2024-05-19",
      "confidence": 0.88
    }
  ],
  "trace_id": "7f0f7d8a0e994a85a0a3c5d509a5bd1a"
}
```

### POST `/ai/sysdesc/`
Erzeugt eine Systembeschreibung zur Weiterleitung an nachgelagerte Agenten (`system_description`).

> **Hinweis:** Der Graph persistiert den Zustand und führt das Request-JSON als
> shallow overwrite mit dem bestehenden State zusammen.

**Headers** wie `/ai/intake/`.

**Body Beispiel**
```json
{
  "context": {
    "tenant_features": ["rag", "chat"],
    "language": "de"
  }
}
```

**Response 200 Beispiel**
```json
{
  "system_prompt": "Du bist der NOESIS-2 Assistent für Tenant acme.",
  "guardrails": ["keine PII ausgeben"],
  "trace_id": "a5d059fbe33b4a7fa6bbfd93a8f41d4e"
}
```
