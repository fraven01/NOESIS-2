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
Lädt Rohdokumente in den Object-Store hoch und legt einen Ingestion-Job an.

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
  "document_id": "doc_8fb6f3f4",
  "ingestion_job_id": "job_7c92f4",
  "idempotent": false,
  "trace_id": "b1ca46f2191b44abbb74116bb6c1b724"
}
```

**Fehler**
- `400 Bad Request`: Kein File-Part oder ungültige Metadaten.
- `409 Conflict`: Wiederholter Upload ohne `Idempotency-Key`.

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
  "top_k": 5
}
```

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
  ]
}
```

**cURL Beispiel**
```bash
curl -X POST "https://api.noesis.example/ai/v1/rag-demo/" \
  -H "X-Tenant-Schema: acme_prod" \
  -H "X-Tenant-Id: acme" \
  -H "X-Case-Id: crm-7421" \
  -H "Content-Type: application/json" \
  -d '{"query": "Wie konfiguriere ich Tenant-Filter?", "top_k": 3}'
```

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
