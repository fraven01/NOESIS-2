# LiteLLM Admin API Referenz

Diese Referenz ergänzt die [NOESIS-2 API Übersicht](./reference.md) und beschreibt ausschließlich die Administrationsendpunkte des LiteLLM-Proxys. Alle Requests müssen über TLS erfolgen und respektieren die gleichen Multi-Tenancy-Header (`X-Tenant-Schema`, `X-Tenant-Id`, `X-Case-Id`) wie im Hauptdokument. Responses werden automatisch in Langfuse erfasst.

## Authentifizierung & Header

LiteLLM Admin-Routen nutzen Master-Key Authentifizierung.

| Header | Typ | Pflicht | Beschreibung |
| --- | --- | --- | --- |
| `Authorization` | string | required | `Bearer <API-Key>` entsprechend der LiteLLM Master-Key-Policy. |
| `X-Tenant-Schema` | string | optional | Überschreibt die automatische Schema-Auflösung (Subdomain). |
| `X-Tenant-Id` | string | optional | Mandantenkontext für Logging & Rate-Limits. |
| `X-Trace-Id` | string | optional (response) | Wird serverseitig generiert und in Langfuse gespiegelt. |
| `Idempotency-Key` | string | optional | Empfohlen für POST-Requests, um Wiederholungen deterministisch zu halten. |

**Sicherheitsrisiken:** Fehlende `Authorization`-Header führen zu `401 Unauthorized`. Wird ein fremder Tenant referenziert, antwortet der Dienst mit `404 Not Found`.

### Beispiel-cURL

```bash
curl -X GET "https://litellm.noesis.example/keys" \
  -H "Authorization: Bearer <MASTER_KEY>" \
  -H "X-Tenant-Schema: acme_prod" \
  -H "X-Tenant-Id: acme"
```

## Endpunkte

### GET `/health`
Health-Status des LiteLLM-Proxys.

**Response 200 Beispiel**
```json
{
  "status": "ok",
  "models": [
    {
      "name": "gpt-4o",
      "healthy": true,
      "current_rpm": 42
    }
  ],
  "trace_url": "https://app.langfuse.com/project/noesis/traces/9967"
}
```

**Fehler**
- `401 Unauthorized`: Fehlender oder ungültiger Master-Key.
- `429 Too Many Requests`: Rate-Limit überschritten.

### GET `/keys`
Listet vorhandene LiteLLM API-Keys samt Metadaten.

**Query Parameter**
- `limit` (optional, integer): Anzahl der zurückgegebenen Keys (Default: 20).
- `offset` (optional, integer): Startindex für Pagination.

**Response 200 Beispiel**
```json
{
  "results": [
    {
      "alias": "agents-default",
      "owner": "platform",
      "rpm_limit": 120,
      "tpm_limit": 90000,
      "created_at": "2024-04-16T12:14:53Z"
    }
  ],
  "count": 1,
  "next": null,
  "previous": null,
  "trace_id": "bd3f013d0242487bb2cdf9b14d73e2a1"
}
```

**Fehler**
- `401 Unauthorized`: Fehlender oder ungültiger Master-Key.

### POST `/keys`
Erzeugt einen neuen LiteLLM API-Key oder aktualisiert die Limits eines bestehenden Schlüssels. Idempotent bei Verwendung desselben `alias` in Kombination mit `Idempotency-Key`.

**Body Schema**
```json
{
  "alias": "agents-default",
  "owner": "platform",
  "rpm_limit": 120,
  "tpm_limit": 90000
}
```

**Response 201 Beispiel**
```json
{
  "alias": "agents-default",
  "owner": "platform",
  "rpm_limit": 120,
  "tpm_limit": 90000,
  "idempotent": true,
  "trace_id": "2f08da701ee44b338eaeb98f61299461"
}
```

**Fehler**
- `400 Bad Request`: Fehlende Pflichtfelder oder ungültige Limits.
- `401 Unauthorized`: Fehlender oder ungültiger Master-Key.
- `409 Conflict`: `alias` bereits vergeben und kein Idempotency-Key gesetzt.

### POST `/rate-limits/check`
Prüft aktuelle Auslastung gegen konfigurierte Rate-Limits.

**Body Schema**
```json
{
  "alias": "agents-default"
}
```

**Response 200 Beispiel**
```json
{
  "alias": "agents-default",
  "rpm_limit": 120,
  "rpm_current": 64,
  "tpm_limit": 90000,
  "tpm_current": 12000,
  "trace_id": "d1d9e76fb2d5496db029f4ec762b2c13"
}
```

**Fehler**
- `401 Unauthorized`: Fehlender oder ungültiger Master-Key.
- `404 Not Found`: Alias unbekannt.

### DELETE `/keys/{alias}`
Löscht einen bestehenden LiteLLM API-Key. Responses beinhalten `trace_id` zur Nachverfolgung in Langfuse.

**Response 204 Beispiel**
```json
{}
```

**Fehler**
- `401 Unauthorized`: Fehlender oder ungültiger Master-Key.
- `404 Not Found`: Alias existiert nicht.
