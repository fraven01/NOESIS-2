# Tool-Verträge des AI Core

Diese Referenz beschreibt die in `ai_core/tool_contracts/base.py` definierten Hüllmodelle.
Sie bilden die verbindliche Schnittstelle zwischen LangGraph-Agenten und ihren Tools.
Alle Modelle sind Pydantic-`BaseModel`-Klassen mit `frozen=True`, d. h. Instanzen sind unveränderlich.

## ToolContext

Laufzeitmetadaten, die jeden Tool-Aufruf begleiten.

| Feld | Typ | Pflicht? | Validierung / Hinweise |
| --- | --- | --- | --- |
| `tenant_id` | `UUID` | ja | Mandantenkennzeichen, wird aus Request-Header übernommen. |
| `trace_id` | `str` | ja | Korrelierte Trace-ID; Middleware liest `X-Trace-Id`, `X-Request-ID`, Query- oder Body-Feld sowie `traceparent`. |
| `invocation_id` | `UUID` | ja | Eindeutige ID pro Tool-Ausführung. |
| `now_iso` | `datetime` | ja | Muss timezone-aware sein; Validator erzwingt UTC-Konvertierung. |
| `run_id` | `str\|None` | bedingt | Graph-Lauf-ID für Agenten/RAG-Flows; genau eins von `run_id`/`ingestion_run_id` muss gesetzt sein. |
| `ingestion_run_id` | `str\|None` | bedingt | Upload-/Ingestion-Lauf-ID; exklusiv zu `run_id`. |
| `workflow_id` | `str\|None` | optional | Optionale Workflow-Kennung (Graph-Name oder Business-Prozess). |
| `collection_id` | `str\|None` | optional | Logischer Dokument-Scope für Deduplikation & Filter. |
| `document_id` | `str\|None` | optional | Dokument-Referenz, falls bekannt. |
| `document_version_id` | `str\|None` | optional | Versionierte Dokument-Kennung. |
| `case_id` | `str\|None` | optional | Business-Kontext (z. B. CRM- oder Ticket-ID). |
| `idempotency_key` | `str\|None` | optional | Wiederverwendungskennung zur Deduplikation. |
| `timeouts_ms` | `PositiveInt\|None` | optional | Gesamtzeitbudget in Millisekunden (>0). |
| `budget_tokens` | `int\|None` | optional | Modellbudget in Token. |
| `locale` | `str\|None` | optional | Locale-Tag (`de-DE`, `en-US`, …). |
| `safety_mode` | `str\|None` | optional | Guardrail-Profilname. |
| `auth` | `dict[str, Any]\|None` | optional | Weitergereichte Auth-/Credential-Daten. |

**Beispiel**

```json
{
  "tenant_id": "5aa31da6-9278-4da0-9f1a-61b8d3edc5cc",
  "request_id": "8607a8d9-0f3f-43df-bf86-37e845e1574c",
  "trace_id": "trace-123",
  "run_id": "graph_run_cfe213",
  "workflow_id": "retrieval_augmented_generation",
  "invocation_id": "0f4e6712-6d04-4514-b6cb-943b0667d45c",
  "now_iso": "2024-05-03T12:34:56.123456+00:00",
  "collection_id": "hr-onboarding",
  "case_id": "crm-7421",
  "locale": "de-DE",
  "timeouts_ms": 120000,
  "budget_tokens": 4096
}
```

**Hinweis:** Ein Modell-Validator erzwingt, dass exakt eine der IDs `run_id` oder `ingestion_run_id` vorhanden ist. Client- und Service-Schichten erzeugen passende Werte (`run_...` für Graph-Ausführungen, `ingest_...` für Uploads) und spiegeln sie in Langfuse-Tags sowie Persistenzschichten wider.

## ToolResult & ToolResultMeta

Erfolgreiche Antworten bestehen aus einem `ToolResult`-Envelope und einem frei definierbaren Datenmodell (`OT`).

### ToolResultMeta

| Feld | Typ | Pflicht? | Validierung / Hinweise |
| --- | --- | --- | --- |
| `took_ms` | `NonNegativeInt` | ja | Dauer des Aufrufs in Millisekunden (>=0). |
| `source_counts` | `dict[str, int]\|None` | optional | Aggregierte Quellinformationen; Werte >=0. |
| `routing` | `dict[str, Any]\|None` | optional | Angaben zur Profilwahl (`embedding_profile`, `vector_space_id`, …). |
| `cache_hit` | `bool\|None` | optional | Kennzeichnet Cache-Treffer. |
| `token_usage` | `dict[str, int]\|None` | optional | Prompt-/Completion-Tokenzählung. |

### ToolResult

| Feld | Typ | Pflicht? | Validierung / Hinweise |
| --- | --- | --- | --- |
| `status` | Literal `"ok"` | ja | Diskriminator für `ToolOutput`. |
| `input` | `IT` | ja | Echo des Eingabemodells. |
| `data` | `OT` | ja | Ergebnismodell des Tools. |
| `meta` | `ToolResultMeta` | ja | Metadaten wie oben beschrieben. |

**Beispiel**

```json
{
  "status": "ok",
  "input": {"query": "hello"},
  "data": {"answer": "Hi there!"},
  "meta": {
    "took_ms": 42,
    "token_usage": {"prompt": 12, "completion": 20}
  }
}
```

## ToolError, ToolErrorDetail & ToolErrorMeta

Fehlerhafte Aufrufe verwenden den `ToolError`-Envelope.
Die Felder `status="error"` und `ToolErrorDetail.type` dienen als Diskriminatoren.

### ToolErrorMeta

| Feld | Typ | Pflicht? | Validierung / Hinweise |
| --- | --- | --- | --- |
| `took_ms` | `NonNegativeInt` | ja | Dauer bis zur Fehlererkennung. |

### ToolErrorDetail

| Feld | Typ | Pflicht? | Validierung / Hinweise |
| --- | --- | --- | --- |
| `type` | `ToolErrorType` | ja | Standardisierte Fehlerklasse (siehe unten). |
| `message` | `str` | ja | Menschlich lesbare Beschreibung. |
| `code` | `str\|None` | optional | Tool-spezifischer Fehlercode. |
| `cause` | `str\|None` | optional | Technische Ursache oder Exception-Name. |
| `details` | `dict[str, Any]\|None` | optional | Kontextobjekt für strukturierte Zusatzinfos. |
| `retry_after_ms` | `int\|None` | optional | Retry-Backoff in Millisekunden. |
| `upstream_status` | `int\|None` | optional | HTTP-/gRPC-Status eines Upstreamdienstes. |
| `endpoint` | `str\|None` | optional | Upstream-Endpunkt oder Queue-Name. |
| `attempt` | `int\|None` | optional | Aktuelle Wiederholungsnummer. |

### ToolError

| Feld | Typ | Pflicht? | Validierung / Hinweise |
| --- | --- | --- | --- |
| `status` | Literal `"error"` | ja | Diskriminator für `ToolOutput`. |
| `input` | `IT` | ja | Echo des Eingabemodells. |
| `error` | `ToolErrorDetail` | ja | Strukturierte Fehlerangaben. |
| `meta` | `ToolErrorMeta` | ja | Messdaten zum fehlgeschlagenen Aufruf. |

**Beispiel**

```json
{
  "status": "error",
  "input": {"query": "hello"},
  "error": {
    "type": "VALIDATION",
    "message": "Missing required field",
    "code": "missing_field",
    "retry_after_ms": null
  },
  "meta": {"took_ms": 10}
}
```

### ToolErrorType

`ToolErrorDetail.type` ist eine `StrEnum`, deren Werte in `ai_core/tools/errors.py` definiert sind:

- `RATE_LIMIT` – Upstream oder Plattform limitiert den Durchsatz.
- `TIMEOUT` – Zeitbudget überschritten oder Upstream-Timeout.
- `UPSTREAM` – Fehler im nachgelagerten Dienst.
- `VALIDATION` – Eingabe oder Antwort verletzt das Schema.
- `RETRYABLE` – Temporärer Fehler, erneuter Versuch empfohlen.
- `FATAL` – Nicht behebbarer Fehler, manuelles Eingreifen nötig.

## ToolOutput (Discriminated Union)

`ToolOutput[IT, OT]` ist eine `typing.Annotated`-Union aus `ToolResult` und `ToolError` mit dem Diskriminatorfeld `status`.
Dadurch können Parser (z. B. LangChain, FastAPI) das Ergebnis typen-sicher deserialisieren.

## JSON-Schema-Exports

Alle Modelle unterstützen `model_json_schema()`.
Damit lassen sich automatisch aktualisierte JSON-Schemas für API- oder Tool-Autoren exportieren:

```python
from ai_core.tool_contracts.base import ToolContext, ToolResult

schema = ToolResult[MyInput, MyOutput].model_json_schema()
```

Die Examples in den Modellen werden im Schema inkludiert und dienen als Test-Fixtures oder Dokumentation.
