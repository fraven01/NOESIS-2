# Dokument-Observability

Dieser Leitfaden bündelt Logging-, Tracing- und Metrik-Konventionen für den Dokument-Stack (Repository, Storage, CLI, Caption-Pipeline). Alle Beispiele entsprechen der aktuellen Implementierung (`documents.logging_utils`, `documents.metrics`, `documents.captioning`, `documents.repository`, `documents.storage`, `documents.cli`).

## Logging-Konventionen

### Pflichtfelder

Jedes Log-Event wird als JSON serialisiert und enthält – sofern im Kontext verfügbar – folgende Schlüssel:

| Feld | Beschreibung |
| --- | --- |
| `event` | Maschinenlesbarer Bezeichner (z. B. `docs.upsert`, `assets.caption.run`). |
| `phase` | Nur beim Eintritt (`"start"`), bevor Status vorliegt. |
| `status` | Abschlussstatus (`"ok"` oder `"error"`). |
| `duration_ms` | Dauer der Operation in Millisekunden (Exit-Log). |
| `tenant_id` | Aktiver Mandant. |
| `document_id`, `collection_id`, `version` | Falls bekannt aus `DocumentRef`. |
| `asset_id` | Falls aus `AssetRef` verfügbar. |
| `source` | Dokumentquelle (`upload`, `crawler`, `integration`, `other`). |
| `size_bytes` | Größe des betroffenen Blobs. |
| `uri_kind` | Speicherklasse (`memory`, `s3`, `gcs`, `http`). |
| `sha256_prefix` | Erste acht Zeichen des Payload-Checksums. |
| `model`, `caption_method`, `caption_confidence` | Caption-spezifische Felder, falls vorhanden. |
| `trace_id`, `span_id` | OpenTelemetry-Kontext zur Trace-Korrelation. |

PII oder Rohinhalte (Base64, Text) werden **nie** geloggt; ausschließlich Metadaten, Grössen und Hash-Präfixe sind zulässig.

### Beispiel-Log

```json
{
  "event": "docs.upsert",
  "phase": "start",
  "tenant_id": "tenant-a",
  "document_id": "7f5e...",
  "trace_id": "2c3d7a5f0d904891a7e3f58c7f0c1234",
  "span_id": "1ab2c3d4e5f60708"
}
```

```json
{
  "event": "docs.upsert",
  "status": "ok",
  "duration_ms": 12.4,
  "tenant_id": "tenant-a",
  "document_id": "7f5e...",
  "collection_id": "25b7...",
  "sha256_prefix": "0f5e1a2b",
  "size_bytes": 16384,
  "asset_count": 2,
  "trace_id": "2c3d7a5f0d904891a7e3f58c7f0c1234",
  "span_id": "1ab2c3d4e5f60708"
}
```

### Event→Span Mapping

| Event/Log-Name | Span-Name | Beschreibung |
| --- | --- | --- |
| `docs.upsert` | `docs.upsert` | Persistiert oder aktualisiert Dokument und Assets. |
| `docs.get` | `docs.get` | Holt Dokument-Version (optional neueste Variante). |
| `docs.list` | `docs.list_by_collection` | Listet Dokumentversionen pro Collection (Cursor). |
| `docs.list_latest` | `docs.list_by_collection.latest` | Aggregiert neueste Version je Dokument. |
| `docs.delete` | `docs.delete` | Soft- oder Hard-Delete eines Dokuments. |
| `assets.add` | `assets.add` | Persistiert ein Asset (inkl. Blob-Materialisierung). |
| `assets.get` | `assets.get` | Liest ein Asset. |
| `assets.list` | `assets.list_by_document` | Listet Assets pro Dokument. |
| `assets.delete` | `assets.delete` | Soft-/Hard-Delete eines Assets. |
| `storage.put` | `storage.put` | Speichert Binärdaten (`memory://`, `s3://`, …). |
| `storage.get` | `storage.get` | Lädt Binärdaten. |
| `pipeline.assets_caption` | `pipeline.assets_caption` | Gesamter Caption-Lauf für ein Dokument. |
| `pipeline.assets_caption.item` | `pipeline.assets_caption.item` | Einzelnes Asset im Caption-Lauf. |
| `assets.caption.run` | `assets.caption.run` | Caption-Orchestrierung auf Dokumentebene. |
| `assets.caption.process_assets` | `assets.caption.process_assets` | Filtert & aktualisiert Assets für Captioning. |
| `assets.caption.process_collection` | `assets.caption.process_collection` | Batch-Processing über Collections. |
| `assets.caption.load_payload` | `assets.caption.load_payload` | Lädt Bildbytes aus Storage. |
| `cli.schema.print` | `cli.schema.print` | CLI Schema-Ausgabe. |
| `cli.docs.add` | `cli.docs.add` | CLI-Dokument anlegen. |
| `cli.docs.get` | `cli.docs.get` | Dokument abrufen. |
| `cli.docs.list` | `cli.docs.list` | Dokumente einer Collection listen. |
| `cli.docs.delete` | `cli.docs.delete` | Dokument löschen. |
| `cli.assets.add` | `cli.assets.add` | Asset hinzufügen (Inline/File). |
| `cli.assets.get` | `cli.assets.get` | Asset abrufen. |
| `cli.assets.list` | `cli.assets.list` | Assets eines Dokuments listen. |
| `cli.assets.delete` | `cli.assets.delete` | Asset löschen. |
| `cli.main` | `cli.main` | Gesamter CLI-Aufruf. |

Alle Events verwenden `log_context`, sodass `tenant_id`, `document_id`, `collection_id` und `asset_id` automatisch in Logs und Spans erscheinen.

### Whitelist & Maskierung

- Zulässige Felder: oben genannte Pflichtfelder sowie optionale Metadaten (`asset_count`, `limit`, `cursor_present`).
- Verbotene Inhalte: Klartext, Base64, OCR/Text-Content, Secrets.
- Hashes werden ausschließlich als Präfix (`sha256_prefix`) ausgegeben.

## Tracing

Jede Operation erzeugt einen OpenTelemetry-Span mit denselben Namen wie das Event. Attribute folgen dem Namespace `noesis.*`:

| Attribut | Bedeutung |
| --- | --- |
| `noesis.tenant_id` | Mandant der Operation. |
| `noesis.collection_id` | Zugehörige Collection, falls vorhanden. |
| `noesis.document_id` | Dokument-UUID. |
| `noesis.asset_id` | Asset-UUID. |
| `noesis.version` | Dokumentversion (String). |
| `noesis.source` | Quelle des Dokuments. |
| `noesis.uri_kind` | Speicherklasse des Blobs. |
| `noesis.size_bytes` | Blob-Größe. |
| `noesis.caption.model` | Verwendetes Caption-Modell. |
| `noesis.caption.method` | Caption-Strategie (`vlm_caption`, `ocr_only`, …). |
| `noesis.caption.confidence` | Zuverlässigkeit (0.0–1.0). |

Fehler markieren den Span-Status `ERROR` und setzen `error.type` sowie `error.message` (gekürzt). Logs verweisen über `trace_id`/`span_id` exakt auf denselben Span.

**Trace/Log-Korrelation:** In jedem Exit-Log stehen `trace_id` und `span_id`. Die gleichen IDs erscheinen in Langfuse/OTel-Backends, wodurch sich Log-Einträge und Spans verlustfrei verknüpfen lassen.

## Metriken

`documents.metrics` exportiert Counter- und Histogram-Serien mit niedriger Kardinalität (Labels: `event`, `status`).

| Name | Typ | Labels | Beschreibung |
| --- | --- | --- | --- |
| `documents_operation_total` | Counter | `event`, `status` | Dokument-Repository-Aufrufe (`docs.*`). |
| `documents_operation_duration_ms` | Histogram | `event`, `status` | Dauer derselben Operationen. |
| `documents_asset_operation_total` | Counter | `event`, `status` | Asset-Repository-Aufrufe (`assets.*`). |
| `documents_asset_operation_duration_ms` | Histogram | `event`, `status` | Dauer der Asset-Operationen. |
| `documents_storage_operation_total` | Counter | `event`, `status` | Storage-Aufrufe (`storage.*`). |
| `documents_storage_operation_duration_ms` | Histogram | `event`, `status` | Storage-Latenzen. |
| `documents_pipeline_operation_total` | Counter | `event`, `status` | Pipeline-Events (`pipeline.*`, `assets.caption.*`). |
| `documents_pipeline_operation_duration_ms` | Histogram | `event`, `status` | Pipeline-Laufzeiten. |
| `documents_cli_operation_total` | Counter | `event`, `status` | CLI-Aufrufe (`cli.*`). |
| `documents_cli_operation_duration_ms` | Histogram | `event`, `status` | CLI-Laufzeiten. |
| `documents_other_operation_total` | Counter | `event`, `status` | Fallback für unbekannte Events. |
| `documents_other_operation_duration_ms` | Histogram | `event`, `status` | Latenzen für Fallback-Events. |
| `documents_caption_runs_total` | Counter | `status` | Anzahl Caption-Läufe (nur `pipeline.assets_caption`). |
| `documents_caption_duration_ms` | Histogram | `status` | Laufzeiten der Caption-Pipeline. |

### Beispielabfragen

```promql
sum by (status) (documents_operation_total{event="docs.upsert"})
```

```promql
histogram_quantile(0.95,
  sum by (le) (rate(documents_storage_operation_duration_ms_bucket[5m])
))
```

## Runbook

### Storage nicht erreichbar

- **Logs:** `storage.get`/`storage.put` mit `status="error"`, `error_kind="FileNotFoundError"` oder Transportfehler, `uri_kind` signalisiert betroffene Storage-Klasse.
- **Spans:** Status `ERROR`, Attribute enthalten `noesis.uri_kind`. Trace-ID mit Repository-Events korreliert den Auslöser.
- **Metriken:** `documents_storage_operation_total{event="storage.get",status="error"}` steigt, ebenso das Dauer-Histogramm mit langen Latenzen.
- **Checks:** Storage-Endpunkt prüfen, ggf. Memory-Adapter resetten. Nach Recovery `documents_storage_operation_total{status="ok"}` beobachten.

### Captioner-Timeout oder leeres Ergebnis

- **Logs:** `pipeline.assets_caption` oder `assets.caption.run` mit `status="error"` (strict mode) oder `status="ok"` plus `caption_method="ocr_only"` als Fallback.
- **Spans:** `pipeline.assets_caption` markiert Fehler, `assets.caption.item` zeigt das spezifische Asset (`noesis.asset_id`).
- **Metriken:** `documents_pipeline_operation_total{event="pipeline.assets_caption",status="error"}` bzw. Counter mit `status="ok"` + Histogram-Spikes. `documents_caption_runs_total` zeigt den Aggregatstand.
- **Checks:** Captioner-Stub (DeterministicCaptioner) oder externe Integration prüfen, OCR-Fallback validieren.

## Preflight-Checkliste vor Livegang

- `LOG_LEVEL` passend zur Umgebung setzen (`INFO` Prod, `DEBUG` Dev).
- Exporter konfigurieren: Prometheus Endpoint, OTel OTLP-Endpunkt, Langfuse API-Key.
- Sampling verifizieren (`OTEL_TRACES_SAMPLER`/`LANGFUSE_SAMPLE_RATE`).
- Alarm-Schwellen für `documents_storage_operation_total{status="error"}` und `documents_caption_runs_total` definieren.
- CLI- und Pipeline-Smoke-Tests mit aktivierten Logs/Spans ausführen.

## Do / Don’t

| Do | Don’t |
| --- | --- |
| IDs, Hash-Präfixe, Größen loggen. | Rohtext, Base64 oder OCR-Ergebnisse loggen. |
| `log_context` vor jeder Operation setzen. | Logs außerhalb des Decorators mit eigenen Trace-IDs erzeugen. |
| Counter/Historgramme nach Tests via `reset_metrics()` leeren. | Zusätzliche Labels ohne Freigabe hinzufügen (Gefahr hoher Kardinalität). |
| Spans/Logs über `trace_id`/`span_id` korrelieren. | Span-Attribute mit personenbezogenen Daten befüllen. |

