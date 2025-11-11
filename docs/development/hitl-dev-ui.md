# Developer HITL UI

Die Entwickleransicht für Human-in-the-Loop Reviews ist ein isoliertes Mock-System,
das den Hybrid-Rerank-Output interaktiv testbar macht. Die Ansicht liegt bewusst
außerhalb der produktiven RAG-Tools und ist nur mit einem Feature-Flag aktivierbar.

## Aktivierung

1. Setze das Feature-Flag in der `.env` oder via Umgebung:

   ```bash
   export DEV_FEATURE_HITL_UI=true
   ```

2. Starte den Django-Server neu. Die Seite ist anschließend unter
   [`/dev/hitl/`](http://localhost:8000/dev/hitl/) erreichbar.

3. Alle API-Aufrufe müssen den Header `_DEV_ONLY_=true` mitsenden. Für den
   Streaming-Endpunkt (`EventSource`) kann alternativ der Query-Parameter
   `?dev_token=true` genutzt werden.

## Endpunkte

| Route | Methode | Beschreibung |
| --- | --- | --- |
| `/dev/hitl/` | `GET` | Rendert die React-freie Dev-Ansicht inkl. initialem Payload (`<script id="hitl-initial-data">`). |
| `/dev/hitl/runs/<run_id>/` | `GET` | Liefert das Mock-Run-JSON (`top_k`, `coverage_delta`, `meta`). |
| `/dev/hitl/approve-candidates/` | `POST` | Nimmt Entscheidungen entgegen (`approved_ids`, `rejected_ids`, `custom_urls`). |
| `/dev/hitl/progress/<run_id>/stream/` | `GET` | Server-Sent Events (SSE) für Ingestion-, Coverage- und Deadline-Updates. |

### Beispielantwort für `GET /dev/hitl/runs/<run_id>/`

```json
{
  "run_id": "demo-run",
  "top_k": [{
    "id": "demo-run-cand-1",
    "title": "Vendor observability guide for release automation",
    "fused_score": 7.9,
    "score": 86,
    "gap_tags": ["MONITORING_SURVEILLANCE"],
    "risk_flags": ["requires_authentication"],
    "domain": "docs.vendor.example.com",
    "detected_date": "2024-10-01T08:00:00Z",
    "source": "web"
  }],
  "coverage_delta": {
    "summary": "Monitoring- und Audit-Facetten verbessern sich signifikant.",
    "facets_before": {"TECHNICAL": 0.52},
    "facets_after": {"TECHNICAL": 0.71}
  },
  "meta": {
    "tenant_id": "tenant-dev",
    "case_id": "case-demo-run",
    "deadline_utc": "2024-01-01T12:00:00Z",
    "min_diversity_buckets": 3,
    "freshness_mode": "software_docs_strict",
    "rag_unavailable": false,
    "llm_timeout": false,
    "cache_hit_rag": true,
    "cache_hit_llm": false
  }
}
```

### Request Schema für `POST /dev/hitl/approve-candidates/`

```json
{
  "run_id": "demo-run",
  "approved_ids": ["demo-run-cand-1", "demo-run-cand-2"],
  "rejected_ids": ["demo-run-cand-3"],
  "custom_urls": ["https://example.com/zusatz"]
}
```

Der Endpunkt ist idempotent: gleiche Payload innerhalb von fünf Minuten liefert
das identische Antwort-JSON mit denselben Task-IDs. Undo-Aktionen werden als
inverse Payload gesendet (approved/rejected getauscht, `custom_urls` leer).

## UI-Verhalten

- Tabelle mit Sortierung nach `fused_score` (Toggle über Header).
- Checkboxen unterstützen Tastaturkürzel (`a`, `Shift+a`, `Enter`, `Backspace`).
- Countdown zeigt „Auto-Approve in mm:ss“, wechselt nach Ablauf auf
  „Auto-approved“. Undo ist fünf Minuten aktiv und zeigt die verbleibende Zeit an.
- Custom-URL-Feld prüft http/https-URLs clientseitig, ungültige Zeilen werden
  markiert und nicht gesendet.
- Progress-Panel zeigt SSE-Updates (`queued`, `running`, `done`, `failed`) und
  fasst den aktuellen Status in einer Badgeleiste zusammen.
- Coverage-Karte visualisiert Facet-Änderungen (Before/After) und den Ingestion-Fortschritt.

## Telemetrie & Logging

- Clientseitig werden Dev-Logs über `console.info('hitl.dev.telemetry', …)`
  geschrieben (Aktion, Anzahl Selektionen, Custom-URL-Anzahl, Undo genutzt?).
- Serverseitig loggt der Store strukturierte Events über `hitl.dev.*` (z. B.
  `approval_received`, `auto_approve_triggered`, `progress_event`).

## Sicherheit

- Feature-Flag standardmäßig aus (`DEV_FEATURE_HITL_UI=false`).
- Kein Zugriff ohne `_DEV_ONLY_`-Header bzw. `dev_token=true`.
- CSRF-Schutz auf POST aktiviert.
- SSE-Stream ignoriert Events anderer Runs automatisch.

Die Dev-HITL-Ansicht ist ausschließlich für lokale Tests gedacht und besitzt
keine produktiven Seiteneffekte.
