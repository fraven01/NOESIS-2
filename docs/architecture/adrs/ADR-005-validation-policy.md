# ADR-005: ID-Validation-Policy

## Status
Accepted

## Kontext
Graph-Inputs nutzen Pydantic mit `extra=forbid`; `assert_case_active` erzwingt aktive Cases; Tools verlangen `ToolContext`. Unklar war, wann `case_id` Pflicht ist und welche Fallbacks gelten.

## Entscheidung
- **Fail-fast**: `tenant_id`, `trace_id`, genau eine Laufzeit-ID (`run_id` XOR `ingestion_run_id`) sind immer Pflicht. `case_id` ist Pflicht für fachliche Workloads, optional für System-Tasks (Cron-Crawler, Backfill).
- **Kein Default-Case**: Fehlendes `case_id` wird nicht auf einen globalen Default gemappt. Auto-Create wird nur genutzt, wenn explizit angefordert.
- **Validation-Ort**: HTTP-Views nutzen `assert_case_active`; Graph-Kontexte (`GraphContextPayload`, `WebSearchContext`) erzwingen Nicht-Leerwerte; Tools validieren Inputs vor externer API-Kommunikation.
- **Error-Pfade**: Fehlende IDs erzeugen 4xx/ValidationErrors; Worker brechen ab, bevor externe Ressourcen angefasst werden. System-Tasks loggen Warnungen, wenn `case_id` fehlt.

## Konsequenzen
- Entwickler:innen müssen in Dispatcher/Views entscheiden, ob ein Flow fachlich oder systemisch ist und `case_id` entsprechend setzen.
- Observability bleibt vollständig, weil `trace_id`/Laufzeit-IDs immer verfügbar sind.
- Retry-Logik wird vereinfacht: Idempotenz-Schlüssel gelten pro (`tenant_id`, `trace_id`, `run_id`/`ingestion_run_id`).

