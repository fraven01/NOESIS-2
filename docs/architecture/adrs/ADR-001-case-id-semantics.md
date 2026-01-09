# ADR-001: Case-ID als stabiler fachlicher Container

## Status
Accepted

## Kontext
Cases (UUID PK, `external_id`, `status`, `phase`) dienen als fachliche Container. Graph-Boundaries (GraphIOSpec + ToolContext) verlangen `case_id`, und `assert_case_active` blockiert geschlossene Cases.

## Entscheidung
- `case_id` repräsentiert den langfristigen fachlichen Vorgang (Wochen/Monate) und bündelt mehrere Workflows.
- Graphen und Tools führen `case_id`, sobald fachlicher Kontext verfügbar ist; reine System-Tasks dürfen `case_id` leer lassen.
- `case_id` wird niemals aus `trace_id` oder Laufzeit-IDs abgeleitet; fehlende Cases führen zu Validierungsfehlern, außer wenn Auto-Create explizit aktiviert ist.

## Konsequenzen
- Event-Schemas (`CaseEvent`) tragen stets `case_id`, auch nach Abschluss des Cases.
- Dispatcher müssen `case_id` verpflichtend setzen, wenn ein fachlicher Nutzer-Flow gestartet wird.
- System-Crons (z. B. Crawler ohne Case) markieren Events mit `system_task=true`.

