# Glossar Kontext-Identitäten

Dieses Glossar definiert die vier Kern-IDs, die APIs, Graphen, Tools und Services gemeinsam verwenden.

## Definitionen

- **tenant_id**: Identifiziert den Mandanten und schaltet Schema, Berechtigungen und Datenräume. Pflicht in jedem Kontext (API-Header, ToolContext, Graph-Inputs).
- **case_id**: Stabile Kennung eines fachlichen Falls innerhalb eines Tenants. Bündelt Workflows, Dokumente, Kontextdaten und Entscheidungen; muss in jedem fachlich zugehörigen Graphlauf gesetzt sein.
- **workflow_id**: Kennzeichnet den logischen Prozessschritt innerhalb eines Cases (z. B. Intake, Bewertung, Dokumentengenerierung). Bleibt über wiederholte Ausführungen hinweg identisch und wird idealerweise vom Aufrufer oder Dispatcher vergeben, nicht vom Graph.
- **run_id**: Technische Laufzeit-ID für eine konkrete Ausführung eines Workflows durch LangGraph. Jede Ausführung erzeugt eine neue, nicht fachlich interpretierbare ID und gehört genau zu einer `workflow_id` und `case_id`.

## Beziehungsmatrix

Tenant → viele Cases → viele Workflows → viele Runs. Tools benötigen zusätzlich immer `trace_id`, `invocation_id` und genau eine Laufzeit-ID (`run_id` oder `ingestion_run_id`). Graphen übernehmen `case_id` und `workflow_id`, sobald der fachliche Kontext vorliegt; `run_id` wird pro Ausführung neu generiert und bleibt strikt technisch.
