# ADR-002: Workflow-ID Vergabe

## Status
Accepted

## Kontext
Graph-Kontexte validieren `workflow_id` als Pflichtfeld. `workflow_id` ist in Logs/Spans und Tool-Metadaten präsent, wird aber aktuell variabel vergeben.

## Entscheidung
- `workflow_id` wird **immer** vom aufrufenden Dispatcher vergeben (HTTP-View, Scheduler oder übergeordneter Graph) und beschreibt den fachlichen Workflow-Typ.
- LangGraph-Implementierungen erzeugen keine neuen `workflow_id`-Werte; Wiederholungen desselben Workflows behalten die ID.
- Sub-Graph-Aufrufe dürfen eine **eigene** `workflow_id` wählen, müssen aber den `trace_id` des Eltern-Workflows beibehalten.

## Konsequenzen
- Observability kann Workflow-Wiederholungen über identische `workflow_id` und wechselnde `run_id` clustern.
- Scheduler-Konfigurationen müssen `workflow_id` setzen; Default-Werte in Code sind nicht zulässig.
- Tool-Schemas dokumentieren `workflow_id` als optionales Feld, das bei vorhandenem fachlichem Kontext gefüllt wird.

