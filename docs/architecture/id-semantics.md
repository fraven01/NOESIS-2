# ID-Semantik und Lebenszyklen

Dieses Dokument definiert die verbindliche Bedeutung jeder ID in der Multi-Tenant-Plattform. Es ergänzt `docs/ids.md` um eindeutige Lebensdauerregeln, Abgrenzungen und Generierungsquellen für Graphen, Worker und Tools.

## Kern- und Laufzeit-IDs

| ID | Lebensdauer | Erzeugung | Fachliche Rolle | Technische Rolle |
| --- | --- | --- | --- | --- |
| `tenant_id` | dauerhaft je Mandant | Header `X-Tenant-ID`, Token oder Worker-Scope | steuert Mandantenschema und Policies | muss end-to-end propagiert werden; keine Ableitung aus anderen IDs |
| `case_id` | Wochen/Monate pro fachlichem Vorgang | Request-Header oder Dispatcher; für System-Tasks optional | bündelt alle Workflows und Events eines Vorgangs | Pflicht, sobald fachlicher Kontext vorliegt; nur systemische Tasks dürfen ohne laufen |
| `workflow_id` | stabil pro fachlichem Workflow innerhalb eines Cases | Call-Site/Dispatcher vergibt; Graph selbst erzeugt **keine** neue ID | beschreibt Prozess-Typ (z. B. `collection_search`, `external_knowledge`) | bleibt über Wiederholungen identisch und wird in Events gespiegelt |
| `run_id` | eine Ausführung eines Workflows (synchroner Graph-Lauf) | vom Graph-Dispatcher oder Worker generiert | keine fachliche Bedeutung | genau eine Laufzeit-ID pro Tool/Span; nie gemeinsam mit `ingestion_run_id` |
| `ingestion_run_id` | eine Ingestion-Ausführung (Upload/Crawler) | der Einstiegspunkt, der Ingestion startet, erzeugt sie | markiert Ingestion-Jobs je Case/Collection | mutual exclusion zu `run_id`; wird in Ingestion-Tools weitergereicht |
| `trace_id` | Request-/Flow-basiert | Web-Layer oder API ruft `X-Trace-ID` ab oder erzeugt sie | Korrelation für Auditing/HITL | muss über HTTP → Django → Celery → LangGraph → Tools unverändert bleiben |
| `workflow_run_id` | aliasfrei nicht verwenden | — | — | nicht befüllen (historische Bezeichnung) |

### Ableitungen und Redundanz

- `tenant_id` darf **nicht** aus anderen IDs abgeleitet werden, um Schema-Vermischung auszuschließen. Sie wird immer explizit übertragen.
- `case_id` wird nie aus `trace_id`/`run_id` rekonstruiert; fehlende Cases lösen Validierungsfehler aus (siehe Validation-ADR).
- `workflow_id` wird nicht aus Funktionsnamen generiert: Web-Layer, Scheduler oder Dispatcher setzt sie vor dem Graph-Start.

### Beziehungen

- 1 `tenant_id` → N `case_id`
- 1 `case_id` → N `workflow_id`
- 1 `workflow_id` → N `run_id`
- Ingestion-Flows nutzen `ingestion_run_id` statt `run_id`; Kanten zu Cases/Workflows werden über Kontextfelder gesetzt, nicht impliziert.

## ID-Quellen je Schicht

- **HTTP/Django**: `normalize_request` erstellt `ScopeContext` (Single Source of Truth).
- **Celery Dispatcher**: Serialisiert `ScopeContext` und übergibt ihn an Worker.
- **LangGraph Runner**: Deserialisiert `ScopeContext` und validiert ihn erneut.
- **Tools**: Erhalten `ToolContext` (abgeleitet aus `ScopeContext`).
- **Observability**: Nutzt `ScopeContext` zur konsistenten Befüllung von Spans.

## ID-Löschung und Archivierung

- `case_id` bleibt über Schließung erhalten; Events nach Abschluss sind erlaubt, aber Write-Workloads werden blockiert (Validation-Policy).
- `run_id` und `ingestion_run_id` werden nicht wiederverwendet; Persistenz nur in Logs/Events.
- `trace_id` kann in Langfuse archiviert werden, bleibt aber in Events referenzierbar.

## XOR-Regel für Laufzeit-IDs

- Jeder `ScopeContext` und `ToolContext` führt **genau eine** Laufzeit-ID: `run_id` **oder** `ingestion_run_id`.
- Diese Regel wird durch Pydantic-Validatoren im `ScopeContext` strikt durchgesetzt.
- Graphen, die Ingestion triggern, erzeugen einen neuen `ScopeContext` mit `ingestion_run_id` am Trigger-Punkt.
