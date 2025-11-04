# ID-Verträge und Semantik

Dieses Dokument definiert die verbindlichen IDs, die im gesamten NOESIS 2-System verwendet werden. Eine klare und konsistente Verwendung dieser IDs ist entscheidend für Tracing, Mandantenfähigkeit, Kostenkontrolle und die allgemeine Systemstabilität.

## Kern-IDs (Core IDs)

Diese IDs bilden das Rückgrat für die Nachverfolgung und Steuerung von Anfragen und Daten.

| ID | Zweck | Beschreibung | Inhaber/Quelle |
|---|---|---|---|
| `tenant_id` | Mandanten-Isolation | Isoliert Daten, Konfigurationen und Berechtigungen pro Mandant. Jede Operation, die mandantenspezifische Daten betrifft, **muss** eine `tenant_id` tragen. | Aufrufer (z.B. Frontend via `X-Tenant-ID` Header) |
| `case_id` | Geschäftsvorfall-Kontext | Kapselt einen vollständigen Geschäftsvorfall (z.B. die Bearbeitung eines Tickets oder einer Anfrage). Alle zugehörigen Traces und Operationen werden mit derselben `case_id` verknüpft. | Aufrufer (z.B. Frontend via `X-Case-ID` Header) |
| `trace_id` | End-to-End-Korrelation | Korreliert eine einzelne Anfrage über alle Systemgrenzen hinweg (Web, Worker, Langfuse). Ermöglicht die lückenlose Nachverfolgung eines Kontrollflusses. | Wird am Systemeingang (z.B. Web-Service) generiert oder vom Aufrufer bereitgestellt. |
| `span_id` | Operations-Segment | Identifiziert eine einzelne Operation (z.B. einen Funktionsaufruf oder einen DB-Zugriff) innerhalb eines `trace_id`. Wird von der Observability-Schicht (OpenTelemetry) erzeugt. | OpenTelemetry/Instrumentation |
| `workflow_id` | Workflow-Identifikation | Identifiziert einen spezifischen Geschäfts-Workflow oder -Graphen. Dient der Abbildung und Nachverfolgung komplexer Prozessketten. | LangGraph/Agenten-Orchestrierung |

## Laufzeit-IDs (Runtime IDs)

Diese IDs steuern und verfolgen die Ausführung von Prozessen, insbesondere von Agenten und Ingestion-Pipelines.

| ID | Zweck | Beschreibung | Inhaber/Quelle |
|---|---|---|---|
| `run_id` | Agenten-Ausführung | Eindeutige ID für eine einzelne Ausführung eines LangGraph-Graphen. Dient der Laufzeit-Telemetrie, dem Status-Tracking und der Wiederholbarkeit von Agenten-Läufen. | LangGraph/Agenten-Orchestrierung |
| `ingestion_run_id` | Daten-Ingestion-Lauf | Spezialisierte ID für Daten-Ingestion-Prozesse. Verfolgt den Lebenszyklus eines Dokuments vom Upload bis zur Indizierung im Vektorstore. | Ingestion-Worker |
| `idempotency_key` | Deduplikation von Anfragen | Ein vom Client bereitgestellter Schlüssel (typischerweise für `POST`-Requests), um eine versehentliche doppelte Ausführung zu verhindern. | Aufrufer (Client) |
| `worker_call_id` | Tool-Aufruf-Identifikation | Eindeutige ID für jeden einzelnen Aufruf eines Tools innerhalb eines Workflows. Dient der Korrelation externer API-Calls und nachgelagerter Ereignisse. | System (Tool-Orchestrierung) |

## Daten-IDs (Data IDs)

Diese IDs identifizieren Datenartefakte innerhalb des Systems.

| ID | Zweck | Beschreibung | Inhaber/Quelle |
|---|---|---|---|
| `document_id` | Dokumenten-Identifikation | Eindeutige ID für ein einzelnes Dokument im System. | Wird bei der Dokumentenerstellung (z.B. Upload) generiert. |
| `collection_id` | Dokumenten-Sammlung | Identifiziert eine logische Sammlung von zusammengehörigen Dokumenten. | Wird bei der Erstellung einer Sammlung generiert oder vom Aufrufer bereitgestellt. |
| `document_version_id` | Dokumenten-Version | Eindeutige ID für eine spezifische Version eines Dokuments. Ermöglicht das Handling und die Nachverfolgung von Dokumentenänderungen. | System (Dokumentenverwaltung) |

## Anwendung und Validierung

- **Header**: Eingehende HTTP-Anfragen an das Backend **müssen** die Header `X-Tenant-ID` und `X-Case-ID` enthalten.
- **Tool-Kontext**: Alle Tool-Aufrufe innerhalb von Agenten erhalten einen `ToolContext`, der `tenant_id`, `trace_id` und entweder `run_id` oder `ingestion_run_id` enthalten muss.
- **Validierung**: Die `require_ids`-Funktion in `ai_core.ids.contracts` wird verwendet, um sicherzustellen, dass die erforderlichen IDs in Metadaten und Kontexten vorhanden sind.

## Kontext-Objekte

Die folgenden Kontextobjekte werden für die Weitergabe von IDs innerhalb des Graphen- und Tool-Layers verwendet.

### ToolContext

| Feld | Pflicht | Beschreibung |
|---|---|---|
| `tenant_id` | ja | Mandantenbezug der aktuellen Ausführung. |
| `trace_id` | ja | End-to-End-Korrelation; wird vom Systemeingang übernommen oder erzeugt. |
| `workflow_id` | optional | Identifiziert den aktiven Prozess- oder LangGraph-Workflow. |
| `case_id` | optional | Geschäftskontext für Rückfragen, HITL und Audits. |
| `run_id` | bedingt | Laufzeit-ID für Graph-Ausführungen; obligatorisch, wenn kein `ingestion_run_id` gesetzt ist. |
| `ingestion_run_id` | bedingt | Laufzeit-ID für Ingestion-Flows; obligatorisch, wenn kein `run_id` gesetzt ist. |
| `worker_call_id` | optional | Eindeutige ID pro Tool-Aufruf zur Korrelation externer API-Calls. |

### GraphContext

| Feld | Pflicht | Beschreibung |
|---|---|---|
| `tenant_id` | ja | Mandantenbezug für den gesamten Graph. |
| `trace_id` | ja | End-to-End-Korrelation; darf nach Initialisierung nicht überschrieben werden. |
| `workflow_id` | ja | Identifiziert den ausgeführten Graphen oder Geschäftsprozess. |
| `case_id` | optional | Geschäftskontext für Agentenentscheidungen. |
| `run_id` | ja | Frische Laufzeit-ID je Graph-Execution, die in alle Spans und Events übernommen wird. |
| `ingestion_run_id` | optional | Wird gesetzt, wenn eine Graph-Ausführung Ingestion anstößt. |
| `collection_id` | optional | Verweist auf die betroffene Dokument-Sammlung. |

## Erzeugung & Immutabilität

- **`trace_id`**: Wird vom aufrufenden System geliefert. Ist beim Graph-Einstieg leer, wird eine neue ID erzeugt und anschließend unverändert propagiert.
- **`run_id`**: Für jede Graph-Ausführung neu erzeugen und in allen Spans/Events verwenden.
- **`worker_call_id`**: Pro Tool-Aufruf neu erzeugen, um einzelne externe API-Calls korrelieren zu können.
- **`ingestion_run_id`**: Nur erzeugen, wenn der Flow tatsächlich Ingestion triggert. Die ID wird von der Stelle erzeugt, die den Ingestion-Lauf startet (nicht vom Search-Worker).

## Konsistente Attributnamen für Spans und Logs

Verwende ausschließlich die folgenden Attribute und vermeide Aliasnamen wie `request_id`:

`tenant_id`, `trace_id`, `workflow_id`, `case_id`, `run_id`, `ingestion_run_id`, `collection_id`, `document_id`, `worker_call_id`, `search_session_id`.

Die Attribute müssen in Telemetrie (Spans, Events, Logs) konsistent benannt und befüllt werden.

## Entscheidungen und Transitionen in Graph-Knoten

Jeder Graph-Knoten gibt eine Transition mit dem Schema `{decision, rationale, meta}` zurück. Das `meta`-Objekt enthält die oben genannten IDs und knoten- bzw. tool-spezifische Felder. Dadurch bleiben Entscheidungswege nachvollziehbar und vollständig rekonstruierbar.

## Human-in-the-Loop (HITL)

Bei manuellen Prüfungen wird der Zustand `PENDING_REVIEW` gesetzt. Der Zustand enthält den vollständigen Kontext inklusive aller relevanten IDs. Freigaben oder Ablehnungen (`review_token`) korrelieren über die Kombination aus `trace_id` und `run_id`.