# Case Subsystem Übersicht

Dieses Dokument bietet einen architektonischen Überblick über das Case-Subsystem und beschreibt dessen Zweck, Datenstrukturen, Lifecycle-Management sowie Integrationspunkte innerhalb der Gesamtplattform.

## Zweck

Das Case-Subsystem führt einen persistenten geschäftlichen Kontext in die Plattform ein und schlägt eine Brücke zwischen flüchtigen technischen Operationen (wie Ingestion-Runs oder Vektorsuchen) und langlebigen Geschäftsprozessen. Ein "Case" dient als stabiler Anker für eine spezifische Geschäftseinheit – beispielsweise einen Versicherungsfall, eine Rechtsangelegenheit oder ein Support-Ticket – und ermöglicht es, disparate Systemaktivitäten über die Zeit hinweg zu korrelieren und nachzuverfolgen.

Durch die Beibehaltung einer stabilen `external_id` stellt das Case-Modell sicher, dass alle zugehörigen Daten, Events und Artefakte konsistent abgerufen und analysiert werden können, unabhängig von den zugrunde liegenden technischen Workflows, die sie generiert haben.

## Datenmodell

Der Kern des Subsystems besteht aus zwei primären Modellen: `Case` und `CaseEvent`.

### Case

Die Entität `Case` repräsentiert das zentrale Register für einen Geschäftsprozess. Sie ist auf einen spezifischen Tenant (Mandanten) bezogen und wird durch eine eindeutige `external_id` identifiziert.

* **Identität**:
  * `id`: Interner UUID-Primärschlüssel.
  * `external_id`: Der benutzerseitige, stabile Identifikator (z. B. "CLM-2024-001"). Diese ID ist innerhalb eines Tenants eindeutig.
  * `tenant`: Referenz auf den besitzenden Tenant.

* **Zustand**:
  * `status`: Übergeordneter operativer Status, entweder `open` oder `closed`.
  * `phase`: Granulare Geschäftsphase (z. B. "intake", "evidence_collection"). Diese wird dynamisch aus der Event-Historie abgeleitet.
  * `metadata`: Flexibler JSON-Speicher für benutzerdefinierte Geschäftsattribute.

* **Zeitstempel**: Erfasst `created_at`, `updated_at` und `closed_at` für Auditing und SLA-Überwachung.

### CaseEvent

Die Entität `CaseEvent` erfasst signifikante Zustandsänderungen oder Aktivitäten im Zusammenhang mit einem Case. Sie dient als unveränderliches Audit-Log, das die Lifecycle-Logik steuert.

* **Kernattribute**:
  * `event_type`: Ein String-Identifikator für die spezifische Aktion (z. B. `ingestion_run_completed`, `collection_search:verified`).
  * `source`: Die Systemkomponente, die das Event ausgelöst hat (z. B. "ingestion", "graph:CollectionSearchGraph").
  * `payload`: Detaillierter Kontext zum Event, wie z. B. Verarbeitungsmetriken oder Transitionsdaten.

* **Korrelationen**:
  * `ingestion_run`: Verknüpft das Event mit einem spezifischen Dokumentenverarbeitungsauftrag.
  * `graph_name`: Identifiziert den beteiligten Workflow-Graphen.
  * `workflow_id` & `trace_id`: Ermöglichen verteiltes Tracing und Debugging über Services hinweg.
  * `collection_id`: Beschränkt das Event auf eine spezifische Dokumentensammlung.

## Lifecycle Management

Der Case-Lifecycle ist kein statischer Zustandsautomat, sondern ein dynamischer, rekonstruktiver Prozess, der durch den Strom von `CaseEvents` gesteuert wird.

### Standardphasen

Obwohl Phasen pro Tenant angepasst werden können, definiert das System einen Standardablauf:

1. **Intake**: Der Anfangszustand, wenn ein Case erstellt wird.
2. **Evidence Collection**: Aktives Sammeln und Verarbeiten von Dokumenten.
3. **Search Completed**: Automatisierte Such-Workflows sind abgeschlossen.
4. **External Review**: Der Fall wartet auf Feedback von Menschen oder externen Systemen.

### Rekonstruktive Logik

Das System bestimmt die aktuelle `phase` eines Cases mithilfe des Mechanismus `apply_lifecycle_definition`. Anstatt den Zustand direkt als Reaktion auf API-Aufrufe zu ändern, geht das System wie folgt vor:

1. Laden der geordneten Historie der `CaseEvents` für den Case.
2. Abrufen der Lifecycle-Definition des Tenants (oder der Standarddefinition).
3. Abspielen der Events gegen die definierten Transitionen, um die aktuelle Phase abzuleiten.
4. Aktualisieren des Feldes `Case.phase` nur dann, wenn sich der abgeleitete Zustand geändert hat.

Dieser Ansatz stellt sicher, dass der Case-Zustand immer konsistent mit seiner Historie ist und ermöglicht rückwirkende Korrekturen, falls sich Definitionen ändern.

## Bridging & Integration

Das Case-Subsystem fungiert als Bindeglied, das Signale aus anderen Domänen abfängt und in standardisierte Case-Events umwandelt.

### Ingestion Integration

Die Funktion `record_ingestion_case_event` verbindet die Dokumenten-Ingestion-Domäne mit dem Case-System. Wenn ein Ingestion-Run seinen Status aktualisiert (z. B. von "running" zu "completed"), erstellt diese Bridge ein entsprechendes `CaseEvent`. Dies ermöglicht es dem Case-Lifecycle, automatisch voranzuschreiten (z. B. zu "Evidence Collection"), ohne dass der Ingestion-Worker Kenntnis von der Case-Logik haben muss.

### Collection Search Integration

Die Funktion `emit_case_lifecycle_for_collection_search` verbindet die Graph-Execution-Engine mit dem Case-System. Während der Collection-Search-Graph Knoten durchläuft (wie "strategy_generated" oder "verified"), übersetzt dieser Hook Graph-Transitionen in `CaseEvents`. Dies ermöglicht es komplexen, mehrstufigen Workflows, den übergeordneten Geschäftsstatus des Cases zu steuern.

### API Kontext

Das System erzwingt eine strikte Zuordnung zwischen dem API-Kontext und dem Case-Modell. Der Standard-HTTP-Header `X-Case-ID` (und die entsprechenden Metadaten-Schlüssel) bildet direkt auf das Feld `Case.external_id` ab. Dies stellt sicher, dass alle API-Anfragen – sei es für Ingestion, Suche oder Abruf – automatisch dem korrekten Geschäftskontext zugeordnet werden, ohne dass der Client interne UUID-Lookups durchführen muss.

### Identifikatoren und Kopplung

Es ist wichtig zu beachten, dass Worker-Prozesse (wie die Dokumenten-Ingestion) oft lose gekoppelt sind und mit der `external_id` (String) arbeiten, um keine harten Abhängigkeiten zur `Case`-Datenbanktabelle zu erzwingen.

* **Ingestion Worker**: Das Modell `DocumentIngestionRun` speichert die `external_id` als String. Dies ermöglicht eine asynchrone Verarbeitung ohne sofortigen Zwang zur Case-Existenz.
* **Core System**: Das `Case`-Modell und eng verknüpfte Entitäten (wie `DocumentCollection`) nutzen die interne UUID als Fremdschlüssel.

Die Brückenfunktionen (wie `record_ingestion_case_event`) sind dafür verantwortlich, den String-Identifikator zur Laufzeit in die korrekte UUID aufzulösen (`get_or_create_case_for`) und so die Konsistenz zwischen den lose gekoppelten Workern und dem relationalen Kernsystem zu gewährleisten.
