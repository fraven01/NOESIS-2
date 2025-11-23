# Architektur-Notiz: LangGraph-Fassade statt Workflow-Engine

## Entscheidung

Wir kapseln die bisherigen `run(state, meta)`-Module über eine schlanke LangGraph-Fassade,
statt die alte Workflow-Engine weiter auszubauen. Damit bleibt die bestehende HTTP-Schnittstelle
funktionsfähig, während neue Graph-Laufzeiten schrittweise integriert werden können.

## Heutiger Scope

- `ai_core/graph/core.py` definiert `GraphRunner`, `Checkpointer`, `GraphContext` und den
dateibasierten `FileCheckpointer` (`.ai_core_store/<tenant>/<case>/state.json`).
- `ai_core/graph/bootstrap.py` registriert über `ai_core/graph/adapters.py` den Intake-Stub
  (`info_intake`) sowie die produktiven Graphen `retrieval_augmented_generation` (als `rag.default`),
  `crawler_ingestion_graph` (als `crawler.ingestion`) und `collection_search`.
- `_GraphView` in `ai_core/views.py` nutzt `normalize_meta`, `merge_state` und die Registry,
um Requests unverändert zu bedienen, inklusive Rate-Limits und Header-Contracts.
- Die ehemaligen `workflows`-Modelle wurden entfernt; LangGraph bedient Cases direkt über `ai_core`.

## Ausblick

- Austausch des `FileCheckpointer` gegen eine Datenbank- oder Objekt-Storage-Lösung
  für parallele Läufe und Replays.
- Einführung echter LangGraph-Nodes (Tool-/LLM-Aufrufe, Guardrails) mit Telemetrie über Langfuse.
- Versionierte Graph-Definitionen und Migrationspfad aus der Legacy-Workflow-Engine.
