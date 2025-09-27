# Architektur-Notiz: LangGraph-Fassade statt Workflow-Engine

## Entscheidung
Wir kapseln die bisherigen `run(state, meta)`-Module über eine schlanke LangGraph-Fassade,
statt die alte Workflow-Engine weiter auszubauen. Damit bleibt die bestehende HTTP-Schnittstelle
funktionsfähig, während neue Graph-Laufzeiten schrittweise integriert werden können.

## Heutiger Scope
- `ai_core/graph/core.py` definiert `GraphRunner`, `Checkpointer`, `GraphContext` und den
dateibasierten `FileCheckpointer` (`.ai_core_store/<tenant>/<case>/state.json`).
- `ai_core/graph/adapters.py` und `graph/bootstrap.py` registrieren die vier Legacy-Module
  (`info_intake`, `scope_check`, `needs_mapping`, `system_description`) als Runner.
- `_GraphView` in `ai_core/views.py` nutzt `normalize_meta`, `merge_state` und die Registry,
um Requests unverändert zu bedienen, inklusive Rate-Limits und Header-Contracts.
- `workflows/` bleibt read-only (Modelle/Admin); keine Trigger oder Worker-Anbindung.

## Ausblick
- Austausch des `FileCheckpointer` gegen eine Datenbank- oder Objekt-Storage-Lösung
  für parallele Läufe und Replays.
- Einführung echter LangGraph-Nodes (Tool-/LLM-Aufrufe, Guardrails) mit Telemetrie über Langfuse.
- Versionierte Graph-Definitionen und Migrationspfad aus der Legacy-Workflow-Engine.
