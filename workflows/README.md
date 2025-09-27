The `workflows` app only exposes Django models and admin registrations for workflow templates, steps, and instances.
There are no triggers or runners implemented here; orchestration is handled by the graph facade under `ai_core/graph/`.
Orchestration findet ausschließlich unter `ai_core/graph/*` statt; diese App bleibt für Datenmodelle.
Use this package purely for data modeling until the workflow runtime is replaced.
