# run_id / ingestion_run_id constraint note (deprecated)

This document is deprecated. The runtime ID contract is:

- At least one of `run_id` or `ingestion_run_id` is required.
- Both may coexist (for example, when a workflow triggers ingestion).

`ScopeContext` enforces this rule in `ai_core/contracts/scope.py`.

If you need historical context, use git history for this file.
