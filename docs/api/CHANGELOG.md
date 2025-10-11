# API Changelog (Placeholder)

> Wird vor dem Production-GoLive aktiviert. Enthält künftig Breaking (⚠️) und Non-Breaking (🟢) Änderungen am OpenAPI-Schema.

- [ ] Aktivieren vor Prod: CI-Job `api:changelog` + `oasdiff`/`openapi-diff`.
- [ ] Versionierungs-Policy (SemVer) beschließen und im Repo verlinken.
- [ ] Pipeline-Gate für Breaking Changes.

- ⚠️ 2025-10-10 – `/v1/ai/rag/query/` liefert Retrieval-Diagnostik (`retrieval`, `snippets`) nun auf Top-Level. Clients, die nur `answer`/`prompt_version` erwarteten, müssen aktualisiert werden.
- ⚠️ 2025-10-10 – Beispiel `docs/api/examples/retrieve_response.json` auf das neue Response-Schema (Top-Level `answer`, `prompt_version`, `retrieval`, `snippets`) aktualisiert. **Breaking (MVP)**.
> 2025-10-10: RAG v2 – 422 `retrieval_inconsistent_metadata` für Chunks ohne tenant_id/case_id (Re-Index erforderlich).
