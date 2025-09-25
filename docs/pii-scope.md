# PII-Scope Playbook

Dieser Leitfaden beschreibt, wie der PII-Session-Scope in NOESIS 2 gesetzt, übertragen und ausgewertet wird. Die Abfolge gilt für den Django-Monolithen ebenso wie für Microservices und bildet die Blaupause für weitere Laufzeitumgebungen (z. B. FastAPI).

## 1. HTTP-Ingress

1. **PII-Session-Middleware** (`ai_core.middleware.PIISessionScopeMiddleware`) liest bei jedem Request die Header `X-Tenant-Id`, `X-Case-Id` und `X-Trace-Id`.
2. Der Scope wird als `(tenant_id, case_id, session_salt)` registriert. `session_salt` entspricht `trace_id || case_id || tenant_id`, sofern kein eigener Wert gesetzt wurde.
3. Die Middleware steht *vor* Logging- und Redaction-Komponenten, damit sämtlicher nachfolgender Code denselben Scope nutzt.
4. Nach Abschluss (Response oder Exception) wird der Scope wieder geleert.

> ✅ **Guard:** `noesis2/settings/base.py` enthält einen Test, der sicherstellt, dass `PIISessionScopeMiddleware` vor den Logging-Middlewares eingehängt bleibt (siehe `common/tests/test_pii_flags.py::test_pii_middleware_order`).

## 2. LLM-Pfad

1. `ai_core.infra.mask_prompt.mask_prompt()` ruft `get_pii_config()` auf, setzt ggf. den HMAC-Schlüssel und maskiert den Prompt *vor* dem LLM-Aufruf.
2. `run_prompt_node()` sowie Compose/Draft-Nodes nutzen dieselbe Utility, sodass der Scope automatisch in deterministischen Tokens berücksichtigt wird.
3. `ai_core.infra.mask_prompt.mask_response()` wird nach dem LLM-Aufruf aufgerufen, sobald `PII_POST_RESPONSE=true` ist.
4. Zusammengefasste `[REDACTED: …]`-Blöcke werden für Prompts standardmäßig angehängt; bei Responses nur, wenn das Flag `include_summary` aktiv ist.

> ✅ **Guard:** `ai_core/tests/test_nodes.py` prüft, dass Prompts vor dem LLM-Aufruf maskiert und Responses gemäß Flag nachbearbeitet werden.

## 3. Logging

1. `common.logging.configure_logging()` hängt – sofern `PII_LOGGING_REDACTION=true` – den `pii_redaction_processor` vor Renderer/Exporter ein.
2. Der Processor ruft `mask_text()` nur auf `str`-Felder auf, respektiert das 64 KB-Limit für strukturierte Daten und bewahrt JSON-Formatierung.
3. Deterministische Tokens werden per Scope stabilisiert (HMAC + Session-Context).

> ✅ **Guard:** `common/tests/test_logging_redaction.py::test_logging_redaction_processor_order` stellt sicher, dass der Processor vor Renderer/Exporter bleibt.

## 4. Async & Tasks

1. `common.celery.ScopedTask` setzt den Scope vor `run()` und räumt ihn in `finally` wieder auf.
2. Produzenten verwenden `with_scope_apply_async()` und injizieren `tenant_id`, `case_id`, `trace_id`, `session_salt` in die Signaturen.
3. Tests (`common/tests/test_celery_scope.py`) verifizieren, dass der Scope im Worker vorhanden ist und nach dem Task verschwindet.

## 5. Egress / Downstream Services

1. HTTP-Clients, die andere Services aufrufen, müssen die Header `X-Tenant-Id`, `X-Case-Id` und `X-Trace-Id` weitergeben.
2. Für externe Systeme ohne PII-Masking wird zusätzlich empfohlen, nur maskierte Payloads auszugeben.

## Review-Checkliste

Bei Änderungen an PII-relevanten Pfaden sollten Reviewer die folgenden Punkte abhaken:

1. [ ] Middleware setzt Scope vor Logging und räumt ihn verlässlich auf.
2. [ ] Prompt-/Response-Pfade nutzen `mask_prompt()` und `mask_response()` mit derselben Config.
3. [ ] Logging-Pipeline enthält den `pii_redaction_processor` vor Renderer/Exporter.
4. [ ] Async/Task-Aufrufe propagieren Scope-Informationen (Producer & Worker).
5. [ ] HTTP-Egress trägt Tenant-/Case-/Trace-Header weiter.

## FastAPI-Referenz

```python
from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from ai_core.infra.policy import clear_session_scope, set_session_scope


class PIISessionScopeMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        tenant_id = request.headers.get("X-Tenant-Id", "")
        case_id = request.headers.get("X-Case-Id", "")
        trace_id = request.headers.get("X-Trace-Id", "")
        session_salt = trace_id or case_id or tenant_id

        if tenant_id or case_id or session_salt:
            set_session_scope(
                tenant_id=tenant_id,
                case_id=case_id,
                session_salt=session_salt,
            )

        try:
            response: Response = await call_next(request)
            return response
        finally:
            clear_session_scope()


app = FastAPI()
app.add_middleware(PIISessionScopeMiddleware)
```

> ℹ️ **Hinweis:** Für Frameworks ohne Middleware-Lifecycle kann dieselbe Logik über Decorators oder Dependency Injection umgesetzt werden.

## Smoke-Test

`common/tests/test_pii_flags.py::test_mask_text_deterministic_under_scope` prüft in CI, dass deterministische Tokens innerhalb eines gesetzten Scopes stabil bleiben.
