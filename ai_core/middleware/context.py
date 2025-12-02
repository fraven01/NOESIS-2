from __future__ import annotations

import json
import uuid
from typing import Any, Mapping, MutableMapping

from django.http import HttpRequest, HttpResponse, JsonResponse
from pydantic import ValidationError
from structlog.contextvars import bind_contextvars, clear_contextvars

from ai_core.ids import (
    coerce_trace_id,
    normalize_request,
)
from ai_core.infra.resp import apply_std_headers
from customers.tenant_context import TenantRequiredError


class RequestContextMiddleware:
    """Bind request metadata to structlog context and response headers."""

    TRACEPARENT_HEADER = "HTTP_TRACEPARENT"
    TRACE_ID_HEADER = "HTTP_X_TRACE_ID"

    CASE_ID_HEADER = "HTTP_X_CASE_ID"
    TENANT_ID_HEADER = "HTTP_X_TENANT_ID"
    KEY_ALIAS_HEADER = "HTTP_X_KEY_ALIAS"
    IDEMPOTENCY_KEY_HEADER = "HTTP_IDEMPOTENCY_KEY"
    INVOCATION_ID_HEADER = "HTTP_X_INVOCATION_ID"
    RUN_ID_HEADER = "HTTP_X_RUN_ID"
    INGESTION_RUN_ID_HEADER = "HTTP_X_INGESTION_RUN_ID"
    WORKFLOW_ID_HEADER = "HTTP_X_WORKFLOW_ID"
    FORWARDED_FOR_HEADER = "HTTP_X_FORWARDED_FOR"
    REMOTE_ADDR_HEADER = "REMOTE_ADDR"

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request: HttpRequest) -> HttpResponse:
        clear_contextvars()
        try:
            meta = self._gather_metadata(request)
        except TenantRequiredError as exc:
            return self._tenant_required_response(exc)
        except ValueError as exc:
            error_msg = str(exc)
            # Use specific error code for case header validation errors
            error_code = (
                "invalid_case_header"
                if "Case header" in error_msg
                else "invalid_request"
            )
            return JsonResponse({"detail": error_msg, "code": error_code}, status=400)
        except ValidationError as exc:
            # Handle Pydantic validation errors from ScopeContext
            # We take the first error message for simplicity
            error_msg = str(exc)
            if exc.errors():
                error_msg = exc.errors()[0]["msg"]
            return JsonResponse(
                {"detail": error_msg, "code": "invalid_request"}, status=400
            )
        else:
            bind_contextvars(**meta["log_context"])

            response = self.get_response(request)
            response = apply_std_headers(response, meta["response_meta"])
            return response
        finally:
            clear_contextvars()

    def _gather_metadata(self, request: HttpRequest) -> dict[str, Mapping[str, str]]:
        headers = request.META

        # Use the normalizer as the Single Source of Truth
        scope_context = normalize_request(request)
        request.scope_context = scope_context

        # Extract values from scope_context for logging and response headers
        trace_id = scope_context.trace_id
        tenant_id = scope_context.tenant_id
        case_id = scope_context.case_id
        key_alias = self._normalize_header(headers.get(self.KEY_ALIAS_HEADER))
        idempotency_key = scope_context.idempotency_key

        # Span ID is not in ScopeContext, so we resolve it separately or keep it if needed
        # The original code had _resolve_trace_and_span, but ScopeContext handles trace_id.
        # We can keep span_id logic if it's critical, but ScopeContext is the authority on trace_id.
        # For now, let's re-use the span_id extraction if trace_id matches, or just skip it if not critical.
        # Actually, let's keep it simple and consistent with the plan: use ScopeContext.
        # If span_id is needed for logging, we can extract it from headers manually or via a helper,
        # but ScopeContext doesn't store it.
        # Let's check if we can get span_id from the traceparent header if present.
        _, span_id = self._resolve_trace_and_span(request)

        traceparent = self._normalize_header(headers.get(self.TRACEPARENT_HEADER))
        http_method = request.method.upper() if request.method else ""
        http_route = self._resolve_route(request)
        client_ip = self._resolve_client_ip(headers)

        log_context: dict[str, str] = {
            "trace.id": trace_id,
            "http.method": http_method,
            "http.route": http_route,
        }
        if span_id:
            log_context["span.id"] = span_id
        if tenant_id:
            log_context["tenant.id"] = tenant_id
        if case_id:
            log_context["case.id"] = case_id
        if key_alias:
            log_context["key.alias"] = key_alias
        if idempotency_key:
            log_context["idempotency.key"] = idempotency_key
        if client_ip:
            log_context["client.ip"] = client_ip

        response_meta: dict[str, str] = {
            "trace_id": trace_id,
        }
        if span_id:
            response_meta["span_id"] = span_id
        if tenant_id:
            response_meta["tenant_id"] = tenant_id
        if case_id:
            response_meta["case_id"] = case_id
        if key_alias:
            response_meta["key_alias"] = key_alias
        if idempotency_key:
            response_meta["idempotency_key"] = idempotency_key
        if traceparent:
            response_meta["traceparent"] = traceparent

        return {"log_context": log_context, "response_meta": response_meta}

    @staticmethod
    def _tenant_required_response(exc: TenantRequiredError) -> HttpResponse:
        return JsonResponse({"detail": str(exc)}, status=403)

    def _resolve_trace_and_span(self, request: HttpRequest) -> tuple[str, str | None]:
        """Resolve trace and span IDs from headers, query params, and body."""
        headers = request.META
        span_id: str | None = None

        try:
            trace_id, span_id = coerce_trace_id(headers)
        except ValueError:
            trace_id = None

        if not trace_id:
            meta: MutableMapping[str, Any] = {}
            query_trace = request.GET.get("trace_id")
            if query_trace:
                meta["trace_id"] = query_trace
            if (
                not meta.get("trace_id")
                and "application/json" in (request.content_type or "")
                and request.body
            ):
                try:
                    data = json.loads(request.body)
                    if isinstance(data, dict):
                        meta["trace_id"] = data.get("trace_id")
                except json.JSONDecodeError:
                    pass  # Ignore malformed JSON

            if meta:
                try:
                    trace_id, _ = coerce_trace_id(meta)  # type: ignore[arg-type]
                except ValueError:
                    trace_id = None

        if not trace_id:
            trace_id = uuid.uuid4().hex
            if not span_id:
                span_id = uuid.uuid4().hex[:16]

        return trace_id, span_id

    @staticmethod
    def _normalize_header(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return str(value)

    @staticmethod
    def _resolve_route(request: HttpRequest) -> str:
        match = getattr(request, "resolver_match", None)
        route = getattr(match, "route", None)
        if route:
            return str(route)
        return request.path

    def _resolve_client_ip(self, headers: Mapping[str, Any]) -> str | None:
        forwarded = self._normalize_header(headers.get(self.FORWARDED_FOR_HEADER))
        if forwarded:
            return forwarded.split(",")[0].strip()
        return self._normalize_header(headers.get(self.REMOTE_ADDR_HEADER))
