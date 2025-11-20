from __future__ import annotations

import json
import uuid
from typing import Any, Mapping, MutableMapping

from django.http import HttpRequest, HttpResponse, JsonResponse
from structlog.contextvars import bind_contextvars, clear_contextvars

from ai_core.ids.contracts import normalize_trace_id
from ai_core.infra.resp import apply_std_headers
from customers.tenant_context import TenantContext, TenantRequiredError


class RequestContextMiddleware:
    """Bind request metadata to structlog context and response headers."""

    TRACEPARENT_HEADER = "HTTP_TRACEPARENT"
    TRACE_ID_HEADER = "HTTP_X_TRACE_ID"

    CASE_ID_HEADER = "HTTP_X_CASE_ID"
    TENANT_ID_HEADER = "HTTP_X_TENANT_ID"
    KEY_ALIAS_HEADER = "HTTP_X_KEY_ALIAS"
    IDEMPOTENCY_KEY_HEADER = "HTTP_IDEMPOTENCY_KEY"
    FORWARDED_FOR_HEADER = "HTTP_X_FORWARDED_FOR"
    REMOTE_ADDR_HEADER = "REMOTE_ADDR"

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request: HttpRequest) -> HttpResponse:
        clear_contextvars()
        try:
            meta = self._gather_metadata(request)
            bind_contextvars(**meta["log_context"])

            response = self.get_response(request)
            response = apply_std_headers(response, meta["response_meta"])
            return response
        except TenantRequiredError as exc:
            return self._tenant_required_response(exc)
        finally:
            clear_contextvars()

    def _gather_metadata(self, request: HttpRequest) -> dict[str, Mapping[str, str]]:
        headers = request.META
        trace_id, span_id = self._resolve_trace_and_span(request)
        traceparent = self._normalize_header(headers.get(self.TRACEPARENT_HEADER))

        tenant = TenantContext.from_request(
            request, allow_headers=False, require=True
        )
        tenant_id = getattr(tenant, "schema_name", None)
        case_id = self._normalize_header(headers.get(self.CASE_ID_HEADER))
        key_alias = self._normalize_header(headers.get(self.KEY_ALIAS_HEADER))
        idempotency_key = self._normalize_header(
            headers.get(self.IDEMPOTENCY_KEY_HEADER)
        )

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
        meta: MutableMapping[str, Any] = {}
        span_id: str | None = None

        # 1. Headers
        headers = request.META
        meta["trace_id"] = self._normalize_header(headers.get(self.TRACE_ID_HEADER))

        traceparent = self._normalize_header(headers.get(self.TRACEPARENT_HEADER))
        if traceparent:
            parts = traceparent.split("-")
            if len(parts) >= 4:
                meta["trace_id"] = self._normalize_w3c_id(parts[1])
                span_id = self._normalize_w3c_id(parts[2])

        # 2. Query Parameters
        if not meta.get("trace_id"):
            meta["trace_id"] = request.GET.get("trace_id")

        # 3. Request Body (for JSON content type)
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

        try:
            trace_id = normalize_trace_id(meta)
        except ValueError:
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
    def _normalize_w3c_id(value: str | None) -> str | None:
        if not value:
            return None
        normalized = value.replace("-", "").strip().lower()
        if normalized:
            return normalized
        return None

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
