from __future__ import annotations

import uuid
from typing import Any, Mapping

from django.http import HttpRequest, HttpResponse
from structlog.contextvars import bind_contextvars, clear_contextvars

from ai_core.infra.resp import apply_std_headers


class RequestContextMiddleware:
    """Bind request metadata to structlog context and response headers."""

    TRACEPARENT_HEADER = "HTTP_TRACEPARENT"
    TRACE_ID_HEADER = "HTTP_X_TRACE_ID"
    CASE_ID_HEADER = "HTTP_X_CASE_ID"
    TENANT_ID_HEADER = "HTTP_X_TENANT_ID"
    KEY_ALIAS_HEADER = "HTTP_X_KEY_ALIAS"
    FORWARDED_FOR_HEADER = "HTTP_X_FORWARDED_FOR"
    REMOTE_ADDR_HEADER = "REMOTE_ADDR"

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request: HttpRequest) -> HttpResponse:
        clear_contextvars()
        meta = self._gather_metadata(request)
        bind_contextvars(**meta["log_context"])

        try:
            response = self.get_response(request)
            response = apply_std_headers(response, meta["response_meta"])
            traceparent = meta["response_meta"].get("traceparent")
            if traceparent:
                response["traceparent"] = traceparent
            return response
        finally:
            clear_contextvars()

    def _gather_metadata(self, request: HttpRequest) -> dict[str, Mapping[str, str]]:
        headers = request.META
        traceparent = self._normalize_header(headers.get(self.TRACEPARENT_HEADER))
        trace_id, span_id = self._resolve_trace_and_span(headers, traceparent)

        tenant_id = self._normalize_header(headers.get(self.TENANT_ID_HEADER))
        case_id = self._normalize_header(headers.get(self.CASE_ID_HEADER))
        key_alias = self._normalize_header(headers.get(self.KEY_ALIAS_HEADER))

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
        if client_ip:
            log_context["client.ip"] = client_ip

        response_meta: dict[str, str] = {
            "trace_id": trace_id,
        }
        if span_id:
            response_meta["span_id"] = span_id
        if tenant_id:
            response_meta["tenant"] = tenant_id
        if case_id:
            response_meta["case"] = case_id
        if key_alias:
            response_meta["key_alias"] = key_alias
        if traceparent:
            response_meta["traceparent"] = traceparent

        return {"log_context": log_context, "response_meta": response_meta}

    def _resolve_trace_and_span(
        self, headers: Mapping[str, Any], traceparent: str | None
    ) -> tuple[str, str | None]:
        if traceparent:
            parts = traceparent.split("-")
            if len(parts) >= 4:
                trace_id = self._normalize_trace_id(parts[1])
                span_id = self._normalize_span_id(parts[2])
                if trace_id and span_id:
                    return trace_id, span_id

        candidate = self._normalize_header(headers.get(self.TRACE_ID_HEADER))
        if candidate:
            normalized = self._normalize_trace_id(candidate)
            if normalized:
                return normalized, None

        generated_trace_id = uuid.uuid4().hex
        return generated_trace_id, uuid.uuid4().hex[:16]

    @staticmethod
    def _normalize_header(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return str(value)

    @staticmethod
    def _normalize_trace_id(value: str | None) -> str | None:
        if not value:
            return None
        normalized = value.replace("-", "").strip().lower()
        if normalized:
            return normalized
        return None

    @staticmethod
    def _normalize_span_id(value: str | None) -> str | None:
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
