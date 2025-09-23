from __future__ import annotations

from typing import Any

from django.utils.deprecation import MiddlewareMixin

from ai_core.infra.pii_flags import (
    clear_pii_config,
    load_tenant_pii_config,
    resolve_tenant_pii_config,
    set_pii_config,
)
from ai_core.infra.policy import (
    clear_session_scope,
    get_session_scope,
    set_session_scope,
)
from common.tenants import get_current_tenant


class PIISessionScopeMiddleware(MiddlewareMixin):
    """Populate the masking session scope from request headers."""

    TENANT_HEADER = "HTTP_X_TENANT_ID"
    CASE_HEADER = "HTTP_X_CASE_ID"
    TRACE_HEADER = "HTTP_X_TRACE_ID"

    def __call__(self, request: Any):
        response = None
        try:
            response = super().__call__(request)
            return response
        finally:
            clear_pii_config()
            clear_session_scope()

    def process_request(self, request: Any) -> None:
        meta = getattr(request, "META", {})
        tenant_id = self._normalize(meta.get(self.TENANT_HEADER))
        case_id = self._normalize(meta.get(self.CASE_HEADER))
        trace_id = self._normalize(meta.get(self.TRACE_HEADER))
        session_salt = self._resolve_session_salt(trace_id, case_id, tenant_id)
        set_session_scope(
            tenant_id=tenant_id,
            case_id=case_id,
            session_salt=session_salt,
        )

        tenant = getattr(request, "tenant", None) or get_current_tenant()
        tenant_config = resolve_tenant_pii_config(tenant)
        if tenant_config is None and tenant is None and tenant_id:
            tenant_config = load_tenant_pii_config(tenant_id)
        if tenant_config:
            scope = get_session_scope()
            if scope:
                scoped_config = dict(tenant_config)
                scoped_config["session_scope"] = scope
            else:
                scoped_config = tenant_config
            set_pii_config(scoped_config)

    @staticmethod
    def _normalize(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            stripped = value.strip()
            return stripped
        return str(value)

    @staticmethod
    def _resolve_session_salt(
        trace_id: str,
        case_id: str,
        tenant_id: str,
    ) -> str:
        parts = [value for value in (trace_id, case_id, tenant_id) if value]
        if parts:
            return "||".join(parts)
        return ""
