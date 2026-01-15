"""WebSocket utilities for RAG Workbench.

DEPRECATED: This module is deprecated. Use `theme.helpers.context.prepare_workbench_context`
instead, which now supports both HTTP requests and WebSocket ASGI scopes.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any
from uuid import UUID, uuid4

from pydantic import ValidationError

from ai_core.contracts.business import BusinessContext
from ai_core.contracts.scope import ScopeContext
from ai_core.tool_contracts import ContextError


def _extract_user_id(request: Mapping[str, Any]) -> str | None:
    user = request.get("user")
    if user is None or not getattr(user, "is_authenticated", False):
        return None
    user_pk = getattr(user, "pk", None)
    if user_pk is None:
        raise ContextError(
            "user_id is required for authenticated requests", field="user_id"
        )
    try:
        return str(UUID(str(user_pk)))
    except (TypeError, ValueError) as exc:
        raise ContextError("user_id must be a UUID string", field="user_id") from exc


def build_websocket_context(
    *,
    request: Mapping[str, Any],
    tenant_id: str,
    tenant_schema: str | None = None,
    case_id: str | None = None,
    collection_id: str | None = None,
    workflow_id: str | None = None,
    thread_id: str | None = None,
) -> tuple[ScopeContext, BusinessContext]:
    """Build ScopeContext and BusinessContext from a WebSocket request.

    .. deprecated:: 2026-01-15
        Use `theme.helpers.context.prepare_workbench_context` instead,
        which now supports both HTTP and WebSocket contexts.
    """
    warnings.warn(
        "build_websocket_context is deprecated. Use "
        "theme.helpers.context.prepare_workbench_context instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    user_id = _extract_user_id(request)
    try:
        scope = ScopeContext(
            tenant_id=tenant_id,
            tenant_schema=tenant_schema or tenant_id,
            trace_id=uuid4().hex,
            invocation_id=uuid4().hex,
            run_id=uuid4().hex,
            user_id=user_id,
        )
    except ValidationError as exc:
        raise ContextError("Invalid websocket scope context", field=None) from exc

    business = BusinessContext(
        case_id=case_id,
        collection_id=collection_id,
        workflow_id=workflow_id,
        thread_id=thread_id,
    )
    return scope, business
