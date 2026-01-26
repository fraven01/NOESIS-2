"""Context helpers for RAG Workbench views."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Union
from uuid import UUID, uuid4

from django.http import HttpRequest
from pydantic import ValidationError

from ai_core.ids.http_scope import normalize_request
from ai_core.contracts.business import BusinessContext
from ai_core.contracts.scope import ScopeContext
from ai_core.tool_contracts import ToolContext, ContextError

logger = logging.getLogger(__name__)


def _extract_user_id_from_asgi(asgi_scope: Mapping[str, Any]) -> str | None:
    """Extract and validate user_id from ASGI scope (WebSocket).

    Args:
        asgi_scope: ASGI scope dict with 'user' key

    Returns:
        UUID string or None if unauthenticated

    Raises:
        ContextError: If user_id is invalid or not a UUID
    """
    user = asgi_scope.get("user")
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


def _build_scope_from_asgi(
    asgi_scope: Mapping[str, Any],
    tenant_id: str,
    tenant_schema: str | None = None,
) -> ScopeContext:
    """Build ScopeContext from ASGI scope (WebSocket).

    For WebSocket connections, we generate new IDs since each message
    is a new invocation (unlike HTTP where we reuse IDs from headers).

    Args:
        asgi_scope: ASGI scope dict
        tenant_id: Tenant UUID
        tenant_schema: Optional tenant schema name

    Returns:
        ScopeContext with new trace/invocation/run IDs

    Raises:
        ContextError: If validation fails
    """
    user_id = _extract_user_id_from_asgi(asgi_scope)
    try:
        return ScopeContext(
            tenant_id=tenant_id,
            tenant_schema=tenant_schema or tenant_id,
            trace_id=uuid4().hex,
            invocation_id=uuid4().hex,
            run_id=uuid4().hex,
            user_id=user_id,
        )
    except ValidationError as exc:
        raise ContextError("Invalid websocket scope context", field=None) from exc


def prepare_workbench_context(
    request_or_scope: Union[HttpRequest, Mapping[str, Any]],
    *,
    workflow_id: str | None = None,
    collection_id: str | None = None,
    case_id: str | None = None,
    thread_id: str | None = None,
    tenant_id: str | None = None,
    tenant_schema: str | None = None,
) -> ToolContext:
    """Build a standardized ToolContext for RAG Workbench views.

    This helper centralizes the logic for:
    1. Extracting ScopeContext from the request (HTTP or WebSocket/ASGI).
    2. Resolving or generating BusinessContext (Case, Workflow, Collection).
    3. Constructing the final ToolContext.

    Supports both HTTP requests (Django HttpRequest) and WebSocket connections
    (ASGI scope dict).

    Args:
        request_or_scope: Django HttpRequest OR ASGI scope dict (WebSocket).
        workflow_id: Optional workflow ID override.
        collection_id: Optional collection ID override.
        case_id: Optional case ID override.
        thread_id: Optional thread ID override.
        tenant_id: Required for WebSocket, ignored for HTTP (extracted from request).
        tenant_schema: Optional tenant schema (WebSocket), ignored for HTTP.

    Returns:
        ToolContext: A fully validated tool context.

    Raises:
        TenantRequiredError: If tenant cannot be resolved (HTTP).
        ContextError: If validation fails (WebSocket).
    """
    # 1. Base Scope (Tenant, User, Trace, Run)
    # Detect input type: HttpRequest vs. ASGI scope (Mapping)
    if isinstance(request_or_scope, HttpRequest):
        # HTTP Request: Use normalize_request (reuses IDs from headers/session)
        scope: ScopeContext = normalize_request(request_or_scope)
        request = request_or_scope
    elif isinstance(request_or_scope, Mapping):
        # ASGI/WebSocket: Generate new IDs per message
        if not tenant_id:
            raise ContextError(
                "tenant_id is required for WebSocket context", field="tenant_id"
            )
        scope = _build_scope_from_asgi(request_or_scope, tenant_id, tenant_schema)
        # For WebSocket, we don't have GET/headers, so use ASGI scope as "request"
        request = None  # type: ignore
    else:
        raise TypeError(
            f"request_or_scope must be HttpRequest or Mapping, got {type(request_or_scope)}"
        )

    # 2. Business Context Resolution
    # Priority: Function Arg -> Request META/Header -> Session -> Default

    # For HTTP: Extract from request GET/headers/session
    # For WebSocket: Use only function args (no request.GET/headers/session available)
    is_http = request is not None

    # Case ID
    final_case_id = case_id
    if not final_case_id and is_http:
        final_case_id = (
            request.GET.get("case_id")
            or request.headers.get("X-Case-ID")
            or request.session.get("dev_case_id")
        )

    # Workflow ID
    final_workflow_id = workflow_id
    if not final_workflow_id and is_http:
        final_workflow_id = (
            request.POST.get("workflow_id")
            or request.GET.get("workflow_id")
            or request.headers.get("X-Workflow-ID")
            or request.session.get("rag_active_workflow_id")
        )
    if isinstance(final_workflow_id, str):
        final_workflow_id = final_workflow_id.strip() or None
    if not final_workflow_id:
        raise ContextError(
            "workflow_id is required for Workbench requests", field="workflow_id"
        )

    # Collection ID
    final_collection_id = collection_id
    if not final_collection_id and is_http:
        final_collection_id = (
            request.GET.get("collection_id")
            or request.headers.get("X-Collection-ID")
            or ""
        )

    # Thread ID (for Chat)
    final_thread_id = thread_id
    if not final_thread_id and is_http:
        final_thread_id = (
            request.GET.get("thread_id")
            or request.headers.get("X-Thread-ID")
            or request.session.get("rag_chat_thread_id")
        )
    if not final_thread_id:
        final_thread_id = str(uuid4())
        # Side-effect: persist to session if missing (HTTP only)
        if is_http:
            try:
                request.session["rag_chat_thread_id"] = final_thread_id
            except Exception:
                # Session might not be available (e.g., test scenarios)
                pass

    business = BusinessContext(
        case_id=final_case_id,
        workflow_id=final_workflow_id,
        collection_id=final_collection_id,
        thread_id=final_thread_id,
    )

    # 3. Tool Context
    return ToolContext(
        scope=scope,
        business=business,
        metadata={"workflow_id": final_workflow_id},
    )
