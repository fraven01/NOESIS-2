"""Context helpers for RAG Workbench views."""

from __future__ import annotations

import logging
from uuid import uuid4

from django.http import HttpRequest

from ai_core.ids.http_scope import normalize_request
from ai_core.contracts.business import BusinessContext
from ai_core.contracts.scope import ScopeContext
from ai_core.tool_contracts import ToolContext

logger = logging.getLogger(__name__)


def prepare_workbench_context(
    request: HttpRequest,
    *,
    workflow_id: str | None = None,
    collection_id: str | None = None,
    case_id: str | None = None,
    thread_id: str | None = None,
) -> ToolContext:
    """Build a standardized ToolContext for RAG Workbench views.

    This helper centralizes the logic for:
    1. Extracting ScopeContext from the request (Tenant, User, Trace).
    2. resolving or generating BusinessContext (Case, Workflow, Collection).
    3. Constructing the final ToolContext.

    Args:
        request: The HTTP request object.
        workflow_id: Optional workflow ID override.
        collection_id: Optional collection ID override.
        case_id: Optional case ID override.

    Returns:
        ToolContext: A fully validated tool context.

    Raises:
        TenantRequiredError: If tenant cannot be resolved (handled by normalize_request).
    """
    # 1. Base Scope (Tenant, User, Trace, Run)
    # normalize_request handles standard headers and session fallback
    scope: ScopeContext = normalize_request(request)

    # 2. Business Context Resolution
    # Priority: Function Arg -> Request META/Header -> Session -> Default

    # Case ID
    final_case_id = case_id
    if not final_case_id:
        final_case_id = (
            request.GET.get("case_id")
            or request.headers.get("X-Case-ID")
            or request.session.get("dev_case_id")
        )

    # Workflow ID
    final_workflow_id = workflow_id
    if not final_workflow_id:
        final_workflow_id = (
            request.GET.get("workflow_id")
            or request.headers.get("X-Workflow-ID")
            or "rag-workbench-manual"
        )

    # Collection ID
    final_collection_id = collection_id
    if not final_collection_id:
        final_collection_id = (
            request.GET.get("collection_id")
            or request.headers.get("X-Collection-ID")
            or ""
        )

    # Thread ID (for Chat)
    final_thread_id = thread_id
    if not final_thread_id:
        final_thread_id = (
            request.GET.get("thread_id")
            or request.headers.get("X-Thread-ID")
            or request.session.get("rag_chat_thread_id")
        )
    if not final_thread_id:
        final_thread_id = str(uuid4())
        # Side-effect: persist to session if missing (common pattern in workbench)
        request.session["rag_chat_thread_id"] = final_thread_id

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
    )
