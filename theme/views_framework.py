from __future__ import annotations

from django.shortcuts import render
from django.views.decorators.http import require_POST
from structlog.stdlib import get_logger

from ai_core.graphs.business.framework_analysis_graph import (
    build_graph as build_framework_graph,
)
from ai_core.contracts.business import BusinessContext
from ai_core.tools.framework_contracts import FrameworkAnalysisInput
from customers.tenant_context import TenantRequiredError

logger = get_logger(__name__)


def _views():
    from theme import views as theme_views

    return theme_views


def framework_analysis_tool(request):
    """Render the framework analysis developer tool."""
    views = _views()
    try:
        scope = views._scope_context_from_request(request)
    except TenantRequiredError as exc:
        return views._tenant_required_response(exc)
    tenant_id = scope.tenant_id
    tenant_schema = scope.tenant_schema or tenant_id

    # Resolve default collection (manual collection for this tenant)
    manual_collection_id, _ = views._resolve_manual_collection(tenant_id, None)

    return render(
        request,
        "theme/framework_analysis.html",
        {
            "tenant_id": tenant_id,
            "tenant_schema": tenant_schema,
            "default_collection_id": manual_collection_id,
        },
    )


@require_POST
def framework_analysis_submit(request):
    """
    Handle HTMX submission for framework analysis.
    Returns a partial HTML response with the analysis result.
    """
    views = _views()
    try:
        scope = views._scope_context_from_request(request)
    except TenantRequiredError as exc:
        return views._tenant_required_response(exc)

    try:
        # Parse form data
        collection_id = request.POST.get("collection_id") or request.POST.get(
            "document_collection_id"
        )
        document_id = request.POST.get("document_id") or None
        force_reanalysis = request.POST.get("force_reanalysis") == "on"
        confidence_threshold = float(request.POST.get("confidence_threshold", 0.7))

        if not collection_id:
            return views._json_error_response(
                "Collection ID missing",
                status_code=400,
                code="invalid_collection_id",
            )

        input_params = FrameworkAnalysisInput(
            force_reanalysis=force_reanalysis,
            confidence_threshold=confidence_threshold,
        )
        business = BusinessContext(
            collection_id=collection_id,
            document_id=document_id,
        )
        tool_context = scope.to_tool_context(business=business)

        graph = build_framework_graph()
        output = graph.run(
            context=tool_context,
            input_params=input_params,
        )

        response_data = output.model_dump(mode="json")

        # Return as generic JSON response partial
        return render(
            request,
            "theme/partials/_generic_json_response.html",
            {"data": response_data},
        )

    except Exception as e:
        logger.exception("framework_analysis_submit_failed")
        return render(
            request,
            "theme/partials/_generic_json_response.html",
            {"data": {"error": str(e)}},
        )
