from __future__ import annotations

import json
from uuid import uuid4

from django.http import JsonResponse
from django.shortcuts import redirect, render
from django.urls import reverse
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_POST
from structlog.stdlib import get_logger

from ai_core.rag.routing_rules import get_routing_table
from customers.tenant_context import TenantRequiredError
from documents.models import DocumentNotification
from theme.helpers.tasks import submit_business_graph
from theme.validators import SearchQualityParams


logger = get_logger(__name__)


def _views():
    from theme import views as theme_views

    return theme_views


@login_required
def rag_tools(request):
    """Render a minimal interface to exercise the RAG endpoints manually."""
    # Strict enforcing of login for the workbench entrypoint
    views = _views()
    blocked = views._rag_tools_gate(request, json_response=False)
    if blocked is not None:
        return blocked
    try:
        scope = views._scope_context_from_request(request)
    except TenantRequiredError as exc:
        return views._tenant_required_response(exc)
    tenant_id = scope.tenant_id
    tenant_schema = scope.tenant_schema or tenant_id

    collection_options: list[str] = []
    resolver_profile_hint: str | None = None
    resolver_collection_hint: str | None = None

    try:
        routing_table = get_routing_table()
    except Exception:
        logger.warning("rag_tools.routing_table.unavailable", exc_info=True)
        routing_table = None
    else:
        unique_collections = {
            rule.collection_id
            for rule in routing_table.rules
            if getattr(rule, "collection_id", None)
        }
        collection_options = sorted(
            value for value in unique_collections if isinstance(value, str)
        )

        try:
            resolution = routing_table.resolve_with_metadata(
                tenant=tenant_id,
                process=None,
                collection_id=None,
                workflow_id=None,
                doc_class=None,
            )
        except Exception:
            logger.info("rag_tools.profile_resolution.failed", exc_info=True)
        else:
            resolver_profile_hint = resolution.profile
            if resolution.rule and resolution.rule.collection_id:
                resolver_collection_hint = resolution.rule.collection_id

    manual_collection_id, _ = views._resolve_manual_collection(tenant_id, None)

    # Context Resolution (Unified Strategy)
    active_collection_id = request.session.get("rag_active_collection_id")
    active_case_id = request.session.get("rag_active_case_id")

    # Fetch available cases for the tenant
    from cases.models import Case

    case_query = Case.objects.filter(tenant__schema_name=tenant_id).values(
        "external_id", "title"
    )
    case_options = [
        {"id": str(c["external_id"]), "label": f"{c['title']} ({c['external_id']})"}
        for c in case_query
    ]

    context = {
        "active_tab": "search",
        "tenant_id": tenant_id,
        "tenant_schema": tenant_schema,
        "collection_options": collection_options,
        "manual_collection_id": manual_collection_id,
        "case_options": case_options,
        "active_collection_id": active_collection_id,
        "active_case_id": active_case_id,
        "simulated_users": views.get_simulated_users(request.user),
        "current_simulated_user_id": request.session.get("rag_tools_simulated_user_id"),
        "resolver_profile_hint": resolver_profile_hint,
        "resolver_collection_hint": resolver_collection_hint,
    }
    return render(request, "theme/workbench.html", context)


@login_required
@require_POST
def rag_tools_set_context(request):
    """
    Update the active Case and Collection context in the session.
    Invoked via HTMX from the sidebar dropdowns.
    """
    collection_id = request.POST.get("collection_id", "").strip()
    case_id = request.POST.get("case_id", "").strip()

    if collection_id:
        request.session["rag_active_collection_id"] = collection_id
    else:
        # If empty, clear it (global mode)
        if "rag_active_collection_id" in request.session:
            del request.session["rag_active_collection_id"]

    if case_id:
        request.session["rag_active_case_id"] = case_id
    else:
        # If empty, clear it (global mode)
        if "rag_active_case_id" in request.session:
            del request.session["rag_active_case_id"]

    # Return empty response with HX-Refresh to reload the page with new context
    response = JsonResponse({"status": "ok"})
    response["HX-Refresh"] = "true"
    return response


@login_required
@require_POST
def rag_tools_identity_switch(request):
    """Switch the simulated user identity for the workbench."""
    user_id = request.POST.get("user_id")
    active_tab = request.POST.get("active_tab", "search")

    if user_id == "anonymous":
        if "rag_tools_simulated_user_id" in request.session:
            del request.session["rag_tools_simulated_user_id"]
    elif user_id:
        request.session["rag_tools_simulated_user_id"] = user_id
    elif "rag_tools_simulated_user_id" in request.session:
        # Empty value means reset to real user
        del request.session["rag_tools_simulated_user_id"]

    redirect_url = reverse("rag-tools")
    if active_tab:
        redirect_url += f"#{active_tab}"

    return redirect(redirect_url)


@login_required
def tool_collaboration(request):
    """Collaboration playground view (HTMX partial)."""
    notifications = []
    if request.user.is_authenticated:
        try:
            notifications = (
                DocumentNotification.objects.filter(user=request.user)
                .select_related("document", "comment", "comment__user")
                .order_by("-created_at")[:20]
            )
        except Exception:
            pass

    return render(
        request,
        "theme/partials/tool_collaboration.html",
        {
            "notifications": notifications,
        },
    )


@login_required
@require_POST
def start_rerank_workflow(request):
    """Start the software_documentation_collection graph asynchronously via worker queue.

    This view accepts a query and collection parameters, then enqueues the graph
    to run in the background. Results will appear in the /dev-hitl/ queue for HITL review.
    """
    views = _views()
    blocked = views._rag_tools_gate(request, json_response=True)
    if blocked is not None:
        return blocked
    try:
        data = json.loads(request.body)
        query = data.get("query", "").strip()
        collection_id = data.get("collection_id", "").strip()
        quality_params = SearchQualityParams.model_validate(data)
        quality_mode = quality_params.quality_mode
        max_candidates = quality_params.max_candidates

        logger.info(
            "start_rerank_workflow.request", query=query, collection_id=collection_id
        )

        if not query:
            return views._json_error_response(
                "Query is required",
                status_code=400,
                code="missing_query",
            )

        if not collection_id:
            return views._json_error_response(
                "Collection ID is required",
                status_code=400,
                code="missing_collection_id",
            )

        from theme.helpers.context import prepare_workbench_context

        # 1. Prepare Context
        tool_context = prepare_workbench_context(
            request,
            case_id=data.get("case_id"),
            collection_id=collection_id,
            workflow_id="rerank-workflow-manual",
        )

        trace_id = tool_context.scope.trace_id
        # Ensure run_id for this specific execution
        if not tool_context.scope.run_id:
            tool_context.scope.run_id = str(uuid4())

        # Build graph input state
        purpose = data.get("purpose", "collection_search")
        quality_mode = SearchQualityParams.model_validate(data).quality_mode
        max_candidates = SearchQualityParams.model_validate(data).max_candidates

        graph_state = {
            "question": query,
            "collection_scope": collection_id,
            "quality_mode": quality_mode,
            "max_candidates": max_candidates,
            "purpose": purpose,
        }

        col_tool_context = tool_context

        # Build GraphIOSpec-compliant request (Hard Enforcement)
        boundary_request = {
            "schema_id": "noesis.graphs.collection_search",
            "schema_version": "1.0.0",
            "input": graph_state,
            "tool_context": col_tool_context.model_dump(mode="json"),
        }

        # Execute task synchronously with REDUCED timeout (QW-4)
        result_payload, completed = submit_business_graph(
            graph_name="collection_search",
            tool_context=col_tool_context,
            state=boundary_request,
            timeout_s=30,
        )

        logger.info(
            "start_rerank_workflow.submitted",
            task_id=result_payload.get("task_id"),
            trace_id=trace_id,
            completed=completed,
        )

        if not completed:
            return JsonResponse(
                {
                    "status": "pending",
                    "graph_name": "collection_search",
                    "task_id": result_payload.get("task_id"),
                    "trace_id": trace_id,
                    "message": "Workflow wurde gestartet und läuft im Hintergrund.",
                }
            )

        graph_result = result_payload.get("result") or {}
        search_payload = graph_result.get("search") or {}
        telemetry_payload = graph_result.get("telemetry") or {}
        outcome_label = graph_result.get("outcome") or "Workflow abgeschlossen"

        response_data = {
            "status": "completed",
            "graph_name": "collection_search",
            "task_id": result_payload.get("task_id"),
            "trace_id": trace_id,
            "results": search_payload.get("results", []),
            "search": search_payload,
            "telemetry": telemetry_payload or None,
            "graph_result": graph_result,
            "cost_summary": result_payload.get("cost_summary"),
            "message": str(outcome_label).replace("_", " ").capitalize(),
        }

        if request.headers.get("HX-Request"):
            return render(
                request,
                "theme/partials/_web_search_results.html",
                {
                    "results": response_data.get("results"),
                    "search": response_data.get("search"),
                    "trace_id": response_data.get("trace_id"),
                    "status": "completed",
                    "task_id": response_data.get("task_id"),
                },
            )

        return JsonResponse(response_data)

    except json.JSONDecodeError:
        return views._json_error_response(
            "Invalid JSON",
            status_code=400,
            code="invalid_json",
        )
    except Exception:
        logger.exception("start_rerank_workflow.failed")
        return views._json_error_response(
            "internal_error",
            status_code=500,
            code="internal_error",
        )


@login_required
def workbench_index(request):
    """Main container for the RAG Command Center."""
    views = _views()
    blocked = views._rag_tools_gate(request, json_response=False)
    if blocked is not None:
        return blocked
    try:
        scope = views._scope_context_from_request(request)
    except TenantRequiredError:
        # Fallback for dev/testing if no tenant context
        tenant_id, tenant_schema = "dev", "public"
    else:
        tenant_id = scope.tenant_id
        tenant_schema = scope.tenant_schema or tenant_id

    case_id = request.GET.get("case_id") or request.headers.get("X-Case-ID")

    context = {
        "tenant_id": tenant_id,
        "tenant_schema": tenant_schema,
        "case_id": case_id,
        "simulated_users": views._get_dev_simulated_users(),
        "current_simulated_user_id": request.session.get("rag_tools_simulated_user_id"),
    }
    return render(request, "theme/workbench.html", context)


@login_required
def tool_search(request):
    """Render the Search & Retrieval workspace partial."""
    views = _views()
    blocked = views._rag_tools_gate(request, json_response=False)
    if blocked is not None:
        return blocked

    try:
        scope = views._scope_context_from_request(request)
    except TenantRequiredError:
        tenant_id, tenant_schema = None, None
    else:
        tenant_id = scope.tenant_id
        tenant_schema = scope.tenant_schema or tenant_id

    case_id = request.GET.get("case_id") or request.headers.get("X-Case-ID")
    active_collection_id = request.session.get("rag_active_collection_id")

    collection_options: list[dict[str, str]] = []
    if tenant_schema:
        try:
            from django_tenants.utils import schema_context
            from documents.models import DocumentCollection

            with schema_context(tenant_schema):
                collections = (
                    DocumentCollection.objects.select_related("case")
                    .order_by("name", "created_at")
                    .all()
                )
                for collection in collections:
                    label = (
                        collection.name
                        or collection.key
                        or str(collection.collection_id)
                    )
                    case_obj = collection.case
                    if case_obj and getattr(case_obj, "external_id", None):
                        label = f"{label} (case {case_obj.external_id})"
                    collection_options.append(
                        {
                            "id": str(collection.collection_id),
                            "label": label,
                        }
                    )
        except Exception:
            logger.exception("tool_search.collection_options_failed")

    return render(
        request,
        "theme/partials/tool_search.html",
        {
            "case_id": case_id,
            "tenant_id": tenant_id,
            "tenant_schema": tenant_schema,
            "active_collection_id": active_collection_id,
            "collection_options": collection_options,
        },
    )


@login_required
def tool_ingestion(request):
    """Render the Ingestion Pipeline workspace partial."""
    views = _views()
    blocked = views._rag_tools_gate(request, json_response=False)
    if blocked is not None:
        return blocked
    case_id = request.GET.get("case_id") or request.headers.get("X-Case-ID")
    return render(request, "theme/partials/tool_ingestion.html", {"case_id": case_id})


@login_required
def tool_crawler(request):
    """Render the Crawler workspace partial."""
    views = _views()
    blocked = views._rag_tools_gate(request, json_response=False)
    if blocked is not None:
        return blocked
    case_id = request.GET.get("case_id") or request.headers.get("X-Case-ID")
    return render(request, "theme/partials/tool_crawler.html", {"case_id": case_id})


@login_required
def tool_framework(request):
    """Render the Framework Analysis workspace partial."""
    views = _views()
    blocked = views._rag_tools_gate(request, json_response=False)
    if blocked is not None:
        return blocked
    case_id = request.GET.get("case_id") or request.headers.get("X-Case-ID")
    return render(request, "theme/partials/tool_framework.html", {"case_id": case_id})


@login_required
def tool_chat(request):
    """Render the RAG Chat workspace partial."""
    views = _views()
    blocked = views._rag_tools_gate(request, json_response=False)
    if blocked is not None:
        return blocked
    case_id = request.GET.get("case_id") or request.headers.get("X-Case-ID")
    thread_id = request.GET.get("thread_id")
    if thread_id:
        request.session["rag_chat_thread_id"] = thread_id
    else:
        thread_id = request.session.get("rag_chat_thread_id")
    if not thread_id:
        thread_id = uuid4().hex
        request.session["rag_chat_thread_id"] = thread_id
    try:
        scope = views._scope_context_from_request(request)
    except TenantRequiredError:
        tenant_id, tenant_schema = None, None
    else:
        tenant_id = scope.tenant_id
        tenant_schema = scope.tenant_schema or tenant_id

    collection_options: list[dict[str, str]] = []
    if tenant_schema:
        try:
            from django_tenants.utils import schema_context
            from documents.models import DocumentCollection

            with schema_context(tenant_schema):
                collections = (
                    DocumentCollection.objects.select_related("case")
                    .order_by("name", "created_at")
                    .all()
                )
                for collection in collections:
                    label = (
                        collection.name
                        or collection.key
                        or str(collection.collection_id)
                    )
                    case_obj = collection.case
                    if case_obj and getattr(case_obj, "external_id", None):
                        label = f"{label} (case {case_obj.external_id})"
                    collection_options.append(
                        {
                            "id": str(collection.collection_id),
                            "label": label,
                        }
                    )
        except Exception:
            logger.exception("tool_chat.collection_options_failed")

    active_collection_id = request.session.get("rag_active_collection_id")
    chat_scope = request.session.get("rag_chat_scope", "collection")
    # Prefer explicit case_id from args, else session
    if not case_id:
        case_id = request.session.get("rag_active_case_id")

    return render(
        request,
        "theme/partials/tool_chat.html",
        {
            "case_id": case_id,
            "active_case_id": case_id,  # for clarity in template
            "thread_id": thread_id,
            "tenant_id": tenant_id,
            "tenant_schema": tenant_schema,
            "collection_options": collection_options,
            "active_collection_id": active_collection_id,
            "chat_scope": chat_scope,
        },
    )
