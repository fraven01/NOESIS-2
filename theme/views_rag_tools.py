from __future__ import annotations

import json
from uuid import uuid4

from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import redirect, render
from django.urls import reverse
from django.views.decorators.http import require_POST
from structlog.stdlib import get_logger

from ai_core.rag.routing_rules import get_routing_table, is_collection_routing_enabled
from cases.services import ensure_case
from customers.tenant_context import TenantRequiredError
from documents.models import DocumentNotification
from theme.validators import SearchQualityParams


logger = get_logger(__name__)


def _views():
    from theme import views as theme_views

    return theme_views


def rag_tools(request):
    """Render a minimal interface to exercise the RAG endpoints manually."""
    views = _views()
    try:
        tenant_id, tenant_schema = views._tenant_context_from_request(request)
    except TenantRequiredError as exc:
        return views._tenant_required_response(exc)

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

    # Manage Dev Session Case ID
    case_id = views.DEV_DEFAULT_CASE_ID
    try:
        request.session["dev_case_id"] = case_id
    except Exception:
        pass
    try:
        tenant_obj = views.TenantContext.resolve_identifier(tenant_id, allow_pk=True)
        if tenant_obj is not None:
            ensure_case(
                tenant_obj,
                case_id,
                title="Dev Local",
                reopen_closed=True,
            )
    except Exception:
        logger.exception(
            "rag_tools.default_case_bootstrap_failed",
            extra={"tenant_id": tenant_id, "case_id": case_id},
        )

    return render(
        request,
        "theme/rag_tools.html",
        {
            "tenant_id": tenant_id,
            "tenant_schema": tenant_schema,
            "case_id": case_id,
            "default_embedding_profile": getattr(
                settings, "RAG_DEFAULT_EMBEDDING_PROFILE", "standard"
            ),
            "collection_options": collection_options,
            "collection_alias_enabled": is_collection_routing_enabled(),
            "resolver_profile_hint": resolver_profile_hint,
            "resolver_collection_hint": resolver_collection_hint,
            "crawler_runner_url": reverse("ai_core:rag_crawler_run"),
            "crawler_default_workflow_id": getattr(
                settings, "CRAWLER_DEFAULT_WORKFLOW_ID", ""
            ),
            "crawler_dry_run_default": bool(
                getattr(settings, "CRAWLER_DRY_RUN_DEFAULT", False)
            ),
            "manual_collection_id": manual_collection_id,
            "simulated_users": views._get_dev_simulated_users(),
            "current_simulated_user_id": request.session.get(
                "rag_tools_simulated_user_id"
            ),
        },
    )


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


@require_POST
def start_rerank_workflow(request):
    """Start the software_documentation_collection graph asynchronously via worker queue.

    This view accepts a query and collection parameters, then enqueues the graph
    to run in the background. Results will appear in the /dev-hitl/ queue for HITL review.
    """
    views = _views()
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

        try:
            tenant_id, tenant_schema = views._tenant_context_from_request(request)
        except TenantRequiredError as exc:
            return views._tenant_required_response(exc)
        user = getattr(request, "user", None)
        user_id = (
            str(user.pk)
            if user
            and getattr(user, "is_authenticated", False)
            and getattr(user, "pk", None) is not None
            else None
        )
        trace_id = str(uuid4())
        run_id = str(uuid4())

        # Build graph input state according to GraphInput schema
        purpose = data.get("purpose", "collection_search")
        if not isinstance(purpose, str):
            purpose = "collection_search"
        purpose = purpose.strip() or "collection_search"

        graph_state = {
            "question": query,
            "collection_scope": collection_id,
            "quality_mode": quality_mode,
            "max_candidates": max_candidates,
            "purpose": purpose,
        }

        # Submit graph task to worker queue without waiting (timeout_s=0)
        task_payload = {
            "state": graph_state,
        }

        scope = {
            "tenant_id": tenant_id,
            "case_id": str(data.get("case_id") or "").strip() or None,
            "trace_id": trace_id,
            "workflow_id": "rerank-workflow-manual",
            "run_id": run_id,
            # Identity ID (Pre-MVP ID Contract)
            "user_id": user_id,
        }

        # Execute task synchronously with extended timeout for pipeline visualization
        result_payload, completed = views.submit_worker_task(
            task_payload=task_payload,
            scope=scope,
            graph_name="collection_search",
            timeout_s=120,
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


def workbench_index(request):
    """Main container for the RAG Command Center."""
    views = _views()
    try:
        tenant_id, tenant_schema = views._tenant_context_from_request(request)
    except TenantRequiredError:
        # Fallback for dev/testing if no tenant context
        tenant_id, tenant_schema = "dev", "public"

    case_id = request.GET.get("case_id") or request.headers.get("X-Case-ID")

    context = {
        "tenant_id": tenant_id,
        "tenant_schema": tenant_schema,
        "case_id": case_id,
        "simulated_users": views._get_dev_simulated_users(),
        "current_simulated_user_id": request.session.get("rag_tools_simulated_user_id"),
    }
    return render(request, "theme/workbench.html", context)


def tool_search(request):
    """Render the Search & Retrieval workspace partial."""
    case_id = request.GET.get("case_id") or request.headers.get("X-Case-ID")
    return render(request, "theme/partials/tool_search.html", {"case_id": case_id})


def tool_ingestion(request):
    """Render the Ingestion Pipeline workspace partial."""
    case_id = request.GET.get("case_id") or request.headers.get("X-Case-ID")
    return render(request, "theme/partials/tool_ingestion.html", {"case_id": case_id})


def tool_crawler(request):
    """Render the Crawler workspace partial."""
    case_id = request.GET.get("case_id") or request.headers.get("X-Case-ID")
    return render(request, "theme/partials/tool_crawler.html", {"case_id": case_id})


def tool_framework(request):
    """Render the Framework Analysis workspace partial."""
    case_id = request.GET.get("case_id") or request.headers.get("X-Case-ID")
    return render(request, "theme/partials/tool_framework.html", {"case_id": case_id})


def tool_chat(request):
    """Render the RAG Chat workspace partial."""
    case_id = request.GET.get("case_id") or request.headers.get("X-Case-ID")
    return render(request, "theme/partials/tool_chat.html", {"case_id": case_id})
