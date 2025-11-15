import json
from uuid import uuid4

from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.urls import reverse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from structlog.stdlib import get_logger

from ai_core.graphs.software_documentation_collection import (
    GraphContextPayload,
    InvalidGraphInput,
    build_graph,
)
from ai_core.rag.collections import (
    MANUAL_COLLECTION_SLUG,
    ensure_manual_collection,
    manual_collection_uuid,
)
from ai_core.rag.routing_rules import (
    get_routing_table,
    is_collection_routing_enabled,
)
from llm_worker.runner import submit_worker_task


logger = get_logger(__name__)

def home(request):
    """Render the homepage."""

    return render(request, "theme/home.html")


def _tenant_context_from_request(request) -> tuple[str, str]:
    """Return the tenant identifier and schema for the current request."""

    tenant = getattr(request, "tenant", None)
    default_schema = getattr(settings, "DEFAULT_TENANT_SCHEMA", None) or "dev"

    tenant_id: str | None = None
    tenant_schema: str | None = None

    if tenant is not None:
        tenant_id = getattr(tenant, "tenant_id", None)
        tenant_schema = getattr(tenant, "schema_name", None)

    if not tenant_schema:
        tenant_schema = default_schema

    if not tenant_id:
        tenant_id = tenant_schema

    return tenant_id, tenant_schema


def _resolve_manual_collection(
    tenant_id: str | None,
    requested: object,
    *,
    ensure: bool = False,
) -> tuple[str | None, str | None]:
    """Return ``(manual_collection_id, resolved_collection_id)`` for the request."""

    manual_id: str | None = None
    if tenant_id:
        try:
            base_value = (
                ensure_manual_collection(tenant_id)
                if ensure
                else manual_collection_uuid(tenant_id)
            )
            manual_id = str(base_value)
        except ValueError:
            logger.info(
                "rag_tools.manual_collection.unavailable",
                extra={"tenant_id": tenant_id},
            )

    requested_text = str(requested or "").strip()
    if not requested_text:
        return manual_id, manual_id

    if manual_id:
        if requested_text.lower() == MANUAL_COLLECTION_SLUG:
            return manual_id, manual_id
        if requested_text == manual_id:
            return manual_id, manual_id

    if requested_text.lower() == MANUAL_COLLECTION_SLUG:
        return manual_id, manual_id

    return manual_id, requested_text


def rag_tools(request):
    """Render a minimal interface to exercise the RAG endpoints manually."""

    tenant_id, tenant_schema = _tenant_context_from_request(request)

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

    manual_collection_id, _ = _resolve_manual_collection(tenant_id, None)

    return render(
        request,
        "theme/rag_tools.html",
        {
            "tenant_id": tenant_id,
            "tenant_schema": tenant_schema,
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
            "crawler_shadow_default": bool(
                getattr(settings, "CRAWLER_SHADOW_MODE_DEFAULT", False)
            ),
            "crawler_dry_run_default": bool(
                getattr(settings, "CRAWLER_DRY_RUN_DEFAULT", False)
            ),
            "manual_collection_id": manual_collection_id,
        },
    )


@require_POST
def web_search(request):
    """Execute the software documentation graph for manual RAG searches."""

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    query = str(data.get("query") or "").strip()
    purpose_value = data.get("purpose")
    purpose = str(purpose_value).strip() if isinstance(purpose_value, str) else ""

    if not query:
        return JsonResponse({"error": "Query is required"}, status=400)
    if not purpose:
        return JsonResponse({"error": "Purpose is required"}, status=400)

    tenant_id, _ = _tenant_context_from_request(request)
    case_id = str(data.get("case_id") or "local").strip() or "local"
    trace_id = str(uuid4())
    run_id = str(uuid4())

    logger.info("web_search.query", query=query, purpose=purpose)

    context = GraphContextPayload(
        tenant_id=tenant_id,
        workflow_id="software-docs-manual",
        case_id=case_id,
        trace_id=trace_id,
        run_id=run_id,
    )

    graph_state = {
        "input": {
            "question": query,
            "purpose": purpose,
            "collection_scope": "default",
        }
    }

    graph = build_graph()

    try:
        _state, result = graph.run(state=graph_state, meta=context.model_dump())
    except InvalidGraphInput as exc:
        logger.info("web_search.invalid_input", error=str(exc))
        return JsonResponse({"error": "Ungültige Eingabe für den Graphen."}, status=400)
    except Exception:
        logger.exception("web_search.failed")
        return JsonResponse({"error": "Graph execution failed."}, status=500)

    search_payload = result.get("search")
    if not isinstance(search_payload, dict):
        search_payload = {}
    results = search_payload.get("results")
    if not isinstance(results, list):
        results = []

    response_data = {
        "outcome": result.get("outcome"),
        "results": results,
        "search": search_payload,
        "telemetry": result.get("telemetry"),
        "trace_id": trace_id,
    }

    return JsonResponse(response_data)


@require_POST
@csrf_exempt
def web_search_ingest_selected(request):
    """Ingest user-selected URLs from web search results via crawler_runner API.

    Uses the internal crawler_runner HTTP API endpoint for proper layer separation.
    Returns a summary of started ingestion tasks.
    """
    import httpx

    try:
        data = json.loads(request.body)
        urls = data.get("urls", [])

        logger.info("web_search_ingest_selected", url_count=len(urls))
        if not urls:
            return JsonResponse({"error": "URLs are required"}, status=400)

        tenant_id, tenant_schema = _tenant_context_from_request(request)
        manual_collection_id, collection_id = _resolve_manual_collection(
            tenant_id, data.get("collection_id"), ensure=True
        )
        if not collection_id:
            return JsonResponse(
                {"error": "Collection ID could not be resolved"}, status=400
            )

        # Build headers for API call
        headers = {
            "Content-Type": "application/json",
            "X-Tenant-ID": tenant_id,
            "X-Tenant-Schema": tenant_schema,
            "X-Case-ID": data.get("case_id", "local"),
            "X-Trace-ID": str(uuid4()),
        }

        # Build payload for crawler_runner endpoint
        crawler_payload = {
            "workflow_id": "web-search-ingestion",
            "mode": "live",
            "origins": [{"url": url} for url in urls],
            "collection_id": collection_id,
        }

        # Call crawler_runner API endpoint
        # Use localhost since we're in the same service
        crawler_url = "http://localhost:8000" + reverse("ai_core:rag_crawler_run")

        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                crawler_url,
                json=crawler_payload,
                headers=headers,
            )

        if response.status_code == 200:
            response_data = response.json()
            return JsonResponse(
                {
                    "status": "completed",
                    "result": response_data,
                    "url_count": len(urls),
                }
            )
        else:
            return JsonResponse(
                {
                    "error": "Crawler API call failed",
                    "status_code": response.status_code,
                    "detail": response.text,
                },
                status=response.status_code,
            )

    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    except Exception as e:
        logger.exception("web_search_ingest_selected.failed")
        return JsonResponse({"error": str(e)}, status=500)


@require_POST
def start_rerank_workflow(request):
    """Start the software_documentation_collection graph asynchronously via worker queue.

    This view accepts a query and collection parameters, then enqueues the graph
    to run in the background. Results will appear in the /dev-hitl/ queue for HITL review.
    """
    try:
        data = json.loads(request.body)
        query = data.get("query", "").strip()
        collection_id = data.get("collection_id", "").strip()
        quality_mode = data.get("quality_mode", "standard").strip().lower()
        max_candidates = data.get("max_candidates", 20)

        logger.info(
            "start_rerank_workflow.request", query=query, collection_id=collection_id
        )

        if not query:
            return JsonResponse({"error": "Query is required"}, status=400)

        if not collection_id:
            return JsonResponse({"error": "Collection ID is required"}, status=400)

        # Validate max_candidates
        try:
            max_candidates = int(max_candidates)
            if max_candidates < 5 or max_candidates > 40:
                max_candidates = 20
        except (ValueError, TypeError):
            max_candidates = 20

        # Validate quality_mode
        if quality_mode not in ("standard", "premium", "fast"):
            quality_mode = "standard"

        tenant_id, tenant_schema = _tenant_context_from_request(request)
        trace_id = str(uuid4())
        run_id = str(uuid4())

        # Build graph input state according to GraphInput schema
        purpose = data.get("purpose", "software_documentation_collection")
        if not isinstance(purpose, str):
            purpose = "software_documentation_collection"
        purpose = purpose.strip() or "software_documentation_collection"

        graph_state = {
            "question": query,
            "collection_scope": collection_id,
            "quality_mode": quality_mode,
            "max_candidates": max_candidates,
            "purpose": purpose,
        }

        # Build metadata for the graph execution
        graph_meta = {
            "tenant_id": tenant_id,
            "trace_id": trace_id,
            "workflow_id": "rerank-workflow-manual",
            "case_id": data.get("case_id", "local"),
            "run_id": run_id,
        }

        # Submit graph task to worker queue without waiting (timeout_s=0)
        task_payload = {
            "state": graph_state,
        }

        scope = {
            "tenant_id": tenant_id,
            "case_id": graph_meta["case_id"],
            "trace_id": trace_id,
        }

        # Enqueue the task without blocking (timeout_s=0 means don't wait)
        result_payload, completed = submit_worker_task(
            task_payload=task_payload,
            scope=scope,
            graph_name="software_documentation_collection",
            timeout_s=0,
        )

        logger.info(
            "start_rerank_workflow.submitted",
            task_id=result_payload.get("task_id"),
            trace_id=trace_id,
            completed=completed,
        )

        return JsonResponse(
            {
                "status": "pending",
                "graph_name": "software_documentation_collection",
                "task_id": result_payload.get("task_id"),
                "trace_id": trace_id,
                "message": "Workflow wurde gestartet. Ergebnisse erscheinen in /dev-hitl/.",
            }
        )

    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    except Exception as e:
        logger.exception("start_rerank_workflow.failed")
        return JsonResponse({"error": str(e)}, status=500)
