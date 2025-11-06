import json
from collections.abc import Mapping
from uuid import uuid4

from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.urls import reverse
from django.views.decorators.http import require_POST
from structlog.stdlib import get_logger

from ai_core.graphs.external_knowledge_graph import (
    CrawlerIngestionOutcome,
    ExternalKnowledgeGraphConfig,
    GraphContextPayload,
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


logger = get_logger(__name__)


class _NoOpIngestionAdapter:
    """No-op adapter for graphs that skip ingestion phase (run_until="after_search").

    This adapter is never actually called since web_search stops before ingestion.
    It exists only to satisfy the build_graph() signature.
    """

    def trigger(
        self,
        *,
        url: str,
        collection_id: str,
        context: Mapping[str, str],
    ) -> CrawlerIngestionOutcome:
        """No-op trigger that should never be called."""
        logger.error(
            "noop_adapter.trigger_called",
            url=url,
            msg="No-op adapter was called unexpectedly. Graph should stop before ingestion.",
        )
        return CrawlerIngestionOutcome(
            decision="skipped",
            crawler_decision="noop",
            document_id=None,
        )


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
    """Handle web search requests from the RAG tools page via ExternalKnowledgeGraph.

    This view only performs the search phase. For ingestion, use the separate
    /web-search/ingest-selected/ endpoint after the user has selected URLs.
    """
    try:
        data = json.loads(request.body)
        query = data.get("query")
        tenant_id, tenant_schema = _tenant_context_from_request(request)
        manual_collection_id, collection_id = _resolve_manual_collection(
            tenant_id, data.get("collection_id")
        )
        if not collection_id:
            collection_id = manual_collection_id

        logger.info("web_search.query", query=query)
        if not query:
            return JsonResponse({"error": "Query is required"}, status=400)

        # Build graph context
        context = GraphContextPayload(
            tenant_id=tenant_id,
            trace_id=str(uuid4()),
            workflow_id="web-search-manual",
            case_id=data.get("case_id", "local"),
            run_id=str(uuid4()),
        )

        # For manual testing, we only run until after search
        # The user will then select URLs and call /web-search/ingest-selected/
        run_until = "after_search"

        # Build graph input
        graph_input = {
            "query": query,
            "collection_id": collection_id,
            "enable_hitl": False,  # No human-in-the-loop for manual testing
            "run_until": run_until,
        }

        # Build graph with no-op adapter (never called since run_until="after_search")
        graph = build_graph(
            ingestion_adapter=_NoOpIngestionAdapter(),
            config=ExternalKnowledgeGraphConfig(
                top_n=10,
                prefer_pdf=data.get("prefer_pdf", False),
                blocked_domains=frozenset(data.get("blocked_domains", [])),
                min_snippet_length=40,
                run_until=run_until,
            ),
        )

        # Run graph
        state, result = graph.run(state=graph_input, meta=context.model_dump())

        # Extract results from state
        search_results = state.get("search", {}).get("results", [])
        selection = state.get("selection", {})
        selected = selection.get("selected", [])
        shortlisted = selection.get("shortlisted", [])
        ingestion = state.get("ingestion", {})
        outcome = result.get("outcome", "unknown")

        # Convert URLs to strings for JSON serialization
        for result_item in search_results:
            if "url" in result_item:
                result_item["url"] = str(result_item["url"])

        response_data = {
            "outcome": outcome,
            "results": search_results,
            "selected": selected,
            "shortlisted": shortlisted,
            "ingestion": ingestion,
            "telemetry": result.get("telemetry"),
            "trace_id": context.trace_id,
        }

        logger.info(
            "web_search.completed", outcome=outcome, result_count=len(search_results)
        )
        return JsonResponse(response_data)

    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    except Exception as e:
        logger.exception("web_search.failed")
        return JsonResponse({"error": str(e)}, status=500)


@require_POST
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
