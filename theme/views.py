import json
from collections.abc import Mapping
from uuid import uuid4

from django.conf import settings
from django.http import JsonResponse, HttpRequest
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
from ai_core.rag.routing_rules import (
    get_routing_table,
    is_collection_routing_enabled,
)


logger = get_logger(__name__)


class SimpleCrawlerIngestionAdapter:
    """Simple adapter for triggering crawler ingestion via Celery task."""

    def trigger(
        self,
        *,
        url: str,
        collection_id: str,
        context: Mapping[str, str],
    ) -> CrawlerIngestionOutcome:
        """Trigger the crawler ingestion for the given URL."""
        from io import BytesIO
        from ai_core import views as ai_core_views

        try:
            # Build request data for crawler
            crawler_request_data = {
                "workflow_id": context.get("workflow_id", "web-search-ingestion"),
                "mode": "live",
                "origins": [{"url": url}],
                "collection_id": collection_id,
            }

            # Encode request body
            body = json.dumps(crawler_request_data).encode("utf-8")

            # Create a mock Django HttpRequest with BytesIO stream
            crawler_request = HttpRequest()
            crawler_request.method = "POST"
            crawler_request._stream = BytesIO(body)
            crawler_request.META = {
                "CONTENT_TYPE": "application/json",
                "CONTENT_LENGTH": str(len(body)),
                "HTTP_X_TENANT_ID": context.get("tenant_id", ""),
                "HTTP_X_TRACE_ID": context.get("trace_id", ""),
            }

            # Create mock tenant object (required by django-tenants)
            from types import SimpleNamespace

            tenant_id = context.get("tenant_id", "")
            crawler_request.tenant = SimpleNamespace(
                schema_name=tenant_id, tenant_id=tenant_id
            )

            # Call crawler_runner directly with Django HttpRequest
            # DRF views wrap the request themselves
            response = ai_core_views.crawler_runner(crawler_request)

            if response.status_code == 200:
                # Extract document_id from response if available
                response_data = response.data if hasattr(response, "data") else {}
                document_id = response_data.get("document_id")

                return CrawlerIngestionOutcome(
                    decision="ingested",
                    crawler_decision="accepted",
                    document_id=document_id,
                )
            else:
                logger.warning(
                    "crawler_ingestion_failed",
                    url=url,
                    status=response.status_code,
                )
                return CrawlerIngestionOutcome(
                    decision="ingestion_error",
                    crawler_decision="rejected",
                    document_id=None,
                )
        except Exception as exc:
            logger.exception("crawler_ingestion_exception", url=url, exc_info=exc)
            return CrawlerIngestionOutcome(
                decision="ingestion_error",
                crawler_decision="exception",
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
        collection_id = data.get("collection_id", "manual-search")

        logger.info("web_search.query", query=query)
        if not query:
            return JsonResponse({"error": "Query is required"}, status=400)

        tenant_id, tenant_schema = _tenant_context_from_request(request)

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

        # Build graph with ingestion adapter and config
        graph = build_graph(
            ingestion_adapter=SimpleCrawlerIngestionAdapter(),
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
    """Ingest user-selected URLs from web search results via CrawlerIngestionGraph."""
    try:
        data = json.loads(request.body)
        urls = data.get("urls", [])
        collection_id = data.get("collection_id", "manual-search")

        logger.info("web_search_ingest_selected", url_count=len(urls))
        if not urls:
            return JsonResponse({"error": "URLs are required"}, status=400)

        tenant_id, tenant_schema = _tenant_context_from_request(request)

        # Build context for crawler ingestion
        context = {
            "tenant_id": tenant_id,
            "trace_id": str(uuid4()),
            "workflow_id": "web-search-ingestion",
            "case_id": data.get("case_id", "local"),
        }

        # Create adapter
        adapter = SimpleCrawlerIngestionAdapter()

        # Trigger ingestion for each URL
        results = []
        for url in urls:
            outcome = adapter.trigger(
                url=url,
                collection_id=collection_id,
                context=context,
            )
            results.append(
                {
                    "url": url,
                    "decision": outcome.decision,
                    "crawler_decision": outcome.crawler_decision,
                    "document_id": outcome.document_id,
                }
            )

        response_data = {
            "status": "completed",
            "results": results,
            "trace_id": context["trace_id"],
        }

        logger.info("web_search_ingest_selected.completed", result_count=len(results))
        return JsonResponse(response_data)

    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    except Exception as e:
        logger.exception("web_search_ingest_selected.failed")
        return JsonResponse({"error": str(e)}, status=500)
