from __future__ import annotations

import json
from uuid import uuid4

from opentelemetry import trace
from opentelemetry.trace import format_trace_id

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from structlog.stdlib import get_logger

from ai_core.contracts import BusinessContext, ScopeContext
from ai_core.graphs.technical.collection_search import (
    GraphInput as CollectionSearchGraphInput,
    build_graph as build_collection_search_graph,
)
from ai_core.schemas import CrawlerRunRequest
from common.logging import bind_log_context
from crawler.manager import CrawlerManager
from customers.tenant_context import TenantRequiredError
from theme.validators import SearchQualityParams

logger = get_logger(__name__)


def _views():
    from theme import views as theme_views

    return theme_views


@require_POST
def web_search(request):
    """Execute the external knowledge graph for manual RAG searches."""
    views = _views()
    blocked = views._rag_tools_gate(json_response=True)
    if blocked is not None:
        return blocked
    try:
        if request.headers.get("HX-Request"):
            # HTMX sends form-encoded data by default unless configured for JSON
            # We'll support both for robustness
            if request.content_type == "application/json":
                data = json.loads(request.body)
            else:
                data = request.POST
        else:
            data = json.loads(request.body)
    except json.JSONDecodeError:
        return views._json_error_response(
            "Invalid JSON",
            status_code=400,
            code="invalid_json",
        )

    query = str(data.get("query") or "").strip()
    if not query:
        return views._json_error_response(
            "Query is required",
            status_code=400,
            code="missing_query",
        )

    try:
        tenant_id, tenant_schema = views._tenant_context_from_request(request)
    except TenantRequiredError as exc:
        return views._tenant_required_response(exc)
    case_id = str(data.get("case_id") or "").strip() or None
    user = getattr(request, "user", None)
    user_id = (
        str(user.pk)
        if user
        and getattr(user, "is_authenticated", False)
        and getattr(user, "pk", None) is not None
        else None
    )

    trace_id = str(uuid4())
    # Try to use active OTel trace
    span = trace.get_current_span()
    ctx = span.get_span_context()
    if ctx.is_valid:
        trace_id = format_trace_id(ctx.trace_id)

    run_id = str(uuid4())

    # Bind log context so all subsequent logger.info() calls include these IDs
    bind_log_context(
        trace_id=trace_id,
        tenant_id=tenant_id,
        case_id=case_id,
        run_id=run_id,
        user_id=user_id,
    )

    logger.info("web_search.query", query=query)

    manual_collection_id, resolved_collection_id = views._resolve_manual_collection(
        tenant_id, data.get("collection_id")
    )
    # Default to manual collection if no specific collection requested
    # But preserve what the user typed if it's a valid alias or ID
    collection_id = resolved_collection_id or manual_collection_id or "default"

    logger.info(
        "web_search.collection_scope",
        collection_id=collection_id,
        requested=data.get("collection_id"),
        manual_collection_id=manual_collection_id,
    )

    search_type = str(data.get("search_type") or "web_acquisition").strip().lower()
    if search_type in {"external_knowledge", "external-knowledge"}:
        search_type = "web_acquisition"
    quality_params = SearchQualityParams.model_validate(data)

    if search_type == "collection_search":
        # Collection Search Graph Execution
        purpose = str(data.get("purpose") or "").strip()
        if not purpose:
            return views._json_error_response(
                "Purpose is required for Collection Search",
                status_code=400,
                code="missing_purpose",
            )

        quality_mode = quality_params.quality_mode
        auto_ingest = str(data.get("auto_ingest") or "").lower() == "on"

        graph_input = {
            "question": query,
            "collection_scope": collection_id,
            "purpose": purpose,
            "quality_mode": quality_mode,
            "auto_ingest": auto_ingest,
        }

        try:
            # Validate input using Pydantic model
            CollectionSearchGraphInput.model_validate(graph_input)
        except Exception as exc:
            logger.info("web_search.collection.invalid_input", error=str(exc))
            return views._json_error_response(
                f"Invalid input: {str(exc)}",
                status_code=400,
                code="invalid_request",
            )

    # Execution logic consolidated below to handle both search types cleanly

    response_data = {}

    if search_type == "collection_search":
        # ... Collection Search Logic (Existing) ...
        # Copied from original file context
        purpose = str(data.get("purpose") or "").strip()
        if not purpose:
            return views._json_error_response(
                "Purpose is required for Collection Search",
                status_code=400,
                code="missing_purpose",
            )

        quality_mode = quality_params.quality_mode
        auto_ingest = str(data.get("auto_ingest") or "").lower() == "on"

        graph_input = {
            "question": query,
            "collection_scope": collection_id,
            "purpose": purpose,
            "quality_mode": quality_mode,
            "auto_ingest": auto_ingest,
        }
        try:
            CollectionSearchGraphInput.model_validate(graph_input)
        except Exception as exc:
            return views._json_error_response(
                f"Invalid input: {str(exc)}",
                status_code=400,
                code="invalid_request",
            )

        # BREAKING CHANGE (Option A - Strict Separation):
        # Build ScopeContext and BusinessContext for Collection Search
        col_scope = ScopeContext(
            tenant_id=tenant_id,
            tenant_schema=tenant_id,
            trace_id=trace_id,
            invocation_id=str(uuid4()),
            run_id=run_id,
            user_id=user_id,  # Pre-MVP ID Contract
        )

        col_business = BusinessContext(
            workflow_id="collection-search-manual",
            case_id=case_id,
            collection_id=collection_id,
        )

        # Build context_payload as nested + flattened dict (hybrid for compatibility)
        col_context_payload = {
            # ToolContext structure (for model_validate)
            "scope": col_scope.model_dump(mode="json"),
            "business": col_business.model_dump(mode="json"),
            "metadata": {},
            # Flattened fields for backward compatibility
            "tenant_id": tenant_id,
            "workflow_id": "collection-search-manual",
            "case_id": case_id,
            "trace_id": trace_id,
            "run_id": run_id,
            "user_id": user_id,
        }

        col_graph = build_collection_search_graph()
        # CollectionSearchGraph.run returns (state, result)
        final_state, result = col_graph.run(state=graph_input, meta=col_context_payload)

        search_payload = final_state.get("search", {})
        results = search_payload.get("results", [])
        telemetry_payload = result.get("telemetry")
        if telemetry_payload:
            telemetry_payload = dict(telemetry_payload)
            if "responses" in search_payload:
                telemetry_payload["search_responses"] = search_payload["responses"]

        response_data = {
            "outcome": result.get("outcome"),
            "results": results,
            "search": search_payload,
            "telemetry": telemetry_payload,
            "trace_id": trace_id,
        }

    else:
        # Web Acquisition Graph (Search Only)
        from ai_core.graphs.web_acquisition_graph import build_web_acquisition_graph
        from ai_core.tool_contracts import ToolContext

        # Parse configurable parameters from request
        try:
            top_n = int(data.get("top_n", 5))
            if top_n < 1 or top_n > 20:
                top_n = 5
        except (ValueError, TypeError):
            top_n = 5

        input_payload = {
            "query": query,
            "mode": data.get("mode", "search_only"),
            "search_config": {
                "top_n": top_n,
                "prefer_pdf": True,
            },
        }

        # Build ToolContext
        business = BusinessContext(
            workflow_id="web-acquisition-manual",
            case_id=case_id,
            collection_id=collection_id,
        )

        tool_context = ToolContext(
            scope={
                "tenant_id": tenant_id,
                "tenant_schema": tenant_schema,
                "trace_id": trace_id,
                "user_id": user_id,
                "invocation_id": str(uuid4()),
                "ingestion_run_id": run_id,
            },
            business=business,
            metadata={},
        )

        # Acquisition State
        graph_state = {
            "input": input_payload,
            "tool_context": tool_context,
        }

        web_graph = build_web_acquisition_graph()
        result_state = web_graph.invoke(graph_state)

        output = result_state.get("output", {})
        decision = output.get("decision", "error")
        error_msg = output.get("error")
        # Legacy UI expects "search.results" structure
        search_results = output.get("search_results") or []

        response_data = {
            # P2 Fix: Both 'acquired' and 'no_results' are successful completion states
            "outcome": (
                "completed" if decision in ("acquired", "no_results") else "error"
            ),
            "results": search_results,
            "search": {"results": search_results},
            "telemetry": {},
            "trace_id": trace_id,
        }

        if error_msg:
            response_data["error"] = error_msg
            response_data["outcome"] = "error"

            if decision == "error":
                logger.warning(
                    "web_search.acquisition_failed", extra={"error": error_msg}
                )

    # Common Logic
    results = response_data.get("results", [])
    search_payload = response_data.get("search", {})
    trace_id = response_data.get("trace_id")

    if data.get("rerank"):
        try:
            rerank_meta, reranked_results = views._run_rerank_workflow(
                request_data=data,
                query=query,
                collection_id=collection_id,
                results=results,
                tenant_id=tenant_id,
                case_id=case_id,
                trace_id=trace_id,
                user_id=user_id,
            )
        except Exception:  # pragma: no cover - defensive
            logger.exception("web_search.rerank_failed")
            rerank_meta = {
                "status": "failed",
                "message": "Rerank konnte nicht gestartet werden.",
            }
            reranked_results = None

        response_data["rerank"] = rerank_meta
        if reranked_results is not None:
            response_data["results"] = reranked_results

    if request.headers.get("HX-Request"):
        return render(
            request,
            "theme/partials/_web_search_results.html",
            {
                "results": response_data.get("results"),
                "search": response_data.get("search"),
                "trace_id": trace_id,
                "collection_id": collection_id,
            },
        )

    return JsonResponse(response_data)


@require_POST
@csrf_exempt
def web_search_ingest_selected(request):
    """Ingest user-selected URLs from web search results via crawler service."""
    views = _views()
    blocked = views._rag_tools_gate(json_response=True)
    if blocked is not None:
        return blocked
    try:
        if request.headers.get("HX-Request"):
            if request.content_type == "application/json":
                data = json.loads(request.body)
            else:
                data = request.POST
                # Handle list of URLs from form data if needed, though usually this endpoint expects JSON
                # For HTMX, we might need to parse 'urls' from a list input or comma string
                if "urls" not in data and "urls[]" in data:
                    data = data.copy()
                    data["urls"] = request.POST.getlist("urls[]")
                elif "urls" in data and isinstance(data["urls"], str):
                    # If it's a single string, wrap it or split it?
                    # The partial sends a JSON string in hx-vals, so it should come as a param
                    # But hx-vals with JSON format usually requires hx-ext="json-enc" or manual parsing
                    # Let's assume standard form data for now, but the partial used hx-vals='{\"urls\": ...}' which sends as params
                    pass
        else:
            data = json.loads(request.body)

        urls = data.get("urls", [])
        if isinstance(urls, str):
            # If it came as a single string (e.g. from hx-vals simple key-value), wrap it
            urls = [urls]

        mode = data.get("mode", "live")  # Pass mode to crawler

        logger.info("web_search_ingest_selected", url_count=len(urls), mode=mode)
        if not urls:
            return views._json_error_response(
                "URLs are required",
                status_code=400,
                code="missing_urls",
            )

        try:
            tenant_id, tenant_schema = views._tenant_context_from_request(request)
        except TenantRequiredError as exc:
            return views._tenant_required_response(exc)
        manual_collection_id, collection_id = views._resolve_manual_collection(
            tenant_id, data.get("collection_id"), ensure=True
        )
        collection_id = views._normalize_collection_id(collection_id, tenant_schema)
        if not collection_id:
            return views._json_error_response(
                "Collection ID could not be resolved",
                status_code=400,
                code="invalid_collection",
            )

        # Build payload for crawler dispatch

        # We must use proper UUIDs for collection_id as verified by the schema
        # manual_collection_id is verified stringified UUID from _resolve_manual_collection

        request_model = CrawlerRunRequest(
            workflow_id="web-search-ingestion",
            mode=mode,
            origins=[{"url": url} for url in urls],
            collection_id=collection_id,
        )

        # L2 -> L3 Dispatch
        # BREAKING CHANGE (Option A - Strict Separation):
        # Separate infrastructure (scope_context) from business (business_context)
        scope = ScopeContext(
            tenant_id=tenant_id,
            tenant_schema=tenant_schema,
            trace_id=str(data.get("trace_id") or "").strip() or str(uuid4()),
            invocation_id=str(uuid4()),
            ingestion_run_id=str(uuid4()),
        )
        business = BusinessContext(
            case_id=str(data.get("case_id") or "").strip() or None,
            collection_id=collection_id,
        )
        tool_context = scope.to_tool_context(business=business)
        meta = {
            "scope_context": scope.model_dump(mode="json", exclude_none=True),
            "business_context": business.model_dump(mode="json", exclude_none=True),
            "tool_context": tool_context.model_dump(mode="json", exclude_none=True),
        }

        manager = CrawlerManager()
        try:
            result = manager.dispatch_crawl_request(request_model, meta)
        except Exception as exc:
            logger.exception("web_search.crawler_dispatch_failed")
            return views._json_error_response(
                str(exc),
                status_code=500,
                code="internal_error",
            )

        response_data = {
            "status": "dispatched",
            "count": result.get("count", 0),
            "task_ids": result.get("tasks", []),
        }

        if request.headers.get("HX-Request"):
            return render(
                request,
                "theme/partials/_ingestion_status.html",
                {
                    "status": "completed",
                    "result": response_data,
                    "task_ids": response_data.get("task_ids"),
                    "url_count": len(urls),
                },
            )

        return JsonResponse(
            {
                "status": "completed",
                "result": response_data,
                "task_ids": response_data.get("task_ids"),
            }
        )

    except json.JSONDecodeError:
        return views._json_error_response(
            "Invalid JSON",
            status_code=400,
            code="invalid_json",
        )
    except Exception as e:
        import traceback

        logger.exception("web_search_ingest_selected.failed")
        return views._json_error_response(
            str(e),
            status_code=500,
            code="internal_error",
            details={"traceback": traceback.format_exc()},
        )
