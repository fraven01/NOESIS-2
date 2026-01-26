from __future__ import annotations

import json
from uuid import uuid4

from opentelemetry import trace
from opentelemetry.trace import format_trace_id

from celery.result import AsyncResult
from django.core.cache import cache
from django.http import JsonResponse
from django.shortcuts import render
from django.urls import reverse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET
from django.views.decorators.http import require_POST
from pydantic import ValidationError
from structlog.stdlib import get_logger

from ai_core.contracts import BusinessContext, ScopeContext
from ai_core.graphs.technical.collection_search import (
    GraphInput as CollectionSearchGraphInput,
)
from theme.helpers.tasks import submit_business_graph
from ai_core.schemas import CrawlerRunRequest
from common.logging import bind_log_context
from common.task_result import TaskResult
from crawler.manager import CrawlerManager
from customers.tenant_context import TenantRequiredError
from theme.validators import SearchQualityParams

logger = get_logger(__name__)
_WEB_SEARCH_CACHE_TTL_S = 900
_WEB_SEARCH_PENDING_STATES = {"PENDING", "RECEIVED", "STARTED", "RETRY"}


def _cache_web_search_request(task_id: str, payload: dict[str, object]) -> None:
    cache.set(f"web_search:request:{task_id}", payload, timeout=_WEB_SEARCH_CACHE_TTL_S)


def _load_web_search_request(task_id: str) -> dict[str, object]:
    cached = cache.get(f"web_search:request:{task_id}")
    return cached if isinstance(cached, dict) else {}


def _cache_web_search_rerank(task_id: str, payload: dict[str, object]) -> None:
    cache.set(f"web_search:rerank:{task_id}", payload, timeout=_WEB_SEARCH_CACHE_TTL_S)


def _load_web_search_rerank(task_id: str) -> dict[str, object]:
    cached = cache.get(f"web_search:rerank:{task_id}")
    return cached if isinstance(cached, dict) else {}


def _normalize_task_payload(payload: object) -> dict[str, object]:
    if isinstance(payload, dict):
        try:
            task_result = TaskResult.model_validate(payload)
        except ValidationError:
            return payload
        normalized: dict[str, object] = {
            "status": task_result.status,
            "data": dict(task_result.data or {}),
        }
        if task_result.error:
            normalized["error"] = task_result.error
        return normalized
    return {"status": "error", "error": "invalid_task_result", "data": {}}


def _build_collection_search_response(
    response_payload: dict[str, object], trace_id: str
) -> dict[str, object]:
    data_res = response_payload.get("data") or {}
    if not isinstance(data_res, dict):
        data_res = {}

    search_payload = data_res.get("search") or {}
    if not isinstance(search_payload, dict):
        search_payload = {}

    results = search_payload.get("results", []) or []
    telemetry_payload = data_res.get("telemetry")
    if telemetry_payload:
        telemetry_payload = dict(telemetry_payload)
        if "responses" in search_payload:
            telemetry_payload["search_responses"] = search_payload["responses"]

    return {
        "outcome": data_res.get("outcome"),
        "results": results,
        "search": search_payload,
        "telemetry": telemetry_payload,
        "trace_id": trace_id,
    }


def _build_web_acquisition_response(
    response_payload: dict[str, object], trace_id: str
) -> dict[str, object]:
    if response_payload.get("status") == "error":
        error_payload = response_payload.get("error")
        return {
            "outcome": "error",
            "results": [],
            "telemetry": {},
            "trace_id": trace_id,
            "error": error_payload,
        }

    result_state = response_payload.get("data", {})
    if not isinstance(result_state, dict):
        result_state = {}

    output = result_state.get("output", {})
    if not isinstance(output, dict):
        output = {}

    decision = output.get("decision", "error")
    error_msg = output.get("error")
    search_results = output.get("search_results") or []

    response_data: dict[str, object] = {
        "outcome": "completed" if decision in ("acquired", "no_results") else "error",
        "results": search_results,
        "telemetry": {},
        "trace_id": trace_id,
    }

    if error_msg:
        response_data["error"] = error_msg
        response_data["outcome"] = "error"

    return response_data


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
        scope = views._scope_context_from_request(request)
    except TenantRequiredError as exc:
        return views._tenant_required_response(exc)
    tenant_id = scope.tenant_id
    tenant_schema = scope.tenant_schema or tenant_id
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

    rerank_requested = str(data.get("rerank") or "").lower() in ("1", "true", "on")
    request_data = {
        "quality_mode": quality_params.quality_mode,
        "max_candidates": quality_params.max_candidates,
        "purpose": str(data.get("purpose") or "").strip() or None,
    }

    if search_type == "collection_search":
        purpose = str(data.get("purpose") or "").strip()
        if not purpose:
            return views._json_error_response(
                "Purpose is required for Collection Search",
                status_code=400,
                code="missing_purpose",
            )

        quality_mode = quality_params.quality_mode

        # Auto-ingest settings (default: enabled)
        auto_ingest_raw = data.get("auto_ingest")
        auto_ingest = str(auto_ingest_raw or "").lower() in ("on", "true", "1")
        if auto_ingest_raw is None:
            auto_ingest = True
        auto_ingest_min_score = float(data.get("auto_ingest_min_score", 60.0))
        auto_ingest_top_k = int(data.get("auto_ingest_top_k", 10))

        graph_input = {
            "question": query,
            "collection_scope": collection_id,
            "purpose": purpose,
            "quality_mode": quality_mode,
            "auto_ingest": auto_ingest,
            "auto_ingest_min_score": auto_ingest_min_score,
            "auto_ingest_top_k": auto_ingest_top_k,
        }

        try:
            CollectionSearchGraphInput.model_validate(graph_input)
        except Exception as exc:
            logger.info("web_search.collection.invalid_input", error=str(exc))
            return views._json_error_response(
                f"Invalid input: {str(exc)}",
                status_code=400,
                code="invalid_request",
            )

        col_scope = ScopeContext(
            tenant_id=tenant_id,
            tenant_schema=tenant_id,
            trace_id=trace_id,
            invocation_id=str(uuid4()),
            run_id=run_id,
            user_id=user_id,
        )

        col_business = BusinessContext(
            workflow_id="collection-search-manual",
            case_id=case_id,
            collection_id=collection_id,
        )

        col_tool_context = col_scope.to_tool_context(business=col_business)

        boundary_request = {
            "schema_id": "noesis.graphs.collection_search",
            "schema_version": "1.0.0",
            "input": graph_input,
            "tool_context": col_tool_context.model_dump(mode="json"),
        }

        response_payload, _completed = submit_business_graph(
            graph_name="collection_search",
            tool_context=col_tool_context,
            state=boundary_request,
        )
    else:
        from ai_core.tool_contracts import ToolContext

        try:
            top_n = int(data.get("top_n", 5))
            if top_n < 1 or top_n > 20:
                top_n = 5
        except (ValueError, TypeError):
            top_n = 5

        input_payload = {
            "query": query,
            "search_config": {
                "top_n": top_n,
                "prefer_pdf": True,
            },
        }

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

        graph_state = {
            "input": input_payload,
            "tool_context": tool_context.model_dump(mode="json"),
        }

        response_payload, _completed = submit_business_graph(
            graph_name="web_acquisition",
            tool_context=tool_context,
            state=graph_state,
        )

    task_id = (
        response_payload.get("task_id") if isinstance(response_payload, dict) else None
    )
    if not task_id:
        return views._json_error_response(
            "Task submission failed",
            status_code=500,
            code="internal_error",
        )

    _cache_web_search_request(
        task_id,
        {
            "search_type": search_type,
            "query": query,
            "collection_id": collection_id,
            "case_id": case_id,
            "trace_id": trace_id,
            "run_id": run_id,
            "request_data": request_data,
            "rerank": rerank_requested,
        },
    )

    status_url = request.build_absolute_uri(
        reverse("web-search-status", kwargs={"task_id": task_id})
    )
    logger.info(
        "web_search.queued",
        task_id=task_id,
        search_type=search_type,
        hx_request=bool(request.headers.get("HX-Request")),
    )

    if request.headers.get("HX-Request"):
        return render(
            request,
            "theme/partials/_web_search_pending.html",
            {
                "task_id": task_id,
                "poll_url": status_url,
            },
            status=202,
        )

    return JsonResponse(
        {
            "status": "queued",
            "task_id": task_id,
            "status_url": status_url,
            "trace_id": trace_id,
        },
        status=202,
    )


@require_GET
def web_search_status(request, task_id: str):
    """Poll the web search task status and render results when complete."""
    views = _views()
    blocked = views._rag_tools_gate(json_response=True)
    if blocked is not None:
        return blocked

    cached = _load_web_search_request(task_id)
    search_type = str(
        cached.get("search_type") or request.GET.get("search_type") or "web_acquisition"
    ).strip()
    search_type = search_type.lower()
    if search_type in {"external_knowledge", "external-knowledge"}:
        search_type = "web_acquisition"

    query = str(cached.get("query") or request.GET.get("query") or "").strip()
    collection_id = cached.get("collection_id") or request.GET.get("collection_id")
    case_id = cached.get("case_id") or request.GET.get("case_id")
    trace_id = str(cached.get("trace_id") or request.GET.get("trace_id") or uuid4())
    run_id = str(cached.get("run_id") or request.GET.get("run_id") or uuid4())
    request_data = cached.get("request_data") or {}
    rerank_requested = bool(cached.get("rerank") or request.GET.get("rerank"))

    status_url = request.build_absolute_uri(
        reverse("web-search-status", kwargs={"task_id": task_id})
    )

    async_result = AsyncResult(task_id)
    state = async_result.state
    if state in _WEB_SEARCH_PENDING_STATES:
        if request.headers.get("HX-Request"):
            return render(
                request,
                "theme/partials/_web_search_pending.html",
                {
                    "task_id": task_id,
                    "poll_url": status_url,
                },
                status=202,
            )
        return JsonResponse(
            {
                "status": "queued",
                "task_id": task_id,
                "status_url": status_url,
                "state": state,
            },
            status=202,
        )

    if state in ("FAILURE", "REVOKED"):
        error_message = "Search task failed."
        if request.headers.get("HX-Request"):
            return render(
                request,
                "theme/partials/_web_search_error.html",
                {"message": error_message},
                status=500,
            )
        return views._json_error_response(
            error_message,
            status_code=500,
            code="internal_error",
        )

    if state != "SUCCESS":
        if request.headers.get("HX-Request"):
            return render(
                request,
                "theme/partials/_web_search_pending.html",
                {
                    "task_id": task_id,
                    "poll_url": status_url,
                },
                status=202,
            )
        return JsonResponse(
            {
                "status": "queued",
                "task_id": task_id,
                "status_url": status_url,
                "state": state,
            },
            status=202,
        )

    response_payload = _normalize_task_payload(async_result.result)
    if response_payload.get("status") == "error":
        error_message = f"Worker Error: {response_payload.get('error')}"
        if request.headers.get("HX-Request"):
            return render(
                request,
                "theme/partials/_web_search_error.html",
                {"message": error_message},
                status=500,
            )
        return views._json_error_response(
            error_message,
            status_code=500,
            code="internal_error",
        )

    if search_type == "collection_search":
        response_data = _build_collection_search_response(response_payload, trace_id)
    else:
        response_data = _build_web_acquisition_response(response_payload, trace_id)

    if rerank_requested and collection_id:
        rerank_cached = _load_web_search_rerank(task_id)
        if rerank_cached:
            response_data["rerank"] = rerank_cached.get("meta")
            cached_results = rerank_cached.get("results")
            if cached_results is not None:
                response_data["results"] = cached_results
        else:
            try:
                scope = views._scope_context_from_request(request)
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

            try:
                rerank_meta, reranked_results = views._run_rerank_workflow(
                    request_data=request_data,
                    query=query,
                    collection_id=collection_id,
                    results=response_data.get("results") or [],
                    tenant_id=scope.tenant_id,
                    case_id=case_id,
                    trace_id=trace_id,
                    run_id=run_id,
                    user_id=user_id,
                )
            except Exception:  # pragma: no cover - defensive
                logger.exception("web_search.rerank_failed")
                rerank_meta = {
                    "status": "failed",
                    "message": "Rerank could not be started.",
                }
                reranked_results = None

            response_data["rerank"] = rerank_meta
            if reranked_results is not None:
                response_data["results"] = reranked_results

            _cache_web_search_rerank(
                task_id,
                {
                    "meta": rerank_meta,
                    "results": reranked_results,
                },
            )
    elif rerank_requested:
        response_data["rerank"] = {
            "status": "skipped",
            "message": "Rerank skipped: missing collection_id.",
        }

    if request.headers.get("HX-Request"):
        return render(
            request,
            "theme/partials/_web_search_results.html",
            {
                "results": response_data.get("results"),
                "search": response_data.get("search"),
                "trace_id": response_data.get("trace_id"),
                "collection_id": collection_id,
                "case_id": case_id,
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
            scope = views._scope_context_from_request(request)
        except TenantRequiredError as exc:
            return views._tenant_required_response(exc)
        tenant_id = scope.tenant_id
        tenant_schema = scope.tenant_schema or tenant_id
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
            workflow_id=(
                str(data.get("workflow_id") or request_model.workflow_id or "").strip()
                or None
            ),
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
