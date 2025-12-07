import json
from typing import Mapping
from uuid import uuid4

from django.conf import settings
from django.core.cache import cache
from django.http import JsonResponse, HttpResponse
from django.shortcuts import render
from django.urls import reverse
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from structlog.stdlib import get_logger

from ai_core.services import _get_documents_repository
from ai_core.graphs.external_knowledge_graph import (
    CrawlerIngestionAdapter,
    CrawlerIngestionOutcome,
    GraphContextPayload,
    InvalidGraphInput,
    build_graph as build_external_knowledge_graph,
)
from ai_core.graphs.collection_search import (
    GraphInput as CollectionSearchGraphInput,
    build_graph as build_collection_search_graph,
)
from ai_core.rag.collections import (
    MANUAL_COLLECTION_SLUG,
    manual_collection_uuid,
)
from documents.collection_service import CollectionService
from ai_core.rag.routing_rules import (
    get_routing_table,
    is_collection_routing_enabled,
)
from ai_core.llm import routing as llm_routing
from llm_worker.runner import submit_worker_task

from customers.tenant_context import TenantContext, TenantRequiredError
from documents.services.document_space_service import (
    DocumentSpaceRequest,
    DocumentSpaceService,
)

from ai_core.views import crawl_selected as _core_crawl_selected
from ai_core.graphs.framework_analysis_graph import build_graph as build_framework_graph
from ai_core.tools.framework_contracts import FrameworkAnalysisInput


logger = get_logger(__name__)
DOCUMENT_SPACE_SERVICE = DocumentSpaceService()
build_graph = build_external_knowledge_graph  # Backwards compatibility for tests
crawl_selected = _core_crawl_selected  # Re-export for tests


def home(request):
    """Render the homepage."""

    return render(request, "theme/home.html")


def _tenant_context_from_request(request) -> tuple[str, str]:
    """Return the tenant identifier and schema for the current request."""

    tenant_obj = TenantContext.from_request(request, allow_headers=False, require=True)
    tenant_schema = getattr(tenant_obj, "schema_name", None)
    if tenant_schema is None:
        tenant_schema = getattr(tenant_obj, "tenant_id", None)

    # Strict Policy: The ID is the Schema Name
    tenant_id = tenant_schema

    if tenant_schema is None or tenant_id is None:
        raise TenantRequiredError("Tenant could not be resolved from request")

    return str(tenant_id), str(tenant_schema)


def _tenant_required_response(exc: TenantRequiredError) -> JsonResponse:
    return JsonResponse({"detail": str(exc)}, status=403)


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
            if ensure:
                service = CollectionService()
                base_value = service.ensure_manual_collection(tenant_id)
            else:
                base_value = manual_collection_uuid(tenant_id)
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
        if requested_text.lower() == "manual":
            return manual_id, manual_id

    if requested_text.lower() == MANUAL_COLLECTION_SLUG:
        return manual_id, manual_id

    return manual_id, requested_text


def _parse_limit(value: object, *, default: int = 25) -> int:
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        numeric = default
    return max(5, min(200, numeric))


def _parse_bool(value: object, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    token = str(value).strip().lower()
    if not token:
        return default
    if token in {"1", "true", "yes", "on"}:
        return True
    if token in {"0", "false", "no", "off"}:
        return False
    return default


def _human_readable_bytes(size: object) -> str:
    try:
        numeric = float(size)
    except (TypeError, ValueError):
        return "—"
    if numeric < 0:
        return "—"
    units = ("B", "KB", "MB", "GB", "TB")
    value = float(numeric)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{int(value)} B"


_RERANK_MODEL_FALLBACK = "fast"
_RERANK_CACHE_PREFIX = "rag_tools.rerank_model"


def _resolve_rerank_model_preset() -> str:
    """Resolve the configured rerank model preset via llm routing with caching."""

    preset_label = getattr(settings, "RERANK_MODEL_PRESET", "") or ""
    preset_label = preset_label.strip()
    if not preset_label:
        return _RERANK_MODEL_FALLBACK

    cache_key = f"{_RERANK_CACHE_PREFIX}:{preset_label}"
    cached = cache.get(cache_key)
    if isinstance(cached, str) and cached:
        return cached

    try:
        resolved = llm_routing.resolve(preset_label)
    except ValueError:
        logger.warning(
            "rag_tools.rerank_model.resolve_failed",
            preset=preset_label,
        )
        resolved = _RERANK_MODEL_FALLBACK

    cache.set(cache_key, resolved, timeout=300)
    return resolved


def _normalise_quality_mode(value: object, default: str = "standard") -> str:
    candidate = str(value or "").strip().lower()
    if candidate in {"standard", "premium", "fast"}:
        return candidate
    return default


def _normalise_max_candidates(value: object, *, default: int = 20) -> int:
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        numeric = default
    numeric = max(5, min(40, numeric))
    return numeric


def _result_identifier(result: Mapping[str, object]) -> str | None:
    """Return the identifier used to merge rerank scores back onto results."""

    for key in ("document_id", "id"):
        candidate = result.get(key)
        if candidate:
            return str(candidate)
    metadata = result.get("meta")
    if isinstance(metadata, Mapping):
        for key in ("document_id", "id"):
            candidate = metadata.get(key)
            if candidate:
                return str(candidate)
    return None


def _apply_rerank_results(
    results: list[Mapping[str, object]],
    rerank_entries: list[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Return ``results`` reordered with rerank scores applied."""

    if not rerank_entries:
        return [dict(item) for item in results]

    remaining = [dict(item) for item in results]
    ordered: list[dict[str, object]] = []
    consumed_ids: set[str] = set()

    for entry in rerank_entries:
        entry_id = entry.get("id")
        if not entry_id:
            continue
        entry_id = str(entry_id)
        for candidate in remaining:
            result_id = _result_identifier(candidate)
            if result_id == entry_id and entry_id not in consumed_ids:
                candidate = dict(candidate)
                candidate["rerank"] = {
                    key: value for key, value in entry.items() if key != "id"
                }
                ordered.append(candidate)
                consumed_ids.add(entry_id)
                break

    for candidate in remaining:
        candidate_id = _result_identifier(candidate)
        if candidate_id and candidate_id in consumed_ids:
            continue
        ordered.append(candidate)

    return ordered


def _build_rerank_state(
    *,
    query: str,
    collection_id: str,
    results: list[Mapping[str, object]],
    request_data: Mapping[str, object],
) -> dict[str, object]:
    quality_mode = _normalise_quality_mode(request_data.get("quality_mode"))
    max_candidates = _normalise_max_candidates(request_data.get("max_candidates"))
    purpose = str(request_data.get("purpose") or "web_search_rerank").strip() or (
        "web_search_rerank"
    )
    return {
        "question": query,
        "collection_scope": collection_id,
        "quality_mode": quality_mode,
        "max_candidates": max_candidates,
        "purpose": purpose,
        "search": {"results": results},
    }


def _run_rerank_workflow(
    *,
    request_data: Mapping[str, object],
    query: str,
    collection_id: str,
    results: list[Mapping[str, object]],
    tenant_id: str,
    case_id: str | None,
    trace_id: str,
) -> tuple[dict[str, object], list[dict[str, object]] | None]:
    """Trigger the rerank worker graph and return ``(meta, updated_results)``."""

    if not results:
        return (
            {
                "status": "skipped",
                "message": "Keine Ergebnisse zum Reranken vorhanden.",
            },
            None,
        )

    state = _build_rerank_state(
        query=query,
        collection_id=collection_id,
        results=results,
        request_data=request_data,
    )
    model_preset = _resolve_rerank_model_preset()
    task_payload = {
        "state": state,
        "workflow_id": "web-search-rerank",
        "control": {
            "model_preset": model_preset,
        },
    }
    scope = {
        "tenant_id": tenant_id,
        "case_id": case_id,
        "trace_id": trace_id,
    }
    rerank_response, completed = submit_worker_task(
        task_payload=task_payload,
        scope=scope,
        graph_name="collection_search",
        timeout_s=60,
    )

    meta = {
        "status": "succeeded" if completed else "queued",
        "task_id": rerank_response.get("task_id"),
        "model_preset": model_preset,
    }
    task_id = meta["task_id"]
    if task_id:
        meta["status_url"] = reverse(
            "llm_worker:task_status",
            kwargs={"task_id": task_id},
        )

    if not completed:
        meta["message"] = "Rerank workflow wurde gestartet und läuft im Hintergrund."
        return meta, None

    result_payload = rerank_response.get("result") or {}
    ranked_entries = result_payload.get("ranked") or []
    merged_results = _apply_rerank_results(results, ranked_entries)

    if "model" in result_payload:
        meta["model"] = result_payload["model"]
    if "latency_s" in result_payload:
        meta["latency_s"] = result_payload["latency_s"]
    meta["graph_name"] = rerank_response.get("graph_name", "collection_search")

    return meta, merged_results


class _ViewCrawlerIngestionAdapter(CrawlerIngestionAdapter):
    """Adapter that triggers crawler ingestion by calling crawler_runner view directly."""

    def trigger(
        self,
        *,
        url: str,
        collection_id: str,
        context: Mapping[str, str],
    ) -> CrawlerIngestionOutcome:
        """Trigger ingestion for the given URL."""
        from ai_core.views import crawler_runner
        from django.http import HttpRequest

        from customers.tenant_context import TenantContext

        tenant_id = context.get("tenant_id", "dev")
        trace_id = context.get("trace_id", "")
        case_id = context.get("case_id")
        mode = context.get("mode", "live")
        tenant_schema = context.get("tenant_schema") or tenant_id

        payload = {
            "workflow_id": "external-knowledge-ingestion",
            "mode": mode,
            "origins": [{"url": url}],
            "collection_id": collection_id,
        }
        body = json.dumps(payload).encode("utf-8")

        try:
            # Create a mock Django request
            django_request = HttpRequest()
            django_request.method = "POST"
            django_request.META = {
                "CONTENT_TYPE": "application/json",
                "HTTP_CONTENT_TYPE": "application/json",
                "CONTENT_LENGTH": str(len(body)),
                "HTTP_X_TENANT_ID": str(tenant_id),
                "HTTP_X_TENANT_SCHEMA": str(tenant_schema),
                "HTTP_X_CASE_ID": str(case_id) if case_id else None,
                "HTTP_X_TRACE_ID": str(trace_id),
            }
            django_request._body = body

            # Attach tenant if available
            tenant = TenantContext.resolve_identifier(tenant_id)
            if tenant:
                django_request.tenant = tenant
            else:
                logger.warning("crawler.adapter.tenant_not_found", tenant_id=tenant_id)

            # Call crawler_runner directly with the constructed HttpRequest
            response = crawler_runner(django_request)
            response_data = response.data if hasattr(response, "data") else {}

            if response.status_code == 200:
                return CrawlerIngestionOutcome(
                    decision="ingested",
                    crawler_decision=response_data.get("decision", "unknown"),
                    document_id=response_data.get("document_id"),
                )
            return CrawlerIngestionOutcome(
                decision="ingestion_error", crawler_decision="http_error"
            )
        except Exception:
            logger.exception("crawler.trigger.failed")
            return CrawlerIngestionOutcome(
                decision="ingestion_error", crawler_decision="trigger_exception"
            )


def rag_tools(request):
    """Render a minimal interface to exercise the RAG endpoints manually."""
    try:
        tenant_id, tenant_schema = _tenant_context_from_request(request)
    except TenantRequiredError as exc:
        return _tenant_required_response(exc)

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

    # Manage Dev Session Case ID
    case_id = request.session.get("dev_case_id")
    if not case_id:
        import time
        from uuid import uuid4

        case_id = f"dev-case-{int(time.time())}-{uuid4().hex[:6]}"
        request.session["dev_case_id"] = case_id

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
            "crawler_shadow_default": bool(
                getattr(settings, "CRAWLER_SHADOW_MODE_DEFAULT", False)
            ),
            "crawler_dry_run_default": bool(
                getattr(settings, "CRAWLER_DRY_RUN_DEFAULT", False)
            ),
            "manual_collection_id": manual_collection_id,
        },
    )


def document_space(request):
    """Expose a developer workbench for inspecting document collections."""

    try:
        tenant_id, tenant_schema = _tenant_context_from_request(request)
    except TenantRequiredError as exc:
        return _tenant_required_response(exc)

    tenant_obj = getattr(request, "tenant", None)
    if tenant_obj is None:
        try:
            tenant_obj = TenantContext.resolve_identifier(tenant_id)
        except Exception:
            tenant_obj = None

    requested_collection = request.GET.get("collection")
    limit = _parse_limit(request.GET.get("limit"))
    limit_options = [10, 25, 50, 100, 200]
    if limit not in limit_options:
        limit_options = sorted(set(limit_options + [limit]))
    latest_only = _parse_bool(request.GET.get("latest"), default=True)
    search_term = str(request.GET.get("q", "") or "").strip()
    cursor_param = str(request.GET.get("cursor", "") or "").strip()
    workflow_filter = str(request.GET.get("workflow", "") or "").strip()

    repository = _get_documents_repository()
    params = DocumentSpaceRequest(
        requested_collection=requested_collection,
        limit=limit,
        latest_only=latest_only,
        cursor=cursor_param or None,
        workflow_filter=workflow_filter or None,
        search_term=search_term,
    )
    result = DOCUMENT_SPACE_SERVICE.build_context(
        tenant_context=tenant_id,
        tenant_obj=tenant_obj,
        params=params,
        repository=repository,
    )

    query_defaults = {
        "collection": result.selected_collection_identifier or "",
        "limit": limit,
        "latest": "1" if latest_only else "0",
        "workflow": workflow_filter,
        "q": search_term,
    }

    return render(
        request,
        "theme/document_space.html",
        {
            "tenant_id": tenant_id,
            "tenant_schema": tenant_schema,
            "search_term": search_term,
            "latest_only": latest_only,
            "limit": limit,
            "limit_options": limit_options,
            "cursor": cursor_param,
            "workflow_filter": workflow_filter,
            "query_defaults": query_defaults,
            "next_query": (
                {**query_defaults, "cursor": result.next_cursor}
                if result.next_cursor
                else None
            ),
            "debug_tenant": str(tenant_obj),
            "debug_collections_count": len(result.collections),
            **result.as_context(),
        },
    )


def document_explorer(request):
    """Developer workbench tool for inspecting document collections (HTMX partial)."""

    try:
        try:
            tenant_id, tenant_schema = _tenant_context_from_request(request)
        except TenantRequiredError as exc:
            return _tenant_required_response(exc)

        tenant_obj = getattr(request, "tenant", None)
        if tenant_obj is None:
            try:
                tenant_obj = TenantContext.resolve_identifier(tenant_id)
            except Exception:
                tenant_obj = None

        requested_collection = request.GET.get("collection")
        limit = _parse_limit(request.GET.get("limit"))
        limit_options = [10, 25, 50, 100, 200]
        if limit not in limit_options:
            limit_options = sorted(set(limit_options + [limit]))
        latest_only = _parse_bool(request.GET.get("latest"), default=True)
        search_term = str(request.GET.get("q", "") or "").strip()
        cursor_param = str(request.GET.get("cursor", "") or "").strip()
        workflow_filter = str(request.GET.get("workflow", "") or "").strip()

        repository = _get_documents_repository()
        params = DocumentSpaceRequest(
            requested_collection=requested_collection,
            limit=limit,
            latest_only=latest_only,
            cursor=cursor_param or None,
            workflow_filter=workflow_filter or None,
            search_term=search_term,
        )
        result = DOCUMENT_SPACE_SERVICE.build_context(
            tenant_context=tenant_id,
            tenant_obj=tenant_obj,
            params=params,
            repository=repository,
        )

        query_defaults = {
            "collection": result.selected_collection_identifier or "",
            "limit": limit,
            "latest": "1" if latest_only else "0",
            "workflow": workflow_filter,
            "q": search_term,
        }

        return render(
            request,
            "theme/partials/tool_documents.html",
            {
                "tenant_id": tenant_id,
                "tenant_schema": tenant_schema,
                "search_term": search_term,
                "latest_only": latest_only,
                "limit": limit,
                "limit_options": limit_options,
                "cursor": cursor_param,
                "workflow_filter": workflow_filter,
                "query_defaults": query_defaults,
                "next_query": (
                    {**query_defaults, "cursor": result.next_cursor}
                    if result.next_cursor
                    else None
                ),
                "debug_tenant": str(tenant_obj),
                "debug_collections_count": len(result.collections),
                **result.as_context(),
            },
        )
    except Exception as exc:
        logger.exception("document_explorer.crashed")
        return render(
            request,
            "theme/partials/tool_documents.html",
            {"documents_error": f"Critical Error: {str(exc)}"},
        )


@require_POST
def web_search(request):
    """Execute the external knowledge graph for manual RAG searches."""

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
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    query = str(data.get("query") or "").strip()
    if not query:
        return JsonResponse({"error": "Query is required"}, status=400)

    try:
        tenant_id, _ = _tenant_context_from_request(request)
    except TenantRequiredError as exc:
        return _tenant_required_response(exc)
    case_id = str(data.get("case_id") or "").strip() or None
    trace_id = str(uuid4())
    run_id = str(uuid4())

    logger.info("web_search.query", query=query)

    context = GraphContextPayload(
        tenant_id=tenant_id,
        workflow_id="external-knowledge-manual",
        case_id=case_id,
        trace_id=trace_id,
        run_id=run_id,
    )

    manual_collection_id, resolved_collection_id = _resolve_manual_collection(
        tenant_id, data.get("collection_id")
    )
    collection_id = resolved_collection_id or manual_collection_id or "default"

    logger.info(
        "web_search.collection_scope",
        collection_id=collection_id,
        requested=data.get("collection_id"),
        manual_collection_id=manual_collection_id,
    )

    search_type = str(data.get("search_type") or "external_knowledge").strip()

    if search_type == "collection_search":
        # Collection Search Graph Execution
        purpose = str(data.get("purpose") or "").strip()
        if not purpose:
            return JsonResponse(
                {"error": "Purpose is required for Collection Search"}, status=400
            )

        quality_mode = _normalise_quality_mode(data.get("quality_mode"))
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
            return JsonResponse({"error": f"Invalid input: {str(exc)}"}, status=400)

        graph = build_collection_search_graph()
        graph_state = graph_input

    else:
        # External Knowledge Graph Execution (Default)
        graph_state = {
            "query": query,
            "collection_id": collection_id,
            "run_until": "after_search",  # Per user request, only run search
        }
        ingestion_adapter = _ViewCrawlerIngestionAdapter()
        graph = build_graph(ingestion_adapter=ingestion_adapter)

    try:
        # The graph returns the final state and a result payload
        final_state, result = graph.run(state=graph_state, meta=context.model_dump())
    except InvalidGraphInput as exc:
        logger.info("web_search.invalid_input", error=str(exc))
        return JsonResponse({"error": "Ungültige Eingabe für den Graphen."}, status=400)
    except Exception:
        logger.exception("web_search.failed")
        return JsonResponse({"error": "Graph execution failed."}, status=500)

    # Extract search results from the final graph state
    search_payload = final_state.get("search", {})
    results = search_payload.get("results", [])

    telemetry_payload = result.get("telemetry")
    if telemetry_payload is not None:
        telemetry_payload = dict(telemetry_payload)
        search_responses = search_payload.get("responses")
        if isinstance(search_responses, list):
            telemetry_payload["search_responses"] = search_responses

    response_data = {
        "outcome": result.get("outcome"),
        "results": results,
        "search": search_payload,
        "telemetry": telemetry_payload,
        "trace_id": trace_id,
    }

    if data.get("rerank"):
        try:
            rerank_meta, reranked_results = _run_rerank_workflow(
                request_data=data,
                query=query,
                collection_id=collection_id,
                results=results,
                tenant_id=tenant_id,
                case_id=case_id,
                trace_id=trace_id,
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
    """Ingest user-selected URLs from web search results via crawler_runner.

    Calls crawler_runner view directly to avoid HTTP overhead and tenant routing issues.
    Returns a summary of started ingestion tasks.
    """
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
                    # Let's assume standard form data for now, but the partial used hx-vals='{"urls": ...}' which sends as params
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
            return JsonResponse({"error": "URLs are required"}, status=400)

        try:
            tenant_id, tenant_schema = _tenant_context_from_request(request)
        except TenantRequiredError as exc:
            return _tenant_required_response(exc)
        manual_collection_id, collection_id = _resolve_manual_collection(
            tenant_id, data.get("collection_id"), ensure=True
        )
        if not collection_id:
            return JsonResponse(
                {"error": "Collection ID could not be resolved"}, status=400
            )

        # Build payload for crawl_selected
        crawl_payload = {
            "urls": urls,
            "workflow_id": "web-search-ingestion",
            "collection_id": collection_id,
            "mode": mode,
        }

        # Create a new request with the crawl payload
        from django.http import HttpRequest

        crawl_request = HttpRequest()
        crawl_request.method = "POST"
        crawl_request.META = request.META.copy()
        crawl_request._body = json.dumps(crawl_payload).encode("utf-8")
        crawl_request.META.setdefault("CONTENT_TYPE", "application/json")
        crawl_request.META.setdefault("HTTP_CONTENT_TYPE", "application/json")
        crawl_request.META.setdefault("CONTENT_LENGTH", str(len(crawl_request._body)))

        # Ensure tenant headers are present for crawler_runner
        crawl_request.META.setdefault("HTTP_X_TENANT_ID", tenant_id)
        crawl_request.META.setdefault("HTTP_X_TENANT_SCHEMA", tenant_schema)
        case_id = str(data.get("case_id") or "").strip() or None
        if case_id:
            crawl_request.META.setdefault("HTTP_X_CASE_ID", case_id)
        crawl_request.META.setdefault("HTTP_X_TRACE_ID", str(uuid4()))

        # Copy tenant context
        if hasattr(request, "tenant"):
            crawl_request.tenant = request.tenant
        if hasattr(request, "tenant_schema"):
            crawl_request.tenant_schema = request.tenant_schema

        # Call crawl_selected
        response = crawl_selected(crawl_request)

        # Parse response
        try:
            response_data = json.loads(response.content.decode())
        except (json.JSONDecodeError, AttributeError):
            response_data = {}

        if response.status_code in (200, 202):
            if request.headers.get("HX-Request"):
                return render(
                    request,
                    "theme/partials/_ingestion_status.html",
                    {
                        "status": (
                            "accepted" if response.status_code == 202 else "completed"
                        ),
                        "result": response_data,
                        "task_ids": response_data.get("task_ids"),
                        "url_count": len(urls),
                    },
                )

            return JsonResponse(
                {
                    "status": (
                        "accepted" if response.status_code == 202 else "completed"
                    ),
                    "result": response_data,
                    "task_ids": response_data.get("task_ids"),
                    "url_count": len(urls),
                },
                status=response.status_code,
            )
        else:
            if request.headers.get("HX-Request"):
                return render(
                    request,
                    "theme/partials/_ingestion_status.html",
                    {
                        "status": (
                            "accepted" if response.status_code == 202 else "completed"
                        ),
                        "result": response_data,
                        "task_ids": response_data.get("task_ids"),
                        "url_count": len(urls),
                        "error": (
                            response_data.get("details")
                            if response.status_code != 200
                            else None
                        ),
                    },
                )
            else:
                return JsonResponse(
                    {
                        "error": "Crawler call failed",
                        "status_code": response.status_code,
                        "detail": response_data.get(
                            "details", response_data.get("detail", str(response_data))
                        ),
                    },
                    status=response.status_code,
                )

    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    except Exception as e:
        import traceback
        import sys

        print("!!! WEB SEARCH INGEST SELECTED FAILED !!!", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        logger.exception("web_search_ingest_selected.failed")
        return JsonResponse(
            {"error": str(e), "traceback": traceback.format_exc()}, status=500
        )


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

        try:
            tenant_id, tenant_schema = _tenant_context_from_request(request)
        except TenantRequiredError as exc:
            return _tenant_required_response(exc)
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

        # Build metadata for the graph execution
        graph_meta = {
            "tenant_id": tenant_id,
            "trace_id": trace_id,
            "workflow_id": "rerank-workflow-manual",
            "case_id": str(data.get("case_id") or "").strip() or None,
            "run_id": run_id,
        }

        # Submit graph task to worker queue without waiting (timeout_s=0)
        task_payload = {
            "state": graph_state,
            **graph_meta,
        }

        scope = {
            "tenant_id": tenant_id,
            "case_id": graph_meta["case_id"],
            "trace_id": trace_id,
        }

        # Execute task synchronously with extended timeout for pipeline visualization
        result_payload, completed = submit_worker_task(
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
        response_payload = {
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
                    "results": response_payload.get("results"),
                    "search": response_payload.get("search"),
                    "trace_id": trace_id,
                    "collection_id": collection_id,
                },
            )

        return JsonResponse(response_payload)

    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    except Exception as e:
        logger.exception("start_rerank_workflow.failed")
        return JsonResponse({"error": str(e)}, status=500)


@require_POST
def crawler_submit(request):
    """Handle crawler form submission via HTMX."""
    if not request.headers.get("HX-Request"):
        return JsonResponse({"error": "HTMX required"}, status=400)

    try:
        # Parse form data
        data = request.POST

        # Build payload (mimic JS buildCrawlerPayload)
        payload = {
            "workflow_id": data.get("workflow_id"),
            "mode": data.get("mode", "live"),
            "origin_url": data.get("origin_url"),
            "document_id": data.get("document_id"),
            "title": data.get("title"),
            "language": data.get("language"),
            "provider": data.get("provider") or "web",
            "content_type": data.get("content_type") or "text/html",
            "content": data.get("content"),
            "collection_id": data.get("collection_id"),
            "manual_review": data.get("review"),
        }

        # Handle booleans (checkboxes send 'on' or nothing)
        payload["fetch"] = data.get("fetch") == "on"
        payload["shadow_mode"] = data.get("shadow_mode") == "on"
        payload["dry_run"] = data.get("dry_run") == "on"
        payload["force_retire"] = data.get("force_retire") == "on"
        payload["recompute_delta"] = data.get("recompute_delta") == "on"

        # Handle snapshot
        if data.get("snapshot") == "on":
            payload["snapshot"] = {"enabled": True}
            if data.get("snapshot_label"):
                payload["snapshot"]["label"] = data.get("snapshot_label")

        # Handle tags
        tags = data.get("tags")
        if tags:
            payload["tags"] = [t.strip() for t in tags.split(",") if t.strip()]

        # Handle max_document_bytes
        if data.get("max_document_bytes"):
            try:
                payload["max_document_bytes"] = int(data.get("max_document_bytes"))
            except (ValueError, TypeError):
                pass

        # Handle origin_urls list
        origin_urls_text = data.get("origin_urls", "")
        if origin_urls_text:
            additional_origins = [
                url.strip() for url in origin_urls_text.splitlines() if url.strip()
            ]
            if additional_origins:
                # If we have multiple origins, we need to structure the 'origins' list
                # The primary 'origin_url' is also included in the logic usually
                origins = []
                if payload.get("origin_url"):
                    origins.append({"url": payload["origin_url"]})

                for url in additional_origins:
                    origins.append({"url": url})

                # Deduplicate based on URL
                seen = set()
                unique_origins = []
                for o in origins:
                    if o["url"] not in seen:
                        seen.add(o["url"])
                        unique_origins.append(o)

                payload["origins"] = unique_origins

        # Call crawler_runner
        from ai_core.views import crawler_runner
        from django.test import RequestFactory

        body = json.dumps(payload).encode("utf-8")

        # Create a mock Django request using RequestFactory
        factory = RequestFactory()
        django_request = factory.post(
            "/ai/ingest/crawler/run/", data=body, content_type="application/json"
        )

        # Ensure tenant headers
        tenant_id, tenant_schema = _tenant_context_from_request(request)
        django_request.META["HTTP_X_TENANT_ID"] = tenant_id
        django_request.META["HTTP_X_TENANT_SCHEMA"] = tenant_schema

        # Propagate case_id if present
        case_id = data.get("case_id")
        if case_id:
            django_request.META["HTTP_X_CASE_ID"] = str(case_id).strip()

        # Copy tenant context
        if hasattr(request, "tenant"):
            django_request.tenant = request.tenant

        response = crawler_runner(django_request)

        if hasattr(response, "data"):
            response_data = response.data
        else:
            try:
                response_data = json.loads(response.content.decode())
            except (json.JSONDecodeError, AttributeError):
                response_data = {}

        return render(
            request,
            "theme/partials/_crawler_status.html",
            {
                "result": response_data,
                "task_ids": response_data.get("task_ids"),
                "error": (
                    response_data.get("detail") if response.status_code >= 400 else None
                ),
            },
        )

    except Exception as e:
        logger.exception("crawler_submit.failed")
        return render(request, "theme/partials/_crawler_status.html", {"error": str(e)})


@require_POST
def ingestion_submit(request):
    """
    Handle HTMX submission for ingestion (file upload + start run).
    Returns a partial HTML response with the ingestion status.
    """
    try:
        from ai_core.services import handle_document_upload, start_ingestion_run

        # 1. Handle File Upload
        if "file" not in request.FILES:
            return render(
                request,
                "theme/partials/_ingestion_status.html",
                {"error": "No file provided."},
            )

        uploaded_file = request.FILES["file"]
        tenant_id, tenant_schema = _tenant_context_from_request(request)
        case_id = request.POST.get("case_id") or request.headers.get("X-Case-ID")

        # Prepare metadata for upload
        meta = {
            "tenant_id": tenant_id,
            "tenant_schema": tenant_schema,
            "case_id": case_id,
            "trace_id": uuid4().hex,
        }

        # Upload document
        upload_response = handle_document_upload(
            upload=uploaded_file, metadata_raw=None, meta=meta, idempotency_key=None
        )

        if upload_response.status_code >= 400:
            return render(
                request,
                "theme/partials/_ingestion_status.html",
                {
                    "error": f"Upload failed: {upload_response.data.get('detail', 'Unknown error')}"
                },
            )

        document_id = upload_response.data.get("document_id")

        # 2. Start Ingestion Run
        run_payload = {
            "document_ids": [document_id],
            "case_id": case_id,
        }

        run_response = start_ingestion_run(
            request_data=run_payload, meta=meta, idempotency_key=None
        )

        if run_response.status_code >= 400:
            return render(
                request,
                "theme/partials/_ingestion_status.html",
                {
                    "error": f"Ingestion start failed: {run_response.data.get('detail', 'Unknown error')}"
                },
            )

        # 3. Return Status Partial
        return render(
            request,
            "theme/partials/_ingestion_status.html",
            {
                "status": "queued",
                "task_ids": [run_response.data.get("ingestion_run_id")],
                "url_count": 1,  # Represents the single file
                "result": True,
                "now": timezone.now(),
            },
        )

    except Exception as e:
        logger.exception("ingestion_submit.failed")
        return render(
            request,
            "theme/partials/_ingestion_status.html",
            {"error": str(e)},
        )


# RAG Workbench Redesign Views


def workbench_index(request):
    """Main container for the RAG Command Center."""
    try:
        tenant_id, tenant_schema = _tenant_context_from_request(request)
    except TenantRequiredError:
        # Fallback for dev/testing if no tenant context
        tenant_id, tenant_schema = "dev", "public"

    case_id = request.GET.get("case_id") or request.headers.get("X-Case-ID")

    context = {
        "tenant_id": tenant_id,
        "tenant_schema": tenant_schema,
        "case_id": case_id,
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


def framework_analysis_tool(request):
    """Render the framework analysis developer tool."""
    try:
        tenant_id, tenant_schema = _tenant_context_from_request(request)
    except TenantRequiredError as exc:
        return _tenant_required_response(exc)

    # Resolve default collection (manual collection for this tenant)
    manual_collection_id, _ = _resolve_manual_collection(tenant_id, None)

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
def chat_submit(request):
    """
    Handle HTMX submission for RAG Chat.
    Invokes the production RAG graph and returns the assistant's reply.
    """
    message = request.POST.get("message")
    case_id = request.POST.get("case_id") or request.headers.get("X-Case-ID")

    if not message:
        return HttpResponse('<div class="text-red-500 p-4">Message is required.</div>')

    try:
        from ai_core.graphs.retrieval_augmented_generation import run as run_rag_graph

        tenant_id, tenant_schema = _tenant_context_from_request(request)

        # 1. Prepare State & Meta
        state = {
            "question": message,
            "query": message,  # Required for retrieval
            "hybrid": {
                "alpha": 0.5,
                "top_k": 5,
                "min_sim": 0.0,
            },
        }

        meta = {
            "tenant_id": tenant_id,
            "tenant_schema": tenant_schema,
            "case_id": case_id,
            "trace_id": uuid4().hex,
            # Ensure we have a valid tool context
            "tool_context": {
                "tenant_id": tenant_id,
                "tenant_schema": tenant_schema,
                "case_id": case_id,
            },
        }

        # 2. Run Graph
        final_state, result_payload = run_rag_graph(state, meta)

        # 3. Extract Answer
        answer = result_payload.get("answer", "No answer generated.")
        snippets = result_payload.get("snippets", [])

        # 4. Render Response Partial
        # We'll inline the HTML for now, or we could create a partial template
        response_html = f"""
        <div class="flex items-start gap-4 justify-end">
            <div class="bg-indigo-600 p-4 rounded-2xl rounded-tr-none shadow-sm text-sm text-white max-w-[80%]">
                <p>{message}</p>
            </div>
            <div class="flex-shrink-0 h-8 w-8 rounded-full bg-slate-200 flex items-center justify-center text-slate-600 font-bold text-xs">You</div>
        </div>
        <div class="flex items-start gap-4">
            <div class="flex-shrink-0 h-8 w-8 rounded-full bg-indigo-100 flex items-center justify-center text-indigo-700 font-bold text-xs">AI</div>
            <div class="bg-white p-4 rounded-2xl rounded-tl-none shadow-sm border border-slate-100 text-sm text-slate-700 max-w-[80%] space-y-2">
                <div class="prose prose-sm max-w-none text-slate-700">
                    {answer}
                </div>
                {'<div class="mt-2 pt-2 border-t border-slate-100"><p class="text-xs font-semibold text-slate-500 mb-1">Sources:</p><ul class="space-y-1">' + ''.join([f'<li class="text-xs text-slate-400 truncate" title="{s.get("text", "")[:200]}">• {s.get("source", "Unknown")} ({int(s.get("score", 0)*100)}%)</li>' for s in snippets[:3]]) + '</ul></div>' if snippets else ''}
            </div>
        </div>
        """
        return HttpResponse(response_html)

    except Exception as e:
        logger.exception("chat_submit.failed")
        return HttpResponse(
            f"""
        <div class="flex items-start gap-4">
            <div class="flex-shrink-0 h-8 w-8 rounded-full bg-red-100 flex items-center justify-center text-red-700 font-bold text-xs">ERR</div>
            <div class="bg-white p-4 rounded-2xl rounded-tl-none shadow-sm border border-red-100 text-sm text-red-600 max-w-[80%]">
                <p>Error processing request: {str(e)}</p>
            </div>
        </div>
        """
        )


@require_POST
def framework_analysis_submit(request):
    """
    Handle HTMX submission for framework analysis.
    Returns a partial HTML response with the analysis result.
    """
    tenant_id = request.headers.get("X-Tenant-ID")
    if not tenant_id:
        # Try to resolve from request context if header missing
        try:
            tenant_id, _ = _tenant_context_from_request(request)
        except TenantRequiredError:
            return JsonResponse({"error": "Tenant ID missing"}, status=400)

    tenant_schema = request.headers.get("X-Tenant-Schema") or "public"
    trace_id = request.headers.get("X-Trace-ID") or uuid4().hex

    try:
        # Parse form data
        collection_id = request.POST.get("collection_id")
        document_id = request.POST.get("document_id") or None
        force_reanalysis = request.POST.get("force_reanalysis") == "on"
        confidence_threshold = float(request.POST.get("confidence_threshold", 0.7))

        input_params = FrameworkAnalysisInput(
            document_collection_id=collection_id,
            document_id=document_id,
            force_reanalysis=force_reanalysis,
            confidence_threshold=confidence_threshold,
        )

        graph = build_framework_graph()
        output = graph.run(
            input_params=input_params,
            tenant_id=tenant_id,
            tenant_schema=tenant_schema,
            trace_id=trace_id,
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


# Force reload
