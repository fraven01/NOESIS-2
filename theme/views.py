import json
from typing import Mapping
from uuid import uuid4

from django.conf import settings
from django.core.cache import cache
from django.http import JsonResponse
from django.shortcuts import render
from django.urls import reverse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from structlog.stdlib import get_logger

from ai_core.graphs.external_knowledge_graph import (
    CrawlerIngestionAdapter,
    CrawlerIngestionOutcome,
    GraphContextPayload,
    InvalidGraphInput,
    build_graph as build_external_knowledge_graph,
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
from ai_core.llm import routing as llm_routing
from llm_worker.runner import submit_worker_task

from customers.tenant_context import TenantContext, TenantRequiredError

from ai_core.views import crawl_selected as _core_crawl_selected


logger = get_logger(__name__)
build_graph = build_external_knowledge_graph  # Backwards compatibility for tests
crawl_selected = _core_crawl_selected  # Re-export for tests


def home(request):
    """Render the homepage."""

    return render(request, "theme/home.html")


def _tenant_context_from_request(request) -> tuple[str, str]:
    """Return the tenant identifier and schema for the current request."""

    tenant_obj = TenantContext.from_request(
        request, allow_headers=False, require=True
    )
    tenant_schema = getattr(tenant_obj, "schema_name", None)
    if tenant_schema is None:
        tenant_schema = getattr(tenant_obj, "tenant_id", None)
    tenant_id = getattr(tenant_obj, "tenant_id", None) or tenant_schema

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
    case_id: str,
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
        case_id = context.get("case_id", "local")
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
                "HTTP_X_CASE_ID": str(case_id),
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
    """Execute the external knowledge graph for manual RAG searches."""

    try:
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
    case_id = str(data.get("case_id") or "local").strip() or "local"
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

    return JsonResponse(response_data)


@require_POST
@csrf_exempt
def web_search_ingest_selected(request):
    """Ingest user-selected URLs from web search results via crawler_runner.

    Calls crawler_runner view directly to avoid HTTP overhead and tenant routing issues.
    Returns a summary of started ingestion tasks.
    """
    try:
        data = json.loads(request.body)
        urls = data.get("urls", [])
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
        case_id = str(data.get("case_id") or "local").strip() or "local"
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
            return JsonResponse(
                {
                    "error": "Crawler call failed",
                    "status_code": response.status_code,
                    "detail": response_data.get("error", str(response_data)),
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
            "case_id": data.get("case_id", "local"),
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
        return JsonResponse(response_payload)

    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    except Exception as e:
        logger.exception("start_rerank_workflow.failed")
        return JsonResponse({"error": str(e)}, status=500)
