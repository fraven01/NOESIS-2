import json
from collections.abc import Mapping
from hashlib import sha256
from typing import Any, Sequence
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
    CrawlerIngestionOutcome,
    ExternalKnowledgeGraphConfig,
    GraphContextPayload,
    build_graph,
)
from ai_core.llm import routing as llm_routing
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
from llm_worker.schemas import ScoreResultsTask


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


_RERANK_CRITERIA = [
    "Relevanz",
    "Deckung der Rechtsfrage",
    "Aktualität",
    "Autorität",
]
_RERANK_MAX_RESULTS = 15
_RERANK_SNIPPET_LIMIT = 400
_RERANK_TITLE_LIMIT = 200
_RERANK_CACHE_PREFIX = "ragtools:rerank"
_DEFAULT_RERANK_MODEL_LABEL = "fast"


def _truncate_text(value: object, limit: int) -> str:
    if value in (None, ""):
        return ""
    text = str(value)
    stripped = text.strip()
    if limit <= 0 or len(stripped) <= limit:
        return stripped
    return f"{stripped[:limit].rstrip()}…"


def _result_identifier(result: Mapping[str, Any], fallback_index: int) -> str:
    for key in ("document_id", "doc_id", "id"):
        candidate = result.get(key)
        if candidate not in (None, ""):
            return str(candidate).strip()
    url = result.get("url")
    if url:
        return str(url).strip()
    return f"result-{fallback_index}"


def _annotate_results_for_rerank(
    results: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    for index, entry in enumerate(results):
        candidate = dict(entry)
        candidate["rerank_id"] = candidate.get("rerank_id") or _result_identifier(
            candidate, index
        )
        candidate["rerank_index"] = index
        if "rerank" not in candidate:
            candidate["rerank"] = {"score": None, "reasons": []}
        annotated.append(candidate)
    return annotated


def _build_rerank_inputs(
    results: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    limited = []
    identifiers: list[str] = []
    for index, entry in enumerate(results[:_RERANK_MAX_RESULTS]):
        rid = entry.get("rerank_id") or _result_identifier(entry, index)
        identifiers.append(rid)
        limited.append(
            {
                "id": rid,
                "title": _truncate_text(entry.get("title"), _RERANK_TITLE_LIMIT),
                "snippet": _truncate_text(entry.get("snippet"), _RERANK_SNIPPET_LIMIT),
                "url": (str(entry.get("url")).strip() if entry.get("url") else None),
            }
        )
    return limited, identifiers


def _rerank_cache_key(
    tenant_id: str | None, query: str, identifiers: Sequence[str]
) -> str:
    payload = json.dumps(
        {
            "tenant": tenant_id or "",
            "query": (query or "").strip(),
            "ids": list(identifiers),
        },
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    digest = sha256(payload).hexdigest()
    return f"{_RERANK_CACHE_PREFIX}:{digest}"


def _apply_rerank_scores(
    results: Sequence[Mapping[str, Any]], ranking: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], int]:
    if not ranking:
        baseline: list[dict[str, Any]] = []
        for entry in results:
            candidate = dict(entry)
            candidate["rerank"] = candidate.get("rerank") or {
                "score": None,
                "reasons": [],
            }
            baseline.append(candidate)
        return baseline, 0

    ranking_map: dict[str, dict[str, Any]] = {}
    for entry in ranking:
        rid = entry.get("id")
        if not isinstance(rid, str):
            continue
        rid = rid.strip()
        if not rid or rid in ranking_map:
            continue
        ranking_map[rid] = {
            "score": entry.get("score"),
            "reasons": entry.get("reasons") or [],
        }

    scored: list[tuple[float, int, dict[str, Any]]] = []
    remainder: list[tuple[int, dict[str, Any]]] = []
    for index, entry in enumerate(results):
        candidate = dict(entry)
        rid = candidate.get("rerank_id") or _result_identifier(candidate, index)
        candidate["rerank_id"] = rid
        candidate["rerank_index"] = index
        info = ranking_map.get(rid)
        if info is not None and info.get("score") is not None:
            score_value = float(info["score"])
            candidate["rerank"] = {
                "score": score_value,
                "reasons": list(info.get("reasons") or []),
            }
            scored.append((score_value, index, candidate))
        else:
            candidate["rerank"] = {
                "score": None,
                "reasons": [],
            }
            remainder.append((index, candidate))

    scored.sort(key=lambda item: (-item[0], item[1]))
    ordered = [entry[2] for entry in scored] + [
        entry[1] for entry in sorted(remainder, key=lambda pair: pair[0])
    ]
    return ordered, len(scored)


def _average_score(ranking: Sequence[Mapping[str, Any]]) -> float | None:
    scores = [
        float(entry.get("score"))
        for entry in ranking
        if isinstance(entry.get("score"), (int, float))
    ]
    if not scores:
        return None
    return sum(scores) / len(scores)


def _determine_rerank_model_label() -> str | None:
    """
    Return a routing label that resolves to a configured model.

    Prefers ``settings.RERANK_MODEL_PRESET`` but falls back to the default
    label when the configured value is missing or unknown.
    """

    configured = getattr(settings, "RERANK_MODEL_PRESET", None)
    candidates: list[str] = []
    if configured:
        text = str(configured).strip()
        if text:
            candidates.append(text)
    fallback_labels = [_DEFAULT_RERANK_MODEL_LABEL, "default"]
    for fallback_label in fallback_labels:
        if fallback_label not in candidates:
            candidates.append(fallback_label)

    for label in candidates:
        try:
            llm_routing.resolve(label)
        except ValueError:
            logger.warning(
                "ragtools.rerank_model.invalid",
                extra={"model_label": label},
            )
            continue
        return label
    return None


def _execute_rerank(
    request,
    *,
    query: str,
    results: Sequence[Mapping[str, Any]],
    context: GraphContextPayload,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    annotated = _annotate_results_for_rerank(results)
    trimmed_inputs, identifiers = _build_rerank_inputs(annotated)
    if not trimmed_inputs:
        return annotated, {
            "status": "skipped",
            "applied": False,
            "reason": "insufficient_results",
        }

    tenant_id = context.tenant_id
    cache_key = _rerank_cache_key(tenant_id, query, identifiers)
    cache_ttl = getattr(settings, "RERANK_CACHE_TTL_SECONDS", 900)

    logger.info(
        "ragtools.rerank.requested",
        extra={
            "tenant_id": tenant_id,
            "case_id": context.case_id,
            "trace_id": context.trace_id,
            "query": query,
            "result_count": len(trimmed_inputs),
        },
    )

    cached_result = cache.get(cache_key)
    if isinstance(cached_result, Mapping):
        ranked = cached_result.get("ranked") or []
        ordered, applied = _apply_rerank_scores(annotated, ranked)
        avg_score = _average_score(ranked)
        meta = {
            "status": "succeeded",
            "applied": bool(applied),
            "source": "cache",
            "latency_s": cached_result.get("latency_s"),
            "model": cached_result.get("model"),
            "avg_score": avg_score,
            "reranked_count": applied,
            "criteria": _RERANK_CRITERIA,
        }
        logger.info(
            "ragtools.rerank.applied",
            extra={
                "tenant_id": tenant_id,
                "source": "cache",
                "reranked_count": applied,
            },
        )
        return ordered, meta

    model_label = _determine_rerank_model_label()
    if not model_label:
        logger.error(
            "ragtools.rerank.timeout_or_fail",
            extra={"tenant_id": tenant_id, "reason": "no_model"},
        )
        return annotated, {
            "status": "error",
            "applied": False,
            "message": "LLM-Rerank nicht konfigurierbar",
            "criteria": _RERANK_CRITERIA,
        }

    control_payload = {
        "tenant_id": tenant_id,
        "case_id": context.case_id,
        "trace_id": context.trace_id,
        "model_preset": model_label,
        "temperature": 0.1,
    }
    task = ScoreResultsTask(
        control=control_payload,
        data={
            "query": query,
            "results": trimmed_inputs,
            "criteria": _RERANK_CRITERIA,
            "k": min(10, len(trimmed_inputs)),
        },
    )
    task_payload = task.model_dump(mode="python")

    try:
        worker_payload, completed = submit_worker_task(
            task_payload=task_payload,
            scope={
                "tenant_id": tenant_id,
                "case_id": context.case_id,
                "trace_id": context.trace_id,
            },
            graph_name="score_results",
        )
    except Exception:
        logger.exception(
            "ragtools.rerank.timeout_or_fail",
            extra={"tenant_id": tenant_id, "reason": "error"},
        )
        return annotated, {
            "status": "error",
            "applied": False,
            "message": "LLM-Rerank nicht verfügbar",
            "criteria": _RERANK_CRITERIA,
        }

    if completed:
        llm_result = worker_payload.get("result") or {}
        ranked = llm_result.get("ranked") or []
        cache.set(cache_key, llm_result, cache_ttl)
        ordered, applied = _apply_rerank_scores(annotated, ranked)
        avg_score = _average_score(ranked)
        latency_s = llm_result.get("latency_s")
        meta = {
            "status": "succeeded",
            "applied": bool(applied),
            "task_id": worker_payload.get("task_id"),
            "latency_s": latency_s,
            "model": llm_result.get("model"),
            "avg_score": avg_score,
            "usage": llm_result.get("usage"),
            "reranked_count": applied,
            "criteria": _RERANK_CRITERIA,
        }
        logger.info(
            "ragtools.rerank.applied",
            extra={
                "tenant_id": tenant_id,
                "source": "worker",
                "reranked_count": applied,
                "latency_s": latency_s,
            },
        )
        return ordered, meta

    status_url = request.build_absolute_uri(
        reverse("llm_worker:task_status", args=[worker_payload["task_id"]])
    )
    logger.info(
        "ragtools.rerank.timeout_or_fail",
        extra={"tenant_id": tenant_id, "reason": "timeout"},
    )
    return annotated, {
        "status": "queued",
        "applied": False,
        "task_id": worker_payload["task_id"],
        "status_url": status_url,
        "criteria": _RERANK_CRITERIA,
        "result_ids": identifiers,
    }


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

        rerank_requested = bool(data.get("rerank"))
        search_results = _annotate_results_for_rerank(search_results)
        rerank_summary: dict[str, Any] | None = None

        if rerank_requested and search_results:
            search_results, rerank_summary = _execute_rerank(
                request,
                query=query,
                results=search_results,
                context=context,
            )
        elif rerank_requested:
            rerank_summary = {
                "status": "skipped",
                "applied": False,
                "reason": "no_results",
            }

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
            "rerank": rerank_summary,
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
