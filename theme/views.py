import json
from typing import Any, Mapping
from uuid import UUID, uuid4

from opentelemetry import trace
from opentelemetry.trace import format_trace_id

from django.conf import settings
from django.core.cache import cache
from django.http import JsonResponse, HttpResponse
from django.shortcuts import render
from django.urls import reverse
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from structlog.stdlib import get_logger

from ai_core.contracts import ScopeContext, BusinessContext
from common.logging import bind_log_context
from ai_core.services import _get_documents_repository
from ai_core.services.crawler_runner import run_crawler_runner
from ai_core.graphs.technical.universal_ingestion_graph import (
    build_universal_ingestion_graph,
    UniversalIngestionState,
)
from ai_core.graphs.technical.collection_search import (
    GraphInput as CollectionSearchGraphInput,
    build_graph as build_collection_search_graph,
)
from ai_core.rag.collections import (
    MANUAL_COLLECTION_SLUG,
    manual_collection_uuid,
)
from documents.collection_service import CollectionService
from documents.models import DocumentCollection
from ai_core.rag.routing_rules import (
    get_routing_table,
    is_collection_routing_enabled,
)
from ai_core.llm import routing as llm_routing
from llm_worker.runner import submit_worker_task
from crawler.manager import CrawlerManager
from ai_core.schemas import CrawlerRunRequest

from customers.tenant_context import TenantContext, TenantRequiredError
from documents.services.document_space_service import (
    DocumentSpaceRequest,
    DocumentSpaceService,
)

from ai_core.views import crawl_selected as _core_crawl_selected
from ai_core.graphs.business.framework_analysis_graph import (
    build_graph as build_framework_graph,
)
from ai_core.tools.framework_contracts import FrameworkAnalysisInput
from cases.services import ensure_case
from pydantic import ValidationError


logger = get_logger(__name__)
DOCUMENT_SPACE_SERVICE = DocumentSpaceService()
# build_graph aliasing removed as build_external_knowledge_graph is gone
crawl_selected = _core_crawl_selected  # Re-export for tests
DEV_DEFAULT_CASE_ID = "dev-case-local"


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


def _extract_user_id(request) -> str | None:
    """Extract user_id from authenticated request (Pre-MVP ID Contract).

    User Request Hop: user_id is required when auth is present.
    Returns None for unauthenticated requests.
    """
    user = getattr(request, "user", None)
    if user is None:
        return None
    if not getattr(user, "is_authenticated", False):
        return None
    user_pk = getattr(user, "pk", None)
    if user_pk is None:
        return None
    return str(user_pk)


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
        if requested_text.lower() == "dev-workbench":
            return manual_id, manual_id

    if requested_text.lower() == MANUAL_COLLECTION_SLUG:
        return manual_id, manual_id

    return manual_id, requested_text


def _normalize_collection_id(
    collection_identifier: str | None, tenant_schema: str
) -> str | None:
    """Return a canonical UUID for collection identifiers provided as keys/aliases."""

    value = (collection_identifier or "").strip()
    if not value:
        return None

    try:
        return str(UUID(value))
    except (TypeError, ValueError):
        pass

    tenant = TenantContext.resolve_identifier(tenant_schema, allow_pk=True)
    if tenant is None:
        return None

    try:
        collection = DocumentCollection.objects.get(tenant=tenant, key=value)
        return str(collection.collection_id)
    except DocumentCollection.DoesNotExist:
        logger.info(
            "collection.lookup.missing",
            extra={
                "tenant_schema": tenant_schema,
                "collection_key": value,
            },
        )
        return None


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
    user_id: str | None = None,
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
        "control": {
            "model_preset": model_preset,
        },
    }
    scope = {
        "tenant_id": tenant_id,
        "case_id": case_id,
        "trace_id": trace_id,
        "workflow_id": "web-search-rerank",
        # Identity ID (Pre-MVP ID Contract)
        "user_id": user_id,
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


class _ViewCrawlerIngestionAdapter:
    """Adapter that triggers crawler ingestion by calling run_crawler_runner."""

    def trigger(
        self,
        *,
        url: str,
        collection_id: str,
        context: Mapping[str, str],
    ) -> Mapping[str, Any]:
        """Trigger ingestion for the given URL."""
        tenant_id = context.get("tenant_id", "dev")
        trace_id = context.get("trace_id", "")

        if not trace_id:
            span = trace.get_current_span()
            ctx = span.get_span_context()
            if ctx.is_valid:
                trace_id = format_trace_id(ctx.trace_id)

        case_id = context.get("case_id")
        mode = context.get("mode", "live")
        tenant_schema = context.get("tenant_schema") or tenant_id

        payload = {
            "workflow_id": "external-knowledge-ingestion",
            "mode": mode,
            "origins": [{"url": url}],
            "collection_id": collection_id,
        }
        try:
            request_model = CrawlerRunRequest.model_validate(payload)
            scope_context = {
                "tenant_id": str(tenant_id),
                "tenant_schema": str(tenant_schema),
                "case_id": str(case_id) if case_id else None,
                "trace_id": str(trace_id) if trace_id else str(uuid4()),
                "invocation_id": str(uuid4()),
                "run_id": str(uuid4()),
                "user_id": context.get("user_id"),
            }
            meta = {"scope_context": scope_context}
            result = run_crawler_runner(
                meta=meta,
                request_model=request_model,
                lifecycle_store=_resolve_lifecycle_store(),
                graph_factory=None,
            )
            response_data = result.payload

            if result.status_code in (200, 202):
                return {
                    "decision": "ingested",
                    "crawler_decision": response_data.get("decision", "unknown"),
                    "document_id": response_data.get("document_id"),
                }
            return {
                "decision": "ingestion_error",
                "crawler_decision": response_data.get("code", "http_error"),
            }
        except Exception:
            logger.exception("crawler.trigger.failed")
            return {
                "decision": "ingestion_error",
                "crawler_decision": "trigger_exception",
            }


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
    case_id = DEV_DEFAULT_CASE_ID
    try:
        request.session["dev_case_id"] = case_id
    except Exception:
        pass
    try:
        tenant_obj = TenantContext.resolve_identifier(tenant_id, allow_pk=True)
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
        show_retired = _parse_bool(request.GET.get("show_retired"), default=False)
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
            show_retired=show_retired,
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
            "show_retired": "true" if show_retired else "false",
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


def _resolve_lifecycle_store() -> object | None:
    """Return the document lifecycle store used for crawler baseline lookups."""
    try:
        from documents import api as documents_api  # local import to avoid cycles
    except Exception:  # pragma: no cover - defensive import guard
        return None
    return getattr(documents_api, "DEFAULT_LIFECYCLE_STORE", None)


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
    # Identity ID (Pre-MVP ID Contract)
    user_id = _extract_user_id(request)

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

    manual_collection_id, resolved_collection_id = _resolve_manual_collection(
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

    # Execution logic consolidated below to handle both search types cleanly

    response_data = {}

    if search_type == "collection_search":
        # ... Collection Search Logic (Existing) ...
        # Copied from original file context
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
            CollectionSearchGraphInput.model_validate(graph_input)
        except Exception as exc:
            return JsonResponse({"error": f"Invalid input: {str(exc)}"}, status=400)

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
        # External Knowledge Graph (LangGraph)
        from ai_core.tools.shared_workers import get_web_search_worker

        search_worker = get_web_search_worker()
        ingestion_adapter = _ViewCrawlerIngestionAdapter()

        # Parse configurable parameters from request
        try:
            top_n = int(data.get("top_n", 5))
            if top_n < 1 or top_n > 20:
                top_n = 5
        except (ValueError, TypeError):
            top_n = 5

        try:
            min_snippet_length = int(data.get("min_snippet_length", 40))
            if min_snippet_length < 10 or min_snippet_length > 500:
                min_snippet_length = 40
        except (ValueError, TypeError):
            min_snippet_length = 40

        # BREAKING CHANGE (Option A - Strict Separation):
        # Build ScopeContext (infrastructure IDs) and BusinessContext (domain IDs)
        # instead of flat dict to match ToolContext structure

        scope = ScopeContext(
            tenant_id=tenant_id,
            tenant_schema=tenant_id,
            trace_id=trace_id,
            invocation_id=str(uuid4()),
            run_id=run_id,
            user_id=user_id,  # Pre-MVP ID Contract
        )

        business = BusinessContext(
            workflow_id="external-knowledge-manual",
            case_id=case_id,
            collection_id=collection_id,
        )

        # Build context_payload with pure ToolContext structure
        # BREAKING CHANGE (Option A - Full Migration):
        # All nodes now use ToolContext, no flattened fields needed anymore!
        context_payload = {
            "scope": scope.model_dump(mode="json"),
            "business": business.model_dump(mode="json"),
            "metadata": {
                # Non-serializable objects - graph nodes access via ToolContext.metadata
                "runtime_worker": search_worker,
                "runtime_trigger": ingestion_adapter,
                # Config
                "top_n": top_n,
                "min_snippet_length": min_snippet_length,
                "prefer_pdf": True,
            },
        }

        # Prepare Search Config
        search_config = {
            "top_n": top_n,
            "min_snippet_length": min_snippet_length,
            "prefer_pdf": True,
        }

        input_payload = {
            "source": "search",
            "mode": "acquire_only",  # Manual search implies viewing results first
            "collection_id": collection_id,
            "search_query": query,
            "search_config": search_config,
            # Required fields for TypedDict but None for search
            "upload_blob": None,
            "metadata_obj": None,
            "normalized_document": None,
        }

        # Universal Ingestion State
        input_state: UniversalIngestionState = {
            "input": input_payload,
            "context": context_payload,
            "normalized_document": None,
            "search_results": [],
            "selected_result": None,
            "processing_result": None,
            "ingestion_result": None,
            "output": None,
            "error": None,
        }

        try:

            # Phase 4 Migration: Use Universal Ingestion Graph
            universal_graph = build_universal_ingestion_graph()
            final_state = universal_graph.invoke(input_state)
        except Exception:
            logger.exception("web_search.failed")
            return JsonResponse({"error": "Graph execution failed."}, status=500)

        # Use filtered_results (after top_n and min_snippet_length filtering)
        # Fallback to search_results if filtering hasn't run
        # Use search_results from Universal Graph
        results = final_state.get("search_results", [])
        # Construct response similar to old format for UI compatibility
        response_data = {
            "outcome": "completed",  # Simple outcome
            "results": results,
            "search": {"results": results},  # UI expects search.results
            "telemetry": {},  # Simplified telemetry for now
            "trace_id": trace_id,
        }

        if final_state.get("error"):
            # If error field is set
            response_data["outcome"] = "error"
            response_data["error"] = final_state["error"]

    # Common Logic
    results = response_data.get("results", [])
    search_payload = response_data.get("search", {})
    trace_id = response_data.get("trace_id")

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
        collection_id = _normalize_collection_id(collection_id, tenant_schema)
        if not collection_id:
            return JsonResponse(
                {"error": "Collection ID could not be resolved"}, status=400
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
        scope_context = {
            "tenant_id": tenant_id,
            "tenant_schema": tenant_schema,
            "case_id": str(data.get("case_id") or "").strip() or None,
            "trace_id": str(data.get("trace_id") or "").strip() or str(uuid4()),
            "ingestion_run_id": str(uuid4()),
        }
        meta = {"scope_context": scope_context}

        manager = CrawlerManager()
        try:
            result = manager.dispatch_crawl_request(request_model, meta)
        except Exception as exc:
            logger.exception("web_search.crawler_dispatch_failed")
            return JsonResponse({"error": str(exc)}, status=500)

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
        # Identity ID (Pre-MVP ID Contract)
        user_id = _extract_user_id(request)
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
            "ingestion_run_id": str(uuid4()),  # Required for crawler ingestion graph
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

        tenant_id, tenant_schema = _tenant_context_from_request(request)
        case_id = str(data.get("case_id") or "").strip() or None
        trace_id = str(uuid4())
        scope_context = {
            "tenant_id": tenant_id,
            "tenant_schema": tenant_schema,
            "case_id": case_id,
            "trace_id": trace_id,
            "invocation_id": str(uuid4()),
            "run_id": str(uuid4()),
            "ingestion_run_id": payload.get("ingestion_run_id"),
            "user_id": _extract_user_id(request),
        }
        meta = {"scope_context": scope_context}

        try:
            request_model = CrawlerRunRequest.model_validate(payload)
        except ValidationError as exc:
            response_data = {"detail": str(exc), "code": "invalid_request"}
            return render(
                request,
                "theme/partials/_crawler_status.html",
                {"result": response_data, "error": response_data["detail"]},
            )

        try:
            result = run_crawler_runner(
                meta=meta,
                request_model=request_model,
                lifecycle_store=_resolve_lifecycle_store(),
                graph_factory=None,
            )
            response_data = result.payload
            status_code = result.status_code
        except Exception as exc:
            response_data = {"detail": str(exc), "code": "crawler_error"}
            status_code = 500

        return render(
            request,
            "theme/partials/_crawler_status.html",
            {
                "result": response_data,
                "task_ids": response_data.get("task_ids"),
                "error": (response_data.get("detail") if status_code >= 400 else None),
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
        from ai_core.services import handle_document_upload

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

        # Resolve manual collection for tenant
        manual_collection_id, _ = _resolve_manual_collection(
            tenant_id, None, ensure=True
        )

        # Prepare metadata for upload
        scope_context = {
            "tenant_id": tenant_id,
            "tenant_schema": tenant_schema,
            "case_id": case_id,
            "trace_id": uuid4().hex,
            "collection_id": manual_collection_id,  # Explicitly set collection
            "workflow_id": "document-upload-manual",  # Workflow type for tracing
            "invocation_id": uuid4().hex,
        }
        meta = {"scope_context": scope_context}

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

        run_id = upload_response.data.get("ingestion_run_id")

        # Extract transition data from graph response for UI
        response_data = upload_response.data
        transition_info = {
            "decision": response_data.get("decision"),
            "reason": response_data.get("reason"),
            "severity": response_data.get("severity"),
            "document_id": response_data.get("document_id"),
        }

        # 3. Return Status Partial
        return render(
            request,
            "theme/partials/_ingestion_status.html",
            {
                "status": "queued",
                "task_ids": [run_id],
                "url_count": 1,  # Represents the single file
                "result": True,
                "now": timezone.now(),
                "transition": transition_info,
            },
        )

    except Exception as e:
        import traceback

        with open("debug_traceback_2.txt", "w") as f:
            f.write(traceback.format_exc())
        tenant_context = {"tenant_id": tenant_id}  # Define tenant_context for logger
        logger.error(
            "ingestion_submit.failed",
            extra={"tenant_id": tenant_context.get("tenant_id")},
            exc_info=True,
        )
        return render(
            request,
            "theme/partials/ingestion_submit_error.html",
            {"error_message": str(e)},
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
    case_id = (
        request.POST.get("case_id")
        or request.headers.get("X-Case-ID")
        or "dev-case-local"
    )

    # Feature: Global Search in RAG Chat (Dev Workbench)
    # If global_search is checked, ignore case_id to search entire tenant
    if request.POST.get("global_search") == "on":
        case_id = None

    if not message:
        return HttpResponse('<div class="text-red-500 p-4">Message is required.</div>')

    try:
        from ai_core.graphs.technical.retrieval_augmented_generation import (
            run as run_rag_graph,
        )

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

        trace_id = uuid4().hex
        run_id = str(uuid4())

        from ai_core.tool_contracts import ToolContext

        tool_context = ToolContext(
            tenant_id=tenant_id,
            tenant_schema=tenant_schema,
            case_id=case_id,
            trace_id=trace_id,
            run_id=run_id,
            workflow_id="rag-chat-manual",
        )

        meta = {
            "tenant_id": tenant_id,
            "tenant_schema": tenant_schema,
            "case_id": case_id,
            "trace_id": trace_id,
            "run_id": run_id,  # Required for ScopeContext validation
            "workflow_id": "rag-chat-manual",  # Workflow type for tracing
            # Ensure we have a valid tool context
            "tool_context": tool_context.model_dump(mode="json", exclude_none=True),
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


@csrf_exempt
def document_delete(request):
    """Handle document deletion via HTMX.

    Query params:
        document_id: UUID of the document to delete
        hard: If 'true', permanently delete. Otherwise soft delete (retire).
    """
    if request.method != "DELETE":
        return HttpResponse(status=405)

    document_id = request.GET.get("document_id")
    hard_delete = request.GET.get("hard", "").lower() == "true"

    if not document_id:
        return HttpResponse(
            '<div class="p-4 text-red-600 text-sm">Document ID required</div>',
            status=400,
        )

    try:
        tenant_id, tenant_schema = _tenant_context_from_request(request)
    except TenantRequiredError as exc:
        return HttpResponse(
            f'<div class="p-4 text-red-600 text-sm">{exc}</div>', status=400
        )

    try:
        from django_tenants.utils import schema_context
        from documents.models import Document

        doc_uuid = UUID(document_id)

        with schema_context(tenant_schema):
            try:
                document = Document.objects.get(pk=doc_uuid)
            except Document.DoesNotExist:
                return HttpResponse(
                    '<div class="p-4 text-amber-600 text-sm">Document not found</div>',
                    status=404,
                )

            doc_title = (
                document.metadata.get("title", str(doc_uuid)[:8])
                if document.metadata
                else str(doc_uuid)[:8]
            )

            if hard_delete:
                try:
                    from ai_core.rag.vector_client import get_default_client

                    vector_client = get_default_client()
                    vector_client.hard_delete_documents(
                        tenant_id=str(tenant_id),
                        document_ids=[doc_uuid],
                    )
                    logger.info(
                        "document_delete.vector_document_removed",
                        extra={"document_id": str(doc_uuid), "tenant": tenant_schema},
                    )
                except Exception as exc:
                    logger.exception(
                        "document_delete.vector_document_remove_failed",
                        extra={"document_id": str(doc_uuid), "tenant": tenant_schema},
                    )
                    return HttpResponse(
                        f'<div class="p-4 text-red-600 text-sm">Vector cleanup failed: {exc}</div>',
                        status=500,
                    )

                document.delete()
                return HttpResponse(
                    f"""<div class="rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700">
                        <strong>Deleted:</strong> {doc_title}<br>
                        <span class="text-xs">Permanently removed from database</span>
                    </div>"""
                )
            else:
                document.lifecycle_state = "retired"
                document.lifecycle_updated_at = timezone.now()

                try:
                    from ai_core.rag.vector_client import get_default_client

                    vector_client = get_default_client()
                    vector_client.update_lifecycle_state(
                        tenant_id=str(tenant_id),
                        document_ids=[doc_uuid],
                        state="retired",
                        reason="soft_delete_from_ui",
                    )
                    logger.info(
                        "document_delete.vector_lifecycle_updated",
                        extra={"document_id": str(doc_uuid), "tenant": tenant_schema},
                    )
                except Exception as exc:
                    logger.exception(
                        "document_delete.vector_lifecycle_update_failed",
                        extra={"document_id": str(doc_uuid), "tenant": tenant_schema},
                    )
                    return HttpResponse(
                        f'<div class="p-4 text-red-600 text-sm">Vector cleanup failed: {exc}</div>',
                        status=500,
                    )

                document.save(update_fields=["lifecycle_state", "lifecycle_updated_at"])

                return HttpResponse(
                    f"""<div class="rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-700">
                        <strong>Archived:</strong> {doc_title}<br>
                        <span class="text-xs">Lifecycle state changed to 'retired'</span>
                    </div>"""
                )
    except Exception as e:
        logger.exception("document_delete.failed")
        return HttpResponse(
            f'<div class="p-4 text-red-600 text-sm">Error: {str(e)}</div>', status=500
        )


@csrf_exempt
def document_restore(request):
    """Restore a retired document to active state via HTMX.

    Query params:
        document_id: UUID of the document to restore
    """
    if request.method != "POST":
        return HttpResponse(status=405)

    document_id = request.GET.get("document_id")

    if not document_id:
        return HttpResponse(
            '<div class="p-4 text-red-600 text-sm">Document ID required</div>',
            status=400,
        )

    try:
        tenant_id, tenant_schema = _tenant_context_from_request(request)
    except TenantRequiredError as exc:
        return HttpResponse(
            f'<div class="p-4 text-red-600 text-sm">{exc}</div>', status=400
        )

    try:
        from django_tenants.utils import schema_context
        from documents.models import Document

        doc_uuid = UUID(document_id)

        with schema_context(tenant_schema):
            try:
                document = Document.objects.get(pk=doc_uuid)
            except Document.DoesNotExist:
                return HttpResponse(
                    '<div class="p-4 text-amber-600 text-sm">Document not found</div>',
                    status=404,
                )

            doc_title = (
                document.metadata.get("title", str(doc_uuid)[:8])
                if document.metadata
                else str(doc_uuid)[:8]
            )
            previous_state = document.lifecycle_state
            previous_updated_at = document.lifecycle_updated_at

            document.lifecycle_state = "active"
            document.lifecycle_updated_at = timezone.now()

            # Also update lifecycle in vector store so RAG search includes restored docs
            try:
                from ai_core.rag.vector_client import get_default_client

                vector_client = get_default_client()
                vector_client.update_lifecycle_state(
                    tenant_id=str(tenant_id),
                    document_ids=[doc_uuid],
                    state="active",
                    reason="restore_from_ui",
                )
                logger.info(
                    "document_restore.vector_lifecycle_updated",
                    extra={"document_id": str(doc_uuid), "tenant": tenant_schema},
                )
            except Exception as exc:
                logger.exception(
                    "document_restore.vector_lifecycle_update_failed",
                    extra={"document_id": str(doc_uuid), "tenant": tenant_schema},
                )
                document.lifecycle_state = previous_state
                document.lifecycle_updated_at = previous_updated_at
                return HttpResponse(
                    f'<div class="p-4 text-red-600 text-sm">Vector lifecycle update failed: {exc}</div>',
                    status=500,
                )

            document.save(update_fields=["lifecycle_state", "lifecycle_updated_at"])

            return HttpResponse(
                f"""<div class="rounded-xl border border-green-200 bg-green-50 p-4 text-sm text-green-700">
                    <strong>Restored:</strong> {doc_title}<br>
                    <span class="text-xs">Lifecycle state changed from '{previous_state}' to 'active'</span>
                </div>"""
            )
    except Exception as e:
        logger.exception("document_restore.failed")
        return HttpResponse(
            f'<div class="p-4 text-red-600 text-sm">Error: {str(e)}</div>', status=500
        )
