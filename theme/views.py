from typing import Any, Mapping
from uuid import UUID, uuid4

from opentelemetry import trace
from opentelemetry.trace import format_trace_id

from django.conf import settings
from django.core.cache import cache
from django.http import JsonResponse
from django.shortcuts import render
from django.urls import reverse
from structlog.stdlib import get_logger

from ai_core.contracts import ScopeContext, BusinessContext
from ai_core.infra.resp import build_tool_error_payload
from ai_core.services.crawler_runner import run_crawler_runner
from ai_core.rag.collections import (
    MANUAL_COLLECTION_SLUG,
    manual_collection_uuid,
)
from documents.collection_service import CollectionService
from documents.models import DocumentCollection
from ai_core.llm import routing as llm_routing
from llm_worker.runner import submit_worker_task
from ai_core.schemas import CrawlerRunRequest

from customers.tenant_context import TenantContext, TenantRequiredError
from documents.services.document_space_service import DocumentSpaceService
from theme.validators import SearchQualityParams

from ai_core.views import crawl_selected as _core_crawl_selected


from django.contrib.auth import get_user_model

logger = get_logger(__name__)
DOCUMENT_SPACE_SERVICE = DocumentSpaceService()
# build_graph aliasing removed as build_external_knowledge_graph is gone
crawl_selected = _core_crawl_selected  # Re-export for tests
DEV_DEFAULT_CASE_ID = "dev-case-local"


def _get_dev_simulated_users():
    User = get_user_model()
    usernames = ["admin", "legal_bob", "alice_stakeholder", "charles_external"]
    users = list(User.objects.filter(username__in=usernames).order_by("username"))
    return users


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
    return _json_error_response(str(exc), status_code=403, code="tenant_not_found")


def _default_error_code(status_code: int) -> str:
    if status_code == 400:
        return "invalid_request"
    if status_code == 403:
        return "forbidden"
    if status_code == 404:
        return "not_found"
    if status_code == 409:
        return "conflict"
    if status_code == 413:
        return "payload_too_large"
    if status_code == 415:
        return "unsupported_media_type"
    if status_code == 429:
        return "rate_limited"
    if status_code == 502:
        return "upstream_error"
    if status_code == 504:
        return "timeout"
    if status_code >= 500:
        return "internal_error"
    return "error"


def _json_error_response(
    message: str,
    *,
    status_code: int,
    code: str | None = None,
    details: dict[str, Any] | None = None,
) -> JsonResponse:
    payload = build_tool_error_payload(
        message=message,
        status_code=status_code,
        code=code or _default_error_code(status_code),
        details=details,
    )
    return JsonResponse(payload, status=status_code)


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
    params = SearchQualityParams.model_validate(request_data)
    purpose = params.purpose or "web_search_rerank"
    return {
        "question": query,
        "collection_scope": collection_id,
        "quality_mode": params.quality_mode,
        "max_candidates": params.max_candidates,
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
            scope = ScopeContext(
                tenant_id=str(tenant_id),
                tenant_schema=str(tenant_schema),
                trace_id=str(trace_id) if trace_id else str(uuid4()),
                invocation_id=str(uuid4()),
                run_id=str(uuid4()),
                user_id=context.get("user_id"),
            )
            business = BusinessContext(case_id=str(case_id) if case_id else None)
            tool_context = scope.to_tool_context(business=business)
            meta = {
                "scope_context": scope.model_dump(mode="json", exclude_none=True),
                "business_context": business.model_dump(mode="json", exclude_none=True),
                "tool_context": tool_context.model_dump(mode="json", exclude_none=True),
            }
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


def _resolve_lifecycle_store() -> object | None:
    """Return the document lifecycle store used for crawler baseline lookups."""
    try:
        from documents import api as documents_api  # local import to avoid cycles
    except Exception:  # pragma: no cover - defensive import guard
        return None
    return getattr(documents_api, "DEFAULT_LIFECYCLE_STORE", None)


from theme.views_rag_tools import (  # noqa: E402,F401
    rag_tools,
    rag_tools_identity_switch,
    tool_collaboration,
    start_rerank_workflow,
    workbench_index,
    tool_search,
    tool_ingestion,
    tool_crawler,
    tool_framework,
    tool_chat,
)
from theme.views_documents import (  # noqa: E402,F401
    document_space,
    document_explorer,
    document_delete,
    document_restore,
)
from theme.views_web_search import (  # noqa: E402,F401
    web_search,
    web_search_ingest_selected,
)
from theme.views_ingestion import (  # noqa: E402,F401
    crawler_submit,
    ingestion_submit,
)
from theme.views_framework import (  # noqa: E402,F401
    framework_analysis_tool,
    framework_analysis_submit,
)
from theme.views_chat import chat_submit  # noqa: E402,F401
