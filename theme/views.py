import json
from collections import Counter
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

from ai_core.services import _get_documents_repository
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

from documents.models import DocumentCollection, DocumentLifecycleState


logger = get_logger(__name__)
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


def _stringify_metadata_value(value: object) -> str:
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        except TypeError:
            return str(value)
    return str(value)


def _serialize_collection(collection: DocumentCollection) -> dict[str, object]:
    """Flatten model attributes for template rendering."""

    case_obj = collection.case
    metadata = collection.metadata or {}
    metadata_items = [
        {"key": str(key), "value": _stringify_metadata_value(value)}
        for key, value in sorted(metadata.items(), key=lambda item: str(item[0]))
    ]

    case_info = None
    if case_obj is not None:
        case_info = {
            "id": str(case_obj.id),
            "external_id": getattr(case_obj, "external_id", ""),
            "title": getattr(case_obj, "title", ""),
            "status": getattr(case_obj, "status", ""),
        }

    return {
        "id": str(collection.id),
        "name": collection.name,
        "key": collection.key,
        "collection_id": str(collection.collection_id),
        "type": collection.type or "",
        "visibility": collection.visibility or "",
        "metadata": metadata_items,
        "case": case_info,
        "created_at": collection.created_at,
        "updated_at": collection.updated_at,
        "selector": str(collection.id),
    }


def _match_collection_identifier(
    collections: list[DocumentCollection],
    identifier: object,
) -> DocumentCollection | None:
    token = str(identifier or "").strip().lower()
    if not token:
        return None

    for collection in collections:
        if token == str(collection.id).lower():
            return collection
        if token == str(collection.collection_id).lower():
            return collection
        key_value = (collection.key or "").strip().lower()
        if key_value and token == key_value:
            return collection
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


def _describe_blob(blob) -> dict[str, object]:
    size = getattr(blob, "size", None)
    return {
        "type": getattr(blob, "type", None),
        "size": size,
        "size_display": _human_readable_bytes(size),
        "sha256": getattr(blob, "sha256", None),
        "media_type": getattr(blob, "media_type", None),
        "uri": getattr(blob, "uri", None),
    }


def _dict_items(mapping: Mapping[str, object] | None) -> list[dict[str, str]]:
    if not isinstance(mapping, Mapping):
        return []
    return [
        {"key": str(key), "value": _stringify_metadata_value(value)}
        for key, value in sorted(mapping.items(), key=lambda item: str(item[0]))
    ]


def _build_search_blob(payload: Mapping[str, object]) -> str:
    helpers = []
    ingestion = payload.get("ingestion", {}) if isinstance(payload, Mapping) else {}
    helpers.extend(
        [
            payload.get("document_id"),
            payload.get("title"),
            payload.get("workflow_id"),
            payload.get("version"),
            payload.get("collection_id"),
            payload.get("document_collection_id"),
            payload.get("origin_uri"),
            payload.get("language"),
            payload.get("source"),
            payload.get("external_provider"),
            payload.get("external_id"),
            ingestion.get("state") if isinstance(ingestion, Mapping) else None,
            ingestion.get("trace_id") if isinstance(ingestion, Mapping) else None,
            ingestion.get("run_id") if isinstance(ingestion, Mapping) else None,
            ingestion.get("ingestion_run_id") if isinstance(ingestion, Mapping) else None,
        ]
    )
    helpers.extend(payload.get("tags", []))
    for item in payload.get("external_ref_items", []):
        helpers.append(item.get("value"))
    normalized_tokens = [
        str(value).strip().lower()
        for value in helpers
        if isinstance(value, str) and value.strip()
    ]
    return " ".join(normalized_tokens)


def _serialize_document_payload(
    doc,
    lifecycle: DocumentLifecycleState | None,
) -> dict[str, object]:
    """Flatten NormalizedDocument + lifecycle metadata for display."""

    external_ref = getattr(doc.meta, "external_ref", None) or {}
    lifecycle_state = lifecycle.state if lifecycle else getattr(
        doc, "lifecycle_state", ""
    )
    ingestion_payload = {
        "state": lifecycle_state,
        "changed_at": getattr(lifecycle, "changed_at", None),
        "trace_id": getattr(lifecycle, "trace_id", ""),
        "run_id": getattr(lifecycle, "run_id", ""),
        "ingestion_run_id": getattr(lifecycle, "ingestion_run_id", ""),
        "reason": getattr(lifecycle, "reason", ""),
        "policy_events": list(getattr(lifecycle, "policy_events", []) or []),
    }

    payload = {
        "document_id": str(doc.ref.document_id),
        "workflow_id": doc.ref.workflow_id,
        "version": doc.ref.version or "",
        "collection_id": (
            str(doc.ref.collection_id) if doc.ref.collection_id else ""
        ),
        "document_collection_id": (
            str(doc.meta.document_collection_id)
            if doc.meta.document_collection_id
            else ""
        ),
        "title": doc.meta.title or "",
        "language": doc.meta.language or "",
        "tags": list(doc.meta.tags or []),
        "origin_uri": doc.meta.origin_uri or "",
        "external_ref_items": _dict_items(external_ref),
        "external_provider": external_ref.get("provider"),
        "external_id": external_ref.get("external_id"),
        "created_at": doc.created_at,
        "source": doc.source or "",
        "checksum": doc.checksum,
        "lifecycle_state": doc.lifecycle_state,
        "blob": _describe_blob(doc.blob),
        "download_url": reverse("documents:download", args=[doc.ref.document_id]),
        "ingestion": ingestion_payload,
        "meta": {
            "crawl_timestamp": doc.meta.crawl_timestamp,
            "pipeline_config": doc.meta.pipeline_config or {},
            "parse_stats": doc.meta.parse_stats or {},
        },
    }
    payload["search_blob"] = _build_search_blob(payload)
    return payload


def _filter_documents(documents: list[dict[str, object]], query: str) -> list[dict[str, object]]:
    normalized = str(query or "").strip().lower()
    if not normalized:
        return documents
    tokens = [token for token in normalized.split() if token]
    if not tokens:
        return documents
    filtered: list[dict[str, object]] = []
    for doc in documents:
        search_blob = doc.get("search_blob", "")
        if not isinstance(search_blob, str):
            continue
        blob = search_blob.lower()
        if all(token in blob for token in tokens):
            filtered.append(doc)
    return filtered


def _summaries_for_documents(documents: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    source_counter: Counter[str] = Counter()
    lifecycle_counter: Counter[str] = Counter()

    for doc in documents:
        source_counter[doc.get("source") or ""] += 1
        ingestion = doc.get("ingestion", {}) if isinstance(doc, Mapping) else {}
        lifecycle_counter[ingestion.get("state") or ""] += 1

    def _serialize(counter: Counter[str]) -> list[dict[str, object]]:
        entries = []
        for key, count in counter.items():
            label = key or "unknown"
            entries.append({"label": label, "count": count})
        entries.sort(key=lambda item: item["label"])
        return entries

    return {
        "sources": _serialize(source_counter),
        "lifecycle": _serialize(lifecycle_counter),
    }


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


def document_space(request):
    """Expose a developer workbench for inspecting document collections."""

    try:
        tenant_id, tenant_schema = _tenant_context_from_request(request)
    except TenantRequiredError as exc:
        return _tenant_required_response(exc)

    try:
        ensure_manual_collection(tenant_id)
    except Exception:
        logger.warning("document_space.ensure_manual_collection_failed", exc_info=True)

    tenant_obj = getattr(request, "tenant", None)
    if tenant_obj is None:
        try:
            tenant_obj = TenantContext.resolve_identifier(tenant_id)
        except Exception:
            tenant_obj = None

    collections_qs = DocumentCollection.objects.select_related("case")
    if tenant_obj is not None:
        collections_qs = collections_qs.filter(tenant=tenant_obj)
    else:
        collections_qs = collections_qs.filter(tenant__schema_name=tenant_schema)

    collections = list(collections_qs.order_by("name", "created_at"))
    serialized_collections = [_serialize_collection(item) for item in collections]

    requested_collection = request.GET.get("collection")
    selected_collection = _match_collection_identifier(
        collections, requested_collection
    )
    collection_warning = bool(requested_collection and not selected_collection)
    if selected_collection is None and collections:
        selected_collection = collections[0]
        requested_collection = str(selected_collection.id)

    limit = _parse_limit(request.GET.get("limit"))
    limit_options = [10, 25, 50, 100, 200]
    if limit not in limit_options:
        limit_options = sorted(set(limit_options + [limit]))
    latest_only = _parse_bool(request.GET.get("latest"), default=True)
    search_term = str(request.GET.get("q", "") or "").strip()
    cursor_param = str(request.GET.get("cursor", "") or "").strip()
    workflow_filter = str(request.GET.get("workflow", "") or "").strip()

    documents_payload: list[dict[str, object]] = []
    documents_error: str | None = None
    next_cursor: str | None = None

    if selected_collection:
        repository = _get_documents_repository()
        list_fn = (
            repository.list_latest_by_collection
            if latest_only
            else repository.list_by_collection
        )
        try:
            document_refs, next_cursor = list_fn(
                tenant_id=tenant_id,
                collection_id=selected_collection.collection_id,
                limit=limit,
                cursor=cursor_param or None,
                workflow_id=workflow_filter or None,
            )
        except Exception:
            logger.exception(
                "document_space.list_failed",
                extra={
                    "tenant_id": tenant_id,
                    "collection_id": str(selected_collection.collection_id),
                },
            )
            documents_error = (
                "Dokumentenliste konnte nicht geladen werden. Prüfe die Logs."
            )
        else:
            fetched_docs = []
            for ref in document_refs:
                try:
                    doc = repository.get(
                        tenant_id=tenant_id,
                        document_id=ref.document_id,
                        version=ref.version,
                        prefer_latest=latest_only or ref.version is None,
                        workflow_id=ref.workflow_id,
                    )
                except Exception:
                    logger.warning(
                        "document_space.document_fetch_failed",
                        exc_info=True,
                        extra={
                            "tenant_id": tenant_id,
                            "document_id": str(ref.document_id),
                        },
                    )
                    continue
                if doc is None:
                    continue
                fetched_docs.append(doc)

            lifecycle_map: dict[tuple[object, str], DocumentLifecycleState] = {}
            if fetched_docs:
                lifecycle_records = DocumentLifecycleState.objects.filter(
                    tenant_id=tenant_id,
                    document_id__in=[doc.ref.document_id for doc in fetched_docs],
                )
                lifecycle_map = {
                    (record.document_id, record.workflow_id or ""): record
                    for record in lifecycle_records
                }

            for doc in fetched_docs:
                lifecycle_key = (doc.ref.document_id, doc.ref.workflow_id)
                payload = _serialize_document_payload(
                    doc, lifecycle_map.get(lifecycle_key)
                )
                documents_payload.append(payload)

    filtered_documents = _filter_documents(documents_payload, search_term)
    summaries = _summaries_for_documents(filtered_documents)
    document_summary = {
        "fetched": len(documents_payload),
        "displayed": len(filtered_documents),
        "limit": limit,
    }

    selected_collection_payload = None
    if selected_collection:
        selected_collection_payload = next(
            (
                entry
                for entry in serialized_collections
                if entry["id"] == str(selected_collection.id)
            ),
            None,
        )

    query_defaults = {
        "collection": requested_collection or "",
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
            "collections": serialized_collections,
            "selected_collection": selected_collection_payload,
            "selected_collection_identifier": requested_collection or "",
            "documents": filtered_documents,
            "document_summary": document_summary,
            "summaries": summaries,
            "search_term": search_term,
            "latest_only": latest_only,
        "limit": limit,
            "limit_options": limit_options,
            "cursor": cursor_param,
            "next_cursor": next_cursor,
            "workflow_filter": workflow_filter,
            "documents_error": documents_error,
            "collection_warning": collection_warning,
            "has_collections": bool(collections),
            "query_defaults": query_defaults,
            "next_query": (
                {**query_defaults, "cursor": next_cursor} if next_cursor else None
            ),
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
