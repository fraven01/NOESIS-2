from __future__ import annotations

from uuid import UUID

from django.conf import settings
from django.http import HttpResponse
from django.shortcuts import render
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from structlog.stdlib import get_logger

from customers.tenant_context import TenantRequiredError
from documents.services.document_space_service import DocumentSpaceRequest
from theme.validators import DocumentSpaceQueryParams

logger = get_logger(__name__)


def _views():
    from theme import views as theme_views

    return theme_views


def document_space(request):
    """Expose a developer workbench for inspecting document collections."""
    views = _views()
    try:
        scope = views._scope_context_from_request(request)
    except TenantRequiredError as exc:
        return views._tenant_required_response(exc)
    tenant_id = scope.tenant_id
    tenant_schema = scope.tenant_schema or tenant_id

    tenant_obj = getattr(request, "tenant", None)
    if tenant_obj is None:
        try:
            tenant_obj = views.TenantContext.resolve_identifier(tenant_id)
        except Exception:
            tenant_obj = None

    query_params = DocumentSpaceQueryParams.model_validate(request.GET)
    requested_collection = query_params.collection
    limit = query_params.limit
    limit_options = [10, 25, 50, 100, 200]
    if limit not in limit_options:
        limit_options = sorted(set(limit_options + [limit]))
    latest_only = query_params.latest
    search_term = query_params.q
    cursor_param = query_params.cursor
    workflow_filter = query_params.workflow

    repository = views._get_documents_repository()
    params = DocumentSpaceRequest(
        requested_collection=requested_collection,
        limit=limit,
        latest_only=latest_only,
        cursor=cursor_param or None,
        workflow_filter=workflow_filter or None,
        search_term=search_term,
    )
    result = views.DOCUMENT_SPACE_SERVICE.build_context(
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
    views = _views()
    try:
        try:
            scope = views._scope_context_from_request(request)
        except TenantRequiredError as exc:
            return views._tenant_required_response(exc)
        tenant_id = scope.tenant_id
        tenant_schema = scope.tenant_schema or tenant_id

        tenant_obj = getattr(request, "tenant", None)
        if tenant_obj is None:
            try:
                tenant_obj = views.TenantContext.resolve_identifier(tenant_id)
            except Exception:
                tenant_obj = None

        query_params = DocumentSpaceQueryParams.model_validate(request.GET)
        case_id = request.GET.get("case_id") or request.headers.get("X-Case-ID")
        requested_collection = query_params.collection
        limit = query_params.limit
        limit_options = [10, 25, 50, 100, 200]
        if limit not in limit_options:
            limit_options = sorted(set(limit_options + [limit]))
        latest_only = query_params.latest
        show_retired = query_params.show_retired
        search_term = query_params.q
        cursor_param = query_params.cursor
        workflow_filter = query_params.workflow

        repository = views._get_documents_repository()
        params = DocumentSpaceRequest(
            requested_collection=requested_collection,
            limit=limit,
            latest_only=latest_only,
            cursor=cursor_param or None,
            workflow_filter=workflow_filter or None,
            search_term=search_term,
            show_retired=show_retired,
        )
        result = views.DOCUMENT_SPACE_SERVICE.build_context(
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
                "case_id": case_id,
                "search_term": search_term,
                "latest_only": latest_only,
                "limit": limit,
                "limit_options": limit_options,
                "cursor": cursor_param,
                "workflow_filter": workflow_filter,
                "default_embedding_profile": getattr(
                    settings, "RAG_DEFAULT_EMBEDDING_PROFILE", "standard"
                ),
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


@csrf_exempt
@require_POST
def document_reingest(request):
    """Trigger re-ingestion (chunking + embedding) for a document."""
    views = _views()
    document_id = request.POST.get("document_id") or request.GET.get("document_id")
    if not document_id:
        return HttpResponse(
            '<div class="text-xs text-red-600">Document ID required.</div>',
            status=400,
        )

    try:
        scope = views._scope_context_from_request(request)
    except TenantRequiredError as exc:
        return HttpResponse(
            f'<div class="text-xs text-red-600">{exc}</div>',
            status=400,
        )
    tenant_id = scope.tenant_id
    tenant_schema = scope.tenant_schema or tenant_id

    case_id = (
        request.POST.get("case_id")
        or request.headers.get("X-Case-ID")
        or request.GET.get("case_id")
    )
    collection_id = request.POST.get("collection_id") or None
    embedding_profile = request.POST.get("embedding_profile") or getattr(
        settings, "RAG_DEFAULT_EMBEDDING_PROFILE", "standard"
    )

    user = getattr(request, "user", None)
    user_id = (
        str(user.pk)
        if user
        and getattr(user, "is_authenticated", False)
        and getattr(user, "pk", None) is not None
        else None
    )

    from uuid import uuid4

    from ai_core.contracts import BusinessContext, ScopeContext
    from ai_core.services import start_ingestion_run

    scope = ScopeContext(
        tenant_id=tenant_id,
        tenant_schema=tenant_schema,
        trace_id=uuid4().hex,
        invocation_id=uuid4().hex,
        run_id=uuid4().hex,
        user_id=user_id,
    )
    business = BusinessContext(
        case_id=case_id,
        collection_id=collection_id,
    )
    tool_context = scope.to_tool_context(business=business)
    meta = {
        "scope_context": scope.model_dump(mode="json", exclude_none=True),
        "business_context": business.model_dump(mode="json", exclude_none=True),
        "tool_context": tool_context.model_dump(mode="json", exclude_none=True),
    }

    request_data = {
        "document_ids": [document_id],
        "embedding_profile": str(embedding_profile).strip() or "standard",
        "source": "document_explorer",
    }
    if collection_id:
        request_data["collection_id"] = collection_id

    response = start_ingestion_run(request_data, meta, idempotency_key=None)
    if response.status_code >= 400:
        detail = ""
        if isinstance(getattr(response, "data", None), dict):
            detail = response.data.get("detail") or response.data.get("error", {}).get(
                "message", ""
            )
        detail = detail or "Re-ingestion failed."
        return render(
            request,
            "theme/partials/_document_reingest_status.html",
            {"status": "error", "message": detail},
            status=response.status_code,
        )

    payload = getattr(response, "data", {}) or {}
    return render(
        request,
        "theme/partials/_document_reingest_status.html",
        {
            "status": "queued",
            "ingestion_run_id": payload.get("ingestion_run_id"),
            "trace_id": payload.get("trace_id"),
        },
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

    views = _views()
    try:
        scope = views._scope_context_from_request(request)
    except TenantRequiredError as exc:
        return HttpResponse(
            f'<div class="p-4 text-red-600 text-sm">{exc}</div>', status=400
        )
    tenant_id = scope.tenant_id
    tenant_schema = scope.tenant_schema or tenant_id

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

    views = _views()
    try:
        scope = views._scope_context_from_request(request)
    except TenantRequiredError as exc:
        return HttpResponse(
            f'<div class="p-4 text-red-600 text-sm">{exc}</div>', status=400
        )
    tenant_id = scope.tenant_id
    tenant_schema = scope.tenant_schema or tenant_id

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
