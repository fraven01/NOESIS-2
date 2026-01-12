from __future__ import annotations

import json
from uuid import uuid4

from django.shortcuts import render
from django.utils import timezone
from django.views.decorators.http import require_POST
from pydantic import ValidationError
from structlog.stdlib import get_logger

from ai_core.contracts import BusinessContext, ScopeContext
from ai_core.schemas import CrawlerRunRequest
from ai_core.services.crawler_runner import run_crawler_runner

logger = get_logger(__name__)


def _views():
    from theme import views as theme_views

    return theme_views


@require_POST
def crawler_submit(request):
    """Handle crawler form submission via HTMX."""
    views = _views()
    if not request.headers.get("HX-Request"):
        return views._json_error_response(
            "HTMX required",
            status_code=400,
            code="htmx_required",
        )

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

        scope = views._scope_context_from_request(request)
        tenant_id = scope.tenant_id
        tenant_schema = scope.tenant_schema or tenant_id
        case_id = str(data.get("case_id") or "").strip() or None
        trace_id = str(uuid4())
        user = getattr(request, "user", None)
        user_id = (
            str(user.pk)
            if user
            and getattr(user, "is_authenticated", False)
            and getattr(user, "pk", None) is not None
            else None
        )
        scope = ScopeContext(
            tenant_id=tenant_id,
            tenant_schema=tenant_schema,
            trace_id=trace_id,
            invocation_id=str(uuid4()),
            run_id=str(uuid4()),
            ingestion_run_id=payload.get("ingestion_run_id"),
            user_id=user_id,
        )
        business = BusinessContext(case_id=case_id)
        tool_context = scope.to_tool_context(business=business)
        meta = {
            "scope_context": scope.model_dump(mode="json", exclude_none=True),
            "business_context": business.model_dump(mode="json", exclude_none=True),
            "tool_context": tool_context.model_dump(mode="json", exclude_none=True),
        }

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
                lifecycle_store=views._resolve_lifecycle_store(),
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
    views = _views()
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
        scope = views._scope_context_from_request(request)
        tenant_id = scope.tenant_id
        tenant_schema = scope.tenant_schema or tenant_id
        case_id = request.POST.get("case_id") or request.headers.get("X-Case-ID")
        workflow_id = str(request.POST.get("workflow_id") or "").strip()
        if not workflow_id:
            workflow_id = "document-upload-manual"

        # Resolve manual collection for tenant
        manual_collection_id, _ = views._resolve_manual_collection(
            tenant_id, None, ensure=True
        )

        # Prepare metadata for upload
        user = getattr(request, "user", None)
        user_id = (
            str(user.pk)
            if user
            and getattr(user, "is_authenticated", False)
            and getattr(user, "pk", None) is not None
            else None
        )
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
            collection_id=manual_collection_id,  # Explicitly set collection
            workflow_id=workflow_id,
        )
        tool_context = scope.to_tool_context(business=business)
        meta = {
            "scope_context": scope.model_dump(mode="json", exclude_none=True),
            "business_context": business.model_dump(mode="json", exclude_none=True),
            "tool_context": tool_context.model_dump(mode="json", exclude_none=True),
        }

        # Check for RAG/Embedding enablement from UI checkbox
        # Checkbox sends "true" when checked, None when unchecked
        enable_embedding = request.POST.get("enable_embedding") == "true"

        # Build pipeline config if embedding is explicitly disabled
        metadata_obj = {}
        if not enable_embedding:
            # Store-Only mode: Skip chunking and embedding
            metadata_obj["pipeline_config"] = {"enable_embedding": False}
        # Note: If enable_embedding=True, we don't need to set it explicitly
        # since the dataclass default is now True (after Fix #1)

        # Upload document
        upload_response = handle_document_upload(
            upload=uploaded_file,
            metadata_raw=json.dumps(metadata_obj) if metadata_obj else None,
            meta=meta,
            idempotency_key=None,
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
