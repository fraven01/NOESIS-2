"""Document download views."""

import time

from django.http import FileResponse, HttpResponse, HttpResponseNotModified
from django.views.decorators.http import require_http_methods
from structlog.stdlib import get_logger

from ai_core.services import _get_documents_repository
from customers.tenant_context import TenantContext, TenantRequiredError
from .utils import (
    detect_content_type,
    sanitize_filename,
    content_disposition,
    extract_filename,
)
from .errors import error
from .access_service import DocumentAccessService
from .http_handlers import (
    HttpRangeHandler,
    CacheControlStrategy,
    CacheMetadata,
)

logger = get_logger(__name__)


def _tenant_context_from_request(request) -> tuple[str, str]:
    """Extract tenant context via canonical TenantContext helper."""

    tenant_obj = getattr(request, "tenant", None)
    if tenant_obj is None or (
        getattr(tenant_obj, "tenant_id", None) is None
        and getattr(tenant_obj, "schema_name", None) is None
    ):
        tenant_obj = TenantContext.from_request(
            request, allow_headers=True, require=False
        )

    tenant_schema = getattr(tenant_obj, "schema_name", None)
    if tenant_schema is None:
        tenant_schema = getattr(tenant_obj, "tenant_id", None)
    tenant_id = getattr(tenant_obj, "tenant_id", None) or tenant_schema

    if tenant_schema is None or tenant_id is None:
        raise TenantRequiredError("Tenant could not be resolved from request")

    return str(tenant_id), str(tenant_schema)


@require_http_methods(["GET", "HEAD"])
def document_download(request, document_id: str):
    """
    Stream binary document with production features.

    Supports: GET, HEAD, ETag, Range, RFC 5987 filenames, CRLF protection.

    Refactored to delegate to:
    - DocumentAccessService: Tenant isolation & file resolution
    - CacheControlStrategy: HTTP caching logic
    - HttpRangeHandler: Range request parsing
    """
    # Validate UUID format early (clean 400 instead of 500)
    from uuid import UUID

    try:
        if isinstance(document_id, str):
            doc_uuid = UUID(document_id)
        else:
            doc_uuid = document_id  # Already a UUID from URL converter
    except (ValueError, AttributeError):
        return error(
            400, "InvalidDocumentId", f"Invalid document ID format: {document_id}"
        )

    start_time = time.time()
    try:
        tenant_id, tenant_schema = _tenant_context_from_request(request)
    except TenantRequiredError as exc:
        return error(403, "TenantRequired", str(exc))

    logger.info(
        "documents.download.started",
        tenant_id=tenant_id,
        document_id=document_id,
        method=request.method,
    )

    try:
        # 1. Access check & file resolution (business logic layer)
        repo = _get_documents_repository()
        access_service = DocumentAccessService(repo)
        access_result, access_error = access_service.get_document_for_download(
            tenant_id, doc_uuid
        )

        if access_error:
            duration_ms = int((time.time() - start_time) * 1000)
            logger.warning(
                "documents.download.access_denied",
                tenant_id=tenant_id,
                document_id=document_id,
                error_code=access_error.error_code,
                duration_ms=duration_ms,
            )
            return error(
                access_error.status_code,
                access_error.error_code,
                access_error.message,
            )

        doc = access_result.document
        blob_path = access_result.blob_path
        file_size = access_result.file_size

        # 2. Cache metadata & conditional request handling
        cache_meta = CacheMetadata.from_file_stats(file_size, access_result.mtime)
        cache_strategy = CacheControlStrategy()

        if cache_strategy.should_return_304(
            cache_meta,
            if_none_match=request.headers.get("If-None-Match"),
            if_modified_since=request.headers.get("If-Modified-Since"),
        ):
            logger.info("documents.download.not_modified", etag=cache_meta.etag)
            resp = HttpResponseNotModified()
            resp["ETag"] = cache_meta.etag
            resp["Last-Modified"] = cache_meta.last_modified
            for key, value in cache_strategy.cache_headers().items():
                resp[key] = value
            return resp

        # 3. Content-Type & Filename
        content_type = detect_content_type(doc.blob, blob_path)
        filename = sanitize_filename(extract_filename(doc, document_id, content_type))

        # 4. Range request handling (206 Partial Content)
        range_header = request.headers.get("Range")
        if range_header and request.method == "GET":
            range_request = HttpRangeHandler.parse(range_header, file_size)

            if range_request is None:
                # Invalid range
                logger.warning("documents.download.range_invalid")
                resp = HttpResponse(status=416)
                resp["Content-Range"] = f"bytes */{file_size}"
                for key, value in cache_strategy.cache_headers().items():
                    resp[key] = value
                return resp

            # Open and seek to start position
            f = open(blob_path, "rb")
            f.seek(range_request.start)

            response = FileResponse(f, content_type=content_type, status=206)
            response["Content-Length"] = range_request.length
            response["Content-Range"] = range_request.content_range_header
            response["Content-Disposition"] = content_disposition(filename)
            response["X-Content-Type-Options"] = "nosniff"
            response["ETag"] = cache_meta.etag
            response["Last-Modified"] = cache_meta.last_modified
            for key, value in cache_strategy.cache_headers().items():
                response[key] = value

            duration_ms = int((time.time() - start_time) * 1000)
            logger.info(
                "documents.download.partial_content",
                tenant_id=tenant_id,
                document_id=document_id,
                duration_ms=duration_ms,
                range_start=range_request.start,
                range_end=range_request.end,
                etag=cache_meta.etag,
            )
            return response

        # 5. HEAD request (metadata only, no body)
        if request.method == "HEAD":
            response = HttpResponse()
            response["Content-Type"] = content_type
            response["Content-Length"] = file_size
            response["Content-Disposition"] = content_disposition(filename)
            response["X-Content-Type-Options"] = "nosniff"
            response["ETag"] = cache_meta.etag
            response["Last-Modified"] = cache_meta.last_modified
            for key, value in cache_strategy.cache_headers().items():
                response[key] = value

            duration_ms = int((time.time() - start_time) * 1000)
            logger.info(
                "documents.download.head_completed",
                tenant_id=tenant_id,
                document_id=document_id,
                duration_ms=duration_ms,
                file_size=file_size,
                etag=cache_meta.etag,
            )
            return response

        # 6. GET request (full content streaming)
        response = FileResponse(open(blob_path, "rb"), content_type=content_type)
        response["Content-Length"] = file_size
        response["Content-Disposition"] = content_disposition(filename)
        response["X-Content-Type-Options"] = "nosniff"
        response["ETag"] = cache_meta.etag
        response["Last-Modified"] = cache_meta.last_modified
        for key, value in cache_strategy.cache_headers().items():
            response[key] = value

        duration_ms = int((time.time() - start_time) * 1000)
        logger.info(
            "documents.download.streaming_started",
            tenant_id=tenant_id,
            document_id=document_id,
            duration_ms=duration_ms,
            file_size=file_size,
            content_type=content_type,
            etag=cache_meta.etag,
        )
        return response

    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        logger.exception(
            "documents.download.failed",
            tenant_id=tenant_id,
            document_id=document_id,
            error_type=type(e).__name__,
            duration_ms=duration_ms,
        )
        return error(500, "InternalError", str(e))


@require_http_methods(["GET", "HEAD"])
def asset_serve(request, document_id: str, asset_id: str):
    """
    Stream asset binary content from object store.

    Supports: GET, HEAD with standard caching headers.
    """
    from uuid import UUID
    import os

    start_time = time.time()

    # Validate UUID formats
    try:
        doc_uuid = UUID(document_id) if isinstance(document_id, str) else document_id
        asset_uuid = UUID(asset_id) if isinstance(asset_id, str) else asset_id
    except (ValueError, AttributeError):
        return error(400, "InvalidId", "Invalid document or asset ID format")

    try:
        tenant_id, tenant_schema = _tenant_context_from_request(request)
    except TenantRequiredError as exc:
        return error(403, "TenantRequired", str(exc))

    logger.info(
        "assets.serve.started",
        tenant_id=tenant_id,
        document_id=document_id,
        asset_id=asset_id,
        method=request.method,
    )

    try:
        # 1. Get document to access its assets
        repo = _get_documents_repository()
        doc = repo.get(tenant_id, doc_uuid)

        if doc is None:
            return error(404, "DocumentNotFound", f"Document {document_id} not found")

        # 2. Find the asset in the document
        assets = getattr(doc, "assets", []) or []
        target_asset = None
        for asset in assets:
            asset_ref = getattr(asset, "ref", None)
            if asset_ref:
                aid = getattr(asset_ref, "asset_id", None)
                if aid and str(aid) == str(asset_uuid):
                    target_asset = asset
                    break

        if target_asset is None:
            return error(
                404, "AssetNotFound", f"Asset {asset_id} not found in document"
            )

        # 3. Get blob info
        blob = getattr(target_asset, "blob", None)
        if blob is None:
            return error(404, "BlobNotFound", "Asset has no blob")

        blob_uri = None
        blob_type = None
        if isinstance(blob, dict):
            blob_uri = blob.get("uri")
            blob_type = blob.get("type")
        else:
            blob_uri = getattr(blob, "uri", None)
            blob_type = getattr(blob, "type", None)

        if blob_type == "external":
            # External asset - redirect to original URL
            external_uri = blob_uri
            logger.info(
                "assets.serve.redirect_external",
                asset_id=asset_id,
                external_uri=external_uri,
            )
            from django.http import HttpResponseRedirect

            return HttpResponseRedirect(external_uri)

        if not blob_uri:
            return error(404, "BlobUriMissing", "Asset blob URI is missing")

        # 4. Resolve blob path and serve
        from django.conf import settings

        base_path = getattr(settings, "OBJECT_STORE_BASE_PATH", ".ai_core_store")

        clean_uri = blob_uri
        if clean_uri and clean_uri.startswith("objectstore://"):
            clean_uri = clean_uri.replace("objectstore://", "", 1)
        
        # Ensure relative path for join
        clean_uri = clean_uri.lstrip("/")

        blob_path = os.path.join(base_path, clean_uri)

        if not os.path.exists(blob_path):
            logger.warning(
                "assets.serve.file_not_found",
                asset_id=asset_id,
                blob_path=blob_path,
            )
            return error(404, "FileNotFound", "Asset file not found on disk")

        # 5. Prepare response
        media_type = getattr(target_asset, "media_type", "application/octet-stream")
        file_size = os.path.getsize(blob_path)

        # HEAD request
        if request.method == "HEAD":
            response = HttpResponse()
            response["Content-Type"] = media_type
            response["Content-Length"] = file_size
            return response

        # GET request
        response = FileResponse(open(blob_path, "rb"), content_type=media_type)
        response["Content-Length"] = file_size
        response["X-Content-Type-Options"] = "nosniff"
        response["Cache-Control"] = "public, max-age=86400"

        duration_ms = int((time.time() - start_time) * 1000)
        logger.info(
            "assets.serve.success",
            tenant_id=tenant_id,
            document_id=document_id,
            asset_id=asset_id,
            duration_ms=duration_ms,
            file_size=file_size,
            media_type=media_type,
        )
        return response

    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        logger.exception(
            "assets.serve.failed",
            tenant_id=tenant_id,
            document_id=document_id,
            asset_id=asset_id,
            error_type=type(e).__name__,
            duration_ms=duration_ms,
        )
        return error(500, "InternalError", str(e))
