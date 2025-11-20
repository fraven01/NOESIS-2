"""Document download views."""

import time
from uuid import UUID

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
    try:
        doc_uuid = UUID(document_id)
    except ValueError:
        return error(400, "InvalidDocumentId", "document_id must be a valid UUID")

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
