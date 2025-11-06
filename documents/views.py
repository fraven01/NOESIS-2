"""Document download views."""

import os
import time
import email.utils as email_utils
from uuid import UUID

from django.http import FileResponse, HttpResponse, HttpResponseNotModified
from django.views.decorators.http import require_http_methods
from structlog.stdlib import get_logger

from ai_core.services import _get_documents_repository
from .utils import (
    detect_content_type,
    sanitize_filename,
    content_disposition,
    get_upload_file_path,
)
from .errors import error

logger = get_logger(__name__)


def _tenant_context_from_request(request) -> tuple[str, str]:
    """Extract tenant context (pattern from theme/views.py)."""
    tenant = getattr(request, "tenant", None)
    tenant_id: str | None = None
    tenant_schema: str | None = None

    if tenant is not None:
        tenant_id = getattr(tenant, "tenant_id", None)
        tenant_schema = getattr(tenant, "schema_name", None)

    if not tenant_id:
        tenant_id = "dev"
    if not tenant_schema:
        tenant_schema = "dev"

    return tenant_id, tenant_schema


@require_http_methods(["GET", "HEAD"])
def document_download(request, document_id: str):
    """
    Stream binary document with production features.

    Supports: GET, HEAD, ETag, Range, RFC 5987 filenames, CRLF protection.
    """
    start_time = time.time()
    tenant_id, tenant_schema = _tenant_context_from_request(request)

    logger.info(
        "documents.download.started",
        tenant_id=tenant_id,
        document_id=document_id,
        method=request.method,
    )

    try:
        # 1. Get document metadata via repository
        repo = _get_documents_repository()
        doc = repo.get(tenant_id, UUID(document_id))

        if not doc:
            duration_ms = int((time.time() - start_time) * 1000)
            logger.warning(
                "documents.download.not_found",
                tenant_id=tenant_id,
                document_id=document_id,
                duration_ms=duration_ms,
            )
            return error(404, "DocumentNotFound", f"Document {document_id} not found")

        # 2. Tenant isolation check
        if doc.ref.tenant_id != tenant_id:
            duration_ms = int((time.time() - start_time) * 1000)
            logger.error(
                "documents.download.tenant_mismatch",
                tenant_id=tenant_id,
                document_tenant_id=doc.ref.tenant_id,
                document_id=document_id,
                duration_ms=duration_ms,
            )
            return error(403, "TenantMismatch", "Access denied")

        # 3. Get physical file path (direct access for streaming)
        blob_path = get_upload_file_path(
            doc.ref.tenant_id, doc.ref.workflow_id, str(doc.ref.document_id)
        )

        if not blob_path.exists():
            logger.error(
                "documents.download.blob_missing",
                tenant_id=tenant_id,
                document_id=document_id,
                blob_path=str(blob_path),
            )
            return error(404, "BlobNotFound", "Document file not found on disk")

        # 4. File stats & ETag generation
        st = os.stat(blob_path)
        file_size = st.st_size
        last_modified = email_utils.formatdate(st.st_mtime, usegmt=True)
        weak_etag = f'W/"{file_size:x}-{int(st.st_mtime):x}"'

        # 5. Conditional requests (304 Not Modified)
        if_none_match = request.headers.get("If-None-Match")
        if_modified_since = request.headers.get("If-Modified-Since")

        if if_none_match == weak_etag:
            logger.info("documents.download.not_modified_etag", etag=weak_etag)
            resp = HttpResponseNotModified()
            resp["ETag"] = weak_etag
            resp["Last-Modified"] = last_modified
            return resp

        if if_modified_since:
            try:
                ims_dt = email_utils.parsedate_to_datetime(if_modified_since)
                if ims_dt.timestamp() >= st.st_mtime:
                    logger.info("documents.download.not_modified_time")
                    resp = HttpResponseNotModified()
                    resp["ETag"] = weak_etag
                    resp["Last-Modified"] = last_modified
                    return resp
            except (ValueError, TypeError):
                pass

        # 6. Content-Type detection
        content_type = detect_content_type(doc.blob, blob_path)

        # 7. Filename extraction & sanitization
        filename = doc.meta.title or f"{document_id}.bin"
        filename = sanitize_filename(filename)

        # 8. Range request handling (206 Partial Content)
        range_header = request.headers.get("Range")
        if range_header and request.method == "GET":
            import re

            match = re.match(r"bytes=(\d+)-(\d*)", range_header)
            if match:
                start = int(match.group(1))
                end = int(match.group(2)) if match.group(2) else file_size - 1

                # Validate range bounds
                if start < 0 or start >= file_size:
                    logger.warning(
                        "documents.download.range_invalid", range_start=start
                    )
                    resp = HttpResponse(status=416)
                    resp["Content-Range"] = f"bytes */{file_size}"
                    return resp

                # Cap end to file size
                end = min(end, file_size - 1)
                length = end - start + 1

                # Open and seek to start position
                f = open(blob_path, "rb")
                f.seek(start)

                response = FileResponse(f, content_type=content_type, status=206)
                response["Content-Length"] = length
                response["Content-Range"] = f"bytes {start}-{end}/{file_size}"
                response["Content-Disposition"] = content_disposition(filename)
                response["X-Content-Type-Options"] = "nosniff"
                response["Cache-Control"] = "private, max-age=3600"
                response["Vary"] = "Authorization, Cookie"
                response["ETag"] = weak_etag
                response["Last-Modified"] = last_modified
                response["Accept-Ranges"] = "bytes"

                duration_ms = int((time.time() - start_time) * 1000)
                logger.info(
                    "documents.download.partial_content",
                    tenant_id=tenant_id,
                    document_id=document_id,
                    duration_ms=duration_ms,
                    range_start=start,
                    range_end=end,
                    etag=weak_etag,
                )

                return response

        # 9. HEAD request (metadata only, no body)
        if request.method == "HEAD":
            response = HttpResponse()
            response["Content-Type"] = content_type
            response["Content-Length"] = file_size
            response["Content-Disposition"] = content_disposition(filename)
            response["X-Content-Type-Options"] = "nosniff"
            response["Cache-Control"] = "private, max-age=3600"
            response["Vary"] = "Authorization, Cookie"
            response["ETag"] = weak_etag
            response["Last-Modified"] = last_modified
            response["Accept-Ranges"] = "bytes"

            duration_ms = int((time.time() - start_time) * 1000)
            logger.info(
                "documents.download.head_completed",
                tenant_id=tenant_id,
                document_id=document_id,
                duration_ms=duration_ms,
                file_size=file_size,
                etag=weak_etag,
            )
            return response

        # 10. GET request (full content streaming)
        response = FileResponse(
            open(blob_path, "rb"),
            content_type=content_type,
        )
        response["Content-Length"] = file_size
        response["Content-Disposition"] = content_disposition(filename)
        response["X-Content-Type-Options"] = "nosniff"
        response["Cache-Control"] = "private, max-age=3600"
        response["Vary"] = "Authorization, Cookie"
        response["ETag"] = weak_etag
        response["Last-Modified"] = last_modified
        response["Accept-Ranges"] = "bytes"

        duration_ms = int((time.time() - start_time) * 1000)
        logger.info(
            "documents.download.streaming_started",
            tenant_id=tenant_id,
            document_id=document_id,
            duration_ms=duration_ms,
            file_size=file_size,
            content_type=content_type,
            etag=weak_etag,
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
