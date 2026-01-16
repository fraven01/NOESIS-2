"""Document download views."""

import hashlib
import json
import time
from collections.abc import Mapping
from uuid import UUID

from django.http import (
    FileResponse,
    HttpResponse,
    HttpResponseNotModified,
    JsonResponse,
)
from django.utils.dateparse import parse_datetime
from django.views.decorators.http import require_http_methods
from django_tenants.utils import schema_context
from structlog.stdlib import get_logger

from ai_core.contracts.scope import ScopeContext
from ai_core.ids import normalize_request
from ai_core.services import _get_documents_repository
from customers.tenant_context import TenantContext, TenantRequiredError
from documents.activity_service import ActivityTracker
from documents.authz import DocumentAuthzService
from documents.models import (
    Document,
    DocumentActivity,
    DocumentPermission,
    DocumentVersion,
)
from documents.serializers import DocumentSerializer
from profiles.models import UserProfile
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


RECENT_ACTIVITY_TYPES = (
    DocumentActivity.ActivityType.VIEW,
    DocumentActivity.ActivityType.DOWNLOAD,
)


def _resolve_tenant_context(request) -> tuple[str, str]:
    """Resolve tenant identifiers from scope context or request metadata."""
    scope = getattr(request, "scope_context", None)
    if isinstance(scope, ScopeContext):
        tenant_id = scope.tenant_id
        tenant_schema = scope.tenant_schema or tenant_id
        if tenant_id and tenant_schema:
            return str(tenant_id), str(tenant_schema)

    tenant_obj = getattr(request, "tenant", None)
    if tenant_obj is not None:
        tenant_schema = getattr(tenant_obj, "schema_name", None)
        if tenant_schema is None:
            tenant_schema = getattr(tenant_obj, "tenant_id", None)
        tenant_id = getattr(tenant_obj, "tenant_id", None) or tenant_schema
        if tenant_schema and tenant_id:
            return str(tenant_id), str(tenant_schema)

    scope = normalize_request(request)
    tenant_id = scope.tenant_id
    tenant_schema = scope.tenant_schema or tenant_id
    if tenant_id is None or tenant_schema is None:
        raise TenantRequiredError("Tenant could not be resolved from request")
    return str(tenant_id), str(tenant_schema)


def _resolve_request_user(request):
    user = getattr(request, "user", None)
    if user and getattr(user, "is_authenticated", False):
        return user
    return None


def _require_authenticated_user(request):
    user = _resolve_request_user(request)
    if user:
        return user, None
    return None, error(403, "AuthenticationRequired", "Authentication required")


def _user_is_tenant_admin(user) -> bool:
    try:
        profile = user.userprofile
    except UserProfile.DoesNotExist:
        return False
    return profile.is_active and profile.role == UserProfile.Roles.TENANT_ADMIN


def _log_download_activity(
    request,
    *,
    document_id,
    tenant_id: str,
    tenant_schema: str,
) -> None:
    user = _resolve_request_user(request)
    try:
        ActivityTracker.log(
            document_id=document_id,
            activity_type="DOWNLOAD",
            user=user,
            request=request,
            tenant_schema=tenant_schema,
        )
    except Exception:
        logger.warning(
            "documents.download.activity_log_failed",
            exc_info=True,
            tenant_id=tenant_id,
            document_id=str(document_id),
        )


def _serialize_document_version(version: DocumentVersion) -> dict[str, object]:
    return {
        "document_version_id": str(version.id),
        "document_id": str(version.document_id),
        "version_label": version.version_label,
        "sequence": version.sequence,
        "label_sequence": version.label_sequence,
        "is_latest": version.is_latest,
        "deleted_at": version.deleted_at.isoformat() if version.deleted_at else None,
        "created_at": version.created_at.isoformat() if version.created_at else None,
        "created_by_user_id": (
            str(version.created_by_id) if version.created_by_id else None
        ),
        "created_by_service_id": version.created_by_service_id,
    }


def _chunk_content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _chunk_identity(chunk: Mapping[str, object]) -> str:
    metadata = chunk.get("metadata")
    if isinstance(metadata, Mapping):
        parent_ref = metadata.get("parent_ref")
        if parent_ref:
            return f"parent_ref:{parent_ref}"
        parent_ids = metadata.get("parent_ids")
        if isinstance(parent_ids, list) and parent_ids:
            return f"parent_ids:{parent_ids[0]}"
    chunk_id = chunk.get("chunk_id") or chunk.get("id")
    return f"chunk_id:{chunk_id}"


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
        tenant_id, tenant_schema = _resolve_tenant_context(request)
    except TenantRequiredError as exc:
        return error(403, "TenantRequired", str(exc))

    user, auth_error = _require_authenticated_user(request)
    if auth_error:
        return auth_error

    tenant_obj = TenantContext.resolve_identifier(tenant_id, allow_pk=True)
    with schema_context(tenant_schema):
        access = DocumentAuthzService.user_can_access_document_id(
            user=user,
            document_id=doc_uuid,
            permission_type=DocumentPermission.PermissionType.DOWNLOAD,
            tenant=tenant_obj,
        )
    if not access.allowed:
        if access.reason == "not_found":
            return error(404, "DocumentNotFound", f"Document {document_id} not found")
        return error(403, "PermissionDenied", "Permission denied")

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

            _log_download_activity(
                request,
                document_id=doc_uuid,
                tenant_id=tenant_id,
                tenant_schema=tenant_schema,
            )

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

        _log_download_activity(
            request,
            document_id=doc_uuid,
            tenant_id=tenant_id,
            tenant_schema=tenant_schema,
        )

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
        tenant_id, tenant_schema = _resolve_tenant_context(request)
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


@require_http_methods(["GET"])
def recent_documents(request):
    """Return the most recent documents accessed by the authenticated user."""
    user = _resolve_request_user(request)
    if not user:
        return JsonResponse(
            {"detail": "Authentication required"},
            status=401,
        )

    try:
        tenant_id, tenant_schema = _resolve_tenant_context(request)
    except TenantRequiredError as exc:
        return error(403, "TenantRequired", str(exc))

    with schema_context(tenant_schema):
        activity_ids = (
            DocumentActivity.objects.filter(
                user=user, activity_type__in=RECENT_ACTIVITY_TYPES
            )
            .order_by("-timestamp")
            .values_list("document_id", flat=True)[:20]
        )

        document_ids = []
        seen = set()
        for document_id in activity_ids:
            if document_id in seen:
                continue
            seen.add(document_id)
            document_ids.append(document_id)
            if len(document_ids) >= 10:
                break

        documents = list(
            Document.objects.filter(id__in=document_ids).select_related(
                "created_by", "updated_by"
            )
        )
        documents_by_id = {doc.id: doc for doc in documents}
        ordered_documents = [
            documents_by_id[doc_id]
            for doc_id in document_ids
            if doc_id in documents_by_id
        ]

    return JsonResponse(
        DocumentSerializer(ordered_documents, many=True).data, safe=False
    )


@require_http_methods(["GET"])
def document_versions(request, document_id: str):
    try:
        doc_uuid = UUID(document_id)
    except (ValueError, AttributeError):
        return error(
            400, "InvalidDocumentId", f"Invalid document ID format: {document_id}"
        )

    try:
        tenant_id, tenant_schema = _resolve_tenant_context(request)
    except TenantRequiredError as exc:
        return error(403, "TenantRequired", str(exc))

    user, auth_error = _require_authenticated_user(request)
    if auth_error:
        return auth_error

    tenant_obj = TenantContext.resolve_identifier(tenant_id, allow_pk=True)
    with schema_context(tenant_schema):
        access = DocumentAuthzService.user_can_access_document_id(
            user=user,
            document_id=doc_uuid,
            permission_type=DocumentPermission.PermissionType.VIEW,
            tenant=tenant_obj,
        )
        if not access.allowed:
            if access.reason == "not_found":
                return error(
                    404, "DocumentNotFound", f"Document {document_id} not found"
                )
            return error(403, "PermissionDenied", "Permission denied")

        versions = list(
            DocumentVersion.objects.filter(document_id=doc_uuid).order_by("-sequence")
        )

    payload = {
        "document_id": str(doc_uuid),
        "versions": [_serialize_document_version(version) for version in versions],
    }
    return JsonResponse(payload)


@require_http_methods(["GET"])
def document_version_chunks(request, document_id: str, version_id: str):
    try:
        doc_uuid = UUID(document_id)
    except (ValueError, AttributeError):
        return error(
            400, "InvalidDocumentId", f"Invalid document ID format: {document_id}"
        )
    try:
        version_uuid = UUID(version_id)
    except (ValueError, AttributeError):
        return error(
            400, "InvalidDocumentVersionId", f"Invalid version ID format: {version_id}"
        )

    try:
        tenant_id, tenant_schema = _resolve_tenant_context(request)
    except TenantRequiredError as exc:
        return error(403, "TenantRequired", str(exc))

    user, auth_error = _require_authenticated_user(request)
    if auth_error:
        return auth_error

    tenant_obj = TenantContext.resolve_identifier(tenant_id, allow_pk=True)
    with schema_context(tenant_schema):
        access = DocumentAuthzService.user_can_access_document_id(
            user=user,
            document_id=doc_uuid,
            permission_type=DocumentPermission.PermissionType.VIEW,
            tenant=tenant_obj,
        )
        if not access.allowed:
            if access.reason == "not_found":
                return error(
                    404, "DocumentNotFound", f"Document {document_id} not found"
                )
            return error(403, "PermissionDenied", "Permission denied")

        version = DocumentVersion.objects.filter(
            id=version_uuid,
            document_id=doc_uuid,
        ).first()
        if version is None:
            return error(
                404,
                "DocumentVersionNotFound",
                f"Document version {version_id} not found",
            )

    try:
        limit = int(request.GET.get("limit", "1000"))
    except (TypeError, ValueError):
        limit = 1000
    limit = max(1, min(limit, 5000))
    try:
        offset = int(request.GET.get("offset", "0"))
    except (TypeError, ValueError):
        offset = 0
    offset = max(0, offset)

    from ai_core.rag.vector_client import get_default_client

    vector_client = get_default_client()
    chunks = vector_client.list_chunks_by_version(
        tenant_id=tenant_id,
        document_id=doc_uuid,
        document_version_id=version_uuid,
        limit=limit,
        offset=offset,
    )

    payload = {
        "document_id": str(doc_uuid),
        "document_version_id": str(version_uuid),
        "chunks": chunks,
    }
    return JsonResponse(payload)


@require_http_methods(["GET"])
def document_version_diff(request, document_id: str, version_id: str):
    compare_to = request.GET.get("other_version_id")
    if not compare_to:
        return error(
            400,
            "MissingVersionId",
            "Query parameter other_version_id is required",
        )

    try:
        doc_uuid = UUID(document_id)
    except (ValueError, AttributeError):
        return error(
            400, "InvalidDocumentId", f"Invalid document ID format: {document_id}"
        )
    try:
        left_version_uuid = UUID(version_id)
        right_version_uuid = UUID(compare_to)
    except (ValueError, AttributeError):
        return error(
            400,
            "InvalidDocumentVersionId",
            "Invalid version ID format",
        )

    try:
        tenant_id, tenant_schema = _resolve_tenant_context(request)
    except TenantRequiredError as exc:
        return error(403, "TenantRequired", str(exc))

    user, auth_error = _require_authenticated_user(request)
    if auth_error:
        return auth_error

    tenant_obj = TenantContext.resolve_identifier(tenant_id, allow_pk=True)
    with schema_context(tenant_schema):
        access = DocumentAuthzService.user_can_access_document_id(
            user=user,
            document_id=doc_uuid,
            permission_type=DocumentPermission.PermissionType.VIEW,
            tenant=tenant_obj,
        )
        if not access.allowed:
            if access.reason == "not_found":
                return error(
                    404, "DocumentNotFound", f"Document {document_id} not found"
                )
            return error(403, "PermissionDenied", "Permission denied")

        versions = {
            version.id
            for version in DocumentVersion.objects.filter(
                document_id=doc_uuid,
                id__in=[left_version_uuid, right_version_uuid],
            )
        }
        if left_version_uuid not in versions or right_version_uuid not in versions:
            return error(
                404,
                "DocumentVersionNotFound",
                "One or more document versions not found",
            )

    from ai_core.rag.vector_client import get_default_client

    vector_client = get_default_client()
    left_chunks = vector_client.list_chunks_by_version(
        tenant_id=tenant_id,
        document_id=doc_uuid,
        document_version_id=left_version_uuid,
    )
    right_chunks = vector_client.list_chunks_by_version(
        tenant_id=tenant_id,
        document_id=doc_uuid,
        document_version_id=right_version_uuid,
    )

    left_index: dict[str, dict[str, object]] = {}
    for chunk in left_chunks:
        identity = _chunk_identity(chunk)
        text = str(chunk.get("text") or "")
        left_index[identity] = {
            "chunk": chunk,
            "hash": _chunk_content_hash(text),
        }

    right_index: dict[str, dict[str, object]] = {}
    for chunk in right_chunks:
        identity = _chunk_identity(chunk)
        text = str(chunk.get("text") or "")
        right_index[identity] = {
            "chunk": chunk,
            "hash": _chunk_content_hash(text),
        }

    added = []
    removed = []
    changed = []
    for identity, right_entry in right_index.items():
        if identity not in left_index:
            added.append(right_entry["chunk"])
            continue
        left_entry = left_index[identity]
        if left_entry["hash"] != right_entry["hash"]:
            changed.append(
                {
                    "key": identity,
                    "before": left_entry["chunk"],
                    "after": right_entry["chunk"],
                }
            )

    for identity, left_entry in left_index.items():
        if identity not in right_index:
            removed.append(left_entry["chunk"])

    payload = {
        "document_id": str(doc_uuid),
        "from_version_id": str(left_version_uuid),
        "to_version_id": str(right_version_uuid),
        "added": added,
        "removed": removed,
        "changed": changed,
    }
    return JsonResponse(payload)


@require_http_methods(["POST"])
def share_document(request, document_id: str):
    """Grant document permission to another user."""
    user, auth_error = _require_authenticated_user(request)
    if auth_error:
        return auth_error

    try:
        tenant_id, tenant_schema = _resolve_tenant_context(request)
    except TenantRequiredError as exc:
        return error(403, "TenantRequired", str(exc))

    try:
        payload = json.loads(request.body or b"{}")
    except json.JSONDecodeError:
        return error(400, "InvalidJSON", "Invalid JSON payload")

    target_user_id = payload.get("user_id")
    if not target_user_id:
        return error(400, "MissingUserId", "user_id is required")

    permission_type = payload.get(
        "permission_type", DocumentPermission.PermissionType.VIEW
    )
    try:
        permission_type = DocumentPermission.PermissionType(permission_type).value
    except ValueError:
        return error(
            400,
            "InvalidPermissionType",
            f"Invalid permission_type: {permission_type}",
        )

    expires_at = None
    expires_at_raw = payload.get("expires_at")
    if expires_at_raw:
        expires_at = parse_datetime(str(expires_at_raw))
        if expires_at is None:
            return error(400, "InvalidExpiresAt", "expires_at must be ISO datetime")

    from django.apps import apps

    User = apps.get_model("users", "User")

    with schema_context(tenant_schema):
        document = (
            Document.objects.filter(id=document_id).select_related("created_by").first()
        )
        if document is None:
            return error(404, "DocumentNotFound", f"Document {document_id} not found")

        if document.created_by_id != user.id and not _user_is_tenant_admin(user):
            return error(403, "PermissionDenied", "Only owner can share document")

        target_user = User.objects.filter(pk=target_user_id).first()
        if target_user is None:
            return error(404, "UserNotFound", f"User {target_user_id} not found")

        permission, _ = DocumentPermission.objects.update_or_create(
            document=document,
            user=target_user,
            permission_type=permission_type,
            defaults={
                "granted_by": user,
                "expires_at": expires_at,
            },
        )

        try:
            ActivityTracker.log(
                document=document,
                activity_type=DocumentActivity.ActivityType.SHARE,
                user=user,
                case_id=document.case_id,
                metadata={
                    "shared_with": str(target_user.id),
                    "permission_type": permission_type,
                },
            )
        except Exception:
            logger.warning(
                "documents.share.activity_log_failed",
                exc_info=True,
                tenant_id=tenant_id,
                document_id=str(document_id),
            )

    return JsonResponse(
        {
            "permission_id": permission.id,
            "document_id": str(document.id),
            "user_id": str(target_user.id),
            "permission_type": permission_type,
            "expires_at": expires_at.isoformat() if expires_at else None,
        },
        status=201,
    )
